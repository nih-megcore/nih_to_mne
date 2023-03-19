#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 19 2023

@author: jstout

This function needs some additional work to run:
    Add freesurfer subprocess - w/24 slurm job
    Need to pull the MEG files and create transforms and forward models for each run
    Add input checks to even submit the job (basically don't submit if already there)
    Add logging to BIDS/logdir/subjid or BIDS/derivatives/logdir (??)
    Fix n_jobs to reference os['...'] - slurm threads

"""
from mne.minimum_norm import make_inverse_operator, apply_inverse
import mne
import numpy as np
import os, os.path as op
import mne_bids
from mne_bids import BIDSPath
import numpy as np

n_jobs=10

import argparse
if __name__=='__main__':
    current_dir=os.getcwd()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject', help='subject id')
    parser.add_argument('-bids_dir', help='bids_directory',
                        default=op.join(current_dir, 'BIDS')
    parser.add_argument('-output_dirname', 
                        help='''The folder name for the output.  This will be
                        created and located in the:
                            bids_dir/derivatives/{output_dirname}'''
                        default='meg_proc')
                        )
    args=parser.parse_args()
    bids_dir = args.bids_dir
    subject = args.subject
    output_dirname = args.output_dirname
else:
    #For testing
    bids_dir = '/data/jstout/MOUS_TMP_GRP/BIDS'
    subject = 'sub-A2003'
    
deriv_dir = op.join(bids_dir, 'derivatives')
output_dir = op.join(deriv_dir, output_dirname)
subjects_dir = op.join(deriv_dir, 'freesurfer', 'subjects')
os.environ['SUBJECTS_DIR']=subjects_dir

bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='meg',
                     task=task_type)
anat_bids_path = BIDSPath(root=bids_dir, subject=subject, datatype='anat',
                          extension='.nii')
deriv_path = bids_path.copy().update(root=output_dir, check=False)
deriv_path.directory.mkdir(exist_ok=True, parents=True)

# =============================================================================
# 
# =============================================================================
bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
src_fname = deriv_path.copy().update(suffix='src', extension='.fif')
trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')

subjects_dir = mne_bids.read.get_subjects_dir()
fs_subject = 'sub-'+bids_path.subject
if not bem_fname.fpath.exists():
    bem = mne.make_bem_model(fs_subject, subjects_dir=f'{subjects_dir}', 
                             conductivity=[0.3])
    bem_sol = mne.make_bem_solution(bem)
    
    mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
else:
    bem_sol = mne.read_bem_solution(bem_fname)
    
if not src_fname.fpath.exists():
    src = mne.setup_source_space(fs_subject, spacing='oct6', add_dist='patch',
                         subjects_dir=subjects_dir)
    src.save(src_fname.fpath, overwrite=True)
else:
    src = mne.read_source_spaces(src_fname.fpath)

if not trans_fname.fpath.exists():
    trans = mne_bids.read.get_head_mri_trans(bids_path, extra_params=dict(system_clock='ignore'),
                                        t1_bids_path=anat_bids_path, fs_subject=fs_subject, 
                                        fs_subjects_dir=subjects_dir)
    mne.write_trans(trans_fname.fpath, trans, overwrite=True)
else:
    trans = mne.read_trans(trans_fname.fpath)
if fwd_fname.fpath.exists():
    fwd = mne.read_forward_solution(fwd_fname)
else:
    fwd = mne.make_forward_solution(raw.info, trans, src, bem_sol, eeg=False, 
                                    n_jobs=n_jobs)
    mne.write_forward_solution(fwd_fname.fpath, fwd)
