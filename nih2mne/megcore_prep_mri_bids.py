#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:58:05 2023

@author: jstout
"""

import mne, mne_bids
from mne_bids import BIDSPath
import os, os.path as op, glob
# from nih2mne import get_njobs

n_jobs=10  #<<<<<<<<Fix 

desc = '''Preprocess all MRI related processing.  \
    To process the whole bids tree, run the following command from the bids_root.  \
        
        for dset in $(find sub-* -name '*.ds'); do megcore_prep_mri_bids.py -filename $(pwd)/${dset}; done
        
'''

    
def check_mri(t1_bids_path):
    for acq in ['MPRAGE', None]:
        t1_bids_path.update(acquisition=acq, extension='.nii')
        if op.exists(t1_bids_path.fpath):
            break
        t1_bids_path.update(extension='.nii.gz')
        if op.exists(t1_bids_path.fpath):
            break
    #Redundant test possibly
    if not op.exists(t1_bids_path.fpath):
        raise ValueError('Can not determine T1 path')
    return t1_bids_path



def mripreproc(bids_path=None,
               t1_bids_path=None, 
               deriv_path=None, 
               surf=True):
    '''
    Generate the typical MRI inputs for processing MEG and saved to the 
    derivatives folder:
        bem_sol, fwd, src, trans

    Parameters
    ----------
    bids_path : mne_bids.BIDSpath, required
        bids_path to raw bids data. The default is None.
    t1_bids_path : mne_bids.BIDSpath, required
        bids_path for anatomy. The default is None.
    deriv_path : mne_bids.BIDSpath, required
        Output derivatives bids_path. The default is None.
    surf : Boolean
        If True (default), use the cortical surface.  Else use the volumetric
        sampling at 5mm

    Returns
    -------
    fwd : 
        MNE surface based forward model.

    '''

        
    bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
    if surf==True:
        fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
        src_fname = deriv_path.copy().update(suffix='src', extension='.fif')
    else:
        fwd_fname = deriv_path.copy().update(suffix='volfwd', extension='.fif')
        src_fname = deriv_path.copy().update(suffix='volsrc', extension='.fif')
    
    trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')
    
    raw_fname = bids_path.copy().update(suffix='meg')
    raw = mne.io.read_raw_ctf(raw_fname, system_clock='ignore')
    subjects_dir = mne_bids.read.get_subjects_dir()
    fs_subject = 'sub-'+bids_path.subject
    
    if not op.exists(op.join(subjects_dir, 'sub-'+bids_path.subject, 'bem','watershed')):
        mne.bem.make_watershed_bem(fs_subject, subjects_dir=subjects_dir)    
    if fwd_fname.fpath.exists():
        fwd = mne.read_forward_solution(fwd_fname)
        return fwd
    if not bem_fname.fpath.exists():
        bem = mne.make_bem_model(fs_subject, subjects_dir=subjects_dir, 
                                 conductivity=[0.3])
        bem_sol = mne.make_bem_solution(bem)
        
        mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
    else:
        bem_sol = mne.read_bem_solution(bem_fname)
        
    if surf==True:
        if not src_fname.fpath.exists():
            src = mne.setup_source_space(fs_subject, spacing='oct6', add_dist='patch',
                                 subjects_dir=subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
    else:
        if not src_fname.fpath.exists():
            src = mne.setup_volume_source_space(fs_subject, pos=5.0, bem=bem_sol,
                                 subjects_dir=subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)        
    
    if not trans_fname.fpath.exists():
        trans = mne_bids.read.get_head_mri_trans(bids_path, extra_params=dict(system_clock='ignore'),
                                            t1_bids_path=t1_bids_path, fs_subject=fs_subject, 
                                            fs_subjects_dir=subjects_dir)
        mne.write_trans(trans_fname.fpath, trans, overwrite=True)
    else:
        trans = mne.read_trans(trans_fname.fpath)
    fwd = mne.make_forward_solution(raw.info, trans, src, bem_sol, eeg=False, 
                                    n_jobs=n_jobs)
    mne.write_forward_solution(fwd_fname.fpath, fwd)
    return fwd


amp_thresh = dict(mag=4000e-15)
def preproc(bids_path=None,
            deriv_path=None):
    '''
    Generate the epochs for input to the dipole model.
    Saves filtered raw data and epochs to the derivatives folder

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath, required
        Bids path to original data. The default is None.
    deriv_path : mne_bids.BIDSPath, required
        Bids path for outputs. The default is None.

    Returns
    -------
    epo : mne.Epochs
        Standard and Deviant stim epochs (also saved to deriv_path).

    '''
    raw = read_raw_bids(bids_path=bids_path, extra_params={'system_clock':'ignore'})
    raw.load_data()
    raw.filter(30, 50, n_jobs=n_jobs)
    raw.notch_filter([60], n_jobs=n_jobs)
    raw.resample(300, n_jobs=n_jobs)
    
    raw_out_fname = deriv_path.copy().update(extension='.fif', suffix='meg',
                                            processing='filt')    
    raw.save(raw_out_fname.fpath, overwrite=True)

    evts, evtsid = mne.events_from_annotations(raw)
    # id_dev, id_std = evtsid['3'], evtsid['4']
    epo = mne.Epochs(raw, evts, # event_id=['3','4'],
                     tmin=-1.2,
                     tmax=1.2,
                     baseline = (-0.5, 0),
                     preload=True,
                     reject=amp_thresh)
    
    epo_out_fname = deriv_path.copy().update(extension='.fif', suffix='epo', 
                                            processing='clean')
    epo.save(epo_out_fname.fpath, overwrite=True) 
    
    cov = mne.compute_covariance(epo)
    cov_out_fname = deriv_path.copy().update(extension='.fif',suffix='cov',
                                             processing='clean')
    cov.save(cov_out_fname.fpath, overwrite=True)
    return epo


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-filename',
                        help='CTF dataset path.')
    parser.add_argument('-project',
                        help='''Output folder name.  This will be generated in the
                        bids derivatives folder''',
                        default='nihmeg')
    parser.add_argument('-volume',
                        help='''Perform the MRI processing in volume space''',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    filename = args.filename
    project_name = args.project
    proc_surf = not args.volume
    
    bids_path = mne_bids.get_bids_path_from_fname(filename)
    deriv_path = bids_path.copy().update(root=op.join(bids_path.root, 'derivatives', project_name), check=False)
    deriv_path.directory.mkdir(parents=True, exist_ok=True)
    subjects_dir=op.join(bids_path.root, 'derivatives', 'freesurfer', 'subjects')
    os.environ['SUBJECTS_DIR']=subjects_dir

    tmp_t1_path = BIDSPath(root=bids_path.root,
                      session=bids_path.session,
                      subject=bids_path.subject,
                      datatype='anat',
                      suffix='T1w',
                      extension='.nii.gz',
                      check=True)    
    t1_bids_path = check_mri(tmp_t1_path)
    mripreproc(bids_path, t1_bids_path, deriv_path, surf=proc_surf)
