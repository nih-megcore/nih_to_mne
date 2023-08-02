#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:40:26 2023

@author: jstout
"""

import os, os.path as op
import mne
import mne_bids
import numpy as np
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)
from mne.beamformer import apply_lcmv_epochs, apply_lcmv
from mne.beamformer import make_lcmv

if 'n_jobs' in os.environ:
    n_jobs = os.environ['n_jobs']
else:
    n_jobs = 4
    

def return_beamformer(epochs=None,
                  bem=None,
                  fwd=None,
                  regularization=0.05,
                  return_stc_epochs=False
                  ):
    '''
    

    Parameters
    ----------
    epochs : TYPE, optional
        DESCRIPTION. The default is None.
    bem : bem_sol, required
        Boundary element solution. Typically 1 layer
    trans : 
        MNE transformation matrix
    src : TYPE, optional
        DESCRIPTION. The default is None.
    hemi : TYPE, optional
        DESCRIPTION. The default is None.
    subjects_dir : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    dip : TYPE
        DESCRIPTION.

    '''
    src = fwd['src']
    ref_chans = epochs.copy().pick_types(ref_meg=True).ch_names #!!!Hack
    chans+=ref_chans #Necessary for 3rd order gradient
    epochs = epochs.copy().pick_channels(chans)
    epochs.info.normalize_proj()
    evoked = epochs.average()
    fwd = fwd.copy().pick_channels(chans)
    
    noise_cov = mne.compute_covariance(epochs, method='shrunk', cv=5, n_jobs=n_jobs,
                                       tmax=0)
    data_cov = mne.compute_covariance(epochs, method='shrunk', cv=5, n_jobs=n_jobs, 
                                     tmin=0)

    # src_anat_ori = src[src_idx]['nn'][src[src_idx]['vertno'],:]
    src_anat_ori = np.vstack([i['nn'][i['vertno']] for i in src])

    #Make hemi-field beamformer in another function
    filters = make_lcmv(epochs.info, fwd, data_cov=data_cov, noise_cov=noise_cov, reg=regularization, 
                        pick_ori='max-power', weight_norm='nai')
    
    #Adjust filters to align with anatomy
    ori_flip = np.sign(np.sum(filters['max_power_ori'] * src_anat_ori, axis=1))
    filters['weights'] *= ori_flip[:,np.newaxis]
    filters['max_power_ori'] *= ori_flip[:,np.newaxis]
    
    
    if return_stc_epochs==True:
        return apply_lcmv_epochs(epochs, filters=filters)
    elif return_stc_epochs=='cond_ave':
        tmp_1 = apply_lcmv_epochs(epochs['2'], filters=filters)
        tmp_2 = apply_lcmv_epochs(epochs['3'], filters=filters)
        return {'2':tmp_1, '3':tmp_2}
    else:
        return apply_lcmv(evoked, filters=filters)

def mripreproc(bids_path=None,
               t1_bids_path=None, 
               deriv_path=None):
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

    Returns
    -------
    fwd : 
        MNE surface based forward model.

    '''
    
    bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
    fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
    src_fname = deriv_path.copy().update(suffix='src', extension='.fif')
    trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')
    
    raw_fname = bids_path.copy().update(suffix='meg')
    raw = mne_bids.read_raw_bids(raw_fname, extra_params=dict(system_clock='ignore'))
    subjects_dir = mne_bids.read.get_subjects_dir()
    fs_subject = 'sub-'+bids_path.subject
    
    os.makedirs( deriv_path.directory, exist_ok=True)
    
    if fwd_fname.fpath.exists():
        fwd = mne.read_forward_solution(fwd_fname)
        return fwd
    if not bem_fname.fpath.exists():
        bem = mne.make_bem_model(fs_subject, subjects_dir=subjects_dir, 
                                 conductivity=[0.3])
        bem_sol = mne.make_bem_solution(bem)
        
        mne.write_bem_solution(bem_fname.fpath, bem_sol, overwrite=True)
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
                                            t1_bids_path=t1_bids_path, fs_subject=fs_subject, 
                                            fs_subjects_dir=subjects_dir)
        mne.write_trans(trans_fname.fpath, trans, overwrite=True)
    else:
        trans = mne.read_trans(trans_fname.fpath)
    fwd = mne.make_forward_solution(raw.info, trans, src, bem_sol, eeg=False, 
                                    n_jobs=n_jobs)
    mne.write_forward_solution(fwd_fname.fpath, fwd)
    return fwd



# =============================================================================
# Data Entry part     
# =============================================================================
bids_root = '/fast/BIDS_HV_V1/bids'
subject='ON02747'
task='airpuff'
run='01'
session='01'
PROJECT_NAME='TEST'


# =============================================================================
# Setup paths
# =============================================================================
bids_path = mne_bids.BIDSPath(root=bids_root, task=task, 
                              run=run, session=session, subject=subject)
t1_bids_path=bids_path.copy().update(datatype='anat', task=None, suffix='T1w',
                                     session=session, subject=subject, 
                                     extension='.nii.gz')
deriv_path=bids_path.copy().update(root=op.join(bids_root, 'derivatives', 
                                                PROJECT_NAME),
                                   check=False)
os.environ['SUBJECTS_DIR']=op.join(op.dirname(deriv_path.root), 'freesurfer',
                                   'subjects')

# =============================================================================
# Process MRI
# =============================================================================
fwd = mripreproc(bids_path=bids_path,
               t1_bids_path=t1_bids_path, 
               deriv_path=deriv_path)
bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
bem_sol = mne.read_bem_solution(bem_fname)



# =============================================================================
# Get beamformer
# =============================================================================


return_beamformer(epochs=epo,
                  bem=bem_sol,
                  fwd=fwd,
                  regularization=0.05,
                  return_stc_epochs=True
                  )




# =============================================================================
# Names
# =============================================================================
entities = mne_bids.get_entities_from_fname(mripath)

anat_bidspath.update(
        session=entities['session'],
        run=entities['run'])

_tmp['anat']=bids_path.copy().update(datatype='anat',extension='.nii')
if not os.path.exists(_tmp['anat'].fpath):
    _tmp['anat']=bids_path.copy().update(datatype='anat',extension='.nii.gz')

# populate the temporary dictionary _tmp with all the filenames

_tmp['raw_rest']=meg_rest_raw
if eroompath!=None:
    _tmp['raw_eroom']=meg_er_raw

rest_deriv = rest_derivpath.copy().update(extension='.fif')
if eroompath!=None:
    eroom_deriv = eroom_derivpath.copy().update(extension='.fif')
  
_tmp['rest_filt']=rest_deriv.copy().update(processing='filt')
if eroompath!=None:
    _tmp['eroom_filt']=eroom_deriv.copy().update(processing='filt')

_tmp['rest_epo']=rest_deriv.copy().update(suffix='epo')
if eroompath!=None:
    _tmp['eroom_epo']=eroom_deriv.copy().update(suffix='epo')

_tmp['rest_csd']=rest_deriv.copy().update(suffix='csd', extension='.h5')
if eroompath!=None:
    _tmp['eroom_csd']=eroom_deriv.copy().update(suffix='csd', extension='.h5')
     
_tmp['rest_fwd']=rest_deriv.copy().update(suffix='fwd') 
_tmp['rest_trans']=rest_deriv.copy().update(suffix='trans')
_tmp['bem'] = deriv_path.copy().update(suffix='bem', extension='.fif')
_tmp['src'] = deriv_path.copy().update(suffix='src', extension='.fif')

_tmp['lcmv'] = deriv_path.copy().update(suffix='lcmv',
                                         run=meg_rest_raw.run,
                                         extension='h5')
fooof_dir = deriv_path.directory / \
    deriv_path.copy().update(datatype=None, extension=None).basename

# Cast all bids paths to paths and save as dictionary
path_dict = {key:str(i.fpath) for key,i in _tmp.items()}

# Additional non-bids path files
path_dict['parc'] = op.join(self.subjects_dir, 'morph-maps', 
                       f'sub-{subject}-fsaverage-morph.fif') 





# =============================================================================
# 
# =============================================================================
mne_bids.find_matching_paths(
    root,
    subjects=None,
    sessions=None,
    tasks=None,
    acquisitions=None,
    runs=None,
    processings=None,
    recordings=None,
    spaces=None,
    splits=None,
    descriptions=None,
    suffixes=None,
    extensions=None,
    datatypes=None,
    check=False,
