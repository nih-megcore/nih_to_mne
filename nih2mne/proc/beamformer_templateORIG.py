#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:37:49 2026

@author: jstout
"""

import mne
import os, os.path as op
import numpy as np
import glob
import pickle
import copy 
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs, apply_lcmv_raw
from mne.datasets import fetch_fsaverage
import pandas as pd
from mne.beamformer import apply_lcmv_cov    
import nibabel as nb
import mne_bids
from mne_bids import BIDSPath
import pathlib
import copy
from mne.preprocessing import annotate_muscle_zscore

n_jobs = -1

#%% Define bids_paths
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root')
    parser.add_argument('-subject')
    parser.add_argument('-run')
    parser.add_argument('-session')
    parser.add_argument('-task')
    parser.add_argument('-project')
    
    args = parser.parse_args()
    
    bids_root = args.bids_root
    subject = args.subject
    run = args.run
    ses = args.session
    task_type = args.task
    project = args.project
    
# bids_root = '/fast/KIDDER/BIDS'
# subject = 'NT01'
# run = '01'
# ses = '1'
# task_type = 'megann'
# project = 'beamDecode'

#%% Proc params 
epo_tmin = -0.25
epo_tmax = 0.25
epo_baseline = None # [None,None] # [-0.1, 0] # None
f_min = 0.5
f_max = 110
# resample = 600 
er_run = '01'

#Rejection Thresholds
reject_dict = dict(mag=3e-12)

cov_cv = 5
cov_method = 'shrunk'

beam_reg = 0
beam_ori = 'max-power'

#Conditions of interest - list
conds_OI = ['locface','locscene']
#Contrasts of interest - list - ['Cond1/Cond2']
contrasts_OI = ['locface/locscence']
#Contrast type : percent / logratio / ...
contrasts_type = 'percent'

overwrite_anats = False
overwrite_preproc = False
overwrite_beam = True

#%% Define Paths

bids_path = BIDSPath(root=bids_root, 
                     subject=subject,
                     run=run, 
                     session=ses, 
                     task=task_type, 
                     datatype='meg')

anat_bids_path = BIDSPath(root=bids_root, subject=subject, datatype='anat',
                          extension='.nii.gz', acquisition = None, 
                          suffix = 'T1w', session = ses)

deriv_path = bids_path.copy().update(root= bids_path.root / 'derivatives',
                                     check=False)
preprocessing_path = deriv_path.copy().update(root = deriv_path.root / 'preproc')
preprocessing_path.root.mkdir(parents=True, exist_ok=True)

output_path = deriv_path.copy().update(root = deriv_path.root / project)
output_path.root.mkdir(parents=True, exist_ok=True)

subjects_dir = deriv_path.root / 'freesurfer' / 'subjects'
fs_subject = 'sub-'+bids_path.subject

raw_fname = bids_path.copy() 
noise_fname = bids_path.copy().update(task='EmptyRoom', run=er_run)

bem_fname = preprocessing_path.copy().update(suffix='bem', extension='.fif', 
                                             run=None)
src_fname = preprocessing_path.copy().update(suffix='src', extension='.fif', 
                                             run=None)
trans_fname = preprocessing_path.copy().update(suffix='trans',extension='.fif', 
                                               run=None)
fwd_fname = preprocessing_path.copy().update(suffix='fwd', extension='.fif')


raw_load_opts = dict(system_clock = 'ignore', clean_names =True, 
                          preload=True)
raw = mne.io.read_raw_ctf(raw_fname.fpath, **raw_load_opts)
noise_raw = mne.io.read_raw_ctf(noise_fname.fpath, **raw_load_opts)

#Add data cropping here to remove zeros




#%% Load or create any MRI related items
if not bem_fname.fpath.exists():
    mne.bem.make_watershed_bem(fs_subject, subjects_dir=subjects_dir, overwrite=True)
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
    mne.write_forward_solution(fwd_fname.fpath, fwd, overwrite=True)
    
#%% Preproc Raw data


_musc_annot = annotate_muscle_zscore(raw, threshold=4, ch_type='mag', min_length_good=0.1, 
                       filter_freq=(110, 140), n_jobs=n_jobs, verbose=None)
raw.set_annotations(raw.annotations + _musc_annot[0])


raw.filter(f_min, f_max, n_jobs=n_jobs)
noise_raw.filter(f_min, f_max, n_jobs=n_jobs)

raw.notch_filter([60,120,180], n_jobs=n_jobs)
noise_raw.notch_filter([60,120,180], n_jobs=n_jobs)

# create epochs
evts, evtsid = mne.events_from_annotations(raw)
epo = mne.Epochs(raw, 
                 preload=True, 
                 tmin=epo_tmin, 
                 tmax=epo_tmax, 
                 baseline=epo_baseline, 
                 events=evts,
                 event_id = evtsid, 
                 reject = reject_dict)


#%% Perform the beamformer actions
def normalize_beamfilt_ori(fwd, filters):
    src = fwd['src']
    src_anat_ori = np.vstack([i['nn'][i['vertno']] for i in src])
    ori_flip = np.sign(np.sum(filters['max_power_ori'] * src_anat_ori, axis=1))
    filters['weights'] *= ori_flip[:,np.newaxis]
    filters['max_power_ori'] *= ori_flip[:,np.newaxis]
    
# compute ICA 

# Create cov
full_cov = mne.compute_covariance(epo[conds_OI], 
                                  cv=cov_cv, 
                                  method=cov_method)
full_cov_bidspath = output_path.copy().update(suffix='cov', 
                                              extension='.fif',
                                              description=f'EPOf{f_min}f{f_max}'
                                              )
full_cov.save(full_cov_bidspath.fpath, overwrite=overwrite_beam)

noise_cov = mne.compute_raw_covariance(noise_raw)
noise_cov_bidspath = output_path.copy().update(suffix='cov', 
                                              extension='.fif',
                                              description=f'NOISEf{f_min}f{f_max}'
                                              )
noise_cov.save(noise_cov_bidspath.fpath, overwrite=overwrite_beam)

# Make wts
filters = make_lcmv(epo.info, fwd, full_cov, noise_cov= noise_cov,
                    reg=beam_reg, pick_ori=beam_ori) #rank=epo_rank,
normalize_beamfilt_ori(fwd, filters)
filters_bidspath = output_path.copy().update(description=f'LCMVf{f_min}f{f_max}',
                                             extension='.h5', 
                                             suffix='lcmv')
filters.save(filters_bidspath.fpath, overwrite=overwrite_beam)
# make_bids_json(filters_bidspath


stc_basename = output_path.copy().update(extension='.stc')
                                         
def stc_proc(epo, taskname, filters, return_abs=False, save_ave=True,
             save_epo=False, overwrite=True):
    _tmp_stcs = apply_lcmv_epochs(epo[taskname], filters)
    _tmp_stcs = [i.apply_baseline([None,0]) for i in _tmp_stcs]
    _ave_stcs = copy.deepcopy(_tmp_stcs[0])
    _ave_stcs._data = np.mean(np.array([i._data for i in _tmp_stcs]), axis=0)
    
    if save_epo:
        pass  #set this up to iterate through epochs and save
    
    
    if save_ave:
        stc_bidspath = stc_basename.copy().update(description=f'{taskname}',
                                                  processing=f'f{f_min}f{f_max}')
        _ave_stcs.save(stc_bidspath)
    return _tmp_stcs, _ave_stcs


for cond in conds_OI:
    _ = stc_proc(epo, cond, filters, save_ave=True, save_epo=True)
    


#%%





# #%% Time series src
# plot_opts = dict(subject = fs_subject, 
#               subjects_dir=subjects_dir, 
#               hemi='both', 
#               surface = 'white',
#               clim={'kind':'percent',
#                     'pos_lims': [80, 90, 100]})



# stcs_locface, stc_ave_locface = src_postproc(epo, 'locface', filters)
# stcs_locscene, stc_ave_locscene = src_postproc(epo, 'locscene', filters)

# stc_ave_locface.plot(**plot_opts)


# #%% Locface / locscene
# cov_locface = mne.compute_covariance(epo['locface'], cv = cov_cv, 
#                                      method=cov_method)
# stc_pow_locface = apply_lcmv_cov(cov_locface, filters)

# cov_locscene = mne.compute_covariance(epo['locscene'], cv = cov_cv, 
#                                      method=cov_method)
# stc_pow_locscene = apply_lcmv_cov(cov_locscene, filters)

# stc_pow_locface.plot(**plot_opts)
# stc_pow_locscene.plot(**plot_opts)

# stc_diff = copy.deepcopy(stc_pow_locface)
# stc_diff._data = np.log(stc_pow_locface._data / stc_pow_locscene._data) #/ stc_pow_locscene._data
# stc_diff.plot(**plot_opts)




# #%% TODO 
# Make all orientations fit inner product of cortical orientation --
# Match all orientations across runs? 



# label_vert_data = get_full_label_ts(label, stcs)
# label_pca = _pca(label_vert_data)

# # Identify the in-phase data to determine the flips
# _tmp = np.dot(label_pca, label_vert_data.T)
# flips = _tmp<0
# label_vert_data[flips,:] *= -1
# return label_vert_data




# #%% Multi session analysis

# decod_mat = np.stack([i._data for i in stim_stcs])








