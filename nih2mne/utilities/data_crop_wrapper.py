#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:38:31 2023

@author: stoutjd
"""
import mne
import numpy as np
import subprocess
fname = '/home/stoutjd/data/BIDS_ZOOM/sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-sternberg_run-01_meg.ds'
raw = mne.io.read_raw_ctf(fname, system_clock='ignore', preload=True)


def get_term_time(raw):
    '''
    the index of 20 consecutive zeros is used as an identifier to a terminated run (when user hit "abort")
    
    '''
    try:
        idx_crop = np.where((np.diff(np.convolve(np.ones(20),raw._data[100,:]==0)))==1)[0][0]
        return idx_crop/raw.info['sfreq']
    except:
        return False

fname_in = 'test1'
fname_out = 'test2'
crop_time = 100
crop_time = str(crop_time)

cmd = f'newDs -f -time 0 {crop_time} {fname_in} {fname_out}'

subprocess.run(cmd.split())
