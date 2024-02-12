#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:38:31 2023

@author: stoutjd and amaia
"""
import os, os.path as op
import mne
import numpy as np
import subprocess
import shutil

def get_term_time(raw, channel_idx=100):
    '''
    the index of 20 consecutive zeros is used as an identifier to a terminated run (when user hit "abort")
    
    '''
    try:
        idx_crop = np.where((np.diff(np.convolve(np.ones(20),raw._data[channel_idx,:]==0)))==1)[0][0]
        return idx_crop/raw.info['sfreq']
    except:
        return False

def return_cropped_ds(fname):
    '''
    Load the raw dataset, check the time where a set of zeros are present
    Pass the time to newDs to crop the datset into a temp subfolder in the 
    directory of the original file.

    Parameters
    ----------
    fname : str
        Path String.

    Raises
    ------
    RuntimeError
        If it cannot find a termination point to the scan it returns error.

    Returns
    -------
    fname_out : str
        Path to new cropped dataset.

    '''
    assert fname.endswith('.ds')
    install_check()
    raw = mne.io.read_raw_ctf(fname, system_clock='ignore', preload=True)
    
    crop_time = get_term_time(raw)
    if crop_time == False:
        raise RuntimeError('Could not find a terminated timepoint')
    base = op.abspath(op.dirname(fname))
    f_ = op.basename(fname)
    outdir = op.join(base, 'bids_prep_temp','tmp_cropped')
    if not op.exists(outdir): os.mkdir(outdir)
    fname_out = op.join(outdir, f_) 
    cmd = f'newDs -f -time 0 {str(crop_time)} {fname} {fname_out}'
    subprocess.run(cmd)
    return fname_out
    

def install_check():
    try:
        assert shutil.which('newDs') != None
    except:
        raise BaseException('''The CTF tools are not installed on this system
                            if on biowulf do module load ctf and rerun''')
    

