#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:27:45 2026

@author: jstout
"""

import pytest
from nih2mne.make_meg_bids import _proc_meg_bids, _proc_mri_bids
import nih2mne


test_data = nih2mne.test_data()
assert test_data.is_present()

from mne_bids import BIDSPath
import os, os.path as op


#%%

def test_proc_meg_bids(tmpdir):
    out_bids_path = op.join(tmpdir, 'BIDS')
    _bids_path = BIDSPath(subject='TEST', session='1', task='airpuff', 
                          datatype='meg', suffix='meg', run='01',
                          root = out_bids_path, extension='.ds'
                          )
    
    meg_fname = str(test_data.meg_airpuff_fname)
    bids_path = BIDSPath
    anonymize = False
    ignore_eroom = True
    crop_trailing_zeros = False

    _proc_meg_bids(meg_fname=meg_fname, bids_path=_bids_path,
                        anonymize=False, tmpdir=None, ignore_eroom=True, 
                        crop_trailing_zeros=False, 
                       )

tmpdir = '/tmp/BIDStest'  # REMOVE ~~~~~~~~~~~~~~~~~~~~~~~~    

def test_proc_mri_bids(tmpdir):
    out_bids_path = op.join(tmpdir, 'BIDS')
    _bids_path = BIDSPath(subject='TEST', session='1',  
                          datatype='anat', extension='.nii.gz',
                          root = out_bids_path, suffix='T1w') #, extension='.ds'
    _bids_path.fpath
    
    _proc_mri_bids(t1_bids_path= _bids_path, temp_dir=tmpdir, 
                   anonymize=False, 
                   mri_bsight=str(test_data.mri_nii), 
                   mri_bsight_elec=str(test_data.bsight_elec), 
                   mri_brik=False, input_id='test')
    
    assert op.exists(_bids_path.fpath)
    json_fname = str(_bids_path.fpath).replace('.nii.gz','.json')
    assert op.exists(json_fname)
    

