#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:50:24 2022

@author: jstout
"""
import os.path as op
from nih2mne.utilities.extract_csv_bids_entries import (read_csv_entries, 
                                                        find_end_hdr,
                                                        make_cmd,
                                                        make_swarm_file)
                                                        

test_csv_filled=op.join(op.dirname(__file__),'test_entry.csv')
bids_entry_template = op.join(op.dirname(__file__),'..','..', 'templates', 'bids_entry_template.csv')

def test_find_end_hdr():
    hdrline=find_end_hdr(bids_entry_template)
    assert hdrline==11

def test_extract_csv():
    dframe = read_csv_entries(test_csv_filled)
    g_truth = ['bids_dir', 'subjid', 'meg_dir', 'afni_brik', 'brainsight_mri',
               'brainsight_electrodes', 'bids_session']
    for i in dframe.columns:
        assert i in g_truth
    assert len(dframe.columns) == len(g_truth)

def test_make_cmd():
    dframe = read_csv_entries(test_csv_filled)
    cmd = make_cmd(dframe.loc[0])
    assert cmd=='make_meg_bids.py -bids_dir bids_dir -subjid test1 -meg_input_dir /data/test -afni_brik /data/mri/mri.BRIK'
    cmd = make_cmd(dframe.loc[1])
    assert cmd=='make_meg_bids.py -bids_dir /tmp/test -subjid test2 -meg_input_dir /tmp -mri_bsight /test/tmp.nii -mri_bsight_elec /test/tmp.txt -bids_session 2'

def test_make_swarm_file():
    swarm=make_swarm_file(test_csv_filled, write=False)
    assert swarm[0] == 'make_meg_bids.py -bids_dir bids_dir -subjid test1 -meg_input_dir /data/test -afni_brik /data/mri/mri.BRIK\n'
    assert swarm[1] == 'make_meg_bids.py -bids_dir /tmp/test -subjid test2 -meg_input_dir /tmp -mri_bsight /test/tmp.nii -mri_bsight_elec /test/tmp.txt -bids_session 2\n'
    
    
