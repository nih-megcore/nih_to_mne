#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:50:24 2022

@author: jstout
"""
import os.path as op
from nih2mne.utilities.extract_csv_bids_entries import read_csv_entries, find_end_hdr



def test_find_end_hdr():
    path=op.join(op.dirname(__file__),'..','..', 'templates', 'bids_entry_template.csv')
    hdrline=find_end_hdr(path)
    assert hdrline==11

def test_extract_csv():
    fname=op.join(op.dirname(__file__),'test_entry.csv')
    dframe = read_csv_entries(fname)
    g_truth = ['bids_dir', 'subjid', 'meg_dir', 'afni_brik', 'brainsight_mri',
               'brainsight_electrodes', 'bids_session']
    for i in dframe.columns:
        assert i in g_truth
    assert len(dframe.columns) == len(g_truth)
