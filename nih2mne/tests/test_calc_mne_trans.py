#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:08:48 2021

@author: jstout
"""

from ..calc_mnetrans import _afni_tags_present
from ..calc_mnetrans import _is_exported_bsight
from ..calc_mnetrans import _is_exported_tag

import pytest 
import os, os.path as op
# =============================================================================
# Tests
# =============================================================================

testpath = op.join(os.path.dirname(os.path.abspath(__file__)),
                   'calc_mne_trans_testfiles')

# Need more tests 
# Assess available localizers

     
def test_afni_tags_present():
    neg_fname = op.join(testpath, 's1+orig.HEAD')
    assert not _afni_tags_present(neg_fname)
    pos_fname = op.join(testpath, 's2+orig.HEAD')
    assert _afni_tags_present(pos_fname)
              
# def test_assess_available_localizers():
#     testpath = '/home/jstout/src/nih_to_mne/nih2mne/tests/calc_mne_trans_testfiles'
#     assess_available_localizers(testpath)
    
def test_is_exported_bsight():
    neg_fname = op.join(testpath,'README.txt') 
    assert not _is_exported_bsight(neg_fname)
    pos_fname = op.join(testpath, 's1.txt')
    assert _is_exported_bsight(pos_fname)

def test_is_tagfile():
    neg_fname = op.join(testpath, 's1_mod.tag')
    assert not _is_exported_tag(neg_fname)
    pos_fname = op.join(testpath, 's1.tag')
    assert _is_exported_tag(pos_fname)
        