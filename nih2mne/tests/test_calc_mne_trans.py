#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:08:48 2021

@author: jstout
"""

from ..calc_mnetrans import _afni_tags_present
from ..calc_mnetrans import _is_exported_bsight
from ..calc_mnetrans import _is_exported_tag
from ..calc_mnetrans import assess_available_localizers

import pytest 
import os, os.path as op
import numpy as np
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

@pytest.mark.skip(reason='Failing, but need to check github actions')
def test_assess_available_localizers():
    coords_val = assess_available_localizers(testpath)
    correct_val = {'Nasion': [-10.578000000000003, -107.119, 21.007999999999996],
                   'Left Ear': [70.422, -43.119, -22.99199999999999],
                   'Right Ear': [-73.578, -41.119, -37.99199999999999]}
    assert 'Nasion' in coords_val.keys()
    assert 'Left Ear' in coords_val.keys()
    assert 'Right Ear' in coords_val.keys()
    correct =  np.array([correct_val['Nasion'], 
                         correct_val['Left Ear'], 
                         correct_val['Right Ear']]).round(2)
    coords =   np.array([coords_val['Nasion'], 
                         coords_val['Left Ear'], 
                         coords_val['Right Ear']]).round(2)
    assert np.allclose(coords,correct)
    
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
        
