#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:08:48 2021

@author: jstout
"""

from ..bstags import txt_to_tag, txt_to_tag_pd
from ..bstags import write_tagfile
import pytest 
import os
# =============================================================================
# Tests
# =============================================================================
test_dir = os.path.dirname(os.path.abspath(__file__))

def test_txt_to_tag():
    txtname=os.path.join(test_dir, 'test_txt.txt')
    
    test_tags = {'Nasion': "'Nasion' -1.5966 -123.1359 2.1943",
                 'Left Ear': "'Left Ear' 80.8481 -39.0185 -48.2379",
                 'Right Ear': "'Right Ear' -75.3443 -44.3777 -48.1843"}
    
    tags = txt_to_tag(txtname)
    assert tags == test_tags
    
def test_write_tagfile(tmpdir):
    txtname=os.path.join(test_dir, 'test_txt.txt')
    tags = txt_to_tag(txtname)
    
    name, ext = os.path.splitext(txtname)
    if ext != ".txt":
        name = txtname
    tagname = "{}.tag".format(name)
    tagname = os.path.basename(tagname)
    outfile = os.path.join(tmpdir, tagname)
    
    write_tagfile(tags, out_fname=outfile)
    
    with open(outfile) as w:
        test_test=w.readlines()
    
    test_tagfile = os.path.join(test_dir, 'test_txt.tag')
    with open(test_tagfile) as w:
        test_valid=w.readlines()
        
    assert test_test == test_valid
    
def test_exported_w_extra_spaces():
    txtname=os.path.join(test_dir, 'Exported_Electrodes.txt')
    tags = txt_to_tag(txtname)
    assert len(tags)==3
    assert 'Nasion' in tags
    assert 'Left Ear' in tags
    assert 'Right Ear' in tags
    assert 'Nasion ' not in tags  #Make sure that extra space not included
    
def test_alt_exported():
    '''Some users put the fiducial labels in the Electrode Type'''
    txtname=os.path.join(test_dir, 'alt_ExportedElectrodes.txt')
    tags=txt_to_tag_pd(txtname)
    assert tags['Nasion']=="'Nasion' -6.0344 -114.7126 -2.6041"
    assert tags['RPA']=="'RPA' -67.2147 -18.6125 -36.5009"
    assert tags['LPA']=="'LPA' 64.1748 -28.3103 -32.4693"
    
        
    
        
    