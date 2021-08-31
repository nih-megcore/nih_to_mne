#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:08:48 2021

@author: jstout
"""

from ..bstags import txt_to_tag
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
        
    
        
    