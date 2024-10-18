#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:04:47 2024

@author: jstout
"""

import yaml
from yaml import Loader
import os, os.path as op
from nih2mne.dataQA.qa_config_reader import read_yml
import nih2mne


def test_yaml_read():
    fname = op.join(nih2mne.__path__[0], 'dataQA', 'config_template.yml')
    dat = read_yml(fname)    
    assert dat['airpuff']['stim']==425
    assert dat['airpuff']['missingstim'] ==75
    assert 'haririhammer' in dat.keys()
    
