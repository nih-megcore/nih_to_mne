#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:35:23 2025

@author: jstout
"""

from nih2mne.utilities.data_crop_wrapper import * 
import pytest 
import numpy as np

def test_get_termtime():
    dat = np.zeros(2000)
    dat[:1700] = np.random.uniform([1700])+1
    idx, t_val = get_term_time(dat, sfreq=100.0)
    assert idx==1700, 'Did not find the correct stop index'
    assert t_val==17.0, 'Did not compute the correct stop time'

