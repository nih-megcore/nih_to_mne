#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:58:45 2025

@author: jstout
"""

import pytest
from nih2mne.utilities.trigger_utilities import correct_to_projector
import nih2mne
import pandas as pd
import os.path as op
import numpy as np

topdir = nih2mne.__path__[0]

def test_correct_to_projector():
    dframe_fname = op.join(topdir, 'utilities','tests', 
                           'test_initial_trigger_flanker_dframe.csv')
    dframe = pd.read_csv(dframe_fname)
    test_dframe_fname = op.join(topdir, 'utilities','tests', 
                           'test_projCorrected_flanker_dframe.csv')
    test_output_dframe = pd.read_csv(test_dframe_fname)
    
    
    event_list = ['fixation','right_con','right_incon','left_incon','left_con']
    
    out_dframe = correct_to_projector(dframe, event_list=event_list)
    
    evt_selection = dframe.query(f'condition in {event_list}')
    unaffected_selection = dframe.query(f'condition not in {event_list}')

    o_evt_selection = out_dframe.query(f'condition in {event_list}')
    o_unaffected_selection = out_dframe.query(f'condition not in {event_list}')

    #Check that all corrected events stay in order        
    assert np.all(evt_selection.condition.values == o_evt_selection.condition.values)
    
    #Check that all non-corrected events stay in order and have preserved time
    assert np.all(unaffected_selection.condition.values == o_unaffected_selection.condition.values)
    assert np.all(unaffected_selection.onset.values == o_unaffected_selection.onset.values)
    
    #Check all corrected onsets match test values
    assert np.allclose(test_output_dframe.onset.values,out_dframe.onset.values, atol=0.0001)
    
