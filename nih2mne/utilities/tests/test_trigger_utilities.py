#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:58:45 2025

@author: jstout
"""

import pytest
from nih2mne.utilities.trigger_utilities import (
    correct_to_projector,
    samples_to_trig_timing,
)
import nih2mne
import pandas as pd
import os.path as op
import numpy as np

topdir = nih2mne.__path__[0]


class _DatasetWithSampleTimes:
    def __init__(self):
        self.sample_indices = []

    def getTimePt(self, sample_idx):
        self.sample_indices.append(sample_idx)
        return sample_idx / 1000


def test_samples_to_trig_timing_passes_scalar_indices():
    dataset = _DatasetWithSampleTimes()
    digital_vector = np.array([0, 11, 0, 23])

    result = samples_to_trig_timing(dataset, digital_vector)

    assert dataset.sample_indices == [1, 3]
    assert all(type(idx) is int for idx in dataset.sample_indices)
    np.testing.assert_array_equal(
        result,
        np.array([[0.001, 11], [0.003, 23]]),
    )


def test_samples_to_trig_timing_handles_no_positive_edges():
    dataset = _DatasetWithSampleTimes()

    result = samples_to_trig_timing(dataset, np.zeros(4))

    assert dataset.sample_indices == []
    assert result.shape == (0, 2)


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
    
