#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:08:48 2021

@author: jstout
"""

import pytest 
import os, os.path as op
import numpy as np
import mne

from nih2mne.utilities.trigger_utilities import correct_events_to_projector
from nih2mne.utilities.trigger_utilities import threshold_detect, append_conditions, parse_marks

from nih2mne.utilities.trigger_utilities import threshold_detect, check_analog_inverted, \
    detect_digital

###  UNHASH these eventually
# from ..trigger_utilities import correct_events_to_projector
# from ..trigger_utilities import threshold_detect, append_conditions, parse_marks

# from ..trigger_utilities import threshold_detect, check_analog_inverted, \
#     detect_digital

# =============================================================================
# Tests
# =============================================================================
test_data_dir='/home/jstout/src/nih_to_mne/nih2mne/test_data'
test_data_dir = op.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                   'test_data')
test_fname = op.join(test_data_dir, '20010101','ABABABAB_haririhammer_20010101_002.ds')
hh_logfile = op.join(test_data_dir, '20010101','ABABABAB_hh_cropped_log.log')



def test_inputs():
    response_l = threshold_detect(dsname=test_fname, mark='response_l', deadTime=0.5, 
                     derivThresh=0.1, ampThresh=0.1, channel='UADC006')
    
    response_r = threshold_detect(dsname=test_fname, mark='response_r', deadTime=0.5, 
                     derivThresh=0.1, ampThresh=0.1, channel='UADC007')    
    
    invert_boolean = check_analog_inverted(test_fname, ch_name='UADC016')
    projector = threshold_detect(dsname=test_fname, mark='projector', deadTime=0.5, 
                     derivThresh=0.1, ampThresh=0.1, channel='UADC016', invert=invert_boolean)
       
    ppt = detect_digital(test_fname, channel='UPPT001')
    
    dframe = append_conditions([projector, ppt])
    
    
    corrected = correct_events_to_projector(dframe, projector_channel='UADC016', 
                                    input_channel='UPPT001')
    
    
    
    print(corrected)
    
    
    
    
    
    
    # print(test_data_dir)
    # mne.io.read_raw_ctf(test_fname)
    # print(test_fname)



     
# def test_afni_tags_present():
#     neg_fname = op.join(testpath, 's1+orig.HEAD')
#     assert not _afni_tags_present(neg_fname)
#     pos_fname = op.join(testpath, 's2+orig.HEAD')
#     assert _afni_tags_present(pos_fname)

# @pytest.mark.skip(reason='Failing, but need to check github actions')
# def test_assess_available_localizers():
#     coords_val = assess_available_localizers(testpath)
#     correct_val = {'Nasion': [-10.578000000000003, -107.119, 21.007999999999996],
#                    'Left Ear': [70.422, -43.119, -22.99199999999999],
#                    'Right Ear': [-73.578, -41.119, -37.99199999999999]}
#     assert 'Nasion' in coords_val.keys()
#     assert 'Left Ear' in coords_val.keys()
#     assert 'Right Ear' in coords_val.keys()
#     correct =  np.array([correct_val['Nasion'], 
#                          correct_val['Left Ear'], 
#                          correct_val['Right Ear']]).round(2)
#     coords =   np.array([coords_val['Nasion'], 
#                          coords_val['Left Ear'], 
#                          coords_val['Right Ear']]).round(2)
#     assert np.allclose(coords,correct)
    
# def test_is_exported_bsight():
#     neg_fname = op.join(testpath,'README.txt') 
#     assert not _is_exported_bsight(neg_fname)
#     pos_fname = op.join(testpath, 's1.txt')
#     assert _is_exported_bsight(pos_fname)

# def test_is_tagfile():
#     neg_fname = op.join(testpath, 's1_mod.tag')
#     assert not _is_exported_tag(neg_fname)
#     pos_fname = op.join(testpath, 's1.tag')
#     assert _is_exported_tag(pos_fname)
        
