#!/usr/bin/env python3

from ..make_meg_bids import sessdir2taskrundict
from ..make_meg_bids import _check_multiple_subjects
from ..make_meg_bids import get_subj_logger
import logging 

import pytest 
import nih2mne
import os.path as op
from nih2mne.make_meg_bids import process_meg_bids
code_path = nih2mne.__path__[0]
data_path = op.join(code_path, 'test_data')



global logger
logger = get_subj_logger('TEST', log_dir='/tmp', loglevel=logging.WARN)

#def test_check_multiple_subjects():
    #Make this extensible - currently limited
    #Make a temp dir and link together the files from two different folders
#    indir=''
#    _check_multiple_subjects(indir)

    
def test_sessdir2taskrundict():
    #Test for single subject folder
    input_list=\
        ['TEST_ASSR_20001010_002.ds',
         'TEST_MMFAU_20001010_009.ds',
         'TEST_M100_20001010_007.ds',
         'TEST_rest_20001010_005.ds',
         'TEST_MMFAU_20001010_003.ds',
         'TEST_rest_20001010_012.ds',
         'TEST_MMFUA_20001010_004.ds',
         'TEST_rest_20001010_011.ds',
         'TEST_M100_20001010_006.ds',
         'TEST_MMFUA_20001010_010.ds',
         'TEST_ASSR_20001010_008.ds',
         'TEST_M100_20001010_001.ds']
    out_dict = sessdir2taskrundict(input_list, subject_in='TEST')
    
    g_truth = \
    {'rest': ['TEST_rest_20001010_005.ds',
              'TEST_rest_20001010_011.ds',
              'TEST_rest_20001010_012.ds'],
     'M100': ['TEST_M100_20001010_001.ds',
              'TEST_M100_20001010_006.ds',
              'TEST_M100_20001010_007.ds'],
     'MMFAU': ['TEST_MMFAU_20001010_003.ds', 'TEST_MMFAU_20001010_009.ds'],
     'MMFUA': ['TEST_MMFUA_20001010_004.ds', 'TEST_MMFUA_20001010_010.ds'],
     'ASSR': ['TEST_ASSR_20001010_002.ds', 'TEST_ASSR_20001010_008.ds']}
    
    assert out_dict == g_truth
    
    #Test for multiple acq in single folder
    input_list2=\
        ['TEST_ASSR_20001010_002.ds',
         'TEST_MMFAU_20001010_009.ds',
         'TEST_M100_20001010_007.ds',
         'TEST_rest_20001010_005.ds',
         'TEST_MMFAU_20001010_003.ds',
         'TEST_rest_20001010_012.ds',
         'TEST_MMFUA_20001010_004.ds',
         'TEST_rest_20001010_011.ds',
         'TEST_M100_20001010_006.ds',
         'TEST_MMFUA_20001010_010.ds',
         'TEST_ASSR_20001010_008.ds',
         'TEST_M100_20001010_001.ds',
         'SUBJ2_MMFAU_20001010_009.ds',
         'SUBJ2_M100_20001010_007.ds',
         'SUBJ2_rest_20001010_005.ds',
         'SUBJ2_MMFAU_20001010_003.ds',
         'SUBJ2_rest_20001010_012.ds',
         'SUBJ2_MMFUA_20001010_004.ds',
         'SUBJ2_rest_20001010_011.ds',
         'SUBJ2_M100_20001010_006.ds']
    out_dict2 = sessdir2taskrundict(input_list2, subject_in='TEST')
    assert out_dict2 == g_truth
    



def test_process_meg_bids(tmp_path):
    out_dir = tmp_path / "bids_test_dir"
    out_dir.mkdir()
    
    #Input setup
    meg_dir = op.join(data_path, '20010101')
    subject_in = 'ABABABAB'
    bids_dir = out_dir / 'bids_dir'
    bids_id = 'S01'
    session = '1'
    
    process_meg_bids(input_path = meg_dir, 
                     subject_in = subject_in,
                     bids_dir = bids_dir,
                     bids_id = bids_id, 
                     session = session, 
                     anonymize = False,
                     ignore_eroom=True, 
                     crop_trailing_zeros= False
                     )
    
    assert op.exists(bids_dir)
    for i in ['dataset_description.json','participants.json','participants.tsv','README']:
        assert op.exists(bids_dir / i)
    assert op.exists(bids_dir / f'sub-{bids_id}')
    assert op.exists(bids_dir / f'sub-{bids_id}' / 'ses-1' /'meg')
    dset_checklist = ['sub-S01_ses-1_task-haririhammer_run-01_meg.json',
                     'sub-S01_ses-1_task-airpuff_run-01_events.tsv',
                     'sub-S01_ses-1_task-airpuff_run-01_events.json',
                     'sub-S01_ses-1_task-airpuff_run-01_meg.ds',
                     'sub-S01_ses-1_coordsystem.json',
                     'sub-S01_ses-1_task-haririhammer_run-01_meg.ds',
                     'sub-S01_ses-1_task-haririhammer_run-01_channels.tsv',
                     'sub-S01_ses-1_task-airpuff_run-01_channels.tsv',
                     'sub-S01_ses-1_task-airpuff_run-01_meg.json',
                     'sub-S01_ses-1_task-haririhammer_run-01_events.tsv',
                     'sub-S01_ses-1_task-haririhammer_run-01_events.json']
    for i in dset_checklist:
        assert op.exists(bids_dir / f'sub-{bids_id}' / 'ses-1' /'meg' / i)

