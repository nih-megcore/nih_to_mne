#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 13:15:15 2025

@author: jstout
"""

import nih2mne
import os.path as op, os
import glob
from nih2mne.make_meg_bids import (make_bids, process_meg_bids, _gen_taskrundict, 
                                   sessdir2taskrundict)
from nih2mne.GUI.templates.bids_creator_gui_control_functions import Args #, test_Args

dset_list = glob.glob(op.join(nih2mne.__path__[0], 'test_data', '20010101', '*.ds') )
bsight_elec = op.join(op.join(nih2mne.__path__[0], 'test_data', 'MRI', 'ABABABAB_elec.txt'))
bsight_mri = op.join(op.join(nih2mne.__path__[0], 'test_data', 'MRI', 'ABABABAB_refaced_T1w.nii.gz'))

                     
                      


def test_Args():
    opts = {'anonymize': True, 'meghash': 'None', 'bids_id': 'S01', 
            'bids_dir': '/tmp/BIDS', 'bids_session': '1', 
            'meg_dataset_list': dset_list, 'mri_none': False, 
            'mri_bsight': bsight_mri, 
            'mri_elec': bsight_elec, 
            'mri_brik': False, 'crop_zeros': True, 'include_empty_room': False, 
            'subjid_input':'TEST5', 'subjid':'TEST5', 
            'meg_input_dir':False}
     
            #'subjid_input' : False}
    args = Args(opts)
    make_bids(args)
    
    
    
    
    
    
    
# def _gen_taskrundict(meg_list=None):
#     '''
#     Helper function.  Separate out the functionality of run sorting and labelling

#     Parameters
#     ----------
#     meg_list : TYPE, optional
#         DESCRIPTION. The default is None.

#     Returns
#     -------
#     out_dict : TYPE
#         DESCRIPTION.

#     '''
#     dsets = sorted(meg_list)
    
#     #Return bids dictionary
#     task_list = [op.basename(i).split('_')[1] for i in dsets]
#     task_set = set(task_list)
    
#     # logger.info(f'Using {len(task_set)} tasks: {task_set}')
    
#     out_dict=dict()
#     for key in task_set:
#         idxs = [i for i,x in enumerate(task_list) if x==key]
#         sublist = [dsets[i] for i in idxs]
#         out_dict[key]=sublist
#     return out_dict
