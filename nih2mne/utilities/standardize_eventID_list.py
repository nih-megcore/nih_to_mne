#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:06:33 2024

@author: jstout
"""

import pandas as pd
import mne
import os, os.path as op
import glob


def write_standardized_event_list(meg_dir, task=None):
    '''
    Read out the annotations from all runs of a task a make a standardized
    list. This function will write out a task_TASK_standardized_EventId_list.csv 
    file for each task set in the folder.

    Parameters
    ----------
    meg_dir : str 
        CTF meg dataset directory.
    task : str
        Specify task or leave empty and it will iterate over all tasks

    Returns
    -------
    None.

    '''
    if task==None:
        tmp_ = glob.glob(op.join(meg_dir, '*.ds'))
        task_list = [op.basename(i).split('_')[1] for i in tmp_]
    else:
        task_list = [task]
        
    for task in task_list:
        print(f'Generating unique entry ID list for {task}')
        task_dsets = glob.glob(op.join(meg_dir, f'*_{task}_*.ds'))
        sorted(task_dsets)
        
        id_list = []
        for dset in task_dsets:
            raw = mne.io.read_raw_ctf(dset, system_clock='ignore')
            id_list.extend(list(raw.annotations.description)[:])
        
        id_list = sorted(set(id_list)) #Sort unique entries
        id_list = pd.DataFrame(zip(id_list,range(1, 1+len(id_list))), 
                               columns=['ID_names', 'ID_vals'])    
        id_list.to_csv(op.join(meg_dir, f'task_{task}_standardized_EventId_list.csv')) 
        
def main_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-megdir', 
                        help='CTF acquisition folder - generally a date')
    parser.add_argument('-task',
                        help='''Specify a task to write the standardized IDs.
                        Leave blank to process all tasks''',
                        default=None)
    args = parser.parse_args()
    write_standardized_event_list(args.megdir, task=args.task)
                        
                        
    
    
    
