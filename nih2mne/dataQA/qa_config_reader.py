#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:04:47 2024

@author: jstout
"""

import yaml
from yaml import Loader
# import os, os.path as op
import pandas as pd

def read_yml(fname):
    with open(fname, 'r') as f:
        dat = yaml.load(f, Loader=Loader)
    return dat

def qa_dataset(raw, task_type=None, qa_dict=None):
    dframe = pd.DataFrame(raw.annotations)
    raw_evts_counts = pd.DataFrame(dframe.description.value_counts().items(), columns=['Condition', 'Raw'])
    
    # raw_conditions = list(raw_evts_counts.keys()) #list(dframe.description.unique())
    
    if task_type not in qa_dict.keys():
        task_qa_dict = {}
    else:
        task_qa_dict = qa_dict[task_type]
        
    tmp_ = pd.DataFrame.from_dict(task_qa_dict, orient='index').reset_index()
    expected_evts_counts = tmp_.rename(columns={'index':'Condition', 0:'Expected'})
    out_dframe = pd.merge(raw_evts_counts, expected_evts_counts,  on='Condition', how='outer')
    
    for idx, row in out_dframe.iterrows():
        if pd.isna(row.Expected):
            out_dframe.loc[idx, 'Status']='Not Tested'
            continue
        if (pd.isna(row.Raw)):
            out_dframe.loc[idx, 'Status'] = 'Fail'
            continue
        if row.Expected <= row.Raw:
            out_dframe.loc[idx, 'Status'] = 'Pass'
        else:
            out_dframe.loc[idx, 'Status'] = 'Fail'
            
    out_dframe.set_index('Condition',drop=True, inplace=True)
    return out_dframe
            
            
            
            
   
        
    
    


#%% Testing
def test_yaml_read():
    from nih2mne.dataQA.qa_config_reader import read_yml
    import nih2mne
    fname = op.join(nih2mne.__path__[0], 'dataQA', 'config_template.yml')
    dat = read_yml(fname)    
    assert dat['airpuff']['stim']==425
    assert dat['airpuff']['missingstim'] ==75
    assert 'haririhammer' in dat.keys()
    
