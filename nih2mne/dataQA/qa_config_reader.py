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
    try:
        raw_evts_counts = pd.DataFrame(dframe.description.value_counts().items(), columns=['Condition', 'Raw'])
    except:
        return None
    
    # raw_conditions = list(raw_evts_counts.keys()) #list(dframe.description.unique())
    
    if task_type not in qa_dict.keys():
        task_qa_dict = {'':None}
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
    return out_dframe.__repr__()
            
            
            
