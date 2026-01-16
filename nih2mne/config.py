#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 09:43:39 2026

_get_default_options: provides the standard dictionary of options
initialize_defaults: used to CREATE or LOAD the defaults file
_update_null_entries: Compare the defaults from _get_default_options 
    and add default lines if not present in current defaults.yml




@author: jstout
"""
import os, os.path as op
import yaml

def _get_default_options():
    'This is to provide the standard input dictionary'
    BIDS_GEN = {'bids_root': op.expanduser(f'~/BIDS'), 
                'bids_session': None,
                'bids_session_list': [None],
                'meg_dir': None, 
                'mri_dir': None, 
                'coreg_type': 'Brainsight', 
                'anonymize': 'N', 
                'crop_zeros': 'N',
                'emptyroom': 'N'
                }
    defaults = {'BIDS_gen': BIDS_GEN}    
    return defaults


def initialize_defaults():
    # Initialize the locations of the trigger files
    TRIG_FILE_LOC = op.expanduser(f'~/megcore/trigproc')
    LOG_FILE_LOC = op.expanduser(f'~/megcore/logs/')
    DEFAULTS_FILE_LOC = op.expanduser(f'~/megcore/defaults.yml')
    if not op.exists(TRIG_FILE_LOC):  os.makedirs(TRIG_FILE_LOC)
    if not op.exists(DEFAULTS_FILE_LOC): 
        _initialize_defaults_file()
    
    # Load and synchronize defaults with updated null keys
    defaults = _load_defaults(DEFAULTS_FILE_LOC)
    return defaults

def _initialize_defaults_file(megcore_dir=op.expanduser(f'~/megcore')):
    'Write out the default settings into a yml file in the users megcore folder'
    defaults = _get_default_options()
    with open(op.join(megcore_dir, 'defaults.yml'), 'w') as f:
        yaml.dump(defaults, f, 
                  sort_keys=False, 
                  default_flow_style=False)
        
def _get_defaults_fname():
    #As a function, this can have a switch to pull from an ENV variable
    megcore_dir=op.expanduser(f'~/megcore')
    defaults_fname = op.join(megcore_dir, 'defaults.yml')
    return defaults_fname

def _synchronize_defaults_file(input_defaults=None):
    null_defaults = _get_default_options()
    #null_defaults is a two layer dictionary
    #navigate two layers and write null values for any defaults not present
    need_write = False
    for major_key in null_defaults.keys():
        if major_key not in input_defaults.keys():
            input_defaults[major_key]=null_defaults[major_key]
            need_write = True
        else:
            for minor_key in null_defaults[major_key].keys():
                if minor_key not in input_defaults.keys():
                    input_defaults[major_key][minor_key] = null_defaults[major_key][minor_key]
                    need_write = True
                    
    if need_write == True:
        defaults_fname = _get_defaults_fname()
        with open(defaults_fname, 'w') as f:
            yaml.dump(input_defaults, f, 
                      sort_keys=False, 
                      default_flow_style=False)
    

def _load_defaults(defaults_file=op.expanduser(f'~/megcore/defaults.yml')):
    '''Load the defaults.yml file from the users megcore folder.
    After loading compare with standard dictionary and write out new options into
    defaults.yml that have been added'''
    with open(defaults_file, 'r') as f:
        defaults = yaml.safe_load(f)
    
    # Write new options back to defaults file
    _synchronize_defaults_file(input_defaults=defaults)
    return defaults
    
DEFAULTS = initialize_defaults() 
TRIG_FILE_LOC = op.expanduser(f'~/megcore/trigproc')