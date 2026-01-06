#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 09:43:39 2026

@author: jstout
"""
import os, os.path as op

def initialize_defaults():
    # Initialize the locations of the trigger files
    TRIG_FILE_LOC = op.expanduser(f'~/megcore/trigproc')
    LOG_FILE_LOC = op.expanduser(f'~/megcore/logs/')
    DEFAULTS_FILE_LOC = op.expanduser(f'~/megcore/defaults.yml')
    if not op.exists(TRIG_FILE_LOC):  os.makedirs(TRIG_FILE_LOC)
    if not op.exists(DEFAULTS_FILE_LOC): 
        _gen_defaults()
    defaults = _load_defaults(DEFAULTS_FILE_LOC)
    return defaults

def _gen_defaults(megcore_dir=op.expanduser(f'~/megcore')):
    'Write out the default settings into a yml file in the users megcore folder'
    import yaml
    BIDS_GEN = {'bids_root': op.expanduser(f'~/BIDS'), 
                'meg_dir': None, 
                'mri_dir': None, 
                'coreg_type': 'Brainsight', 
                'anonymize': 'N', 
                'crop_zeros': 'N',
                'emptyroom': 'N'
                }
    
    
    defaults = {'BIDS_gen': BIDS_GEN}
    with open(op.join(megcore_dir, 'defaults.yml'), 'w') as f:
        yaml.dump(defaults, f, 
                  sort_keys=False, 
                  default_flow_style=False)

def _load_defaults(defaults_file=op.expanduser(f'~/megcore/defaults.yml')):
    'Load the defaults.yml file from the users megcore folder'
    import yaml 
    with open(defaults_file, 'r') as f:
        defaults = yaml.safe_load(f)
    return defaults
    
DEFAULTS = initialize_defaults() 
TRIG_FILE_LOC = op.expanduser(f'~/megcore/trigproc')