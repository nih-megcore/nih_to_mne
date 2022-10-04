#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:45:18 2022

@author: jstout
"""

import pandas as pd
import nih2mne
import os.path as op
import os 
import numpy as np

dirpath = op.join(nih2mne.__path__[0], 'templates', 'bids_entry_template.csv')
#   -bids_dir BIDS_DIR    Output bids_dir path
#   -meg_input_dir MEG_INPUT_DIR
#                         'Acquisition directory - typically designated by the acquisition date
#   -mri_brik MRI_BRIK    Afni coregistered MRI
#   -mri_bsight MRI_BSIGHT
#                         Brainsight mri. This should be a .nii file. The exported electrodes text file must be in the same folder and end in .txt. Otherwise, provide the
#                         mri_sight_elec flag
#   -mri_bsight_elec MRI_BSIGHT_ELEC
#                         Exported electrodes file from brainsight. This has the locations of the fiducials
#   -bids_session BIDS_SESSION
#                         Data acquisition session. This is set to 1 by default. If the same subject had multiple sessions this must be set manually
#   -subjid SUBJID 

csv_cmdline_mapping = dict(bids_dir='bids_dir',
                        meg_dir='meg_input_dir',
                        mri_brik='mri_brik',
                        brainsight_mri='mri_bsight',
                        brainsight_electrodes='mri_bsight_elec',
                        bids_session='bids_session',
                        subjid='subjid')

mapping_dtypes = dict(bids_dir=str,
                        meg_input_dir=str,
                        mri_brik=str,
                        mri_bsight=str,
                        mri_bsight_elec=str,
                        bids_session='Int16',
                        subjid=str)

# =============================================================================
# 
# =============================================================================
def find_end_hdr(csvfile):
    i=0
    with open(csvfile) as w:
        lines=w.readlines()
    for row in lines:
        if 'EndOfHeader' in row:
            break
        i+=1
    return i
            
def get_version(csvfile):
    i=0
    with open(csvfile) as w:
        lines=w.readlines()
    for row in lines:
        if 'version' in row:
            break
        i+=1
    return row.split(':')[-1].replace(' ','').replace(',','').replace('\n','')    


def read_csv_entries(csvfile):
    hdrline = find_end_hdr(csvfile)+1
    dframe = pd.read_csv(csvfile, skiprows=hdrline, dtype=mapping_dtypes)
    dframe = dframe.replace(r'^\s*$', np.nan, regex=True)
    # dframe.fillna(False, inplace=True)
    return dframe


def make_cmd(row):
    '''Assemble a make_meg_bids commandline entry'''
    tmp=row.copy()
    tmp.rename(csv_cmdline_mapping, inplace=True) 
    tmp.dropna(inplace=True) #First round cleaning  
    cmd = ['make_meg_bids.py']
    for option,value in tmp.iteritems():
        cmd.append(f'-{option} {value}')
    cmd = ' '.join(cmd)
    return cmd.replace('"','') 

        
def make_swarm_file(csvfile, swarmfile='megbids_swarm.sh', write=False):
    dframe = read_csv_entries(csvfile)
    swarm = []
    for i,row in dframe.iterrows():
        cmd = make_cmd(row)
        swarm.append(cmd)
    swarm = [i+'\n' for i in swarm]
    if write==True:
        with open(swarmfile,'w') as f:
            f.writelines(swarm)
    else:
        return swarm
        
    
    

# def test_







# bids_dir / meg_input_dir / mri_brik / mri_bsight / mri_electrodes.txt / bids session / subj

# usage: 
#         Convert MEG dataset to default Bids format using the MEG hash ID or 
#         entered subject ID as the bids ID.        
        

# WARNING: This does NOT anonymize the data!!!
        
#        [-h] [-bids_dir BIDS_DIR] -meg_input_dir MEG_INPUT_DIR [-mri_brik MRI_BRIK] [-mri_bsight MRI_BSIGHT] [-mri_bsight_elec MRI_BSIGHT_ELEC] [-bids_session BIDS_SESSION]
#        [-subjid SUBJID]

# optional arguments:
#   -h, --help            show this help message and exit
#   -bids_dir BIDS_DIR    Output bids_dir path
#   -meg_input_dir MEG_INPUT_DIR
#                         'Acquisition directory - typically designated by the acquisition date
#   -mri_brik MRI_BRIK    Afni coregistered MRI
#   -mri_bsight MRI_BSIGHT
#                         Brainsight mri. This should be a .nii file. The exported electrodes text file must be in the same folder and end in .txt. Otherwise, provide the
#                         mri_sight_elec flag
#   -mri_bsight_elec MRI_BSIGHT_ELEC
#                         Exported electrodes file from brainsight. This has the locations of the fiducials
#   -bids_session BIDS_SESSION
#                         Data acquisition session. This is set to 1 by default. If the same subject had multiple sessions this must be set manually
#   -subjid SUBJID        The default subject ID is given by the MEG hash. To override the default subject ID, use this flag
