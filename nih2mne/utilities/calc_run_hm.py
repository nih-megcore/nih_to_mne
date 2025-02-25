#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:21:09 2025

@author: jstout
"""

import os, sys
import os.path as op
import subprocess
import shutil
import pandas as pd


def reformat_locs(output, start_idx=None):
    '''
    Return a dictionary w/ each trial listed
    Start_idx sets where to start in the list 
    Loop runs until it hits an emptystring (which would have been a newline)

    Parameters
    ----------
    output : list
        output from calcHeadPos w/o tabs and split on newlines
    start_idx : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    trial_vals : dict
        trials as keys w/ x,y,z values for Nas/L/R coil locs
    dset : str
        relative path to hz file

    '''
    trial_vals = {}
    idx=start_idx
    while idx < len(output):
        val = output[idx]
        if val.endswith('.ds'):
            dset=val
        if val.startswith('Trial '):
            curr_trial=val.replace(':','')
            trial_vals[curr_trial]={}
        if val=='':
            if not output[idx+1].startswith('Trial '):
                break
        if val in ['nasion:', 'left ear:', 'right ear:']:
            fid=val.replace(':','')
            idx+=1
            x=output[idx].split()[-1]
            idx+=1
            y=output[idx].split()[-1]
            idx+=1
            z=output[idx].split()[-1]
            x,y,z = float(x),float(y),float(z)
            trial_vals[curr_trial][fid]=[x,y,z]
        idx+=1
    return trial_vals, dset
        
def return_dframe(out_vals):
    dframe = pd.DataFrame(columns=['dset','trial','nas_x','nas_y','nas_z',
                                   'lpa_x','lpa_y','lpa_z','rpa_x','rpa_y','rpa_z'])
    for fidloc in out_vals:
        trial_val=fidloc[0]
        if len(trial_val)==0:
            continue
        dset=fidloc[1]
        
        for trial_num in trial_val.keys():
            idx=len(dframe)
            fid_dict = trial_val[trial_num]
            dframe.loc[idx,'dset']=dset 
            dframe.loc[idx, 'trial']=trial_num.split()[-1]
            dframe.loc[idx,['nas_x','nas_y','nas_z']]=fid_dict['nasion']
            dframe.loc[idx,['lpa_x','lpa_y','lpa_z']]=fid_dict['left ear']
            dframe.loc[idx,['rpa_x','rpa_y','rpa_z']]=fid_dict['right ear']
            del fid_dict, idx
        del trial_val, dset
    return dframe
        

# def compute_movement(dframe):
    
    
    

def main():
    fname = sys.argv[1]
    hzfile = op.join(fname, 'hz.ds')
    acqfile = op.join(fname, 'hz.ds/hz.acq')

    cmd = f'calcHeadPos {fname} {acqfile}'

    if shutil.which('calcHeadPos') == '':
        raise EnvironmentError('CTF tools do not appear to be installed')

    submission = subprocess.run(cmd.split(),
                                input='echo 0',
                                capture_output=True,
                                text=True,
                                encoding="utf-8")

    output = submission.stdout.split('\n')
    output = [i.replace('\t','') for i in output]

    header_idxs, trial_idxs = [], []
    for idx, val in enumerate(output):
        if val.endswith('.ds'):
            header_idxs.append(idx)
        if val.startswith('Trial '):
            trial_idxs.append(idx)
    
    out_vals = []
    for start_idx in header_idxs:
        out_vals.append(reformat_locs(output, start_idx=start_idx))
    
    dframe = return_dframe(out_vals)
        
        
        

    

