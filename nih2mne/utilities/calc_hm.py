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
import numpy as np

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
    dframe = pd.DataFrame(columns=['dset','trial','hz_val','nas_x','nas_y','nas_z',
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
            dframe.loc[idx,'hz_val']=op.basename(dset).replace('.ds','')
            dframe.loc[idx, 'trial']=trial_num.split()[-1]
            dframe.loc[idx,['nas_x','nas_y','nas_z']]=fid_dict['nasion']
            dframe.loc[idx,['lpa_x','lpa_y','lpa_z']]=fid_dict['left ear']
            dframe.loc[idx,['rpa_x','rpa_y','rpa_z']]=fid_dict['right ear']
            del fid_dict, idx
        del trial_val, dset
    return dframe

def _dist(val1, val2, num_dec=4):
    'Compute Euclidean Distance'
    dist_val = ((val1[0]-val2[0])**2 + (val1[1]-val2[1])**2 + (val1[2]-val2[2])**2)**0.5
    return round(dist_val, num_dec)

def calc_movement(row1, row2):
    'Compute the distance between two rows of dataframe'
    nas1 = row1[['nas_x','nas_y','nas_z']].values
    lpa1 = row1[['lpa_x','lpa_y','lpa_z']].values
    rpa1 = row1[['rpa_x','rpa_y','rpa_z']].values
    
    nas2 = row2[['nas_x','nas_y','nas_z']].values
    lpa2 = row2[['lpa_x','lpa_y','lpa_z']].values
    rpa2 = row2[['rpa_x','rpa_y','rpa_z']].values
    
    nas_m = _dist(nas1,nas2)
    lpa_m = _dist(lpa1,lpa2)
    rpa_m = _dist(rpa1,rpa2)
    av_move = np.round(np.mean([nas_m,lpa_m, rpa_m]) ,5)
    
    print('Values in cm:')                  
    print(f'NAS move: {nas_m}')
    print(f'LPA move: {lpa_m}')
    print(f'RPA move: {rpa_m}')
    print(f'Average movement on coils: {av_move}')
    
    bads = {j:i for i,j in zip([nas_m, lpa_m, rpa_m, av_move],['nas','lpa','rpa','ave']) if i>0.5}
    if len(bads)>0:
        print(f'\nWarning: The following are over the standard limit (0.5cm): {list(bads.keys())}')
        print('If you think this is a data error and not a subject movement related error: see calcHeadPos and changeHeadPos')
    
    return {'N':nas_m, 'L':lpa_m, 'R':rpa_m, 'Ave':av_move}
    


def compute_movement(dframe):
    last_hz_trial = dframe.query('hz_val=="hz"').trial.astype(int).max()
    last_hz_trial = str(last_hz_trial)
    
    row1_idx = dframe.query(f'hz_val=="hz" and trial=="{last_hz_trial}"').index[0]
    row2_idx = dframe.query('hz_val=="hz2" and trial=="1"').index[0]
    row1 = dframe.loc[row1_idx]
    row2 = dframe.loc[row2_idx]
    move_dict = calc_movement(row1, row2)
    return move_dict
    

def main(fname, csv_fname=None):
    hzfile = op.join(fname, 'hz.ds')
    acqfile = op.join(fname, 'hz.ds/hz.acq')

    #Check if CTF tools install
    if shutil.which('calcHeadPos') == '':
        raise EnvironmentError('CTF tools do not appear to be installed')
        

    #Run command and capture stdout
    cmd = f'calcHeadPos {fname} {acqfile}'        
    submission = subprocess.run(cmd.split(),
                                input='echo 0',
                                capture_output=True,
                                text=True,
                                encoding="utf-8")

    output = submission.stdout.split('\n')
    output = [i.replace('\t','') for i in output]
    
    #Capture section headings
    header_idxs, trial_idxs = [], []
    for idx, val in enumerate(output):
        if val.endswith('.ds'):
            header_idxs.append(idx)
        if val.startswith('Trial '):
            trial_idxs.append(idx)
    
    # Create a dictionary at the header breaks 
    out_vals = []
    for start_idx in header_idxs:
        out_vals.append(reformat_locs(output, start_idx=start_idx))
    
    dframe = return_dframe(out_vals)
    move_dict = compute_movement(dframe)
    if csv_fname != None:
        dframe.to_csv(csv_fname)
        print(f'Wrote output file to {csv_fname}')
    
    
def entrypoint():
    import argparse
    parser = argparse.ArgumentParser(description='''Compute the head movement
                                     between beginning and end of run.  This is set
                                     as the last hz.ds trial and hz2.ds(Trial1).''')
    parser.add_argument('-fname', 
                        help='filename to report ')
    parser.add_argument('-to_csv', 
                        help='output fname entry of csv file',
                        default=None)
    args = parser.parse_args()
    main(args.fname, csv_fname=args.to_csv)
    
    
    
if __name__=='__main__':
    entrypoint()
        
        
        

    

