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
import glob

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

def calc_movement(row1, row2, verbose=True):
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
    max_move = np.round(np.max([nas_m,lpa_m, rpa_m]) ,5)
    
    if verbose==True:
        print('Values in cm:')                  
        print(f'NAS move: {nas_m}')
        print(f'LPA move: {lpa_m}')
        print(f'RPA move: {rpa_m}')
        print(f'Max movement on coils: {max_move}')
        
        bads = {j:i for i,j in zip([nas_m, lpa_m, rpa_m, max_move],['nas','lpa','rpa','max']) if i>0.5}
        if len(bads)>0:
            print(f'\n!!Warning!!: The following are over the standard limit (0.5cm): {list(bads.keys())}')
            print('This can represent bad data that should not be part of your analysis')
            print('If you think this is a data error and not a subject movement related error: see calcHeadPos and changeHeadPos')
        
    return {'N':nas_m, 'L':lpa_m, 'R':rpa_m, 'Max':max_move}
    


def compute_movement(dframe, dframe2=None, verbose=True):
    '''If only one dframe set, pull the last trial from hz.ds and the only trial
    from hz2.ds.
    
    If a second dframe is present - this is comparing two runs.
    Assess the last trial of both hz.ds files'''
    if dframe2 is None:  
        #Single Run pre/post acq
        last_hz_trial = dframe.query('hz_val=="hz"').trial.astype(int).max()
        last_hz_trial = str(last_hz_trial)
        row1_idx = dframe.query(f'hz_val=="hz" and trial=="{last_hz_trial}"').index[0]
        row1 = dframe.loc[row1_idx]
        try:
            row2_idx = dframe.query('hz_val=="hz2" and trial=="1"').index[0]
            row2 = dframe.loc[row2_idx]
        except IndexError as e:
            print(f"{dframe.loc[0,'dset']}: Can't assess movement.  It is possible the run was terminated early")
            return {'N':None, 'L':None, 'R':None, 'Max':None}
        move_dict = calc_movement(row1, row2, verbose=verbose)
    else:  
        #Tow run comparison
        last_hz_trial1 = dframe.query('hz_val=="hz"').trial.astype(int).max()
        last_hz_trial1 = str(last_hz_trial1)
        row1_idx = dframe.query(f'hz_val=="hz" and trial=="{last_hz_trial1}"').index[0]

        last_hz_trial2 = dframe2.query('hz_val=="hz"').trial.astype(int).max()
        last_hz_trial2 = str(last_hz_trial2)
        row2_idx = dframe2.query(f'hz_val=="hz" and trial=="{last_hz_trial2}"').index[0]
        
        row1 = dframe.loc[row1_idx]
        row2 = dframe2.loc[row2_idx]
        move_dict = calc_movement(row1, row2, verbose=False)
    return move_dict
    

def get_localizer_dframe(fname):
    hzfile = op.join(fname, 'hz.ds')
    acqfile = op.join(fname, 'hz.ds/hz.acq')
    if not op.exists(hzfile):
        raise IOError(f'{hzfile} does not exist')
    if not op.exists(acqfile):
        raise IOError(f'{acqfile} does not exist')

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
    return dframe


        
def main(fname=None, csv_fname=None, task_name=None, data_dir=None):
    '''
    Run the 

    Parameters
    ----------
    fname : path/str
        Filename path as input
    csv_fname : path, optional
        Output csv file. The default is None.
    task_name : str, optional
        Task to filter datasets. The default is None.
    data_dir : path
        Folder to find datasets.
        If task_name is provided, it is added to the glob filtering

    Raises
    ------
    EnvironmentError
        If ctftools not available.

    Returns
    -------
    pandas dataframe

    '''
    if  fname != None:
        dframe = get_localizer_dframe(fname)
        move_dict = compute_movement(dframe)
        if csv_fname != None:
            dframe.to_csv(csv_fname)
            print(f'Wrote output file to {csv_fname}')
    if task_name != None:
        #Perform run to run movement within a task
        if data_dir == None:
            raise ValueError(f'If assess_task is used, data_dir must be assigned as well')
        dsets = glob.glob(op.join(data_dir, f'*_{task_name}_*.ds'))
        dframe_list = []
        for dset in dsets:
            dframe_list.append(get_localizer_dframe(dset))
        
        from itertools import combinations
        compare_dframe = pd.DataFrame(columns=['RunNum1','RunNum2','distance','>0.5'])

        for df1_2 in combinations(dframe_list,2):
            df1, df2 = df1_2
            runval1 = op.dirname(df1.loc[0].dset).split('_')[-1].replace('.ds','')
            runval2 = op.dirname(df2.loc[0].dset).split('_')[-1].replace('.ds','')
            move_dict = compute_movement(dframe=df1, dframe2=df2)
            
            idx=len(compare_dframe)
            if move_dict['Max']>0.5: 
                exceeds='X' 
            else:
                exceeds=''
            compare_dframe.loc[idx]=runval1, runval2, move_dict['Max'], exceeds
        compare_dframe.RunNum1 = compare_dframe.RunNum1.astype(int)
        compare_dframe.RunNum2 = compare_dframe.RunNum2.astype(int)            
        compare_dframe = compare_dframe.sort_values(by=['RunNum1','RunNum2'])
        compare_dframe.reset_index(drop=True, inplace=True)
        
        print(compare_dframe)
        
        if csv_fname != None:
            dframe.to_csv(csv_fname)
            print(f'Wrote output file to {csv_fname}')
    if (task_name==None) and (data_dir != None):
        #Loop over all datasets and find within run max movement calc
        print(f'Running movement calc over all dsets: {data_dir}')
        dsets = glob.glob(op.join(data_dir, '*.ds'))
        print(f'Found {len(dsets)} datasets')
        dframe_list = []
        movement_dframe = pd.DataFrame(columns=['fname', 'max_move'])
        for dset in dsets:
            try:
                tmp_dframe = get_localizer_dframe(dset)
                move_dict = compute_movement(tmp_dframe, verbose=False)
                fname = op.basename(dset)
                movement_dframe.loc[len(movement_dframe)]=fname,move_dict['Max']
            except IOError as e:
                fname = op.basename(dset)
                movement_dframe.loc[len(movement_dframe)]=fname,'CalcError - MissingFile'
        bad_dsets=0
        for idx,row in movement_dframe.iterrows():
            try:
                val = float(row.max_move)
                if val > 0.5:
                    movement_dframe.loc[idx, '>0.5cm']='X'
                    bad_dsets+=1
                else:
                    movement_dframe.loc[idx, '>0.5cm']=''
            except:
                movement_dframe.loc[idx, '>0.5cm']='???'
        print(movement_dframe)
        if bad_dsets > 0:
            print(f'\n!!Warning!! There are potentially {bad_dsets}')
            print('Combining these with other data can possibly lead to erroneous statistics')
        if csv_fname != None:
            movement_dframe.to_csv(csv_fname)
            print(f'Wrote output file to {csv_fname}')
        
        
        
            
    
def entrypoint():
    import argparse
    parser = argparse.ArgumentParser(description='''Compute the head movement
                                     between beginning and end of run.  This is set
                                     as the last hz.ds trial and hz2.ds(Trial1).
                                     Requires CTF tools package''')
    parser.add_argument('-to_csv', 
                        help='output fname entry of csv file (optional)(works with single or multirun)',
                        default=None)
    single = parser.add_argument_group('Pre/Post Single run')
    single.add_argument('-fname', 
                        help='filename to report head movement',
                        default=None)

    other = parser.add_argument_group('Multirun  options')    
    other.add_argument('-task', 
                        help='Assess movement BETWEEN different runs of a task',
                        default=None)
    other.add_argument('-data_dir',
                        help='''Folder to find datasets.  Does not need to be used with fname flag.\n
                        If given without task flag, it will list WITHIN run pre/post max movement of all runs in the folder''',
                        default=None)
    args = parser.parse_args()
    main(fname=args.fname, csv_fname=args.to_csv, task_name=args.task, 
         data_dir=args.data_dir)
    
    
    
if __name__=='__main__':
    entrypoint()
        

    

