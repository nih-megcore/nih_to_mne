#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:33:33 2022

@author: stoutjd
"""

import pandas as pd 
import mne
import mne_bids
import os, os.path as op 
import glob
import re
import copy
import numpy as np
import logging
import subprocess

import wget #  !pip install wget
import gzip

import shutil
import matplotlib
matplotlib.use('Qt5agg'); 
import matplotlib.pyplot as plt; 
from multiprocessing import Pool

from mne_bids import write_anat, BIDSPath, write_raw_bids


logger = logging.getLogger()

# =============================================================================
# make_meg_bids.py
# 
# According to typical NIMH acquisition
# Take meg file as input
# Assume no repeats
# =============================================================================

'''
cmd flags:
-brainsight MRI
-exported electrodes textfile

-meg folder
-bids_dir




# Current limitiation - Only does 1 session
# Save the error log


'''



def sessdir2taskrundict(session_dir=None):
    '''
    Convert a subject session to a dictionary with each run input name
    as the dict key and the bids task and run as values

    Parameters
    ----------
    session_dir : str, optional
        Path to subject data collection folder.  This is typically a folder
        designated by the date of acquisition

    Returns
    -------
    dict

    '''
    #Test for correct input type
    if (type(session_dir) is str) or (type(session_dir) is os.path):
        dsets=os.listdir(session_dir)
    elif type(session_dir) is list:
        dsets=session_dir
    else:
        logging.exception(f'session_dir variable is not a valid path or \
                          dataset list: {session_dir}')
    
    #Verify that these are meg datasets
    for dset in dsets:
        if os.path.splitext(dset)[-1] != '.ds':
            logging.warning(f'{dset} does not end in .ds and will be ignored')
            
    dsets = sorted(dsets)
    
    #Return bids dictionary
    task_list = [i.split('_')[1] for i in dsets]
    task_set = set(task_list)
    
    logging.info(f'Using {len(task_set)} tasks: {task_set}')
    
    out_dict=dict()
    for key in task_set:
        idxs = [i for i,x in enumerate(task_list) if x==key]
        sublist = [dsets[i] for i in idxs]
        out_dict[key]=sublist
    
    return out_dict
        
def test_sessdir2taskrundict():
    input_list=\
        ['DEC105_ASSR_20220225_002.ds',
         'DEC105_MMFAU_20220225_009.ds',
         'DEC105_M100_20220225_007.ds',
         'DEC105_rest_20220225_005.ds',
         'DEC105_MMFAU_20220225_003.ds',
         'DEC105_rest_20220225_012.ds',
         'DEC105_MMFUA_20220225_004.ds',
         'DEC105_rest_20220225_011.ds',
         'DEC105_M100_20220225_006.ds',
         'DEC105_MMFUA_20220225_010.ds',
         'DEC105_ASSR_20220225_008.ds',
         'DEC105_M100_20220225_001.ds']
    out_dict = sessdir2taskrundict(input_list)
    
    g_truth = \
    {'rest': ['DEC105_rest_20220225_005.ds',
              'DEC105_rest_20220225_011.ds',
              'DEC105_rest_20220225_012.ds'],
     'M100': ['DEC105_M100_20220225_001.ds',
              'DEC105_M100_20220225_006.ds',
              'DEC105_M100_20220225_007.ds'],
     'MMFAU': ['DEC105_MMFAU_20220225_003.ds', 'DEC105_MMFAU_20220225_009.ds'],
     'MMFUA': ['DEC105_MMFUA_20220225_004.ds', 'DEC105_MMFUA_20220225_010.ds'],
     'ASSR': ['DEC105_ASSR_20220225_002.ds', 'DEC105_ASSR_20220225_008.ds']}
    
    assert out_dict == g_truth
    



def process_meg_bids(input_path=None, bids_dir=None, mri_fname=None):
    '''
    Process the MEG component of the data into bids.
    Calls sessdir2taskrundict to get the task IDs and sort according to run #
    Output is the data in bids format in the assigned bids_dir
    
    !Warning - this assumes only 1 session of data acq currently 
    !Warning - this does not parse the events from your dataset

    Parameters
    ----------
    input_path : str, optional
        Path to the MEG folder - typically designated by a Date.
    bids_dir : str, optional
        Output path for your bids data.
    mri_fname : TYPE, optional
        T1.nii or T1.nii.gz from brainsight coreg or Afni coreg

    Returns
    -------
    None.

    '''
    if bids_dir==None:
        raise ValueError('No bids_dir output directory given')
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    dset_dict = sessdir2taskrundict(session_dir=input_path)
    
    error_count=0
    for task, task_sublist in dset_dict.items():
        for run, base_meg_fname in enumerate(task_sublist):
            meg_fname = op.join(input_path, base_meg_fname)
            try:
                subject = op.basename(meg_fname).split('_')[0]
                raw = mne.io.read_raw_ctf(meg_fname, system_clock='ignore')  
                raw.info['line_freq'] = 60 
                
                ses = '01'   ######### !!!! Hack 
                run = str(run) 
                if len(run)==1: run='0'+run
                bids_path = BIDSPath(subject=subject, session=ses, task=task,
                                      run=run, root=bids_dir, suffix='meg')
                write_raw_bids(raw, bids_path, overwrite=True)
                logger.info(f'Successful MNE BIDS: {meg_fname} to {bids_path}')
            except BaseException as e:
                logger.exception('MEG BIDS PROCESSING:', e)
                error_count+=1
    if error_count > 0:
        logger.info(f'There were {error_count} errors in your processing, \
                    check the error log for more information')  #!!! print the error log location
    else:
        logger.info('SUCCESS: There were no errors!')
    
    
    
        
# process_meg_bids(input_path='./20220225', bids_dir='./bids_test', mri_fname='/fast/BIDS_nuge/bidsout/sub-107/ses-01/anat/sub-107_ses-01_T1w.nii.gz')

def process_nih_transforms(topdir=None):
    csv_fname = op.join(topdir, 'MasterList.csv')
    dframe=pd.read_csv(csv_fname)
    from nih2mne.calc_mnetrans import write_mne_fiducials 
    from nih2mne.calc_mnetrans import write_mne_trans
    
    if not os.path.exists(f'{topdir}/trans_mats'): os.mkdir(f'{topdir}/trans_mats')
    
    for idx, row in dframe.iterrows():
        subj_logger=get_subj_logger(row['bids_subjid'], log_dir=f'{topdir}/logs')
        if op.splitext(row['full_mri_path'])[-1] == '.gz':
            afni_fname=row['full_mri_path'].replace('.nii.gz','+orig.HEAD')
        else:
            afni_fname=row['full_mri_path'].replace('.nii','+orig.HEAD')
        fid_path = op.join('./trans_mats', f'{row["bids_subjid"]}_{str(int(row["meg_session"]))}-fiducials.fif')
        try:
            write_mne_fiducials(subject=row['bids_subjid'],
                                subjects_dir=subjects_dir, 
                                searchpath = os.path.dirname(afni_fname),
                                output_fid_path=fid_path)
        except BaseException as e:
            subj_logger.error('Error in write_mne_fiducials', e)
            continue  #No need to write trans if fiducials can't be written
        try:              
            trans_fname=op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')
            write_mne_trans(mne_fids_path=fid_path,
                            dsname=row['full_meg_path'], 
                            output_name=trans_fname, 
                            subjects_dir=subjects_dir)
            dframe.loc[idx,'trans_fname']=trans_fname
        except BaseException as e:
            subj_logger.error('Error in write_mne_trans', e)
            print('error in trans calculation '+row['bids_subjid'])
    dframe.to_csv('MasterList_final.csv', index=False)  




# def subjid_from_filename(filename, position=[0], split_on='_', multi_index_cat='_'):
#     filename = os.path.basename(filename)
#     tmp = os.path.splitext(filename)
#     if len(tmp)>2:
#         return 'Error on split extension - Possibly mutiple "." in filename'
#     if not isinstance(position, list):
#         print('The position variable must be a list even if a single entry')
#         raise ValueError
#     filename = os.path.splitext(filename)[0]
#     filename_parts = filename.split(split_on)    
#     subjid_components = [filename_parts[idx] for idx in position]
#     if isinstance(subjid_components, str) or len(subjid_components)==1:
#         return subjid_components[0]
#     elif len(subjid_components)>1:
#         return multi_index_cat.join(subjid_components)
#     else:
#         return 'Error'



def convert_brik(mri_fname):
    if op.splitext(mri_fname)[-1] not in ['.BRIK', '.HEAD']:
        raise(TypeError('Must be an afni BRIK or HEAD file to convert'))
    import shutil
    if shutil.which('3dAFNItoNIFTI') is None:
        raise(SystemError('It does not appear Afni is installed, cannot call\
                          3dAFNItoNIFTI'))
    basename = op.basename(mri_fname)
    dirname = op.dirname(mri_fname)
    outname = basename.split('+')[0]+'.nii'
    outname = op.join(dirname, outname)
    subcmd = f'3dAFNItoNIFTI {mri_fname} {outname}'
    subprocess.run(subcmd.split())
    print(f'Converted {mri_fname} to nifti')
    


        

#%% Create the bids from the anonymized MRI
def process_mri_bids(bids_dir=None, topdir=None):
    bids_dir = f'{topdir}/bids'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)

    for idx, row in dframe.iterrows():
        subj_logger = get_subj_logger(row.bids_subjid, log_dir=f'{topdir}/logs')
        try:
            sub=row['bids_subjid'][4:] 
            ses='01'
            output_path = f'{topdir}/bids_out'
            
            raw = read_meg(row['full_meg_path'])          #FIX currently should be full_meg_path - need to verify anon
            trans = mne.read_trans(row['trans_fname'])
            t1_path = row['T1anon']
            
            t1w_bids_path = \
                BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
        
            landmarks = mne_bids.get_anat_landmarks(
                image=row['T1anon'],
                info=raw.info,
                trans=trans,
                fs_subject=row['bids_subjid']+'_defaced',
                fs_subjects_dir=subjects_dir
                )
            
            # Write regular
            t1w_bids_path = write_anat(
                image=row['T1anon'],
                bids_path=t1w_bids_path,
                landmarks=landmarks,
                deface=False,  #Deface already done
                overwrite=True
                )
            
            anat_dir = t1w_bids_path.directory   
        except BaseException as e:
            subj_logger.exception('MRI BIDS PROCESSING', e)
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('''
                                     Convert MEG dataset to default Bids 
                                     format using the MEG hash ID or entered
                                     subject ID as the bids ID.
                                     
                                     \n\nWARNING: This does NOT anonymize the data
                                     !!!
                                     ''')
    parser.add_argument('-bids_dir', help='Output bids_dir path', 
                        default='./bids_dir')
    parser.add_argument('-meg_input_dir', 
                        help=''''Acquisition directory - typically designated
                        by the acquisition date''', required=True)
    parser.add_argument('-mri_brik', 
                        help='''Afni coregistered MRI''')
    parser.add_argument('-mri_bsight',
                        help='''Brainsight mri.  This should be a .nii file.
                        The exported electrodes text file must be in the same 
                        folder and end in .txt.  Otherwise, provide the 
                        mri_sight_elec flag''')
    parser.add_argment('-mri_bsight_elec',
                       help='''Exported electrodes file from brainsight.
                       This has the locations of the fiducials''', 
                       required=False)




    




    
            

 