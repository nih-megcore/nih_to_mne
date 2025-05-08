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
from pathlib import Path
import sys
import json

import shutil
import matplotlib
import matplotlib.pyplot as plt; 
from multiprocessing import Pool

from mne_bids import write_anat, BIDSPath, write_raw_bids
from nih2mne.calc_mnetrans import write_mne_fiducials 
from nih2mne.calc_mnetrans import write_mne_trans, coords_from_afni
import nih2mne
from nih2mne.utilities.clear_mrk_path import (calc_extra_mark_filelist,
                                              remove_extra_mrk_files, 
                                              clean_filepath_header)
from nih2mne.utilities.mri_defacing import mri_deface
from nih2mne.utilities.qa_fids import plot_fids_qa

import nibabel as nib

global logger
global err_logger
global temp_subjects_dir

root_logger = logging.getLogger()
logger = logging.getLogger()
err_logger = logging.getLogger()

# =============================================================================
# make_meg_bids.py
# 
# According to typical NIMH acquisition
# Take meg file as input
# Assume no repeats
# =============================================================================

#Set some parameters for anonmizing the MEG data
scrub_list_general = ['MarkerFile.mrk', 'ClassFile.cls']
include_list_general = ['BadChannels', 'ClassFile.cls', 'MarkerFile.mrk', 'params.dsc', 
                'processing.cfg', '*.hc', '*.res4', '*.meg4',  
                '*.infods']  #'*.acq' -- this contains redundant info

def sessdir2taskrundict(session_dir=None, subject_in=None):
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
        dsets=glob.glob(op.join(session_dir, '*ds'))
        dsets=[op.basename(i) for i in dsets]
    elif type(session_dir) is list:
        dsets=session_dir
    else:
        logger.error(f'session_dir variable is not a valid path or \
                          dataset list: {session_dir}')
        err_logger.error(f'session_dir variable is not a valid path or \
                          dataset list: {session_dir}')
    
    #Verify that these are meg datasets
    tmp_=[]
    for dset in dsets:
        if not dset.endswith('.ds'):
            logger.warning(f'{dset} does not end in .ds and will be ignored')
        elif subject_in not in dset:
            continue  #Ignore datasets that are not this subjects
        else:
            tmp_.append(dset)
    dsets = sorted(tmp_)
    
    #Return bids dictionary
    task_list = [i.split('_')[1] for i in dsets]
    task_set = set(task_list)
    
    logger.info(f'Using {len(task_set)} tasks: {task_set}')
    
    out_dict=dict()
    for key in task_set:
        idxs = [i for i,x in enumerate(task_list) if x==key]
        sublist = [dsets[i] for i in idxs]
        out_dict[key]=sublist
    
    return out_dict

def get_eroom(meg_fname, tmpdir=None):
    '''
    Find the closest emptyroom.  
    Download it to a local temp folder.  
    Untar/zip the file and return the new emptyroom for BIDS

    Parameters
    ----------
    meg_fname : str
        Path to MEG file.
    tmpdir : str, optional
        Tempdir to download the emptyroom file.  Default is None.

    Returns
    -------
    er_fname : str
        Path to emptyroom file.

    '''
    from nih2mne.utilities.emptyroom_utilities import get_closest_eroom, pull_eroom
    er_fname = get_closest_eroom(meg_fname)
    print('Pulling and untar/unzip emptyroom')
    pull_eroom(er_fname, tmpdir=tmpdir)
    er_fname = op.join(tmpdir, op.basename(er_fname)).replace('.tgz','.ds')
    #Sometimes the compiled emptyroom file is not a functional file - add failover
    if op.exists(er_fname):
        logger.info(f'Using {er_fname} for emptyroom')
        return er_fname
    else:
        er_fname = get_closest_eroom(meg_fname, failover=True)
        pull_eroom(er_fname, tmpdir=tmpdir)
        er_fname = op.join(tmpdir, op.basename(er_fname)).replace('.tgz','.ds')
        return er_fname 

def _check_markerfile(ds_fname):
    '''
    CTF newDs software (as of 03/08/2024) will not create a new anonymized 
    dataset if there is not a Markerfile present.  This will create a null 
    Markerfile to prevent CTF issues.
    
    Parameters
    ----------
    ds_fname : str
        Path to CTF ds file.

    Returns
    -------
    None.

    '''
    mrk_fname = op.join(ds_fname, 'MarkerFile.mrk')
    if not op.exists(mrk_fname):
        mrk_template = op.join(nih2mne.__path__[0], 'templates', 'MarkerFile.mrk')
        import shutil
        shutil.copy(mrk_template, mrk_fname)
        logger.info(f'Using template markerfile for {ds_fname}')

def _clear_ClassFile(meg_fname):
    '''
    The MEG file gets populated with a BAD Trial number that needs to be cleared
    out.  Check if there are trials listed and clear this entry.
    
    Nothing will change if the Classfile doesnt have an entry in this position

    Parameters
    ----------
    meg_fname : path str
        CTF dataset.

    Returns
    -------
    None.

    '''
    class_file = op.join(meg_fname, 'ClassFile.cls')
    with open(class_file, 'r') as f:
        lines = f.readlines()
    abort_idx = [j for j,i in enumerate(lines) if i=='Aborted\n']
    if len(abort_idx)>0:
        new_lines = lines[abort_idx[0]:]
    else:
        return
    trialN_list_idx = new_lines.index('TRIAL NUMBER\n')
    clear_idx = abort_idx[0] + trialN_list_idx + 1
    if clear_idx == '\n':
        return
    else:
        lines.pop(clear_idx)
    print('Clearing trail number issue from ClassFile')
    with open(class_file, 'w') as f:
        f.writelines(lines)

        
def anonymize_meg(meg_fname, tmpdir=None):
    '''
    Run the standard anonymization from the CTF tools
    

    Parameters
    ----------
    meg_input_dir : TYPE
        DESCRIPTION.
    tmpdir : str
        Path to the temporary directory.  Should be automatically generated
        by input function

    Returns
    -------
    anonymized meg path : str

    '''
    if tmpdir == None:
        raise ValueError
    out_fname = op.join(tmpdir, op.basename(meg_fname))
    cmd = f'newDs -anon {meg_fname} {out_fname}'
    try:
        subprocess.run(cmd.split(), check=True)
    except BaseException as e:
        logger.error(f'Error with CTF tools anonymization')
        err_logger.error(f'Error with CTF tools anonymization: {str(e)}')
    return out_fname

def anonymize_finalize(meg_fname):
    '''This is assumed to be the tempdir directory of the meg
    Clean up extra text files that may have IDs in the history or path etc'''
    include_list = []
    scrub_list = []
    # Fill out lists with wildcards
    all_files = set(glob.glob(op.join(meg_fname, '*')))
    for fname in include_list_general:
        tmp_ = glob.glob(op.join(meg_fname, fname))
        if (tmp_ != []) | ('*' not in tmp_):
            include_list.extend(tmp_)
    for fname in scrub_list_general:
        tmp_ = glob.glob(op.join(meg_fname, fname))
        if (tmp_ != []) | ('*' not in tmp_):
            scrub_list.extend(tmp_)    
    
    #Remove extra files
    remove_files = all_files.difference(set(include_list))
    for rem_file in remove_files:
        if op.isdir(rem_file):
            shutil.rmtree(rem_file)
        else:
            os.remove(rem_file)
    
    #Scrub path info from the header of scrub files (could potentially have date info)
    for scrub_file in scrub_list:
        clean_filepath_header(scrub_file)
    logger.info('Completed Scrubbing MEG files')
    

def process_meg_bids(input_path=None, subject_in=None, bids_id=None,
                     bids_dir=None, session=1, 
                     anonymize=False, tmpdir=None, ignore_eroom=None, 
                     crop_trailing_zeros=False, eventID_csv=None):
    '''
    Process the MEG component of the data into bids.
    Calls sessdir2taskrundict to get the task IDs and sort according to run #
    Output is the data in bids format in the assigned bids_dir
    
    !Warning - this does not parse the events from your dataset
    !Use parse marks or other tools -  preferably before doing the bids proc.

    Parameters
    ----------
    input_path : str, optional
        Path to the MEG folder - typically designated by a Date.
    bids_dir : str, optional
        Output path for your bids data.
    bids_id : str, optional 
        BIDS subject ID
    subject_in : str, 
        MEG subject ID to search for if multiples in folder
    session : int
        Session number for data acquisition.  Defaults to 1 if not set
    crop_trailing_zeros: BOOL
        Some sessions are terminated early, leaving a sizeable amount of zeroed
        data at the end.  This will determine the stop time and crop out the rest.
        Leaving the zero data at the end will cause some issues in the data 
        processing.
    eventID_csv: csv file
        Provide the eventID to event value mapping file.
        This can be generated using standardize_eventID_list.py

    '''
    if bids_dir==None:
        raise ValueError('No bids_dir output directory given')
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    dset_dict = sessdir2taskrundict(session_dir=input_path, subject_in=subject_in)
    
    session = str(int(session)) #Confirm no leading zeros
    #if len(session)==1: session = '0'+session
    
    error_count=0
    for task, task_sublist in dset_dict.items():
        for run, base_meg_fname in enumerate(task_sublist, start=1):
            meg_fname = op.join(input_path, base_meg_fname)
            try:
                _clear_ClassFile(meg_fname) #Remove Trials that fail CTFtools
            except:
                logger.warn(f'''Could not clear the Classfile, this may cause
                             issues with processing.  If so, change the following
                             file to write permissions and retry: 
                                 {str(op.join(meg_fname, 'ClassFile.cls'))}''')
            
            if crop_trailing_zeros==True:
                # This is necessary for trial based acq that is terminated early
                from nih2mne.utilities.data_crop_wrapper import return_cropped_ds
                meg_fname = return_cropped_ds(meg_fname)
            
            if anonymize==True:
                _check_markerfile(meg_fname)
                #Anonymize file and ref new dset off of the output fname
                meg_fname = anonymize_meg(meg_fname, tmpdir=tmpdir) 
                anonymize_finalize(meg_fname) #Scrub or remove extra text files
            
            #Special case for pre/post intervention in same session
            testval_case = base_meg_fname.replace('.ds','').split('_')[-1]
            if testval_case.lower() == 'pre':
                run=1
                logging.info(f'Special case pre assigned to run 1: {meg_fname}')
            elif testval_case.lower() == 'post':
                run=2
                logging.info(f'Special case post assigned to run2: {meg_fname}')
            
            try:
                raw = mne.io.read_raw_ctf(meg_fname, system_clock='ignore', 
                                          clean_names=True)  
                raw.info['line_freq'] = 60 
                
                if eventID_csv != None:
                    evts_dframe = pd.read_csv(eventID_csv)
                    out_dict = {row.ID_names:row.ID_vals for idx,row in evts_dframe.iterrows()}
                    evts_vals, evts_ids = mne.events_from_annotations(raw, event_id=out_dict)
                    raw.annotations.delete(range(len(raw.annotations)))
                else:
                    evts_vals=None
                    evts_ids=None
                
                ses = session
                run = str(run) 
                if len(run)==1: run='0'+run
                bids_path = BIDSPath(subject=bids_id, session=ses, task=task,
                                      run=run, root=bids_dir, suffix='meg')
                write_raw_bids(raw, bids_path, overwrite=True, 
                               events=evts_vals, event_id=evts_ids)
                logger.info(f'Successful MNE BIDS: {meg_fname} to {bids_path}')
            except BaseException as e:
                logger.error(f'MEG BIDS PROCESSING: {meg_fname}')
                err_logger.error('MEG BIDS PROCESSING:', e)
                error_count+=1
    #
    #Include the emptyroom dataset   
    #
    if ignore_eroom != True:
        try:
            tmp_ = str(int(np.random.uniform(0, 1e10)))  #Make a random name
            tmpdir=op.join(temp_dir, f'er_{tmp_}')
            if not op.exists(tmpdir): os.mkdir(tmpdir)
            er_fname = get_eroom(meg_fname, tmpdir=tmpdir) 
            
            #Required to make a new dir for eroom anonymization
            newtmp_ = str(int(np.random.uniform(0, 1e10)))  #Make a random name
            newtmpdir=op.join(temp_dir, f'er_{newtmp_}') 
            if anonymize==True:
                #Anonymize file and ref new dset off of the output fname
                er_fname = anonymize_meg(er_fname, tmpdir=newtmpdir) 
                anonymize_finalize(er_fname) #Scrub or remove extra text files

            raw = mne.io.read_raw_ctf(er_fname, system_clock='ignore', 
                                      clean_names=True)  
            raw.info['line_freq'] = 60 
            
            ses = session
            task = 'noise'
            run = '01'
            bids_path = BIDSPath(subject=bids_id, session=ses, task=task,
                                  run=run, root=bids_dir, suffix='meg')
            write_raw_bids(raw, bids_path, overwrite=True)
            logger.info(f'Successful MNE BIDS: {er_fname} to {bids_path}')
        except BaseException as e:
            logger.error('MEG BIDS PROCESSING EMPTY ROOM') 
            err_logger.error('MEG BIDS PROCESSING EMPTY ROOM:', e)
            error_count+=1
    else:
        logger.info('Ignore ERoom set -- not finding emptyroom')

    if error_count > 0:
        logger.info(f'There were {error_count} errors in your processing, \
                    check the error log for more information')  #!!! print the error log location
    else:
        logger.info('SUCCESS: There were no errors!')
    
    

def freesurfer_import(mri=None, subjid=None, tmp_subjects_dir=None,
                      afni_fname=None,
                      bsight_elec=None, 
                      meg_fname=None):
    '''
    only select afni_fname if the coreg is in the AFNI.HEAD file
    
    
    '''
    os.environ['SUBJECTS_DIR']=str(tmp_subjects_dir)
    tmp_ = op.join(tmp_subjects_dir, subjid)
    if op.exists(tmp_): shutil.rmtree(tmp_)
    
    try:
        subprocess.run(f'recon-all -i {mri} -s {subjid}'.split(),
                        check=True)
        logger.info('RECON_ALL IMPORT FINISHED')
    except BaseException as e:
        logger.error('RECON_ALL IMPORT: {mri}') 
        err_logger.error(f'RECONALL_IMPORT: {str(e)}')
        raise
    try:
        subprocess.run(f'recon-all -autorecon1 -noskullstrip -s {subjid}'.split(),
                        check=True)
    except BaseException as e:
        logger.error('RECON_ALL T1 Processing Error')
        err_logger.error(f'RECON_ALL T1 Processing Error {str(e)}')
        raise
    return op.join(tmp_subjects_dir, subjid)


def make_trans_mat(mri=None, subjid=None, tmp_subjects_dir=None,
                      afni_fname=None,
                      bsight_elec=None, 
                      meg_fname=None):
    tmp_ = tmp_subjects_dir.parent / 'trans_mats'
    if not os.path.exists(tmp_): os.mkdir(tmp_)
    trans_dir = tmp_subjects_dir.parent / 'trans_mats' / subjid
    if not os.path.exists(trans_dir): trans_dir.mkdir()
    fid_path = trans_dir / f'{subjid}-fiducials.fif'
    if (afni_fname is not None) and (bsight_elec is not None):
        logger.error(f'''Brainsight and Afni Brik Coreg can not both be chosen:
                     AFNI: {afni_fname}  Brainsight: {bsight_elec}''')
        err_logger.error(f'''Brainsight and Afni Brik Coreg can not both be chosen:
                     AFNI: {afni_fname}  Brainsight: {bsight_elec}''')                 
    #Now that either afni_fname or bsight_elec is None
    #Both can be passed into the function
    try:
        write_mne_fiducials(subject=subjid,
                            subjects_dir=str(tmp_subjects_dir), 
                            afni_fname=afni_fname,
                            bsight_txt_fname=bsight_elec,
                            output_fid_path=str(fid_path))
    except BaseException as e:
        logger.error('Error in write_mne_fiducials')
        err_logger.error(f'Error in write_mne_fiducials: {str(e)}')
        raise
        # continue  #No need to write trans if fiducials can't be written
    try:              
        trans_fname=trans_dir / (subjid +'-trans.fif')
        if op.exists(trans_fname): os.remove(trans_fname)
        write_mne_trans(mne_fids_path=str(fid_path),
                        dsname=meg_fname, 
                        output_name=str(trans_fname),
                        subjects_dir=str(tmp_subjects_dir))
    except BaseException as e:
        logger.error('Error in write_mne_trans')
        err_logger.error(f'Error in write_mne_trans: {str(e)}')
        print(f'error in trans calculation {subjid}')
    return str(trans_fname)
    
def convert_brik(mri_fname, outdir=None):
    '''Convert the afni file to nifti
    The outdir should be the tempdir/mri_temp folder
    Returns the converted afni file for input to bids'''
    if op.splitext(mri_fname)[-1] not in ['.BRIK', '.HEAD', '.gz']:
        raise(TypeError('Must be an afni BRIK or HEAD file to convert'))
    if op.splitext(mri_fname)[-1]=='.gz':
        if not mri_fname.lower().endswith('.brik.gz'):
            raise(TypeError('.gz file must be a .BRIK.gz file'))
    basename = op.basename(mri_fname)
    if mri_fname.endswith('.HEAD'):
        if op.exists(mri_fname.replace('.HEAD','.BRIK')):
            in_brk = mri_fname.replace('.HEAD','.BRIK')
        elif op.exists(mri_fname.replace('.HEAD','.BRIK.gz')):
            in_brk = mri_fname.replace('.HEAD','.BRIK.gz')
        else:
            raise ValueError(f'Cannot find .BRIK or .BRIK.gz associated with {mri_fname}')
    else:
        in_brk = mri_fname
    
    outname = basename.split('+')[0]+'.nii'
    outname = op.join(outdir, outname)   
    
    # Load BRIk / Convert To NIfti / Save
    mr_dset = nib.load(in_brk)
    mr_nii = nib.Nifti1Image(mr_dset.get_fdata(), mr_dset.affine)
    mr_nii.to_filename(outname)
    return outname
    
    
#Currently only supports 1 session of MRI
def process_mri_bids_fs(bids_dir=None,
                     subjid=None,
                     bids_id=None, 
                     trans_fname=None,
                     meg_fname=None,
                     session=None):
    #This is no longer the default/preferred method - use process_mri_bids
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    
    try:
        ses=str(int(session)) #Confirm no leading zeros
        raw = mne.io.read_raw_ctf(meg_fname, system_clock='ignore')
        trans = mne.read_trans(trans_fname)
        
        t1w_bids_path = \
            BIDSPath(subject=bids_id, session=ses, root=bids_dir, suffix='T1w')
    
        landmarks = mne_bids.get_anat_landmarks(
            image=op.join(temp_subjects_dir, subjid, 'mri','T1.mgz'),
            info=raw.info,
            trans=trans,
            fs_subject=subjid,
            fs_subjects_dir=temp_subjects_dir
            )
        
        # Write regular
        t1w_bids_path = write_anat(
            image=op.join(temp_subjects_dir, subjid, 'mri','T1.mgz'),
            bids_path=t1w_bids_path,
            landmarks=landmarks,
            deface=False, 
            overwrite=True
            )
        
    except BaseException as e:
        logger.error('MRI BIDS PROCESSING')
        err_logger.error(f'MRI BIDS PROCESSING: {str(e)}')

def process_mri_bids(bids_dir=None,
                     bids_id=None, 
                     nii_mri = None,
                     session=None):
    'This function directly writes the brainsight mri without freesurfer processing'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    
    try:
        ses=str(int(session)) #Confirm no leading zeros
        t1w_bids_path = \
            BIDSPath(subject=bids_id, session=ses, root=bids_dir, suffix='T1w')
    
        # Write regular
        t1w_bids_path = write_anat(
            image=nii_mri,
            bids_path=t1w_bids_path,
            deface=False, 
            overwrite=True
            )
        
        return t1w_bids_path
        
    except BaseException as e:
        logger.error('MRI BIDS PROCESSING')
        err_logger.error(f'MRI BIDS PROCESSING: {str(e)}')

def _read_electrodes_file(elec_fname=None): 
    assert op.exists(elec_fname), f'The {elec_fname} does not exist'
    dframe = pd.read_csv(elec_fname, skiprows=6, sep='\t')
    locs_ras = {}
    try:
        for val in ['Nasion', 'Left Ear', 'Right Ear']:
            row = dframe[dframe['# Electrode Name']==val]
            tmp = row['Loc. X'], row['Loc. Y'], row['Loc. Z']
            output = [i.values[0] for i in tmp]
            locs_ras[val] = np.array(output)
    except BaseException() as e:
        print('Cannot process the electrodes file - appears not to use the correct format in template')
    return locs_ras

def process_mri_json(elec_fname=None,
                     mri_fname = None,
                     ras_coords = None
                     ):
    mri = nib.load(mri_fname)
    if elec_fname != None:
        locs_ras = _read_electrodes_file(elec_fname)
    elif ras_coords != None:
        print('Runnig ras coords')
        assert 'Nasion' in ras_coords.keys()
        assert 'Left Ear' in ras_coords.keys()
        assert 'Right Ear' in ras_coords.keys()
        locs_ras = ras_coords
    else:
        raise ValueError('Either elec_fname or ras_coords must be supplied')
    
    # set the fids as voxel coords
    inv_rot = np.linalg.inv(mri.affine[0:3,0:3])
    translation =  mri.affine[0:3,3]
    nas_vox = np.matmul(inv_rot, locs_ras['Nasion']) - translation
    lpa_vox = np.matmul(inv_rot, locs_ras['Left Ear']) - translation
    rpa_vox = np.matmul(inv_rot, locs_ras['Right Ear']) - translation
    
    fids_json_out = {"AnatomicalLandmarkCoordinates": {
        "NAS":list(nas_vox),
        "LPA":list(lpa_vox),
        "RPA":list(rpa_vox)
        }}
    
    if mri_fname.endswith('.gz'):
        json_fname = mri_fname.replace('.nii.gz', '.json')
    else:
        json_fname = mri_fname.replace('.nii','.json')
    
    # Write the json
    with open(json_fname, 'w') as f:
        json.dump(fids_json_out, f)


def _check_multiple_subjects(meg_input_dir):
    '''Checks to see if multiple subjects were acquired in teh same dated 
    folder.  If multiple subjects found - the output will require manual
    choice to determine the correct subject'''
    #Filter for CTF datasets - Ignore folders with MRIs for example
    meglist = glob.glob(op.join(meg_input_dir, '*.ds')) 
    meglist = [op.basename(i) for i in meglist]
    subjects = set([i.split('_')[0] for i in meglist])
    subjects = list(subjects)
    if len(subjects) == 1:
        subjid = subjects
        return subjid[0]
    elif len(subjects) > 1:
        subjid=input(f'Which subject do you want to process (Do not use quotes)\
                     :\n{subjects}\n')
        if subjid in subjects:
            return subjid
        else:
            raise ValueError(f'User provided {subjid} not in {subjects}')
    elif len(subjects) ==0:
        raise ValueError(f'''Could not extract any subjects from the list of 
                         files {meglist}''')

def get_subj_logger(subjid, log_dir=None, loglevel=logging.INFO):
    '''Return the subject specific logger.
    This is particularly useful in the multiprocessing where logging is not
    necessarily in order'''
    _logger = logging.getLogger(subjid)
    _logger.setLevel(level=loglevel)
    if _logger.handlers != []:
        # Check to make sure that more than one file handler is not added
        tmp_ = [type(i) for i in _logger.handlers ]
        if logging.FileHandler in tmp_:
            return _logger
    fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_log.txt')
    fmt = logging.Formatter(fmt=f'%(asctime)s - %(levelname)s - {subjid} - %(message)s')
    fileHandle.setFormatter(fmt=fmt) 
    _logger.addHandler(fileHandle)
    streamHandle=logging.StreamHandler(sys.stdout)
    streamHandle.setFormatter(fmt=fmt)
    _logger.addHandler(streamHandle)
    _logger.info('Initializing subject level HV log')
    return _logger

def _input_checks(args):
    '''Perform minimal checks of existing data to fail early before starting 
    the processing'''
    if not op.exists(args.meg_input_dir):
        raise ValueError(f'{args.meg_input_dir} does not exist')
    if args.mri_bsight !=None:
        if len(args.mri_bsight_elec.split()) > 1:
            raise ValueError(f'Make sure there is not a space in filename {args.mri_bsight_elec}')
        if not op.exists(args.mri_bsight_elec):
            raise ValueError(f'{args.mri_bsight_elec}: does not exist')
        if not op.exists(args.mri_bsight):
            raise ValueError(f'{args.mri_bsight} : does not exist')
        with open(args.mri_bsight_elec, 'r') as f:
            tmp = f.readlines()
        _coord = [i for i in tmp if i.startswith('# Coordinate system:')][0]
        vals = _coord.strip().split(':')[1:]
        if len(vals)==2:
            val1, val2 = vals
        if len(vals)==3:
            val1, val2 = vals[0], vals[2]
        if val1.strip().upper()!='NIFTI' : raise ValueError('The brainsight electrodes file does not appear to be exported in the correct format.  Must be Nifti:scanner')
        if val2.strip().upper()!='SCANNER': raise ValueError('The brainsight electrodes file does not appear to be exported in the correct format.  Must be Nifti:scanner')
    else:
        if not op.exists(args.mri_brik):
            raise ValueError(f'{args.mri_brik} : does not exist')
        
def _output_checks(meg_conv_dict):
    '''Check that all of the datasets have been converted and mri+json w/Fids'''
    _errs = {}
    _goods = {}
    for input_meg, output_meg in meg_conv_dict.items():
        if not op.exists(output_meg):
            _errs[input_meg]=str(output_meg) #f'No output file: {output_meg}'
        else:
            _goods[input_meg]=output_meg
    #! TODO check for the MRI outputs
    return {'errors':_errs, 'good':_goods}

#!!! TODO - integrate this into the meg bids conversion so there isn't redundant code
def _get_conversion_dict(input_path=None, subject_in=None, bids_id=None,
                     bids_dir=None, session=1, 
                     ignore_eroom=None, 
                     ):
    '''
    Provides the conversion information for later checks.  
    

    Parameters
    ----------
    input_path : str,
        MEG directory (typically a date). 
    subject_in : str,
        MEGHASH ID typically. 
    bids_id : str,
        BIDS output ID
    bids_dir : str,
        BIDS output directory
    session : str | int 
        BIDS session
    ignore_eroom : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    conversion_dict : dict
        Dictionary of inputs to outputs

    '''
    dset_dict = sessdir2taskrundict(session_dir=input_path, subject_in=subject_in)
    session = str(int(session)) #Confirm no leading zeros
    conversion_dict={} 
    for task, task_sublist in dset_dict.items():
        for run, base_meg_fname in enumerate(task_sublist, start=1):
            print(run, base_meg_fname)
            meg_fname = op.join(input_path, base_meg_fname)
            ses = session
            run = str(run) 
            if len(run)==1: run='0'+run
            bids_path = BIDSPath(subject=bids_id, session=ses, task=task,
                                  run=run, root=bids_dir, suffix='meg')
            conversion_dict[meg_fname] = bids_path.fpath
    return conversion_dict

            
# =============================================================================
# Commandline Options
# =============================================================================

def make_bids(args):
    
    #Set if not called through the cmdline
    args = _clean_python_args(args) 
    
    #Initialize
    if not op.exists(args.bids_dir): os.mkdir(args.bids_dir)
    
    if args.anonymize==True:
        try:
            assert shutil.which('newDs') is not None  #Check for CTF tools
        except ProcessLookupError as e:
            print('''CTF tools are not detected on this system.  These are required
                  to anonymize the data''')
    else:
        if (args.subjid_input != None) and (args.bids_id == None):
            args.bids_id = args.subjid_input
        notanon_fname = op.join(args.bids_dir, 'NOT_ANONYMIZED!!!.txt')
        with open(notanon_fname, 'a') as w:
            w.write(args.meg_input_dir + '\n')

    #Establish Logging
    # global logger
    logger_dir = Path(args.bids_dir).parent / 'bids_prep_logs'
    logger_dir.mkdir(exist_ok=True)
    
    if args.subjid_input:
        subjid=args.subjid_input
    else:
        subjid = _check_multiple_subjects(args.meg_input_dir)
    
    global logger
    global err_logger
    
    logger = get_subj_logger(subjid, log_dir=logger_dir, loglevel=logging.INFO)
    err_logger = get_subj_logger(subjid+'_err', log_dir=logger_dir, loglevel=logging.WARN)
    
    #Create temporary directories at the parent directory of the bids dir
    global temp_dir
    temp_dir=Path(args.bids_dir).parent / 'bids_prep_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
        
    #
    #   Input Checks
    #
    if args.ignore_mri_checks==False:
        _input_checks(args)
    
    #
    #   Process MEG
    #
    if args.anonymize:
        #Create temp dir for MEG anonymization
        temp_meg_dir = temp_dir / 'meg_tmp' 
        temp_meg_dir.mkdir(parents=True, exist_ok=True)
        temp_meg_prep = temp_dir / 'meg_tmp' / subjid
        if op.exists(temp_meg_prep): shutil.rmtree(temp_meg_prep)
        temp_meg_prep.mkdir(parents=True)
        kwargs={'tmpdir':temp_meg_prep}
        bids_id = args.bids_id
    else:
        kwargs={}
        if hasattr(args, 'bids_id'):
            bids_id = args.bids_id
        else:
            bids_id = subjid
    
    
    process_meg_bids(input_path=args.meg_input_dir,
                                subject_in=subjid,
                                 bids_dir=args.bids_dir,
                                 bids_id = args.bids_id, 
                                 session=args.bids_session, 
                                 anonymize=args.anonymize,
                                 ignore_eroom=args.ignore_eroom,
                                 crop_trailing_zeros=args.autocrop_zeros,
                                 eventID_csv=args.eventID_csv,
                                 **kwargs)
    
    #
    #   Prep MRI
    #
    if args.ignore_mri_checks != True:
        #Create temp dir for MRI
        global temp_subjects_dir
        temp_subjects_dir = temp_dir / 'subjects_tmp' 
        temp_subjects_dir.mkdir(parents=True, exist_ok=True)
        temp_mri_prep = temp_dir / 'mri_tmp' / subjid
        if op.exists(temp_mri_prep): shutil.rmtree(temp_mri_prep)
        temp_mri_prep.mkdir(parents=True)
        
        #    
        #Check for Afni and convert the mri to nifti
        #
        if args.mri_brik:
            host = os.uname().nodename
            
            #Convert the mri to nifti
            nii_mri = convert_brik(args.mri_brik, outdir=temp_mri_prep)
            logger.info(f'Converted {args.mri_brik} to {nii_mri}')
            
            #Extract FIDS from Afni Head and convert to RAS
            coords_lps = coords_from_afni(args.mri_brik)
            coords_ras = {}
            for key in coords_lps.keys():
                tmp = np.array(coords_lps[key])
                tmp[0:2]*=-1
                coords_ras[key]=tmp
              
        #
        # Proc Brainsight Data
        #
        if args.mri_bsight:
            if (args.mri_bsight.endswith('.nii')) or (args.mri_bsight.endswith('.nii.gz')):
                nii_mri = args.mri_bsight
            else:
                raise ValueError(f'mri_bsight entry does not end with (nii or nii.gz): {args.mri_bsight}')
        
        
        #
        #   Anonymize/Deface MRI if set
        #
        if args.anonymize==True:
            nii_mri = mri_deface(nii_mri, topdir=temp_mri_prep)
      
        #
        # Finish MRI prep
        #
        # Get a template MEG dataset by filtering out noise and emptyroom datasets
        _dsets = glob.glob(op.join(args.meg_input_dir, f'{subjid}*.ds'))
        _dsets = [i for i in _dsets if (('noise' not in op.basename(i).lower()) and ('empty' not in op.basename(i).lower())) ]
        template_meg = _dsets[0]
        
        t1_bids_path = process_mri_bids(bids_dir=args.bids_dir,
                                     bids_id=bids_id, 
                                     nii_mri = nii_mri,
                                     session=args.bids_session)
        if args.mri_bsight_elec != None:
            process_mri_json(elec_fname=args.mri_bsight_elec,
                                 mri_fname = str(t1_bids_path))
        elif args.mri_brik != None:
            process_mri_json(elec_fname=args.mri_bsight_elec,
                                 mri_fname = str(t1_bids_path), 
                                 ras_coords=coords_ras)

            
        
        

    
    #
    # Check results
    #
    meg_conv_dict = _get_conversion_dict(input_path=args.meg_input_dir, 
                                         subject_in=args.subjid_input,
                                         bids_id=args.bids_id,
                                         bids_dir=args.bids_dir)
    _tmp = _output_checks(meg_conv_dict)
    errors = _tmp['errors']
    good = _tmp['good']
    logger.info('########### SUMMARY #################')
    for key in good.keys():
        logger.info(f'SUCCESS :: {key} converted to {good[key]}')
    for key in errors.keys():
        logger.warning(f'ERROR :: {key} did not convert : {errors[key]}')
        
    #
    # Plot QA images
    #
    if args.ignore_mri_checks != True:
        out_fids_qa_image = op.join(logger_dir, f'{args.subjid_input}_fids_qa.png')
        plot_fids_qa(subjid=args.bids_id, bids_root=args.bids_dir, 
                     outfile=out_fids_qa_image)
    
    #
    # Downstream Processing
    #
    fs_subjects_dir=op.join(args.bids_dir, 'derivatives','freesurfer','subjects')
    if args.freesurfer:
        nii_fnames = glob.glob(op.join(args.bids_dir, 'sub-'+args.bids_id, 'ses-1','anat','*T1w.nii'))
        nii_fnames += glob.glob(op.join(args.bids_dir, 'sub-'+args.bids_id, 'ses-1','anat','*T1w.nii.gz'))
        nii_fnames = [i for i in nii_fnames if len(i) != 0]
        assert len(nii_fnames)==1
        nii_fname=nii_fnames[0]
        os.makedirs(fs_subjects_dir, exist_ok=True)
        cmd = f"export SUBJECTS_DIR={fs_subjects_dir}; recon-all -all  -i {nii_mri} -s  sub-{subjid}"                            
        script = f'#! /bin/bash\n {cmd}\n'
        submission = subprocess.run(["sbatch", "--mem=6g", "--time=24:00:00"],
                                    input=script,
                                    capture_output=True,
                                    text=True,
                                    encoding="utf-8")
        if submission.returncode == 0:
            print(f"slurm job id: {submission.stdout}")
            sbatch1_ID=submission.stdout
        else:
            print(f"sbatch error: {submission.stderr}")
        
    if (args.mri_prep_s) or (args.mri_prep_v):
        from nih2mne.megcore_prep_mri_bids import mripreproc
        if args.mri_prep_s==True:
            surf=True
        if args.mri_prep_v==True:
            surf=False
        project_name=args.project
        #Loop over all filenames in bids path and generate forward model in 
        #the project derivatives folder
        filenames=glob.glob(op.join(args.bids_dir, 'sub-'+args.bids_id,
                                    'ses-'+str(args.bids_session),'meg', '*.ds'))
        cmd = f"export SUBJECTS_DIR={fs_subjects_dir}; "                            
        for filename in filenames:
            bids_path = mne_bids.get_bids_path_from_fname(filename)
            deriv_path = bids_path.copy().update(root=op.join(bids_path.root, 
                                                              'derivatives', 
                                                              project_name), 
                                                 check=False)
            deriv_path.directory.mkdir(parents=True, exist_ok=True)    
            tmp_t1_path = BIDSPath(root=bids_path.root,
                              session=bids_path.session,
                              subject=bids_path.subject,
                              datatype='anat',
                              suffix='T1w',
                              extension='.nii.gz',
                              check=True)    
            from nih2mne.megcore_prep_mri_bids import check_mri
            t1_bids_path = check_mri(tmp_t1_path)
            if surf:
                cmd += f'megcore_prep_mri_bids.py -filename {filename} -project {project_name} ; '
            else:
                cmd += f'megcore_prep_mri_bids.py -filename {filename} -project {project_name} -volume ; '
        
        #Submit the sbatch script
        script = f'#! /bin/bash\n {cmd}\n'
        submission = subprocess.run(["sbatch", "--mem=6g", "--time=06:00:00",f"--dependency=afterok:{sbatch1_ID}"],
                                    input=script,
                                    capture_output=True,
                                    text=True,
                                    encoding="utf-8")
        if submission.returncode == 0:
            print(f"slurm job id: {submission.stdout}")
        else:
            print(f"sbatch error: {submission.stderr}")
        #mri_preproc(bids_path=bids_path, 
        #            t1_bids_path=tmp_t1_path, 
        #            deriv_path=deriv_path, 
        #            surf=surf)
    


def main():
    import argparse
    parser = argparse.ArgumentParser('''
        Convert MEG dataset to default Bids format using the MEG hash ID or 
        entered subject ID as the bids ID.        
        \n\nWARNING: Must use the -anonymize flag to anonymize otherwise this does NOT anonymize the data!!!
        ''')
    parser.add_argument('-bids_dir', help='Output bids_dir path', 
                        default=op.join(os.getcwd(),'bids_dir'))
    parser.add_argument('-meg_input_dir', 
                        help='''Acquisition directory - typically designated
                        by the acquisition date''', required=True)
    parser.add_argument('-anonymize', 
                        help='''Strip out subject ID information from the MEG
                        data.  Currently this does not anonymize the MRI.
                        Requires the CTF tools.''',
                        default=False,
                        action='store_true')
    group1 = parser.add_argument_group('Afni Coreg')
    group1.add_argument('-mri_brik', 
                        help='''Afni coregistered MRI''')
    group2 = parser.add_argument_group('Brainsight Coreg')
    group2.add_argument('-mri_bsight',
                        help='''Brainsight mri.  This should be a .nii file.
                        The exported electrodes text file must be in the same 
                        folder and end in .txt.  Otherwise, provide the 
                        mri_sight_elec flag''')
    group2.add_argument('-mri_bsight_elec',
                       help='''Exported electrodes file from brainsight.
                       This has the locations of the fiducials''', 
                       required=False)
    group2.add_argument('-ignore_mri_checks', action='store_true', default=False)
    parser.add_argument('-bids_session',
                        help='''Data acquisition session.  This is set to 1
                        by default.  If the same subject had multiple sessions
                        this must be set manually''',
                        default=1,
                        required=False)
    parser.add_argument('-subjid_input',
                        help='''The default subject ID is given by the MEG hash.
                        If more than one subject is present in a folder, this
                        option can be set to select a single subjects dataset.
                        ''',
                        required=False
                        )
    parser.add_argument('-bids_id',
                        help='''The default subject ID is given by the MEG hash.
                        To override the default subject ID, use this flag.\n\n                        
                        If -anonymize is used, you must set the subjid'''
                        )
    parser.add_argument('-autocrop_zeros',
                        help='''If files are terminated early, leaving zeros
                        at the end of the file - this will detect and remove
                        the trailing zeros.  !!!Files larger than 2G will Fail!!!''',
                        action='store_true'
                        )
    group3 = parser.add_argument_group('Optional Overrides')
    group3.add_argument('-ignore_eroom', 
                        help='''If you are Not on Biowulf, use this option
                        to prevent an error. Or if you collected your own empty
                        room data with your dataset''',
                        action='store_true', 
                        default=False)
    group3.add_argument('-supplement_eroom', 
                        help='''If emptyroom present - ignore, else add emptyroom
                        from the biowulf repository.''',
                        action='store_true',
                        default=False)
    group3.add_argument('-freesurfer',
                        help='''Perform recon-all pipeline on the T1w.
                        This is required for the mri_prep portions below''', 
                        action='store_true'
                        )
    group3.add_argument('-eventID_csv', 
                        help='''Provide the standardized event IDs.
                        This can be produced by running: standardize_eventID_list.py
                        ''', 
                        default=None)    
    group4 = parser.add_argument_group('UNDER Construction - BIDS PostProcessing')
    group4.add_argument('-project',
                        help='''Output project name for the mri processing from mri_prep''', 
                        default='megprocessing'
                        )
    group4.add_argument('-mri_prep_s',
                        help='''Perform the standard SURFACE processing for
                        meg analysis (watershed/bem/src/fwd)''', 
                        action='store_true'
                        )
    group4.add_argument('-mri_prep_v',
                        help='''Perform the standard VOLUME processing for
                        meg analysis (watershed/bem/src/fwd)''', 
                        action='store_true'
                        )
    
    
    args=parser.parse_args()
    args = _clean_python_args(args)    

    
    if (not args.mri_brik) and (not args.mri_bsight) and (not args.ignore_mri_checks):
        raise ValueError('Must supply afni or brainsight coregistration')
        
    if args.anonymize==True and args.bids_id is None:
        parser.error("-anonymize requires -bids_id")
        
    make_bids(args)
        
def _clean_python_args(args):
    #Handle None passing from cmdline to python
    _named_arglist = [i for i in dir(args) if not i.startswith('_')]
    for i in _named_arglist:
        if getattr(args, i)=='None':
            setattr(args, i, None)
    return args

        
if __name__ == '__main__':
    main()
     
