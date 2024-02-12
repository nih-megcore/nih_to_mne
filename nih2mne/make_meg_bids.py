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


import shutil
import matplotlib
import matplotlib.pyplot as plt; 
from multiprocessing import Pool

from mne_bids import write_anat, BIDSPath, write_raw_bids
from nih2mne.calc_mnetrans import write_mne_fiducials 
from nih2mne.calc_mnetrans import write_mne_trans

from nih2mne.utilities.clear_mrk_path import (calc_extra_mark_filelist,
                                              remove_extra_mrk_files, 
                                              clean_filepath_header)

global logger
logger = logging.getLogger('__main__')

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
        logger.exception(f'session_dir variable is not a valid path or \
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
    logger.info(f'Using {er_fname} for emptyroom')
    return er_fname   
        
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
    subprocess.run(cmd.split())
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
                     crop_trailing_zeros=False):
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

    Returns
    -------
    None.

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
            
            if crop_trailing_zeros==True:
                # This is necessary for trial based acq that is terminated early
                from nih2mne.utilities.data_crop_wrapper import return_cropped_ds
                meg_fname = return_cropped_ds(meg_fname)
            
            if anonymize==True:
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
                
                ses = session
                run = str(run) 
                if len(run)==1: run='0'+run
                bids_path = BIDSPath(subject=bids_id, session=ses, task=task,
                                      run=run, root=bids_dir, suffix='meg')
                write_raw_bids(raw, bids_path, overwrite=True)
                logger.info(f'Successful MNE BIDS: {meg_fname} to {bids_path}')
            except BaseException as e:
                logger.exception('MEG BIDS PROCESSING:', e)
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
            logger.exception('MEG BIDS PROCESSING EMPTY ROOM:', e)
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
        subprocess.run(f'recon-all -autorecon1 -noskullstrip -s {subjid}'.split(),
                        check=True)
        logger.info('RECON_ALL IMPORT FINISHED')
    except BaseException as e:
        logger.error('RECON_ALL IMPORT')
        logger.error(e)
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
    #Now that either afni_fname or bsight_elec is None
    #Both can be passed into the function
    try:
        write_mne_fiducials(subject=subjid,
                            subjects_dir=str(tmp_subjects_dir), 
                            afni_fname=afni_fname,
                            bsight_txt_fname=bsight_elec,
                            output_fid_path=str(fid_path))
    except BaseException as e:
        logger.error('Error in write_mne_fiducials', e)
        # continue  #No need to write trans if fiducials can't be written
    try:              
        trans_fname=trans_dir / (subjid +'-trans.fif')
        if op.exists(trans_fname): os.remove(trans_fname)
        write_mne_trans(mne_fids_path=str(fid_path),
                        dsname=meg_fname, 
                        output_name=str(trans_fname),
                        subjects_dir=str(tmp_subjects_dir))
    except BaseException as e:
        logger.error('Error in write_mne_trans', e)
        print(f'error in trans calculation {subjid}')
    return str(trans_fname)
    
def convert_brik(mri_fname, outdir=None):
    '''Convert the afni file to nifti
    The outdir should be the tempdir/mri_temp folder
    Returns the converted afni file for input to freesurfer'''
    if op.splitext(mri_fname)[-1] not in ['.BRIK', '.HEAD']:
        raise(TypeError('Must be an afni BRIK or HEAD file to convert'))
    import shutil
    if shutil.which('3dAFNItoNIFTI') is None:
        raise(SystemError('It does not appear Afni is installed, cannot call\
                          3dAFNItoNIFTI'))
    basename = op.basename(mri_fname)
    in_hdr,in_brk = mri_fname[:-4]+'HEAD', mri_fname[:-4]+'BRIK'
    out_hdr = op.join(outdir, op.basename(in_hdr))
    out_brk = op.join(outdir, op.basename(in_brk))
    shutil.copy(in_hdr, out_hdr)
    shutil.copy(in_brk, out_brk)
    
    #Required because afni only outputs to current directory
    init_dir=os.getcwd()
    try:
        os.chdir(outdir)
        outname = basename.split('+')[0]+'.nii'
        outname = op.join(outdir, outname)
        subcmd = f'3dAFNItoNIFTI {op.basename(out_brk)}'
        subprocess.run(subcmd.split())
        print(f'Converted {mri_fname} to nifti')
    finally:
        os.chdir(init_dir)
    return outname
    
#Currently only supports 1 session of MRI
def process_mri_bids(bids_dir=None,
                     subjid=None,
                     bids_id=None, 
                     trans_fname=None,
                     meg_fname=None,
                     session=None):
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
        logger.exception('MRI BIDS PROCESSING', e)

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
    logger = logging.getLogger(subjid)
    logger.setLevel(level=loglevel)
    if logger.handlers != []:
        # Check to make sure that more than one file handler is not added
        tmp_ = [type(i) for i in logger.handlers ]
        if logging.FileHandler in tmp_:
            return logger
    fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_log.txt')
    fmt = logging.Formatter(fmt=f'%(asctime)s - %(levelname)s - {subjid} - %(message)s')
    fileHandle.setFormatter(fmt=fmt) 
    logger.addHandler(fileHandle)
    return logger
            
# =============================================================================
# Commandline Options
# =============================================================================
if __name__ == '__main__':
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
                        the trailing zeros''',
                        action='store_true'
                        )
    group3 = parser.add_argument_group('Optional Overrides')
    group3.add_argument('-ignore_eroom', 
                        help='''If you are Not on Biowulf, use this option
                        to prevent an error. Or if you collected your own empty
                        room data with your dataset''',
                        action='store_true', 
                        default=False)
    group4 = parser.add_argument_group('UNDER Construction - BIDS MRI PostProcessing - CURRENTLY NOT ANONYMIZED')
    group4.add_argument('-freesurfer',
                        help='''Perform recon-all pipeline on the T1w.
                        This is required for the mri_prep portions below''', 
                        action='store_true'
                        )
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
    if (not args.mri_brik) and (not args.mri_bsight):
        raise ValueError('Must supply afni or brainsight coregistration')
        
    #Initialize
    if not op.exists(args.bids_dir): os.mkdir(args.bids_dir)
    if args.anonymize==True and args.bids_id is None:
        parser.error("-anonymize requires -bids_id")
    
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
    logger = get_subj_logger(subjid, log_dir=logger_dir, loglevel=logging.DEBUG)
    
    #Create temporary directories at the parent directory of the bids dir
    global temp_dir
    temp_dir=Path(args.bids_dir).parent / 'bids_prep_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
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
                      **kwargs)
    
    #
    #   Prep MRI
    #
    #Create temp dir for MRI
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
        
        #Determine if on a biowulf node
        if (host[0:2]=='cn') and (len(host)==6):
            if 'LOADEDMODULES' in os.environ: lmods=os.environ['LOADEDMODULES']
            
            lmods = lmods.split(':')
            lmods = [i.split('/')[0] for i in lmods]
            if 'afni' not in lmods:
                raise ValueError('Load the afni module before performing')
        elif not shutil.which('afni'):
            raise ValueError('It does not look like Afni can be found')
        
        #Convert the mri to nifti
        nii_mri = convert_brik(args.mri_brik, outdir=temp_mri_prep)
        logger.info(f'Converted {args.mri_brik} to {nii_mri}')
        
    #
    # Proc Brainsight Data
    #
    if args.mri_bsight:
        assert op.splitext(args.mri_bsight)[-1] in ['.nii','.nii.gz']
        nii_mri = args.mri_bsight
        
    template_meg = glob.glob(op.join(args.meg_input_dir, '*.ds'))[0]
    freesurfer_import(mri=nii_mri, 
                      subjid=subjid, 
                      tmp_subjects_dir=temp_subjects_dir, 
                      afni_fname=args.mri_brik, 
                      meg_fname=template_meg)
    
    trans_fname = make_trans_mat(mri=nii_mri, subjid=subjid, 
                                 tmp_subjects_dir=temp_subjects_dir,
                      afni_fname=args.mri_brik,
                      bsight_elec=args.mri_bsight_elec, 
                      meg_fname=template_meg)
    
    process_mri_bids(bids_dir=args.bids_dir,
                     subjid=subjid,
                     bids_id=bids_id,  
                     trans_fname=trans_fname,
                     meg_fname=template_meg,
                     session=args.bids_session)
    
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
        cmd = f"export SUBJECTS_DIR={fs_subjects_dir}; recon-all -all  -i {nii_fname} -s  sub-{subjid}"                            
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
        
 
    
    



         

                        



def test_bsight():
    subjid = 'APBWVFAR'
    mr_dir = '/fast/OPEN/APBWVFAR'
    nii_mri = f'{mr_dir}/APBWVFAR.nii'
    mri_bsight_elec = f'{mr_dir}/APBWVFAR.txt'
    meg_input_dir = '/fast/OPEN/20200122'
    
    global temp_dir
    temp_dir=Path(f'{mr_dir}').parent / 'bids_prep_temp'
    if op.exists(temp_dir): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    temp_subjects_dir = temp_dir / 'subjects_tmp'
    temp_subjects_dir.mkdir()
    temp_mri_prep = temp_dir / 'mri_tmp'
    temp_mri_prep.mkdir()


    
    template_meg = glob.glob(op.join(meg_input_dir, subjid+'*.ds'))[0]
    freesurfer_import(mri=nii_mri, 
                      subjid=subjid, 
                      tmp_subjects_dir=temp_subjects_dir, 
                      bsight_elec=mri_bsight_elec, 
                      meg_fname=template_meg)
    
    trans_fname = make_trans_mat(mri=nii_mri, subjid=subjid, 
                                 tmp_subjects_dir=temp_subjects_dir,
                      afni_fname=None,
                      bsight_elec=mri_bsight_elec, 
                      meg_fname=template_meg)
    
    process_mri_bids(bids_dir='./bids_dir',
                     subjid=subjid, 
                     trans_fname=trans_fname,
                     meg_fname=template_meg)

 
