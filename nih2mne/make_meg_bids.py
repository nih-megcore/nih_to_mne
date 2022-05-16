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

# logger = logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

# =============================================================================
# make_meg_bids.py
# 
# According to typical NIMH acquisition
# Take meg file as input
# Assume no repeats
# =============================================================================




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
        logger.exception(f'session_dir variable is not a valid path or \
                          dataset list: {session_dir}')
    
    #Verify that these are meg datasets
    for dset in dsets:
        if os.path.splitext(dset)[-1] != '.ds':
            logger.warning(f'{dset} does not end in .ds and will be ignored')
            
    dsets = sorted(dsets)
    
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
        

def process_meg_bids(input_path=None, bids_dir=None, session=1):
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
    session : int
        Session number for data acquisition.  Defaults to 1 if not set

    Returns
    -------
    None.

    '''
    if bids_dir==None:
        raise ValueError('No bids_dir output directory given')
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    dset_dict = sessdir2taskrundict(session_dir=input_path)
    
    session = str(session)
    if len(session)==1: session = '0'+session
    
    error_count=0
    for task, task_sublist in dset_dict.items():
        for run, base_meg_fname in enumerate(task_sublist, start=1):
            meg_fname = op.join(input_path, base_meg_fname)
            try:
                subject = op.basename(meg_fname).split('_')[0]
                raw = mne.io.read_raw_ctf(meg_fname, system_clock='ignore')  
                raw.info['line_freq'] = 60 
                
                ses = session
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
    

def freesurfer_import(mri=None, subjid=None, tmp_subjects_dir=None,
                      afni_fname=None,
                      bsight_elec=None, 
                      meg_fname=None):
    '''
    only select afni_fname if the coreg is in the AFNI.HEAD file
    
    
    '''
    os.environ['SUBJECTS_DIR']=str(tmp_subjects_dir)
    
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
    trans_dir = tmp_subjects_dir.parent / 'trans_mats'
    trans_dir.mkdir()
    fid_path = trans_dir / f'{subjid}-fiducials.fif'
    try:
        write_mne_fiducials(subject=subjid,
                            subjects_dir=str(tmp_subjects_dir), 
                            afni_fname=afni_fname,
                            output_fid_path=str(fid_path))
    except BaseException as e:
        logger.error('Error in write_mne_fiducials', e)
        # continue  #No need to write trans if fiducials can't be written
    try:              
        trans_fname=trans_dir / (subjid +'-trans.fif')
        # op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')
        write_mne_trans(mne_fids_path=str(fid_path),
                        dsname=meg_fname, 
                        output_name=str(trans_fname), 
                        subjects_dir=str(temp_subjects_dir))
    except BaseException as e:
        logger.error('Error in write_mne_trans', e)
        print(f'error in trans calculation {subjid}')
    return str(trans_fname)
    
    
    # try:
    #     subprocess.run(f"mkheadsurf -s {subjid}".split(), check=True)
    #     logger.info('MKHEADSURF FINISHED')
    # except:
    #     try:
    #         proc_cmd = f"mkheadsurf -i {op.join(subjects_dir, subjid, 'mri', 'T1.mgz')} \
    #             -o {op.join(subjects_dir, subjid, 'mri', 'seghead.mgz')} \
    #             -surf {op.join(subjects_dir, subjid, 'surf', 'lh.seghead')}"
    #         subprocess.run(proc_cmd.split(), check=True)
    #     except BaseException as e:
    #         subj_logger.error('MKHEADSURF')
    #         subj_logger.error(e)

    
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
        subcmd = f'3dAFNItoNIFTI {out_brk}'
        subprocess.run(subcmd.split())
        print(f'Converted {mri_fname} to nifti')
    finally:
        os.chdir(init_dir)
    return outname
    
#Currently only supports 1 session of MRI
def process_mri_bids(bids_dir=None,
                     subjid=None, 
                     trans_fname=None,
                     meg_fname=None):
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    
    try:
        ses='01'
        raw = mne.io.read_raw_ctf(meg_fname, system_clock='ignore')
        trans = mne.read_trans(trans_fname)
        
        t1w_bids_path = \
            BIDSPath(subject=subjid, session=ses, root=bids_dir, suffix='T1w')
    
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
    meglist = os.listdir(meg_input_dir)
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
            logger.exception(f'User provided {subjid} not in {subjects}')
    elif len(subjects) ==0:
        logger.exception(f'''Could not extract any subjects from the list of 
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
        \n\nWARNING: This does NOT anonymize the data!!!
        ''')
    parser.add_argument('-bids_dir', help='Output bids_dir path', 
                        default=op.join(os.getcwd(),'bids_dir'))
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
    parser.add_argument('-mri_bsight_elec',
                       help='''Exported electrodes file from brainsight.
                       This has the locations of the fiducials''', 
                       required=False)
    parser.add_argument('-bids_session',
                        help='''Data acquisition session.  This is set to 1
                        by default.  If the same subject had multiple sessions
                        this must be set manually''',
                        default=1,
                        required=False)
    parser.add_argument('-subjid',
                        help='''The default subject ID is given by the MEG hash.
                        To override the default subject ID, use this flag'''
                        )
    args=parser.parse_args()
    
    #Initialize
    if not op.exists(args.bids_dir): os.mkdir(args.bids_dir)
    notanon_fname = op.join(args.bids_dir, 'NOT_ANONYMIZED!!!.txt')
    with open(notanon_fname, 'a') as w:
        w.write(args.meg_input_dir + '\n')

    #Establish Logging
    global logger
    logger_dir = Path(args.bids_dir).parent / 'bids_prep_logs'
    logger_dir.mkdir(exist_ok=True)
    
    if args.subjid:
        subjid=args.subjid
    else:
        subjid = _check_multiple_subjects(args.meg_input_dir)
    logger = get_subj_logger(subjid, log_dir=logger_dir, loglevel=logging.DEBUG)
    
    #
    #   Process MEG
    #
    process_meg_bids(input_path=args.meg_input_dir,
                      bids_dir=args.bids_dir, 
                      session=args.bids_session)
    
    #
    #   Prep MRI
    #
    #Create temporary MRI directories at the parent directory of the bids dir
    global temp_dir
    temp_dir=Path(args.bids_dir).parent / 'bids_prep_temp'
    if op.exists(temp_dir): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    temp_subjects_dir = temp_dir / 'subjects_tmp'
    temp_subjects_dir.mkdir()
    temp_mri_prep = temp_dir / 'mri_tmp'
    temp_mri_prep.mkdir()

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
        
    template_meg = glob.glob(op.join(args.meg_input_dir, subjid+'*.ds'))[0]
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
                     trans_fname=trans_fname,
                     meg_fname=template_meg)
    
    
    
    
# def test_bsight_inputs():
#     anat = 'IIARVYKX/IIARVYKX_anat.nii'
#     meg_input_dir = '20190328'
    
            



         
        
'''
Cannot be parallelized because deletes folder - not subj specific
 - Can fix by making a folder with subjid

Still need to code for bsight 

If 2 subjects in 1 folder - still writes out 2 subjects of MEGdata even if
only 1 subject chosen

If successful clean up bids_prep folder

Still need to make tests for 
copy afni and convert to nii
check for multiple subjects in folder



'''
    
    
    
                        



# =============================================================================
# TESTS - move these to another file
# =============================================================================
def test_check_multiple_subjects():
    #Make this extensible - currently limited
    #Make a temp dir and link together the files from two different folders
    indir='/fast/oberman_test/TEMP'
    _check_multiple_subjects(indir)

    
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
 



    
            

 
