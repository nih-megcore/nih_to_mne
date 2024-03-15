#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:38:31 2023

@author: stoutjd and amaia
"""
import os, os.path as op
import mne
import numpy as np
import subprocess
import shutil
import pyctf
from pyctf.ctf_res4 import *
from pyctf.util import *
from struct import Struct
import textwrap

def get_term_time(data, sfreq):
    '''
    the index of 20 consecutive zeros is used as an identifier to a terminated run (when user hit "abort")
    
    '''
    try:
        idx_crop = np.where((np.diff(np.convolve(np.ones(20),data==0)))==1)[0][0]
        return idx_crop, idx_crop/sfreq
    except:
        return False
    

def checkClassFile(fname):

    ''' 
    Load ClassFile.cls. 
    If "Aborted" class is found in file, 
    temporarily rename file so newDs does not use it
    
    Parameters
    ----------
    fname : str
        Path String.

    '''

    # Open the file
    with open(os.path.join(fname,'ClassFile.cls'), "r") as file:
        # Initialize variables to store class information
        number_of_class = None
        names = []

        # Read the file line by line
        for line in file:
            # Strip whitespace characters from the beginning and end of each line
            line = line.strip()

            # Check if the line contains "NUMBER OF CLASSES"
            if line.startswith("NUMBER OF CLASSES:"):
                # The actual number is on the next line
                number_of_class = int(next(file).strip())
            # Check if the line contains "NAME"
            elif line.startswith("NAME:"):
                # The actual name is on the next line
                names.append(next(file).strip())
                if "Aborted" in names:
                    to_rename = True
                # Break the loop if all names are collected
                if len(names) == number_of_class:
                    break

    return to_rename 


def return_cropped_ds(fname):
    '''
    Load the raw dataset, check the time where a set of zeros are present
    Pass the time to newDs to crop the datset into a temp subfolder in the 
    directory of the original file.

    Parameters
    ----------
    fname : str
        Path String.

    Raises
    ------
    RuntimeError
        If it cannot find a termination point to the scan it returns error.

    Returns
    -------
    fname_out : str
        Path to new cropped dataset.

    '''
    assert fname.endswith('.ds')
    install_check()
    raw = mne.io.read_raw_ctf(fname, system_clock='ignore', preload=True)

    channel_idx = 100
    data = raw._data[channel_idx]
    _, crop_time = get_term_time(data, raw.info['sfreq'])
    if crop_time == False:
        raise RuntimeError('Could not find a terminated timepoint')
    
    base = op.abspath(op.dirname(fname))
    f_ = op.basename(fname)
    outdir = op.join(base, 'bids_prep_temp','tmp_cropped')
    if not op.exists(outdir): os.mkdir(outdir)
    fname_out = op.join(outdir, f_)

    if op.exists(op.join(fname,'ClassFile.cls')): 
        if checkClassFile(fname):
            os.rename(os.path.join(fname,'ClassFile.cls'), os.path.join(fname,'__ClassFile.cls'))

    cmd = f'newDs -f -time 0 {str(crop_time)} {fname} {fname_out}'
    subprocess.run(cmd, shell=True)

    if op.exists(op.join(fname,'__ClassFile.cls')): 
        os.rename(os.path.join(fname,'__ClassFile.cls'), os.path.join(fname,'ClassFile.cls'))

    return fname_out
    

def install_check():
    try:
        assert shutil.which('newDs') != None
    except:
        raise BaseException('''The CTF tools are not installed on this system
                            if on biowulf do module load ctf and rerun''')
    


def writeClassFile(fname_out):       

    file_content = f"""\
    PATH OF DATASET:
    {fname_out}

    
    NUMBER OF CLASSES:
    1


    CLASSGROUPID:
    3
    NAME:
    BAD
    COMMENT:

    COLOR:
    Red
    EDITABLE:
    Yes
    CLASSID:
    1
    NUMBER OF TRIALS:
    0
    LIST OF TRIALS:
    TRIAL NUMBER


    """

    file_content = textwrap.dedent(file_content)

    # Write the file content to a file
    with open(f"{fname_out}/ClassFile.cls", "w") as file:
        file.write(file_content)



def crop_ds(fname):

    '''
    Crop a raw dataset file (.ds) by identifying the termination point of the scan, and write the cropped data to new files.

    Parameters
    ----------
    fname : str
        Path to the raw dataset file (.ds) to be cropped.

    Raises
    ------
    RuntimeError
        If the function cannot find a termination point to the scan, it raises a RuntimeError.

    Returns
    -------
    fname_out : str
        Path to the new cropped dataset.

    Notes
    -----
    This function loads the raw dataset file specified by `fname`, identifies the termination point of the scan, 
    and crops the dataset accordingly. The cropped data is then written to new .res4 and .meg4 files, along with 
    any additional files from the original dataset directory. The output dataset is stored in a temporary 
    subfolder within the directory of the original file.

    '''

    try:
        # Load ds data using pyctf
        ds = pyctf.dsopen(fname) # Load original data

        base = os.path.abspath(os.path.dirname(fname))
        f_ = os.path.basename(fname)
        outdir = os.path.join(base, 'bids_prep_temp', 'tmp_cropped')

        # Create output directory if it doesn't exist already
        os.makedirs(outdir, exist_ok=True)
        print("Directory created successfully:", outdir)

        fname_out = os.path.join(outdir, f_)
        if not os.path.exists(fname_out):
            os.mkdir(fname_out)

        print('Retrieving timepoint...')
        # Get max sample and time for cropping
        channel_idx = 100
        data = ds.getDsRawData(0, channel_idx)
        sfreq = ds.r.genRes[gr_sampleRate]

        idx_crop, crop_time = get_term_time(data, sfreq)
        if not crop_time:
            raise RuntimeError('Could not find a terminated timepoint')
        print('[done]\n')

        n_times = idx_crop
        nchan = ds.getNumberOfChannels()

        print('Writing .res4 file...')
        # Create empty res4 structure
        r = res4data() 

        # Copy fields from original ds file onto new res4 file
        # and only change the necessary fields
        r = ds.r
        r.numSamples = n_times

        # Potential things that could have changed:
        genRes = [None] * 29  # This is the size of genRes according to CTF
        for i in range(len(ds.r.genRes)):
            if i == gr_numSamples:
                genRes[gr_numSamples] = n_times
            elif i == gr_sampleRate:
                genRes[gr_sampleRate] = sfreq
            elif i == gr_epochTime:
                genRes[gr_epochTime] = crop_time
            else:
                genRes[i] = ds.r.genRes[i]
            
        r.genRes = genRes

        write_res4_structs(os.path.join(fname_out, f"{f_.split('.')[0]}.res4"), r)
        print('[done]\n')

        # Write meg4 file
        # Format to write big endian 32-bit integers.
        print('Writing .meg4 file...')
        be_int = Struct(">%di" % n_times)

        meg4Name = os.path.join(fname_out, f"{f_.split('.')[0]}.meg4")
        with open(meg4Name, "wb") as f:
            f.write(b"MEG41CP\x00")
            for i in range(nchan):
                j = ds.getDsRawData(0, i) / ds.getChannelGain(i)[0]
                j = j[:n_times]
                k = j.astype('i')
                f.write(be_int.pack(*k))
        print('[done]\n')

        # Copy remaining files onto outdir
        acceptedExtensions = ['mrk', 'infods', 'hc', 'acq', 'hist', 'infods', 'infods.bak', 'txt', 'cfg']
        for file_ in os.listdir(fname):
            fileP = os.path.join(fname, file_)
            if os.path.isdir(fileP) and not file_.endswith('meg.ds'):
                shutil.copytree(fileP, os.path.join(fname_out, file_))
            else:
                if file_.split('.')[-1] in acceptedExtensions or file_ == 'BadChannels':
                    shutil.copy(fileP, fname_out)
        
        writeClassFile(fname_out)

    except Exception as e:
        print("Error:", e)
