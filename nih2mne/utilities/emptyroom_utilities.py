#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:47:59 2023

@author: jstout
"""
import datetime
import os, os.path as op
import glob
import subprocess
import copy

def compile_erooms(eroom_location='/data/MEGmodules/extras/EmptyRoom'):
    '''Simple list of files in the eroom_location folder and converts these
    to datetime objects'''
    eroom_list = glob.glob(op.join(eroom_location, '*.tgz'))
    eroom_dict={convert_meg_datetime(i):i for i in eroom_list}
    return eroom_dict

def convert_meg_datetime(meg_fname):
    '''
    Function to extract the date string from the CTF dataset and then convert 
    it to a datetime object and return

    Parameters
    ----------
    meg_fname : str
        Path to a meg file

    Returns
    -------
    datetime.datetime
        Datetime object

    '''
    meg_fname= op.basename(copy.copy(meg_fname))
    i=meg_fname.split('_')[2]
    return datetime.datetime(int(i[0:4]), int(i[4:6]), int(i[6:8]))

def get_closest_eroom(meg_fname, eroom_dict=None, eroom_location=None):
    '''
    Return the path of the emptyroom file that matches closest the date of 
    your current meg file.

    Parameters
    ----------
    meg_fname : str
        CTF dataset name - expects date in the name.
    eroom_dict : TYPE, optional
        Override if not using biowulf. The default is None.
    eroom_location : TYPE, optional
        Override if not using biowulf. The default is None.

    Returns
    -------
    closest_eroom : str
        Path to the emptyroom dataset.

    '''
    if eroom_dict==None:
        if eroom_location==None:
            eroom_dict = compile_erooms()
        else:
            eroom_dict = compile_erooms(eroom_location)
    megdate = convert_meg_datetime(meg_fname)
    res = min(eroom_dict.keys(), key=lambda curr: abs(curr - megdate))
    closest_eroom = eroom_dict[res]
    return closest_eroom

def pull_eroom(eroom_fname, tmpdir=None):
    cmd=f'tar -xf {eroom_fname} -C {tmpdir}'
    subprocess.run(cmd.split())


    
    
