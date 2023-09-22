#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:25:34 2023

@author: jstout
"""

import mne
import mne_bids
import os.path as op
from functools import partial

def get_project(bids_root, project=None):
    '''
    Simple search and check for the project root

    Parameters
    ----------
    bids_root : path
        Top directory of bids folder
    project : str, optional
        Project name.  If left blank it will try to determine the project.
        For this to work, there must only be 1 extra folder in derivatives
        except for the freesurfer directory

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    project_root : str path
        Path to the bids project folder.

    '''
    #Identify the derivatives folder
    if type(bids_root)==mne_bids.BIDSPath:
        bids_root=bids_root.root
    if op.basename(bids_root)=='derivatives':
        print('Assuming bids_root is derivatives root - because dirname is derivatives')
        deriv_root=bids_root
    else:
        deriv_root=op.join(bids_root, 'derivatives')
        if not op.exists(deriv_root):
            raise ValueError(f"Can't find the derivatives folder: {deriv_root}")
    #Set and check project folder based on function input 
    if project != None:
        project_root = op.join(deriv_root, project)
        if not op.exists(project_root):
            raise ValueError(f"Can't find the folder: {project_root}")
    #Check for possible project folders
    tmp_=glob.glob(op.join(deriv_root, '*'))
    _=[tmp_.remove(i) for i in tmp_ if op.basename(i) == 'freesurfer']
    if len(tmp_)==0:
        raise ValueError(f"Can't find the project folder in: {deriv_root}")
    elif len(tmp_)>1:
        raise ValueError(f"""Indeterminant amount of project folders: {tmp_}.
                         Specify at the commandline the expected project""")
    else:
        project_root = tmp_[0]
    return project_root

class data_getter():
    def __init__(
            self, 
            subject=None, 
            bids_path=None, 
            ):
        '''
        Find BIDS derivatives in project folder.  Project will be guessed if 
        not provided (also assumes only 1 project present beyond freesurfer).
        
        Use with either filename to assess OR search type.  Will return the
        path and data loader.

        Parameters
        ----------
        subject : TYPE, optional
            DESCRIPTION. The default is None.
        bids_root : TYPE, optional
            DESCRIPTION. The default is None.
        session : TYPE, optional
            DESCRIPTION. The default is '01'.
        run : TYPE, optional
            DESCRIPTION. The default is '01'.
        project : TYPE, optional
            DESCRIPTION. The default is None.
        filename : TYPE, optional
            DESCRIPTION. The default is None.
        search_type : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        # self.subject=subject.replace('sub-','')  # Strip sub- if present
        self.bids_path=bids_path
        self._determine_type()
        
    def _determine_type(self):
        tmp_ = op.splitext(self.bids_path.fpath)[0].split('_')[-1]
        if tmp_ in ['bem','fwd','trans','stc','cov','src']:
            self.type = tmp_
    def _get_loader(self):
        if self.type == 'bem':
            self.loader = mne.bem.read_bem_solution
        if self.type == 'fwd':
            self.loader = mne.read_forward_solution
        if self.type == 'src':
            self.loader = mne.read_source_spaces
        if self.type == 'trans':
            self.loader = mne.read_trans
    def load(self):
        self._get_loader()
        self.data = self.loader(self.bids_path.fpath)
        
#%%


def get_mri_dict(subject, bids_root=None, project=None, session='01', task=None):
    project_root = get_project(bids_root, project)
    subj_deriv = op.join(bids_root, 'derivatives',project, 'sub-'+subject)
    tmp=mne_bids.find_matching_paths(project_root, ['ON02811'], tasks=[task], datatypes='meg')
    data_getter(tmp)
    # ['bem','fwd','src','trans'
    
    
# =============================================================================
#   
# =============================================================================
test = data_getter(subject='ON02811', bids_path = tmp[0])
test.load()
