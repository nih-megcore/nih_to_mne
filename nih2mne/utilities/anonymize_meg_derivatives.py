#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:52:11 2025

@author: jstout
"""

import os, os.path as op
import glob
import mne
import re
import warnings
import argparse
from mne.io.meas_info import anonymize_info


class deriv_anon():
    def __init__(self,deriv_root=None, anon_root=None, fif_list=None, 
                 overwrite=False):
        '''
        Anonymize the derivatives folder of any subject/date information. 
        
        Currently setup for raw / epochs / ica / 
        
        Example usage: 
            da = deriv_anon('/bids_root/derivatives', anon_root='/tmp/anon')
            da.check_types()
            da.anonymize_data()

        Parameters
        ----------
        deriv_root : str, optional
            Path to derivative root. The default is None.
        anon_root : str, optional
            Path to output direoctory. The default is None.
        fif_list : list, optional
            Provide a list of files to anonymize.  This is useful if there were 
            failures in previous attempts (eg. permission issues). The default is None.

        Returns
        -------
        None.

        '''
        assert deriv_root!=None, 'Must assign deriv_root'
        assert deriv_root!='/', 'Recursive search will do the full computer - do not use / as your deriv root'

        self.deriv_root=deriv_root
        if fif_list==None:
            self.fif_list = glob.glob(op.join(self.deriv_root,'**','*.fif'), recursive=True)
        else: 
            self.fif_list = fif_list
        self.anon_root=anon_root
        if anon_root is not None:
            self.initialize_outdir()
        self.overwrite=overwrite
        
    def initialize_outdir(self):
        os.makedirs(self.anon_root, exist_ok=True)
            
    def check_types(self):
        type_dict = {}
        for dset in self.fif_list:
            if not os.access(dset, os.R_OK):
                warnings.warning(f'You do not have read access to file: {dset}')
            basename = op.basename(dset)
            _deriv_type, _loader = self._return_fif_type(basename)
            _dset_dict = dict(deriv_type=_deriv_type,
                              loader=_loader)
            type_dict[dset]=_dset_dict
        self.type_dict = type_dict
        print(f'Found {len(self.type_dict.keys())}: files')
        
    def anonymize_data(self, outdir=None):
        if outdir == None:
            outdir=self.anon_root
        assert outdir != None, "Must declare a method variable outdir or anon_root (during class init)"
        if not hasattr(self, 'type_dict'): raise RuntimeError('Run check_types first to stage the type_dict variable')
        i=0
        for dset in self.type_dict.keys():
            outfname = dset.replace(self.deriv_root, outdir)
            loader = self.type_dict[dset]['loader']
            datobj = loader(dset)
            #Hack because of MNE issue
            if hasattr(datobj, 'annotations'):
                if datobj.annotations is None:
                    delattr(datobj, '_annotations')
            
            if not hasattr(datobj, 'anonymize') & hasattr(datobj, 'info'):
                datobj.info = anonymize_info(datobj.info)
            else:
                datobj.anonymize()
            datobj.save(outfname, overwrite=self.overwrite)
            i+=1
        print(f'Anonymized {i} datasets into {outdir}')
    
    def _get_basenames(self):
        return [op.basename(i) for i in self.fif_list]
    
    def _filter(self,fname, filttype=None):
        'Helper function '        
        assert filttype!=None
        #Currently limited to single 9 partial files of fif
        if (f'-{filttype}.fif' in fname) or re.search(f"-{filttype}-[0-9].fif$", fname):
            return True
        else:
            return False
    
    def _return_fif_type(self, fname):
        '''
        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.
    
        Returns
        -------
        str, 
            DESCRIPTION.
        loader, 
            Data loader 
    
        '''
        if self._filter(fname, filttype='raw'):
            return 'raw', mne.io.read_raw_fif
        elif self._filter(fname, filttype='meg'):
            return 'raw', mne.io.read_raw_fif
        elif self._filter(fname, filttype='epo'):
            return 'epo', mne.read_epochs
        elif self._filter(fname, filttype='ica'):
            return 'ica', mne.preprocessing.read_ica
        else:
            return None, None
        
        
        # fwd
        # bem
        # freesurfer
    

    

#%%
da = deriv_anon(deriv_root='/tmp/preprocessed/ICA', anon_root='/fast2/ANONTEST')

#%%  Quick test name identifiers
da.check_types()
da.anonymize_data()

_testtype, _loader = da._return_fif_type('test-meg.fif')
assert _testtype=='raw'
_testtype, _loader = da._return_fif_type('test-epo.fif')
assert _testtype=='epo'
_testtype, _loader = da._return_fif_type('test-epo-1.fif')
assert _testtype=='epo'
_testtype, _loader = da._return_fif_type('test-ica.fif')
assert _testtype=='ica'


#%% 
da = deriv_anon(deriv_root='/tmp/preprocessed/ICA', anon_root='/fast2/ANONTEST', 
                fif_list=['/tmp/preprocessed/ICA/sub-S19_ses-1_task-OrientationImagery_run-03_step1a-ica.fif'],
                overwrite=True)
da.check_types()
da.anonymize_data()


#%%
key = '/tmp/preprocessed/ICA/sub-S14_ses-1_task-OrientationImageryDynamicDynamic_run-03_step1a-raw.fif'
loader = da.type_dict[key]['loader']


dat = loader(key)

da.initialize_outdir()
da._save()


#%% 
test_list = [
 '/tmp/preprocessed/sub-S16_Still_preprocessed-epo.fif',
 '/tmp/preprocessed/sub-S12_miniEpochs_preprocessed-epo.fif',
 '/tmp/preprocessed/sub-S01_Dynamic_preprocessed-epo.fif',
 '/tmp/preprocessed/ICA/sub-S19_ses-1_task-OrientationImagery_run-03_step1a-ica.fif',
 '/tmp/preprocessed/ICA/sub-S13_ses-1_task-OrientationImagery_run-04_step1a-ica.fif',
 '/tmp/preprocessed/ICA/sub-S03_ses-1_task-OrientationImageryDynamicDynamic_run-01_step1a-raw.fif',
 '/tmp/preprocessed/ICA/sub-S07_ses-1_task-OrientationImagery_run-01_step1a-ica.fif'
]