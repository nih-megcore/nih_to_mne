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
from mne.io import anonymize_info


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

        self.deriv_root=op.abspath(deriv_root)
        if fif_list==None:
            self.fif_list = glob.glob(op.join(self.deriv_root,'**','*.fif'), recursive=True)
        else: 
            self.fif_list = fif_list
        self.anon_root=op.abspath(anon_root)
        if anon_root is not None:
            self.initialize_outdir()
        self.overwrite=overwrite
        
    def initialize_outdir(self):
        os.makedirs(self.anon_root, exist_ok=True)
        tmp_ = set([op.dirname(i) for i in self.fif_list])
        outdirs = [i.replace(self.deriv_root, self.anon_root) for i in tmp_]
        for i in outdirs:
            os.makedirs(i, exist_ok=True)
        print(f'Created {len(outdirs)} sub directories in the anonymization folder')
            
    def check_types(self):
        type_dict = {}
        for dset in self.fif_list:
            if not os.access(dset, os.R_OK):
                warnings.warn(f'You do not have read access to file: {dset}')
            basename = op.basename(dset)
            _deriv_type, _loader = self._return_fif_type(basename)
            _dset_dict = dict(deriv_type=_deriv_type,
                              loader=_loader)
            type_dict[dset]=_dset_dict
        self.type_dict = type_dict
        print(f'Found {len(self.type_dict.keys())}: files')
        
    def anonymize_data(self, outdir=None):
        self.anonymized_dict = {}
        self.anonymized_failed = {}
        if outdir == None:
            outdir=self.anon_root
        assert outdir != None, "Must declare a method variable outdir or anon_root (during class init)"
        if not hasattr(self, 'type_dict'): raise RuntimeError('Run check_types first to stage the type_dict variable')
        i=0
        for dset in self.type_dict.keys():
            try:
                outfname = dset.replace(self.deriv_root, outdir)
                loader = self.type_dict[dset]['loader']
                datobj = loader(dset)
                
                #Hack because of MNE issue
                if hasattr(datobj, 'annotations'):
                    if datobj.annotations is None:
                        delattr(datobj, '_annotations')
                
                #Manually anonymize info if object doesn't have an anonymize option
                if not hasattr(datobj, 'anonymize') & hasattr(datobj, 'info'):
                    datobj.info = anonymize_info(datobj.info)
                else:
                    datobj.anonymize()
                    
                datobj.save(outfname, overwrite=self.overwrite)
                i+=1
                self.anonymized_dict[dset]=outfname
            except BaseException as e:
                self.anonymized_failed[dset] = str(e)
        if len(self.anonymized_failed)==0:                
            print(f'Successfully anonymized ALL {i} datasets into {outdir}')
        else:
            print(f'Could not anonymize {len(self.anonymized_failed)} files')
            print(f'Successful anonymization of {i} out of {len(self.fif_list)} files')
            
    
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
        
    def _postprocessing_checks(self):
        
        ## THIS PART IS NOT FINISHED
        
        if len(self.anonymized_failed) > 0:
            for i in self.anonymized_failed.keys():
                print(f'Failed: {i}')
        _double_check = set(self.fif_list).difference(set(self.anonymized_dict.keys()))
        if len(_double_check) > 0:
            print(f"There looks to be an error - {len(_double_check)} didn't run through process")
        
                  
    

