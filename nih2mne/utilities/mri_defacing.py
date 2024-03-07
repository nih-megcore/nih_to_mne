#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:30:20 2024

@author: jstout
"""

import shutil
import mne  
import os, os.path as op
import subprocess
import logging
import wget
import gzip

logger = logging.getLogger('__main__')

def download_deface_templates(code_topdir):
    '''Download and unzip the templates for freesurfer defacing algorithm'''
    logger = logging.getLogger('process_logger')
    if not op.exists(code_topdir): os.mkdir(code_topdir)
    if not op.exists(f'{code_topdir}/face.gca'):
        wget.download('https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/face.gca.gz',
             out=code_topdir)
        with gzip.open(f'{code_topdir}/face.gca.gz', 'rb') as f_in:
            with open(f'{code_topdir}/face.gca', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    if not op.exists(f'{code_topdir}/talairach_mixed_with_skull.gca'):
        wget.download('https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/talairach_mixed_with_skull.gca.gz',
             out=code_topdir)
        with gzip.open(f'{code_topdir}/talairach_mixed_with_skull.gca.gz', 'rb') as f_in:
            with open(f'{code_topdir}/talairach_mixed_with_skull.gca', 'wb') as f_out: 
                shutil.copyfileobj(f_in, f_out)
    try:
        assert os.path.exists(f'{code_topdir}/face.gca')  
        logger.debug('Face template present')
    except:
        logger.error('Face template does not exist')
    try:
        assert os.path.exists(f'{code_topdir}/talairach_mixed_with_skull.gca')
        logger.debug('Talairach mixed template present')
    except:
        logger.error('Talairach mixed template does not exist')
        
def mri_deface(mri, topdir=None):
    face_template = '/vf/users/MEGmodules/extras/fs_deface_templates/face.gca'
    brain_template = '/vf/users/MEGmodules/extras/fs_deface_templates/talairach_mixed_with_skull.gca'
    
    if not op.exists(face_template) or not op.exists(brain_template):
        logging.info('Using local resources -- downloading deface templates')
        code_topdir=f'{topdir}/staging_dir'
        download_deface_templates(code_topdir)
        brain_template=f'{topdir}/staging_dir/talairach_mixed_with_skull.gca'
        face_template=f'{topdir}/staging_dir/face.gca'
    
    anon_mri = mri.replace('.nii','_anon.nii')        
    subprocess.run(f'mri_deface {mri} {brain_template} {face_template} {anon_mri}'.split(),
                   check=True) 
    return anon_mri

 