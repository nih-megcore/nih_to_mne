#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:58:05 2023

@author: jstout
"""

import mne, mne_bids
from mne_bids import BIDSPath
import os, os.path as op, glob
from pathlib import Path
import numpy as np
import nibabel as nib
import subprocess
# from nih2mne import get_njobs

n_jobs=-1 

desc = '''Preprocess all MRI related processing.  To generate a swarmfile, 
use the swarmfile specific options (without setting filename).
'''

def get_dset_list(bids_root=None, src_type='surf', task=None, session=None, 
                  subject=None, run=None):
    search_str = f'{op.abspath(bids_root)}/sub-{subject}/ses-{session}/meg/sub-{subject}_*task-{task}_run-{run}.ds'
    search_str = search_str.replace('None', '*')
    return glob.glob(search_str) 
    

def make_swarm(bids_root=None, src_type='surf', project=None, 
               task=None, session=None, subject=None, run=None,
               out_fname='swarm_mriprep.sh'):
    '''
    Make a swarm file to process all of the bids datasets

    Parameters
    ----------
    bids_root : str,
        top level path to the bids directory. The default is None.
    task : str, 
        DESCRIPTION. The default is None.
    src_type : str, optional
        either vol or surf. The default is 'surf'.
    subject : str
        

    Returns
    -------
    None.

    '''
    ## This would be the optimal answer - but not working currently (mne-bids issue)
    # entities = {'datatypes':'meg', 'extensions':['.ds']}
    # ent_type = {'tasks':task, 'sessions':session, 'runs':run}
    # for tmp, func_input in ent_type.items():
    #     if func_input!=None:
    #         entities[tmp]=func_input
    # bids_paths = mne_bids.find_matching_paths(bids_root, **entities)
    dsets = get_dset_list(bids_root, task=task, session=session, subject=subject, run=run)
    swarm_dict = {}
    swarm_list = []
    for dset in dsets:
        bids_path = mne_bids.get_bids_path_from_fname(dset)
        cmd_str = f'megcore_prep_mri_bids.py -filename {str(bids_path.fpath)} -project {project}'
        if src_type=='vol':
            cmd_str += ' -volume'
        if bids_path.subject in swarm_dict.keys():
            #Subject level data needs to be on the same line so BEM proc conflicts do not occur
            swarm_dict[bids_path.subject] += f'; {cmd_str}' 
        else:
            swarm_dict[bids_path.subject] = cmd_str
    
    for subjid, cmd in swarm_dict.items():
        cmd +=' \n' #Enter a new line
        swarm_list.append(cmd)
    
    with open(out_fname, 'w') as f:
        f.writelines(swarm_list)
    print('')
    print('')
    print(f'''Wrote {len(dsets)} files to process for {len(swarm_list)} subjects into the swarmfile: {out_fname}''')
    print('This text file can be edited before submitting.')
    print('')
    print('To run on biowulf:')
    print('    module purge ') 
    print('    module load mne')
    print(f'    swarm -f {out_fname} -t 4 -g 6 --logdir=logdir')
    
    
def check_mri(t1_bids_path):
    for acq in ['MPRAGE', 'FSPGR', 'mprage', 'fspgr', None]:
        t1_bids_path.update(acquisition=acq, extension='.nii')
        if op.exists(t1_bids_path.fpath):
            break
        t1_bids_path.update(acquisition=acq, extension='.nii.gz')
        if op.exists(t1_bids_path.fpath):
            break
    #Redundant test possibly
    if not op.exists(t1_bids_path.fpath):
        raise ValueError('Can not determine T1 path')
    return t1_bids_path


def _gen_expanded_src(subject, subjects_dir, dilation_iter=8):
    '''
    Expand the brainmask out to use as a source model.  This is used to 
    prevent voxel cropping on the edges of the brain.

    Parameters
    ----------
    subject : str
        freesurfer subject id -- must include the sub- if using BIDS format
    subjects_dir : str
        freesurfer subjects dir  -- BIDS this would be bids_root/derivatives/freesurfer/subjects
    dilation_iter : int
        see scipy.ndimage.binary_dilation for more info.
        The default is 4.

    Returns
    -------
    mr_out, out_surf

    '''
    import scipy
    infile = op.join(subjects_dir, subject, 'mri','brainmask.mgz')
    mr = nib.load(infile)
    dat = mr.get_fdata()
    mask = scipy.ndimage.binary_dilation(dat, iterations=dilation_iter)
    
    dat_out = np.zeros(dat.shape, dtype=np.int16)
    dat_out[mask]=1
    
    mr_out = nib.MGHImage(dat_out, mr.affine) 
    mr_out_fname = op.join(subjects_dir, subject, 'mri','expanded_brainmask.mgz')
    mr_out.to_filename(mr_out_fname)
    
    in_mri = op.join(subjects_dir, subject, 'mri','expanded_brainmask.mgz')
    out_surf = op.join(subjects_dir, subject, 'surf','expanded_brainmask.surf')
    subprocess.run(f'mri_tessellate {in_mri} 1 {out_surf}'.split(), check=True)
    in_surf = op.join(subjects_dir, subject, 'surf','expanded_brainmask.surf')
    out_surf = op.join(subjects_dir, subject, 'surf','expanded_brainmask_smoothed.surf')
    subprocess.run(f'mris_smooth -n 10 {in_surf} {out_surf}'.split(), check=True)
    print(f'Wrote expanded brainmask to: {mr_out_fname}')
    return mr_out_fname, out_surf
    

def mripreproc(bids_path=None,
               t1_bids_path=None, 
               deriv_path=None, 
               surf=True, 
               subjects_dir=None):
    '''
    Generate the typical MRI inputs for processing MEG and saved to the 
    derivatives folder:
        bem_sol, fwd, src, trans

    Parameters
    ----------
    bids_path : mne_bids.BIDSpath, required
        bids_path to raw bids data. The default is None.
    t1_bids_path : mne_bids.BIDSpath, required
        bids_path for anatomy. The default is None.
    deriv_path : mne_bids.BIDSpath, required
        Output derivatives bids_path. The default is None.
    surf : Boolean
        If True (default), use the cortical surface.  Else use the volumetric
        sampling at 5mm

    Returns
    -------
    fwd : 
        MNE surface based forward model.

    '''

        
    bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif')
    if surf==True:
        fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
        src_fname = deriv_path.copy().update(suffix='src', extension='.fif')
    else:
        fwd_fname = deriv_path.copy().update(suffix='volfwd', extension='.fif')
        src_fname = deriv_path.copy().update(suffix='volsrc', extension='.fif')
    
    trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')
    
    raw_fname = bids_path.copy().update(suffix='meg')
    raw = mne.io.read_raw_ctf(raw_fname, system_clock='ignore', clean_names=True)
    if subjects_dir == None:
        subjects_dir = mne_bids.read.get_subjects_dir()
    fs_subject = 'sub-'+bids_path.subject
    
    if not op.exists(op.join(subjects_dir, 'sub-'+bids_path.subject, 'bem','watershed')):
        mne.bem.make_watershed_bem(fs_subject, subjects_dir=subjects_dir, gcaatlas = True)  #gcaatlas added to prevent cerebellum cropping  
    if fwd_fname.fpath.exists():
        fwd = mne.read_forward_solution(fwd_fname)
        return fwd
    if not bem_fname.fpath.exists():
        bem = mne.make_bem_model(fs_subject, subjects_dir=subjects_dir, 
                                 conductivity=[0.3])
        bem_sol = mne.make_bem_solution(bem)
        
        mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
    else:
        bem_sol = mne.read_bem_solution(bem_fname)
        
    if surf==True:
        if not src_fname.fpath.exists():
            src = mne.setup_source_space(fs_subject, spacing='oct6', add_dist='patch',
                                 subjects_dir=subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
    else:
        if not src_fname.fpath.exists():
            exp_mri, exp_surf = _gen_expanded_src(fs_subject, subjects_dir=subjects_dir)
            src = mne.setup_volume_source_space(subject=fs_subject, pos=5.0, 
                                          mri=exp_mri, surface=exp_surf,
                                          mindist=0.0, subjects_dir=subjects_dir
                                          )
            # BEM restricted below
            # src = mne.setup_volume_source_space(fs_subject, pos=5.0, bem=bem_sol,
            #                      subjects_dir=subjects_dir, mindist=0.0)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)        
    
    if not trans_fname.fpath.exists():
        trans = mne_bids.read.get_head_mri_trans(bids_path, extra_params=dict(system_clock='ignore'),
                                            t1_bids_path=t1_bids_path, fs_subject=fs_subject, 
                                            fs_subjects_dir=subjects_dir)
        mne.write_trans(trans_fname.fpath, trans, overwrite=True)
    else:
        trans = mne.read_trans(trans_fname.fpath)
    fwd = mne.make_forward_solution(raw.info, trans, src, bem_sol, eeg=False, 
                                    n_jobs=n_jobs)
    mne.write_forward_solution(fwd_fname.fpath, fwd)
    return fwd


amp_thresh = dict(mag=4000e-15)
def preproc(bids_path=None,
            deriv_path=None):
    '''
    Generate the epochs for input to the dipole model.
    Saves filtered raw data and epochs to the derivatives folder

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath, required
        Bids path to original data. The default is None.
    deriv_path : mne_bids.BIDSPath, required
        Bids path for outputs. The default is None.

    Returns
    -------
    epo : mne.Epochs
        Standard and Deviant stim epochs (also saved to deriv_path).

    '''
    raw = mne.io.read_raw_ctf(bids_path, system_clock='ignore', clean_names=True)
    raw.load_data()
    raw.filter(30, 50, n_jobs=n_jobs)
    raw.notch_filter([60], n_jobs=n_jobs)
    raw.resample(300, n_jobs=n_jobs)
    
    raw_out_fname = deriv_path.copy().update(extension='.fif', suffix='meg',
                                            processing='filt')    
    raw.save(raw_out_fname.fpath, overwrite=True)

    evts, evtsid = mne.events_from_annotations(raw)
    # id_dev, id_std = evtsid['3'], evtsid['4']
    epo = mne.Epochs(raw, evts, # event_id=['3','4'],
                     tmin=-1.2,
                     tmax=1.2,
                     baseline = (-0.5, 0),
                     preload=True,
                     reject=amp_thresh)
    
    epo_out_fname = deriv_path.copy().update(extension='.fif', suffix='epo', 
                                            processing='clean')
    epo.save(epo_out_fname.fpath, overwrite=True) 
    
    cov = mne.compute_covariance(epo)
    cov_out_fname = deriv_path.copy().update(extension='.fif',suffix='cov',
                                             processing='clean')
    cov.save(cov_out_fname.fpath, overwrite=True)
    return epo


#%% Commandline Options parsing
def main():
    import argparse
    parser = argparse.ArgumentParser(description=desc)
    single_parser = parser.add_argument_group('Single File only')
    single_parser.add_argument('-filename',
                        help='CTF dataset path.', 
                        default=None)
    general_parser = parser.add_argument_group('General Options')
    general_parser.add_argument('-project',
                        help='''Output folder name.  This will be generated in the
                        bids derivatives folder. (default=nihmeg)''',
                        default='nihmeg')
    general_parser.add_argument('-volume',
                        help='''Perform the MRI processing in volume space (default=False / surface-based)''',
                        action='store_true',
                        default=False)
    
    swarm_parser = parser.add_argument_group(
    '''Inputs for swarmfile generation.''',
    '''Not setting the following flags will find all possible combinations (subject/session/run/task)'''
        )
    swarm_parser.add_argument('-gen_swarmfile', action='store_true',
                              default=False
                              )
    swarm_parser.add_argument('-bids_root', default=os.getcwd(),
                              help='Top level bids folder'
                              )
    swarm_parser.add_argument('-swarm_fname', default='swarm_mriprep.sh', 
                              help='(default=swarm_mriprep.sh)'
                              )
    swarm_parser.add_argument('-subject', default=None,
                              help='Subject ID'
                              )
    swarm_parser.add_argument('-run', default=None, 
                              help='''Run number (NOTE: 01 and 1 are different)'''
                              )
    swarm_parser.add_argument('-session', default=None,
                              help='''Session ID'''
                              )
    swarm_parser.add_argument('-task', default=None, 
                              help='''Task ID'''
                              )
    
    args = parser.parse_args()
    filename = args.filename
    project_name = args.project
    proc_surf = not args.volume
    
    if args.filename!=None:
        bids_path = mne_bids.get_bids_path_from_fname(filename)
    else:
        bids_path = mne_bids.BIDSPath(root=args.bids_root)
    deriv_path = bids_path.copy().update(root=op.join(bids_path.root, 'derivatives', project_name), check=False)
    deriv_path.directory.mkdir(parents=True, exist_ok=True)
    subjects_dir=op.join(bids_path.root, 'derivatives', 'freesurfer', 'subjects')
    os.environ['SUBJECTS_DIR']=subjects_dir
    
    if args.volume == True:
        src_type = 'vol'
    else:
        src_type = 'surf'
            
    #%% The main processing component 
    if args.gen_swarmfile ==True:
        make_swarm(bids_root=args.bids_root, src_type=src_type, project=args.project, 
                       task=args.task, session=args.session, subject=args.subject, 
                       run=args.run, out_fname=args.swarm_fname)
    else:  #The Main Processing Occurs here
        tmp_t1_path = BIDSPath(root=bids_path.root,
                          session=bids_path.session,
                          subject=bids_path.subject,
                          datatype='anat',
                          suffix='T1w',
                          extension='.nii.gz',
                          check=True)    
        t1_bids_path = check_mri(tmp_t1_path)
        mripreproc(bids_path, t1_bids_path, deriv_path, surf=proc_surf)
        
        
if __name__=='__main__':
    main()
    
#%% Some minimal tests
#$(pwd)/nih2mne/megcore_prep_mri_bids.py -gen_swarmfile -swarm_fname TEST.sh -bids_root /fast/BIDS_HV_V1/bids/

