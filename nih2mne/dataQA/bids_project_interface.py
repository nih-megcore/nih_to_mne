#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jstout

TODO: 
    1)Button to launch full logfile read

####### BASIC Class explanation ###########
qa_mri_class - mri finder and json QA and freesurfer qa
meg_class - minimal wrapper
meg_list_class - List of meg_class
_subject_bids_info  - MIXIN of qa_mri_class/meg_class/meg_list_class
subject_bids_info  - Factory method to load saved or generate from new
subject_tile                         
"""
import os,os.path as op
import glob
import subprocess
import mne
import numpy as np
import pickle
import dill
from scipy.stats import zscore, trim_mean
import pandas as pd
import pyctf
import mne_bids
from nih2mne.megcore_prep_mri_bids import mripreproc
from collections import OrderedDict

CFG_VERSION = 1.0

jump_thresh = 1.5e-07 #Abs value thresh

# Configure for slurm setup
if 'OMP_NUM_THREADS' in os.environ:
    n_jobs = int(os.environ['OMP_NUM_THREADS'])
else: 
    n_jobs = -1


                           
def run_sbatch(cmd=None, mem=None, threads=None, time="02:00:00", sbatch_dep=None):
    '''
    Wrap an sbatch submission

    Parameters
    ----------
    cmd : str, optional
        Command to be run. The default is None.
    mem : str, optional
        Memory in Gigs. The default is None.
    threads : str, optional
        Number of cpus. The default is None.
    time : str, optional
        Max time allotted for compute. The default is "02:00:00".
    sbatch_dep : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    sbatch1_ID : int
        Use to chain dependencies.

    '''
    script = f'#! /bin/bash\n {cmd}\n'
    submission = subprocess.run(["sbatch", f"--mem={mem}g", f"--time={time}", f"--cpus-per-task={threads}"],
                                input=script,
                                capture_output=True,
                                text=True,
                                encoding="utf-8")
    if submission.returncode == 0:
        print(f"slurm job id: {submission.stdout}")
        sbatch1_ID=submission.stdout
        return sbatch1_ID
    else:
        print(f"sbatch error: {submission.stderr}")


#%%
class qa_megraw_object:
    '''Current minimal template - add more later'''
    def __init__(self, fname, run_qa=False):
        self.rel_path = fname
        self.fname = op.basename(fname)
        self._get_task()
        self._is_emptyroom()
        self.BADS = {}
        if run_qa==True:
            self.qa_dset()
        
        
    def qa_dset(self):
        'Run the quality assurance on the dataset'
        # Identify and drop channel jumps
        self.BADS['JUMPS'] = self._check_jumps()
        
        # Calculate the 10s epoch trim mean (60% mid) average power
        self._calc_chan_power()
        
        # Compute PSD
        self._compute_psd()
    
    def _get_task(self):
        tmp = self.fname.split('_')
        task_stub = [i for i in tmp if i[0:4]=='task'][0]
        self.task = task_stub.split('-')[-1]
        
    def _is_emptyroom(self):
        eroom_tags = ['empty', 'er','emptyroom', 'noise']
        if self.task.lower() in eroom_tags:
            self.is_emptyroom = True
        else:
            self.is_emptyroom = False
    
    def load(self, load_val=False):
        self.raw = mne.io.read_raw_ctf(self.rel_path, 
                                       preload=load_val,
                                       system_clock='ignore',
                                       clean_names=True, 
                                       verbose=False)
        if self.raw.compensation_grade != 3:
            self.raw.apply_gradient_compensation(3)
        
        # Define useable chans
        self._ref_picks = [i for i in self.raw.ch_names if i[0] in ['B', 'G', 'P','Q','R']]
        chan_picks = [i for i in self.raw.ch_names if i[0]=='M']
        self._chan_picks = [i for i in chan_picks if len(i)==5]  #Some datasets have extra odd chans
        self._megall_picks = self._ref_picks+self._chan_picks
        self._orig_bads = self.raw.info['bads']
        
    def _calc_bad_segments(self):
        'Template for calculation'
    
    
    def _check_jumps(self):
        '''
        Jump artifacts with np.diff and abs
        Returns dictionary with jumps segmented into refs and data chans
        '''
        if not hasattr(self, 'raw'):
            self.load(load_val=True)

        ref_picks = [i for i in self.raw.ch_names if i[0] in ['B', 'G', 'P','Q','R']]
        chan_picks = [i for i in self.raw.ch_names if i[0]=='M']
        chan_picks = [i for i in chan_picks if len(i)==5]  #Some datasets have extra odd chans
        megall_picks = ref_picks+chan_picks

        _tmp = self.raw.copy().pick(megall_picks)            
        # _tmp = self.raw.copy().pick(ref_picks)
        dif = mne.io.RawArray(np.diff(_tmp._data), info=_tmp.info)
        
        ch_idx, bad_samps = np.where(np.abs(dif._data) > jump_thresh)
        jump_chans = {dif.ch_names[i]:timeval for i,timeval in zip(set(ch_idx),bad_samps)}
        jump_megs = [i for i in jump_chans if i in chan_picks]
        jump_refs = [i for i in jump_chans if i in ref_picks]
        print(f'Jumps found: {len(jump_chans)}: Refs: {len(jump_refs)}: Grads: {len(jump_megs)}') 
        return {'CHANS': jump_chans, 'TSTEP': bad_samps}
    
    def _calc_chan_power(self):
        if not hasattr(self, 'raw'):
            self.load(load_val=True)
        
        epochs = mne.make_fixed_length_epochs(self.raw, duration=10.0, preload=True)
        epo_pow = epochs._data**2
        epo_pow_av = epo_pow.mean(axis=-1)  #average over time in block
        epo_robust_av = trim_mean(epo_pow_av, proportiontocut=0.2, axis=0)
        self.chan_power = epo_robust_av
        
    def _compute_psd(self):
        # epochs = mne.make_fixed_length_epochs(self.raw, duration=10.0, preload=True)
        self.psd = self.raw.compute_psd()
    
    def _is_valid(self, set_value=None):
        '''Fill in more of this -- maybe '''
        if set_value != None:
            self.is_valid = set_value
        else:
            self.is_valid = True
    
    def button_set_status(self):
        '''Set up toggle for Unchecked/GOOD/BAD'''
        if self.status=='Unchecked':
            self.status = 'GOOD'
        elif self.status=='GOOD':
            self.status = 'BAD'
        elif self.status=='BAD':
            self.status = 'Unchecked'
    
    def set_status(self, status):
        self.status=status
        
    @property
    def coil_locs_dewar(self):
        return pyctf.getHC.getHC(op.join(self.fname, op.basename(self.fname).replace('.ds','.hc')), 'dewar')
    
    #Currently not working -- these get stripped out in BIDS
    @property
    def coil_locs_head(self):
        return pyctf.getHC.getHC(op.join(self.fname, op.basename(self.fname).replace('.ds','.hc')), 'head')

    @property    
    def event_counts(self):
        if not hasattr(self, 'raw'):
            self.load()
        return pd.DataFrame(self.raw.annotations).description.value_counts()

    def __repr__(self):
        tmp_ = f'megraw: {self.task} : {self.fname}'
        if hasattr(self, 'BADS'):
            if 'JUMPS' in self.BADS:
                bads_len = self.BADS['JUMPS']['CHANS'].__len__()
            else:
               bads_len='Untested' 
        else:
            bads_len='Untested'
        if bads_len !=0:
            tmp_ += f' :: JUMPS: {bads_len}'
        return tmp_
            
#%%
class meglist_class:
    def __init__(self, subject=None, bids_root=None):
        dsets = glob.glob(f'{op.join(bids_root, subject, "**", "*.ds")}',
                          recursive=True)
        tmp = [qa_megraw_object(i) for i in dsets if op.basename(i) not in ['hz.ds','hz2.ds']]
        self.meg_list = tmp
        self.meg_emptyroom = [i for i in self.meg_list if i.is_emptyroom]
    
    def _pick_meg_from_list(self, choice_quote='Choice:\n', add_allchoice=False, 
                            idx=None):
        if idx != None:
            # Don't show the prompt - this is useful for GUI
            return self.meg_list[idx]
        for idx, dset in enumerate(self.meg_list):
            print(f'{idx}: {dset.fname}')
        if add_allchoice==True:
            print(f'{idx+1}: Choose all datasets')
        if choice_quote.replace(' ','').endswith('\n'):
            choice_quote+='\n'
        dset_idx = int(input(choice_quote))
        if str(dset_idx) == str(idx+1):
            return 'all'
        else:
            return self.meg_list[dset_idx]
    
    def plot_meg(self, idx=None, hp=None, lp=None, montage=None, f_mains=False):
        if idx == None:
            dset = self._pick_meg_from_list('Enter the number associated with the MEG dataset to plot: ')
        else:
            dset = self.meg_list[idx]
        dset.load()
        self.current_meg_dset = dset.raw.copy()
        if montage != None:
            if type(montage) != list:
                _tmp = montage(self.current_meg_dset)  #Call the function to eval montage
                montage = _tmp
            if len(montage) < 80:
                num_chans = len(montage)
            else:
                num_chans = 20 
            self.current_meg_dset.pick_channels(montage, ordered=True)
            
            # Conditionally Filter 60Hz if notch button checked.  
            self.current_meg_dset.load_data()
            # Check if any main "M" channels available - notch will fail
            misc_only=True if len([i for i in montage if i.startswith('M')])==0 else False
            if (f_mains!=False) and (not misc_only): self.current_meg_dset.notch_filter(float(f_mains), n_jobs=n_jobs)
            
            # Plot
            test_plot = self.current_meg_dset.plot(highpass=hp, lowpass=lp, n_channels=num_chans)
        else:
            self.current_meg_dset.load_data()
            # Conditionally filter 60hz if notch button checked
            if f_mains!=False: self.current_meg_dset.notch_filter(float(f_mains), n_jobs=n_jobs)
            # Plot
            test_plot = self.current_meg_dset.plot(highpass=hp, lowpass=lp) 
        return test_plot
    
    @property
    def meg_count(self):
        return len(self.meg_list)
    
class qa_mri_class:    
    def __init__(self, subject=None, bids_root=None):
        mr_list = glob.glob(f'{op.join(bids_root, subject,"**", "anat/*")}', recursive=True)
        all_mri_list = []
        for i in mr_list:
            if (i[-4:]=='.nii') or (i[-7:]=='.nii.gz'):
                all_mri_list.append(i)
        self.all_mris = all_mri_list
        if len(self.all_mris)==0:
            self.mri = None
            self.mri_json_qa = 'No MRIs'
        elif len(self.all_mris)==1:
            self.mri = self.all_mris[0]
            self.mri_json = self._get_matching_mr_json()
        else:
            self.mri = 'Multiple'
            self.mri_json_qa = 'Undetermined - Multiple MRIs'
        
        if (self.mri != 'Multiple') and (self.mri != None):
            self._valid_fids()
            
    def mri_selection_override(self, override_mri=None):
        if override_mri == None:
            for idx, fname in enumerate(self.all_mris):
                print(f'{idx}: {fname}')
            fname_idx = int(input('Choose an MRI to use in processing:\n'))
            self.mri = self.all_mris[fname_idx]
        else:
            self.mri = override_mri
        self.mri_json = self._get_matching_mr_json()
        assert len(self.mri_json)>3
        self._valid_fids()
        print('Updated MRI')
    
    def _get_matching_mr_json(self):
        if self.mri.endswith('.nii'):
            self.mri_json = self.mri.replace('.nii','.json')
        else:
            self.mri_json = self.mri.replace('.nii.gz','.json')
        return self.mri_json
        
        
    def _sort_T1(self):
        pass
    
    def _valid_fids(self):
        import json
        if not op.exists(self.mri_json):
            self.mri_json_qa = 'No MRI JSON'
            return 
        with open(self.mri_json) as f:
            json_data = json.load(f)
        if 'AnatomicalLandmarkCoordinates' not in json_data.keys():
            self.mri_json_qa = 'BAD'
            return
        fids = json_data['AnatomicalLandmarkCoordinates']
        fids_keys = sorted(fids.keys())
        if fids_keys == ['LPA', 'NAS', 'RPA']:
            self.mri_json_qa = 'GOOD'
        else:
            self.mri_json_qa = 'BAD'

    def check_fs_recon(self):
        '''
        Returns status of freesurfer reconstruction
    
        Parameters
        ----------
        subjid : str
            Subject ID
        subjects_dir : str
            Freesurfer subjects dir
    
        Returns
        -------
        out_dict: dict
            
    
        '''
        logfile = op.join(self.subjects_dir, self.subject, 'scripts', 'recon-all.log')
        if not op.exists(logfile):
            finished = False
            started = False
            fs_success_line=[]
        else:
            started = True
            with open(logfile) as f:
                fs_success_line = f.readlines()[-1]
        if 'finished without error' in fs_success_line:
            finished = True
        else:
            finished = False
        has_lhpial = os.path.exists(op.join(self.subjects_dir, self.subject, 'surf', 'lh.pial'))
        has_rhpial = os.path.exists(op.join(self.subjects_dir, self.subject, 'surf', 'rh.pial'))
        out_dict = dict(fs_success = finished,
                        fs_started = started,
                    lhpial = has_lhpial, 
                    rhpial = has_rhpial)
        return out_dict
        

class _subject_bids_info(qa_mri_class, meglist_class):
    '''Subject Status Mixin of MRI and MEG classes'''
    def __init__(self, subject, bids_root=None, subjects_dir=None, 
                 deriv_project=None):
        if subject[0:4]=='sub-':
            self.subject = subject
            self.bids_id = subject[4:]
        else:
            self.subject = 'sub-'+subject
            self.bids_id = subject
        if bids_root==None:
            self.bids_root=os.getcwd()
        else:
            self.bids_root = bids_root
        if deriv_project == None:
            self.deriv_project = 'nihmeg'
        else:
            self.deriv_project = deriv_project
        self.deriv_root = op.join(self.bids_root, 'derivatives', self.deriv_project)
        
        # Save variables
        self.qa_output_dir = op.join(self.bids_root, 'derivatives', 'megQA')
        self.qa_default_fname = op.join(self.qa_output_dir, self.subject + '.pkl')
        
        if not op.exists(op.join(bids_root, self.subject)):
            raise ValueError(f'Subject {self.subject} does not exist in {bids_root}')
        
        if subjects_dir==None:
            self.subjects_dir = op.join(bids_root, 'derivatives','freesurfer','subjects')
        else:
            self.subjects_dir = subjects_dir
        
        # MEG Component
        meglist_class.__init__(self, self.subject, self.bids_root)
        
        # MRI Component
        qa_mri_class.__init__(self, subject=self.subject, bids_root=self.bids_root)
        
        # Freesurfer Component
        self.fs_recon = self.check_fs_recon()
        
    def _reload_info(self):
        self.fs_recon = self.check_fs_recon()
        
    def proc_freesurfer(self): 
        cmd = f"export SUBJECTS_DIR={self.subjects_dir}; recon-all -all -i {self.mri} -s {self.subject}" 
        os.makedirs(self.subjects_dir, exist_ok=True)
        outcode = run_sbatch(cmd, mem=6, threads=2, time="24:00:00")
        return outcode    
    
    def plot_mri_fids(self):
        ''' Open a triaxial image of the fiducial locations'''
        from nih2mne.utilities.qa_fids import plot_fids_qa
        plot_fids_qa(subjid=self.subject,
                     bids_root=self.bids_root, 
                     outfile=None, block=True, mri_override=[self.mri])
        # tmp_ = input('Hit any button to close')
    
    def plot_3D_coreg(self, idx=None):
        dset = self._pick_meg_from_list(choice_quote='Enter the number associated with the MEG dataset to plot coreg: \n',
                                        idx=idx)
        dset.load()
        bids_path = mne_bids.get_bids_path_from_fname(dset.fname)
        t1_bids_path = mne_bids.get_bids_path_from_fname(self.mri)
        trans = mne_bids.get_head_mri_trans(bids_path, t1_bids_path=t1_bids_path, 
                                            extra_params=dict(system_clock='ignore'),
                                            fs_subject=self.subject, fs_subjects_dir=self.subjects_dir)
        mne.viz.plot_alignment(dset.raw.info, 
                               trans=trans,subject=self.subject, 
                               subjects_dir = self.subjects_dir, dig=True)
        
    @property
    def info(self):
        tmp = f'Subject {self.subject}\n'
        tmp += f'MEG Scans: {self.meg_count}\n'
        if len(self.meg_emptyroom) == 0:
            tmp += 'MEG Emptyroom: None\n'
        else:
            tmp += f'MEG Emptyroom: ({len(self.meg_emptyroom)})\n'
            for i in self.meg_emptyroom:
                tmp += f'   {i.fname}\n'
        tmp += f'MRI Used: {self.mri}\n'
        tmp += f'MRI fiducials: {self.mri_json_qa}\n'
        if self.fs_recon['fs_success']==True:
            tmp += 'Freesurfer: Successful Recon'
        else:
            if self.fs_recon['fs_started']==False:
                tmp += 'Freesurfer: Has not been performed'
            else:
                logfile = op.join(self.subjects_dir, self.subject, 'scripts', 'recon-all.log')
                tmp += f'Freesurfer: ERROR : Check log {logfile}'
        return tmp
    
    def mri_preproc(self, surf=True, fname=None):
        '''
        Perform mri preprocessing  (bem / src / trans / fwd models)
        If an fname is provided, this dataset will be run, otherwise a menu
        with the different runs will be provided to choose.
        
        fname can be "all" to loop over all meg datasets, ignoring emptyroom

        Parameters
        ----------
        surf : BOOL, optional
            Surface (True) or Volumetric (False). The default is True.
        fname : str, optional
            Path to meg dataset. The default is None.

        Raises
        ------
        ValueError
            If no freesurfer, this will raise an error.

        Returns
        -------
        qa_megraw_object or "all"

        '''
        
        if self.fs_recon['fs_started'] != True:
            raise ValueError('Freesurfer processing of the data has not been performed')
        if (self.fs_recon['fs_success'] != True) and (surf==True):
            raise ValueError('Freesurfer did not complete successfully')
        if fname == None:
            dset = self._pick_meg_from_list(choice_quote='Pick an index to process the MEG src/fwd/trans/bem: \n',
                                            add_allchoice=True)
        else:
            dset = fname

        if dset == 'all':
            dsets = self.meg_list
        else:
            dsets = [dset]
        del dset
        
        for dset in dsets:
            if dset.is_emptyroom:
                continue
            bids_path_meg = mne_bids.get_bids_path_from_fname(dset.fname)
            deriv_path =  bids_path_meg.copy().update(root=self.deriv_root, check=False)
            deriv_path.directory.mkdir(parents=True, exist_ok=True)
            mripreproc(bids_path=bids_path_meg,
                       t1_bids_path= mne_bids.get_bids_path_from_fname(self.mri),
                       deriv_path = deriv_path, 
                       surf=surf, subjects_dir=self.subjects_dir)
                   
    def __repr__(self):
        return self.info
    
    def save(self, fname=None, overwrite=False):
        if not op.exists(self.qa_output_dir): os.makedirs(self.qa_output_dir)
        if fname == None:
            fname = self.qa_default_fname
        
        #Remove fully loaded meg before saving
        for meg_dset in self.meg_list:
            if hasattr(meg_dset, 'raw'):
                del meg_dset.raw
        
        fname_exists = op.exists(fname)
        if fname_exists and overwrite==False:
            overwrite = input(f'{fname} exists. \n Do you want to write over (y/n): \n')
            if overwrite[0].lower()=='y':
                overwrite=True
            else:
                return
            
        if (fname_exists==False) or (overwrite==True):
            with open(fname, 'wb') as f:
                dill.dump(self, f)
            if overwrite==True:
                print(f'Overwrote: {fname}')
    
        
        


def subject_bids_info( subject=None, bids_root=None, subjects_dir=None, 
                              deriv_project=None, force_update=False):
    '''
    Main entrypoint for subject bids infor (Factory method) to initialize 
    subject_bids_info class. This is necessary to be able to preload 
    saved projects

    Parameters
    ----------
    subject : str, required
        Subject ID. The default is None.
    bids_root : str, optional
        Path. The default is None.
    subjects_dir : str, optional
        Override Freesurfer subjects dir. The default is None.
    deriv_project : str, optional
        Project output in derivatives folder. The default is None.
    force_update : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    QA object
        QA object for BIDS.

    '''
    if subject[0:4]!='sub-':
        subject = 'sub-'+subject
    qa_default_fname = op.join(bids_root, 'derivatives', 'megQA', subject+'.pkl')
    if op.exists(qa_default_fname) and (force_update==False):
        with open(qa_default_fname, 'rb') as f:
            bids_info = dill.load(f)
            
        bids_info._reload_info()
        return bids_info
    else:
        tmp_ = _subject_bids_info(subject=subject, bids_root=bids_root, 
                          subjects_dir=subjects_dir,
                          deriv_project=deriv_project)
        tmp_.save(overwrite=True)
        return tmp_

class _bids_subject_list():
    def __init__(self, subject_list, bids_root):
        for subject in subject_list:
            self.subjects={}
            self.subjects[subject]= subject_bids_info( subject=subject, bids_root=bids_root)
    
    def __repr__(self):
        return [i.subject for i in self.subjects]
            
# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
        

#%%    
#For initializing and maintaining a larger project
# This will loop over all subjects and save out the pkl file.
# TODO: 
    # Make dot accessible subject with object attributes - Munch does not allow that
class bids_project():
    def __init__(self, bids_root=None, project_root=None, force_update=False):
        _subjects = glob.glob('sub-*', root_dir=bids_root)
        self.subjects = OrderedDict()  
        self.error_subjects= []
        self.bids_root = bids_root
        if project_root == None:
            self.project_root = op.join(self.bids_root, 'derivatives', 'nihmeg')
        for subject in _subjects:
            try:
                tmp_ = subject_bids_info( subject=subject, bids_root=bids_root, 
                                         force_update=force_update)
                self.subjects[subject] = tmp_
                if not op.exists(tmp_.qa_default_fname): tmp_.save()
            except:
                self.error_subjects.append(subject)
        
        self._fs_status()
        self.compile_issues()
    
    def __repr__(self):
        txt = f'BIDS root: {self.bids_root}\n'
        txt += f'Project root: {self.project_root}\n'
        txt += f'There are {len(self.subjects)} subjects in BIDS folder {op.basename(self.bids_root)}\n'
        if len(self.error_subjects) > 0:
            txt += f'!! There were {len(self.error_subjects)} subject errors during reading !!\n'
        #Freesurfer
        s_1, f_1, ns_1 = len(self.fs_recon['success']), len(self.fs_recon['failed']), len(self.fs_recon['notStarted'])
        txt += f'Freesurfer:: (Success {s_1}) / (Failed {f_1}) / (NotStarted {ns_1})\n' 
        
        # MRIs
        txt += '\n########## ISSUES ##########\n'
        txt += f'MRI: {len(self.issues["MRI"])} \n'
        txt += f'MRI FIDS: {len(self.issues["MRI_FIDS"])} \n'
        txt += f'Freesurfer: {f_1+ns_1}'

        return txt
    
    def compile_issues(self):
        self.issues = OrderedDict()
        self.issues['MRI'] = [i.subject for i in self.subjects.values() if i.mri in ['Multiple',None]]
        self.issues['MRI_FIDS'] = [i.subject for i in self.subjects.values() if i.mri_json_qa == 'GOOD']
        
        self.issues['Freesurfer_failed'] = self.fs_recon['failed']
        self.issues['Freesurfer_notStarted'] = self.fs_recon['notStarted']
        # self.issues['Bad_Subjects'] = 
        
    def run_anat_pipeline(self):
        return None
        
    
    
    def _fs_status(self):
        self.fs_recon = {}
        self.fs_recon['success'] = []
        self.fs_recon['failed'] = []
        self.fs_recon['notStarted'] = []
        for subj_id in self.subjects.keys():
            bidsi = self.subjects[subj_id]
            if bidsi.fs_recon['fs_started']==False:
                self.fs_recon['notStarted'].append(bidsi.subject)
                continue
            if bidsi.fs_recon['fs_success']==True:
                self.fs_recon['success'].append(bidsi.subject)
            else:
                self.fs_recon['failed'].append(bidsi.subject)
    
    # def prep_freesurfer(self):
    #     for i in self.subjects:
            
            
        
            
    
    # # def _get_missing(self):
    # #     dframe_missing = 
        
    
    # def _check(self):
    #     return None
    
    # def _return_issue(self):
    #     return None
        
    
                
                
# bids_pro = bids_project(bids_root = '/fast2/BIDS')     
# bids_pro.compile_issues()  
# bids_pro.issues     
        
        
        
        

