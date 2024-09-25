#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:50:57 2024

@author: jstout

TODO: 
    1)Functionalize the window loop operations, so they can be called outside 
    of the while loop - basically this is preventing the read_cfg operation from
    changing the layout.
    2)Button to launch full logfile read
    3) Button to view MRI coreg
    4) Button to view 3D coreg in helmet (requires runtime to eval headsurf)

Code layout:
    1) Define call options (opts)
        a) Override default opts with config file
    2) Define layout based on options    
    3) Read / Write Config options from opts object
    4) RUN: Format opts object into commandline string and execute
        After running - open Summary text
    
                       
"""
import PySimpleGUI as sg
import os,os.path as op
import glob
import subprocess
import mne
import numpy as np
from scipy.stats import zscore, trim_mean
import pandas as pd
import pyctf
import mne_bids

CFG_VERSION = 1.0

font = ("Arial", 25)
sg.set_options(font=font)

global size_mult, font_size, x_size, y_size
size_mult=2
font_size=12  
x_size = 500*size_mult
y_size = 600*size_mult

jump_thresh = 1.5e-07 #Abs value thresh


def collapse(layout, key, visible):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this section visible / invisible
    :param visible: visible determines if section is rendered visible or invisible on initialization
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key, visible=visible, pad=(0,0)))

class window_opts:
    def __init__(self, config=False):
        # Button Entries
        self.anonymize = False
        self.ignore_mri_checks = False

        # Standard Entries
        self.bids_dir = op.join(os.getcwd(), 'bids_dir')
        self.subjects = self.get_subjs()
        
        
        self.meg_input_dir = None
        self.bids_session = 1
        self.subjid_input = None
        self.bids_id = None
        self.coreg = 'Brainsight'

        ## Afni Coreg:
        self.mri_brik = None

        ## Brainsight Coreg:
        self.mri_bsight = None
        self.mri_bsight_elec = None

        ## Optional Overrides:
        self.ignore_eroom = False
        self.autocrop_zeros = False
        self.freesurfer = None
        self.eventID_csv = None
        # Run standardize_eventID_list.py
        
        self.config = config
        if config != False:
            write_opts = read_cfg(config)
            self.update_opts(opts=write_opts)
    
    def update_opts(self, opts=None):
        for key, val in opts.items():
            if hasattr(self, key):
                setattr(self, key, val)
    
    def get_subjs(self):
        '''List the subjects in the bids directory'''
        tmp_ = glob.glob(op.join(self.bids_dir, 'sub-*'))
        return [op.basename(i) for i in tmp_]
    
    
    

def read_cfg(cfg_fname):
    with open(cfg_fname, 'r') as f:
        lines = f.readlines()
    lines = [i.strip('\n') for i in lines]
    write_opts={}
    for i in lines:
        _key, _val = i.split(':')
        if _val == 'True':
            _val=True
        elif _val == 'None':
            _val=None
        elif _val == 'False':
            _val=False
        write_opts[_key]=_val
    return write_opts
    
        

def make_layout(opts=None):
    '''Generate 3 sections - Project / Subject_RAW / Subject_Preproc'''  
    required_opts = [
        [sg.Text('BIDS DIR:'), 
         sg.InputText(key='-BIDS_DIR-', enable_events=True),
         sg.FolderBrowse(target='-BIDS_DIR-')
         ],
        ]
           
    project_opts = [
        [sg.Text(' -- PROJECT_QA -- ')],
        [sg.Button('Compute Table'), sg.Button('View Table', disabled=True)], 
        [sg.Button('Missing Data', disabled=True)], 
        [sg.Button('Assess Noise Levels', disabled=True)]
        ]
    
    subject_required_opts = [
        [sg.Text(' -- Subject Section -- ')],
        [sg.Button('Select Subject')]
         ]
    subject_raw_opts = [
        [sg.Text(' -- SUBJECT INPUT QA -- ')],
        [sg.Button('MRI ANON', disabled=True),
         sg.Button('Fiducials', disabled=True)],
        [sg.Button('3D Coreg', disabled=True),
         sg.Button('Data QA', disabled=True)]
        ]
    
    subject_preproc_opts = [
        [sg.Text('  -- SUBJECT Preproc QA --')],
        [sg.Button('Run MR Prep', disabled=False), sg.Button('Run MR Prep (VOL)', disabled=False)]
        ]    
         
    # -- Assemble Layout --
    layout = required_opts
    layout.append([sg.Text(' ')])
    layout.append(project_opts)
    layout.append([sg.Text(' ')])
    layout.append(subject_raw_opts)
    layout.append([sg.Text(' ')])
    layout.append(subject_preproc_opts)
    layout.append([sg.Text(' ')])
    layout.append([sg.Button('Print CMD', key='-PRINT_CMD-'), sg.Button('RUN', key='-RUN-'), 
                   sg.Button('Write Cfg', key='-WRITE_CFG-'), sg.Button('EXIT')])
    return layout
        
def selector_POPUP(data, text_item='Select'):
    layout = [
        [sg.Text(text_item)],
        [sg.Listbox(data, size=(20,5), key='SELECTED')],
        [sg.Button('OK')],
    ]
    window = sg.Window('POPUP', layout).Finalize()
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'OK':
            break
        else:
            print('OVER')
    window.close()
    if values and values['SELECTED']:
        return values['SELECTED']
    
def write_cfg(opts, fname=None):
    '''
    From a list of writeable options - save these in key:value pair format sep=:
    '''
    save_opt_list = ['anonymize', 'autocrop_zeros', 'freesurfer', 'ignore_eroom',
                    'ignore_mri_checks', 'coreg', 'bids_dir', 'bids_session']
    write_opts = []
    for opt_tag in save_opt_list:
        if hasattr(opts, opt_tag):
            write_opts.append(f'{opt_tag}:{getattr(opts,opt_tag)}\n')
    with open(fname, 'w') as f:
        f.writelines(write_opts)

def get_window(options=None):
    layout = make_layout(opts=options)
    window = sg.Window('MEG BIDS conversion', layout, resizable=True, auto_size_buttons=True, 
                   scaling=True, size=(x_size, y_size))
    del layout
    return window

single_flag_list = ['anonymize', 'autocrop_zeros', 'freesurfer', 'ignore_eroom',
                    'ignore_mri_checks']
drop_flag_list = ['coreg', 'read_from_config', 'config', 'update_opts', 'error_log', 'full_log', 'fids_qa']
def format_cmd(opts):
    '''
    Write out the commandline options from the opts object.  Special cases 
    for the single flag vs flag w/option entry.

    Parameters
    ----------
    opts : opt object
        DESCRIPTION.

    Returns
    -------
    cmd : str

    '''
    arglist = ['make_meg_bids.py']
    for i in value_writedict.values():
        if i in drop_flag_list:
            if (i == 'coreg') and (opts.coreg == 'None'):
                arglist.append('-ignore_mri_checks')
                continue
            else:
                continue
        flag_val =  getattr(opts, i)
        if i in single_flag_list:
            if flag_val == True:
                arglist += [f'-{i}']
            else:
                continue
        else:
            if flag_val != None:
                arglist += [f'-{i} {getattr(opts, i)}']
    cmd = ' '.join(arglist)
    return cmd 

config_fname = False
if __name__=='__main__':
    ''' Get config file from the commandline - preset to False above'''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='Config file: typically ending in cfg', 
                        default=False)
    args = parser.parse_args()
    if hasattr(args, 'config'):
        config_fname = args.config

def qa_gui(config_fname=False):    
    ## Setup and run gui
    opts = window_opts(config=config_fname) #This defaults to False if not set
    window = get_window(opts)
    
    # Conversion dictionary between GUI variables and cmdline vars
    _tmp = [i for i in dir(opts) if not i.startswith('_')]
    global value_writedict
    value_writedict = {f'-{i.upper()}-':i for i in _tmp}
    
    while True:
        event, values = window.read()
        
        if event == '-READ_CFG-':
            cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', 
                                          default_extension='.cfg')
            write_opts = read_cfg(cfg_fname)
            opts.update_opts(write_opts)
        
        # Update object options if event triggered
        if event in value_writedict.keys():
            setattr(opts, value_writedict[event], values[event])
        
        ## BIDS PROJECT LEVEL
        if event == 'Compute Table':
            from nih2mne.utilities.print_bids_table import gui_interface as pbids_table
            opts.bids_table_fname = op.join(op.dirname(opts.bids_dir), 'bids_prep_logs','BIDS_table.csv')
            pbids_table(opts.bids_dir, out_fname =opts.bids_table_fname)
            window['View Table'].update(disabled=False)
        
        if event == 'View Table':
            cmd = f'xdg-open {opts.bids_table_fname}'
            subprocess.run(cmd.split())
            
        ## BIDS SUBJECT LEVEL QA 
        
        
        ## MRI Preprocess
        if (event == 'Run MR Prep') or (event == 'Run MR Prep (VOL)'):
            subjects = sorted(glob.glob('sub-*', root_dir=opts.bids_dir))
            subject = selector_POPUP(subjects, text_item='Select a subject')[0]
            dsets = glob.glob(f'**/{subject}_*.ds', recursive=True, root_dir=op.join(opts.bids_dir, subject))
            dsets = ['ALL'] + dsets
            dsets_sel = selector_POPUP(dsets, text_item = 'Select a dataset to run MR preprocessing')
            if dsets_sel[0] == 'ALL':
                dsets.remove('ALL')
                dsets_sel = dsets
            #Add full path for processing
            dsets_sel = [op.join(opts.bids_dir, subject, i) for i in dsets_sel]
            for dset in dsets_sel:
                cmd = f'megcore_prep_mri_bids.py -bids_root {opts.bids_dir} -filename {dset}'
                if event == 'Run MR Prep (VOL)':
                    cmd+=' -volume'
                subprocess.run(cmd.split())
            
        
        # Logic for displaying coreg options
        if event == '-COREG-':
            if values['-COREG-'] == 'BrainSight':
                window['-COREG_BSIGHT-'].update(visible=True)
                window['-COREG_AFNI-'].update(visible=False)
            elif values['-COREG-'] == 'Afni':
                window['-COREG_AFNI-'].update(visible=True)
                window['-COREG_BSIGHT-'].update(visible=False)
            elif values['-COREG-'] == 'None':
                window['-COREG_BSIGHT-'].update(visible=False)
                window['-COREG_AFNI-'].update(visible=False)
        
        # If input directory is chosen - automatically select subject id and set
        if event == '-MEG_INPUT_DIR-':
            search_dir = values['-MEG_INPUT_DIR-']
            tmp = glob.glob(op.join(search_dir, '*.ds'))
            tmp = [op.basename(i) for i in tmp]
            tmp = list(set([i.split('_')[0] for i in tmp]))
            if len(tmp) == 1:
                values['-SUBJID_INPUT-']=tmp[0]
                opts.subjid_input =tmp[0]
                window['-SUBJID_INPUT-'].update(tmp[0])
            if len(tmp) > 1:
                tmp = subject_selector_POPUP(tmp)
                window['-SUBJID_INPUT-'].update(tmp[0])
                opts.subjid_input=tmp[0]
                
        if event == '-PRINT_CMD-':
            cmd = format_cmd(opts)
            print(cmd)
        
        if event == '-WRITE_CFG-':
            cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', default_path='nih_bids_gui.cfg', 
                                          default_extension='.cfg')
            write_cfg(opts, fname=cfg_fname)
        
        
        if event == '-RUN-':
            print(f'Running the command: {cmd}')
            cmd = format_cmd(opts)
            out_txt = subprocess.run(cmd.split(), check=True, capture_output=True)
            summary = []
            _start = False
            for i in str(out_txt.stdout).split('\\n'):
                if '########### SUMMARY #################' in i:
                    _start = True
                if _start:
                    summary.append(i)
            sg.popup_get_text('\n'.join(summary), title='SUMMARY')
            _tmp = op.dirname(opts.bids_dir)
            setattr(opts, 'error_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_err_log.txt'))
            setattr(opts, 'full_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_log.txt'))
            setattr(opts, 'fids_qa',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_fids_qa.png'))  
            window['-CHECK_TRIAX_COREG-'].update(disabled=False)
            print('FINISHED')
            
        if event == '-CHECK_TRIAX_COREG-':
            subprocess.run(f'xdg-open {opts.fids_qa}'.split())
        if event == sg.WIN_CLOSED or event == 'EXIT': # if user closes window or clicks cancel
            break
        
    window.close()

#%%
# Main window
# Check bids data
'''
qa_mri_class - mri finder and json QA and freesurfer qa
meg_class - minimal wrapper
meg_list_class - List of meg_class
subject_bids_info  - MIXIN
subject_tile  

'''

# bidsroot_template = 'bidsroot'
# projectroot_template = 'projectroot'


# # Data checks 
# data_checks = {
#     'MEGraw':{'key':
#               '*.ds'
#               },
#     'MRIraw':{'key':
#               ['*.nii.gz', '*.nii']
#               },
#     'MRIfree':{'key':
#                    ['surf/lh.pial', 'surf/rh.pial'],
#                'logkey':[]
#                    },
#     'Coreg':[   ], 
#     'MRIprep':[  ],
#     }
    
def check_fs_recon(subjid, subjects_dir):
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
    logfile = op.join(subjects_dir, subjid, 'scripts', 'recon-all.log')
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
    has_lhpial = os.path.exists(op.join(subjects_dir, subjid, 'surf', 'lh.pial'))
    has_rhpial = os.path.exists(op.join(subjects_dir, subjid, 'surf', 'rh.pial'))
    out_dict = dict(fs_success = finished,
                    fs_started = started,
                lhpial = has_lhpial, 
                rhpial = has_rhpial)
                
    return out_dict

#%%
class qa_megraw_object:
    '''Current minimal template - add more later'''
    def __init__(self, fname):
        self.rel_path = fname
        self.fname = op.basename(fname)
        self._get_task()
        self._is_emptyroom()
        
        # Identify and drop channel jumps
        self.BADS = {}
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
        return pd.DataFrame(self.raw.annotations).description.value_counts()

    def __repr__(self):
        tmp_ = f'megraw: {self.task} : {self.fname}'
        bads_len = self.BADS['JUMPS']['CHANS'].__len__()
        if bads_len !=0:
            tmp_ += f' :: JUMPS: {bads_len}'
        return tmp_
            
#%%
class meglist_class:
    def __init__(self, subject=None, bids_root=None):
        dsets = glob.glob(f'{op.join(bids_root, subject, "**", "*.ds")}',
                          recursive=True)
        tmp = [qa_megraw_object(i) for i in dsets]
        self.meg_list = tmp
        self.meg_emptyroom = [i for i in self.meg_list if i.is_emptyroom]
    
    def _print_meg_list_idxs(self):
        for idx, dset in enumerate(self.meg_list):
            print(f'{idx}: {dset.fname}')
    
    def plot_meg(self):
        self._print_meg_list_idxs()
        dset_idx = input('Enter the number associated with the MEG dataset to plot: \n')
        dset_idx = int(dset_idx)
        self.meg_list[dset_idx].raw.plot()    
    
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
            if self.mri.endswith('.nii'):
                self.mri_json = self.mri.replace('.nii','.json')
            else:
                self.mri_json = self.mri.replace('.nii.gz','.json')
        else:
            self.mri = 'Multiple'
            self.mri_json_qa = 'Undetermined - Multiple MRIs'
        
        if (self.mri != 'Multiple') and (self.mri != None):
            self._valid_fids()
            
        
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
        

class subject_bids_info(qa_mri_class, meglist_class):
    '''Subject Status Mixin of MRI and MEG classes'''
    def __init__(self, subject, bids_root=None, subjects_dir=None):
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
        self.fs_recon = check_fs_recon(self.subject, self.subjects_dir)
    
    def plot_mri_fids(self):
        ''' Open a triaxial image of the fiducial locations'''
        from nih2mne.utilities.qa_fids import plot_fids_qa
        plot_fids_qa(subjid=self.subject,
                     bids_root=self.bids_root, 
                     outfile=None, block=True)
        # tmp_ = input('Hit any button to close')
    
    def plot_3D_coreg(self):
        self._print_meg_list_idxs()
        dset_idx = input('Enter the number associated with the MEG dataset to plot coreg: \n')
        dset_idx = int(dset_idx)
        bids_path = mne_bids.get_bids_path_from_fname(self.meg_list[dset_idx].fname)
        t1_bids_path = mne_bids.get_bids_path_from_fname(self.mri)
        trans = mne_bids.get_head_mri_trans(bids_path, t1_bids_path=t1_bids_path, 
                                            extra_params=dict(system_clock='ignore'),
                                            fs_subject=self.subject, fs_subjects_dir=self.subjects_dir)
        mne.viz.plot_alignment(self.meg_list[dset_idx].raw.info, 
                               trans=trans,subject=self.subject, 
                               subjects_dir = self.subjects_dir)
        
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
        
    def __repr__(self):
        return self.info
    
    def save(self, fname=None, overwrite=False):
        import pickle
        if fname == None:
            raise ValueError('fname must be set during save')
        
        #Remove fully loaded meg before saving
        for meg_dset in self.meg_list:
            if hasattr(meg_dset, 'raw'):
                del meg_dset.raw
        
        if op.exists(fname) and overwrite==False:
            raise ValueError(f'The fname already exists: {fname}')
        else:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)

class subject_tile(subject_bids_info):
    '''Attach GUI tile properties to bids information'''
    def __init__(self, subject=None, bids_root=None, subjects_dir=None):
        subject_bids_info.__init__(self, subject=subject, bids_root=bids_root, 
                                   subjects_dir=subjects_dir)
    
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
        
    # def set_type(self, qa_type):
    #     self.qa_type = qa_type
        
    
#%% 

def test_subject_bids_info():
    import nih2mne
    bids_root = op.join(nih2mne.__path__[0], 'test_data','BIDS_test')
    test = subject_bids_info('sub-S01', bids_root=bids_root)
    assert test.meg_count == 2
    tmp_ = test.mri
    tmp_ = tmp_.split('test_data')[-1]
    assert tmp_ == '/BIDS_test/sub-S01/ses-1/anat/sub-S01_ses-1_T1w.nii.gz'
    
    airpuff = test.meg_list[0]
    val_counts = airpuff.event_counts
    assert val_counts['stim']==103
    assert val_counts['missingstim']==17
        
    
test = subject_bids_info('sub-ON02811', bids_root=os.getcwd())
test = subject_bids_info('sub-ON69163', bids_root=os.getcwd())
test.save(op.join('QA_objs', 'sub-ON69163.pkl'), overwrite=True)

test = subject_tile(subject='sub-ON11394', bids_root=os.getcwd())
        
test = subject_tile(subject='sub-ON08710', bids_root=os.getcwd())        
subject_tile_list = [subject_tile(i, bids_root=bids_root) for i in glob.glob('sub-*')]


fail=[]
for i in glob.glob('sub-*'):
    try:
        tmp = subject_bids_info(i, bids_root='/fast2/BIDS')
        tmp.save(op.join('QA_objs', f'{i}_bidsqa.pkl'), overwrite=True)
        del tmp
    except:
        fail.append(i)

# power_stack = {}
# psd_stack = {}
psd_dframe_list = []
pow_dframe_list = []
fail = []
for i in glob.glob('QA_objs/*bidsqa.pkl'):
    with open(i, 'rb') as f:
        qaobj = pickle.load(f)
        # power_stack[qaobj.subject]={}
        # psd_stack[qaobj.subject]={}        
        for dset in qaobj.meg_list:
            try:
                if (not dset.is_emptyroom) and (dset.task != 'artifact'):
                    #power_stack[qaobj.subject][dset.task]=dset.chan_power
                    dset.bids_root=os.getcwd()
                    dset.load()
                    pow_tmp = pd.DataFrame(dset.chan_power[np.newaxis,:], columns=[dset.raw.ch_names])
                    pow_tmp['subject'] = qaobj.subject
                    pow_tmp['task'] = dset.task
                    pow_dframe_list.append(pow_tmp)
                    # psd_stack[qaobj.subject][dset.task]=dset.psd
                    tmp = pd.DataFrame(dset.psd._data, columns=dset.psd.freqs, index=dset.psd.ch_names)
                    tmp['subject']=qaobj.subject
                    tmp['task'] = dset.task
                    psd_dframe_list.append(tmp)
                    del tmp, pow_tmp
            except: 
                fail.append(f'{qaobj.subject} : {dset.task}')

psd_dframe = pd.concat                
power_dframe = pd.concat(pow_dframe_list)
power_dframe.to_csv('/home/jstout/src/nih_to_mne/nih2mne/dataQA/power_hv_092424.csv', index=False)

def make_bids_subject_layout(row_num=6, col_num=4, subject_list=None, opts=None):
    '''Generate a Grid of datasets'''  
    idx = 0
    layout = []
    #Preallocate Layout --- !! WILL need to zero/Null pad the matrix
    for row in range(row_num):
        layout.append([None]*col_num)
    
    #Fill in the current grid with the appropriate subjects
    tile_idxs = np.arange(row_num*col_num)
    tile_idxs_grid = tile_idxs.reshape(row_num, col_num)
    row_idxs, col_idxs = np.unravel_index(tile_idxs, [row_num, col_num])
    for row_idx, col_idx in zip(row_idxs, col_idxs):
        layout[row_idx][col_idx] = subject_list[tile_idxs_grid[row_idx, col_idx]]
        
    # for row in range(row_num):
    #     for col in range(col_num):
    #         if idx<len(subject_tile_list):
    #             layout
    #     layout[idx]
    
    
    for row in range(row_num):
        row_layout = []
        for col in range(col_num):
            subject_tile = subject_list[idx]
            row_layout.append(subject_tile)
            idx+=1
    
    
    required_opts = [
        [sg.Text('BIDS DIR:'), 
         sg.InputText(key='-BIDS_DIR-', enable_events=True),
         sg.FolderBrowse(target='-BIDS_DIR-')
         ],
        ]
           
    project_opts = [
        [sg.Text(' -- PROJECT_QA -- ')],
        [sg.Button('Compute Table'), sg.Button('View Table', disabled=True)], 
        [sg.Button('Missing Data', disabled=True)], 
        [sg.Button('Assess Noise Levels', disabled=True)]
        ]
    
    subject_required_opts = [
        [sg.Text(' -- Subject Section -- ')],
        [sg.Button('Select Subject')]
         ]
    subject_raw_opts = [
        [sg.Text(' -- SUBJECT INPUT QA -- ')],
        [sg.Button('MRI ANON', disabled=True),
         sg.Button('Fiducials', disabled=True)],
        [sg.Button('3D Coreg', disabled=True),
         sg.Button('Data QA', disabled=True)]
        ]
    
    subject_preproc_opts = [
        [sg.Text('  -- SUBJECT Preproc QA --')],
        [sg.Button('Run MR Prep', disabled=False), sg.Button('Run MR Prep (VOL)', disabled=False)]
        ]    
         
    # -- Assemble Layout --
    layout = required_opts
    layout.append([sg.Text(' ')])
    layout.append(project_opts)
    layout.append([sg.Text(' ')])
    layout.append(subject_raw_opts)
    layout.append([sg.Text(' ')])
    layout.append(subject_preproc_opts)
    layout.append([sg.Text(' ')])
    layout.append([sg.Button('Print CMD', key='-PRINT_CMD-'), sg.Button('RUN', key='-RUN-'), 
                   sg.Button('Write Cfg', key='-WRITE_CFG-'), sg.Button('EXIT')])
    return layout        
        
def qa_subject_selector(config_fname=False):    
    ## Setup and run gui
    opts = window_opts(config=config_fname) #This defaults to False if not set
    window = get_window(opts)
    
    # Conversion dictionary between GUI variables and cmdline vars
    _tmp = [i for i in dir(opts) if not i.startswith('_')]
    global value_writedict
    value_writedict = {f'-{i.upper()}-':i for i in _tmp}
    
    while True:
        event, values = window.read()
        
        if event == '-READ_CFG-':
            cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', 
                                          default_extension='.cfg')
            write_opts = read_cfg(cfg_fname)
            opts.update_opts(write_opts)
        
        # Update object options if event triggered
        if event in value_writedict.keys():
            setattr(opts, value_writedict[event], values[event])
        
        ## BIDS PROJECT LEVEL
        if event == 'Compute Table':
            from nih2mne.utilities.print_bids_table import gui_interface as pbids_table
            opts.bids_table_fname = op.join(op.dirname(opts.bids_dir), 'bids_prep_logs','BIDS_table.csv')
            pbids_table(opts.bids_dir, out_fname =opts.bids_table_fname)
            window['View Table'].update(disabled=False)
        
        if event == 'View Table':
            cmd = f'xdg-open {opts.bids_table_fname}'
            subprocess.run(cmd.split())
            
        ## BIDS SUBJECT LEVEL QA 
        
        
        ## MRI Preprocess
        if (event == 'Run MR Prep') or (event == 'Run MR Prep (VOL)'):
            subjects = sorted(glob.glob('sub-*', root_dir=opts.bids_dir))
            subject = selector_POPUP(subjects, text_item='Select a subject')[0]
            dsets = glob.glob(f'**/{subject}_*.ds', recursive=True, root_dir=op.join(opts.bids_dir, subject))
            dsets = ['ALL'] + dsets
            dsets_sel = selector_POPUP(dsets, text_item = 'Select a dataset to run MR preprocessing')
            if dsets_sel[0] == 'ALL':
                dsets.remove('ALL')
                dsets_sel = dsets
            #Add full path for processing
            dsets_sel = [op.join(opts.bids_dir, subject, i) for i in dsets_sel]
            for dset in dsets_sel:
                cmd = f'megcore_prep_mri_bids.py -bids_root {opts.bids_dir} -filename {dset}'
                if event == 'Run MR Prep (VOL)':
                    cmd+=' -volume'
                subprocess.run(cmd.split())
            
        
        # Logic for displaying coreg options
        if event == '-COREG-':
            if values['-COREG-'] == 'BrainSight':
                window['-COREG_BSIGHT-'].update(visible=True)
                window['-COREG_AFNI-'].update(visible=False)
            elif values['-COREG-'] == 'Afni':
                window['-COREG_AFNI-'].update(visible=True)
                window['-COREG_BSIGHT-'].update(visible=False)
            elif values['-COREG-'] == 'None':
                window['-COREG_BSIGHT-'].update(visible=False)
                window['-COREG_AFNI-'].update(visible=False)
        
        # If input directory is chosen - automatically select subject id and set
        if event == '-MEG_INPUT_DIR-':
            search_dir = values['-MEG_INPUT_DIR-']
            tmp = glob.glob(op.join(search_dir, '*.ds'))
            tmp = [op.basename(i) for i in tmp]
            tmp = list(set([i.split('_')[0] for i in tmp]))
            if len(tmp) == 1:
                values['-SUBJID_INPUT-']=tmp[0]
                opts.subjid_input =tmp[0]
                window['-SUBJID_INPUT-'].update(tmp[0])
            if len(tmp) > 1:
                tmp = subject_selector_POPUP(tmp)
                window['-SUBJID_INPUT-'].update(tmp[0])
                opts.subjid_input=tmp[0]
                
        if event == '-PRINT_CMD-':
            cmd = format_cmd(opts)
            print(cmd)
        
        if event == '-WRITE_CFG-':
            cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', default_path='nih_bids_gui.cfg', 
                                          default_extension='.cfg')
            write_cfg(opts, fname=cfg_fname)
        
        
        if event == '-RUN-':
            print(f'Running the command: {cmd}')
            cmd = format_cmd(opts)
            out_txt = subprocess.run(cmd.split(), check=True, capture_output=True)
            summary = []
            _start = False
            for i in str(out_txt.stdout).split('\\n'):
                if '########### SUMMARY #################' in i:
                    _start = True
                if _start:
                    summary.append(i)
            sg.popup_get_text('\n'.join(summary), title='SUMMARY')
            _tmp = op.dirname(opts.bids_dir)
            setattr(opts, 'error_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_err_log.txt'))
            setattr(opts, 'full_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_log.txt'))
            setattr(opts, 'fids_qa',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_fids_qa.png'))  
            window['-CHECK_TRIAX_COREG-'].update(disabled=False)
            print('FINISHED')
            
        if event == '-CHECK_TRIAX_COREG-':
            subprocess.run(f'xdg-open {opts.fids_qa}'.split())
        if event == sg.WIN_CLOSED or event == 'EXIT': # if user closes window or clicks cancel
            break
        
    window.close()
        

# Test Section
# topdir = '/fast2/BIDS'
# tmp = meglist_class(subject='ON08710')
# tmp.meg_list
# test= tmp.meg_list[0]
# test.load()

tmp2 = subject_tile(subject='ON08710', bids_root='/fast2/BIDS')
tmp2 = subject_bids_info(subject='ON08710', bids_root='/fast2/BIDS')
# tmp = qa_mri_class(subject='sub-ON08710', bids_root='/fast2/BIDS') 














