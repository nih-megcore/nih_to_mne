#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:50:57 2024

@author: jstout
"""
import PySimpleGUI as sg
import os,os.path as op

font = ("Arial", 25)
sg.set_options(font=font)



class window_opts:
    def __init__(self):
        # Button Entries
        self.anonymize = False
        self.ignore_mri_checks = False

        # Standard Entries
        self.bids_dir = op.join(os.getcwd(), 'bids_dir')
        self.meg_input_dir = None
        self.bids_session = 1
        self.subjid_input = None
        self.bids_id = None
        self.coreg = 'brainsight'

        ## Afni Coreg:
        self.mri_brik = None

        ## Brainsight Coreg:
        self.mri_bsight = None
        self.mri_bsight_elec = None

        ## Optional Overrides:
        self.ignore_eroom = None
        self.autocrop_zeros = None
        self.freesurfer = None
        self.eventID_csv = None
        # Run standardize_eventID_list.py
        

def make_layout(options=None):
    init_layout = [  
        [sg.Text('MEG Conversion to BIDS')],
        [sg.Text('Warning: currently works with {DATE/dsets} format from scanner')]
        ]
    
    if options.anonymize==True:
        l_button_opts = [
            [sg.Button('Anonymize: Y', key='anonymize', button_color='green')], 
            ]
    else:
        l_button_opts = [
            [sg.Button('Anonymize: N', key='anonymize', button_color='red')], 
            ]
             
    standard_opts = [
        [sg.Text('Standard Options')], 
        [sg.Text('BIDS Output Directory'), sg.InputText(key=options.bids_dir)],
        [sg.Text('BIDS Session'), sg.InputText(default_text=options.bids_session)]
        ]

    # MRI opts
    coreg_opts = [
        [sg.Combo(['BrainSight', 'Afni', 'None'], enable_events=True, readonly=True, k='-COREG-', 
                  default_value='BrainSight')]
        ]
         
    coreg_afni_opts = [
        [sg.Text('Brik File', key='brik_file')]
        ]

    coreg_bsight_opts = [
        [sg.Text('Nifti MRI', key='brik_file')]
        ]
     
    layout = init_layout
    layout.append(l_button_opts)
    layout.append(standard_opts)
    if options.ignore_mri_checks != True:
        layout.append(coreg_opts)
    layout.append(coreg_afni_opts)
    layout.append(coreg_bsight_opts)
    layout.append([sg.Button('EXIT')])
    return layout
        

def get_window(options=None):
    layout = make_layout(options=options)
    window = sg.Window('MEG BIDS conversion', layout, resizable=True, auto_size_buttons=True, 
                   scaling=True)
    del layout
    return window

opts = window_opts()
window = get_window(opts)

while True:
    event, values = window.read()
    if event == 'anonymize': 
        if opts.anonymize == False:
            window['anonymize'].update(button_color='green')
            window['anonymize'].update(text='Anonymize: Y')
            opts.anonymize = not opts.anonymize
        else:
            window['anonymize'].update(button_color='red')
            window['anonymize'].update(text='Anonymize: N')
            opts.anonymize = not opts.anonymize
        
    if event == 'set_coreg_afni':
        set_coreg_afni = True
        set_coreg_bsight = False
    if event == 'set_coreg_bsight':
        set_coreg_afni = False
        set_coreg_bsight = True        
    if event == sg.WIN_CLOSED or event == 'EXIT': # if user closes window or clicks cancel
        break
    print('You entered ', values[0])
    print(f'Anonymize is {opts.anonymize}')


window.close()





#%% 
options:
  -h, --help            show this help message and exit
  -bids_dir BIDS_DIR    Output bids_dir path
  -meg_input_dir MEG_INPUT_DIR
                        Acquisition directory - typically designated by the acquisition date
  -anonymize            Strip out subject ID information from the MEG data. Currently this does not
                        anonymize the MRI. Requires the CTF tools.
  -bids_session BIDS_SESSION
                        Data acquisition session. This is set to 1 by default. If the same subject had
                        multiple sessions this must be set manually
  -subjid_input SUBJID_INPUT
                        The default subject ID is given by the MEG hash. If more than one subject is
                        present in a folder, this option can be set to select a single subjects
                        dataset.
  -bids_id BIDS_ID      The default subject ID is given by the MEG hash. To override the default
                        subject ID, use this flag. If -anonymize is used, you must set the subjid
  -autocrop_zeros       If files are terminated early, leaving zeros at the end of the file - this
                        will detect and remove the trailing zeros. !!!Files larger than 2G will
                        Fail!!!

Afni Coreg:
  -mri_brik MRI_BRIK    Afni coregistered MRI

Brainsight Coreg:
  -mri_bsight MRI_BSIGHT
                        Brainsight mri. This should be a .nii file. The exported electrodes text file
                        must be in the same folder and end in .txt. Otherwise, provide the
                        mri_sight_elec flag
  -mri_bsight_elec MRI_BSIGHT_ELEC
                        Exported electrodes file from brainsight. This has the locations of the
                        fiducials
  -ignore_mri_checks

Optional Overrides:
  -ignore_eroom         If you are Not on Biowulf, use this option to prevent an error. Or if you
                        collected your own empty room data with your dataset
  -supplement_eroom     If emptyroom present - ignore, else add emptyroom from the biowulf repository.
  -freesurfer           Perform recon-all pipeline on the T1w. This is required for the mri_prep
                        portions below
  -eventID_csv EVENTID_CSV
                        Provide the standardized event IDs. This can be produced by running:
                        standardize_eventID_list.py

UNDER Construction - BIDS PostProcessing:
  -project PROJECT      Output project name for the mri processing from mri_prep
  -mri_prep_s           Perform the standard SURFACE processing for meg analysis
                        (watershed/bem/src/fwd)
  -mri_prep_v           Perform the standard VOLUME processing for meg analysis
                        (watershed/bem/src/fwd)

















