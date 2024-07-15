#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:50:57 2024

@author: jstout
"""
import PySimpleGUI as sg
import os,os.path as op
import glob
import subprocess

font = ("Arial", 25)
sg.set_options(font=font)

global size_mult, font_size, x_size, y_size
size_mult=2
font_size=12   ### NOT SURE I ACTUALLY USE THESE -- FUTURE IMPLEMENTATION
x_size = 500*size_mult
y_size = 600*size_mult

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
        [sg.Text('MEG input DIR:'), 
         sg.InputText(key='-MEG_INPUT_DIR-', enable_events=True),
         sg.FolderBrowse(target='-MEG_INPUT_DIR-')
         ],
        [sg.Text('MEG Hashcode:'),
         sg.InputText(key='-SUBJID_INPUT-', enable_events=True)
         ],
        [sg.Text('BIDS ID:'),
         sg.InputText(key='-BIDS_ID-', enable_events=True)
         ],
        [sg.Text('BIDS Output Directory'), 
         sg.InputText(key='-BIDS_DIR-', enable_events=True), 
         sg.FolderBrowse(target='-BIDS_DIR-')],
        [sg.Text('BIDS Session'), 
         sg.InputText(default_text=opts.bids_session, enable_events=True, 
                      key='-BIDS_SESSION-')]
        ]

    # MRI opts
    coreg_bsight_opts = [
        [sg.Text('Nifti MRI:'), 
         sg.InputText('', key='-MRI_BSIGHT-', enable_events=True), 
         sg.FileBrowse(target='-MRI_BSIGHT-')], 
        [sg.Text('Bsight Elec file:'), 
         sg.InputText('',key='-MRI_BSIGHT_ELEC-', enable_events=True), 
         sg.FileBrowse(target='-MRI_BSIGHT_ELEC-')]
                        ]
    
    coreg_afni_opts = [
        [sg.Text('Brik File:'), 
         sg.InputText('', key='-MRI_BRIK-', enable_events=True), 
         sg.FileBrowse(target='-MRI_BRIK-')]] 
    
    # Fold down menu for coregistration options
    coreg_opts = [
        [sg.Text('COREG:'), sg.Combo(['BrainSight', 'Afni', 'None'], enable_events=True, readonly=True, k='-COREG-', 
                  default_value='BrainSight')],
        collapse(coreg_bsight_opts, '-COREG_BSIGHT-', True), 
        collapse(coreg_afni_opts, '-COREG_AFNI-', False),
        ]
         
    # -- Assemble Layout --
     
    layout = init_layout
    layout.append(l_button_opts)
    layout.append(standard_opts)
    # if options.ignore_mri_checks != True:
    layout.append(coreg_opts)
    layout.append([sg.Button('Print CMD', key='-PRINT_CMD-'), sg.Button('RUN', key='-RUN-'), sg.Button('EXIT')])
    return layout
        
def subject_selector_POPUP(data):
    layout = [
        [sg.Text('Select a subject')],
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


def get_window(options=None):
    layout = make_layout(options=options)
    window = sg.Window('MEG BIDS conversion', layout, resizable=True, auto_size_buttons=True, 
                   scaling=True, size=(x_size, y_size))
    del layout
    return window

single_flag_list = ['anonymize', 'autocrop_zeros', 'freesurfer', 'ignore_eroom',
                    'ignore_mri_checks']
drop_flag_list = ['coreg']
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
    
## Setup and run gui
opts = window_opts()
window = get_window(opts)
coreg_toggle = False

# Conversion dictionary between GUI variables and cmdline vars
_tmp = [i for i in dir(opts) if not i.startswith('_')]
value_writedict = {f'-{i.upper()}-':i for i in _tmp}

while True:
    event, values = window.read()
    if event in value_writedict.keys():
        setattr(opts, value_writedict[event], values[event])
    
    if event == 'anonymize': 
        if opts.anonymize == False:
            window['anonymize'].update(button_color='green')
            window['anonymize'].update(text='Anonymize: Y')
            opts.anonymize = not opts.anonymize
        else:
            window['anonymize'].update(button_color='red')
            window['anonymize'].update(text='Anonymize: N')
            opts.anonymize = not opts.anonymize
    
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
    
    if event == '-RUN-':
        print(f'Running the command: {cmd}')
        cmd = format_cmd(opts)
        subprocess.run(cmd.split(), check=True, capture_output=True)
        print('FINISHED')
        
    if event == 'set_coreg_afni':
        set_coreg_afni = True
        set_coreg_bsight = False
    if event == 'set_coreg_bsight':
        set_coreg_afni = False
        set_coreg_bsight = True        
    if event == sg.WIN_CLOSED or event == 'EXIT': # if user closes window or clicks cancel
        break
    
window.close()





















