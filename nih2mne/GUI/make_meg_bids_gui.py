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

CFG_VERSION = 1.0

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
    def __init__(self, config=False):
        # Button Entries
        self.anonymize = False
        self.ignore_mri_checks = False

        # Standard Entries
        self.bids_dir = op.join(os.getcwd(), 'bids_dir')
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
    init_layout = [  
        [sg.Text('MEG Conversion to BIDS')],
        [sg.Text('Warning: currently works with {DATE/dsets} format from scanner')]
        ]
    
    if opts.anonymize==True:
        l_button_opts = [
            [sg.Button('Anonymize: Y', key='anonymize', button_color='green'), 
            sg.Button('Read Cfg', key='-READ_CFG-', disabled=True)],
            ]
    else:
        l_button_opts = [
            [sg.Button('Anonymize: N', key='anonymize', button_color='red'), 
            sg.Button('Read Cfg', key='-READ_CFG-', disabled=True)],
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
         sg.InputText(key='-BIDS_DIR-', enable_events=True, default_text=opts.bids_dir), 
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
                  default_value=opts.coreg)],
        collapse(coreg_bsight_opts, '-COREG_BSIGHT-', opts.coreg=='Brainsight'), 
        collapse(coreg_afni_opts, '-COREG_AFNI-', opts.coreg=='Afni'),
        ]
    
    # Additional Options
    additional_opts = [
        [sg.Text('OPTIONAL:'),
         sg.Checkbox('CropZeros', key='-AUTOCROP_ZEROS-', 
                     enable_events=True, default=opts.autocrop_zeros),
         sg.Checkbox('No_EmptyRoom', key='-IGNORE_EROOM-', 
                     enable_events=True, default=opts.ignore_eroom), 
         sg.Checkbox('Freesurfer', key='-FREESURFER-', 
                     enable_events=True, default=opts.freesurfer, 
                     disabled=True)
         ]
        ]
         
    # -- Assemble Layout --
     
    layout = init_layout
    layout.append(l_button_opts)
    layout.append(standard_opts)
    # if options.ignore_mri_checks != True:
    layout.append(coreg_opts)
    layout.append(additional_opts)
    layout.append([sg.Button('Print CMD', key='-PRINT_CMD-'), sg.Button('RUN', key='-RUN-'), 
                   sg.Button('Write Cfg', key='-WRITE_CFG-'), sg.Button('EXIT')])
    layout.append([sg.Button('QA_coreg', key='-CHECK_TRIAX_COREG-', disabled=True)])
    return layout
        
def subject_selector_POPUP(data):
    '''
    If multiple datasets are found - create a popup selector to select the 
    correct subject
    '''
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

def bids_gui(config_fname=False):    
    ## Setup and run gui
    opts = window_opts(config=config_fname) #This defaults to False if not set
    window = get_window(opts)
    coreg_toggle = False
    
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
        
        if event == '-WRITE_CFG-':
            cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', default_path='nih_bids_gui.cfg', 
                                          default_extension='.cfg')
            write_cfg(opts, fname=cfg_fname)
        
        
        if event == '-RUN-':
            cmd = format_cmd(opts)
            print(f'Running the command: {cmd}')
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
    bids_gui(config_fname)





















