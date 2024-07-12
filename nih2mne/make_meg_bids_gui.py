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
        [sg.Text('BIDS Output Directory'), sg.InputText(key=options.bids_dir)],
        [sg.Text('BIDS Session'), sg.InputText(default_text=options.bids_session)]
        ]

    # MRI opts
    coreg_bsight_opts = [[sg.Text('Nifti MRI', key='bsight_file')], 
                         [sg.Text('Bsight Elec file', key='bsight_elec')]
                        ]
    coreg_afni_opts = [[sg.Text('Brik File', key='brik_file')]]
    
    
    coreg_opts = [
        [sg.Combo(['BrainSight', 'Afni', 'None'], enable_events=True, readonly=True, k='-COREG-', 
                  default_value='BrainSight')],
        collapse(coreg_bsight_opts, '-COREG_BSIGHT-', False), 
        collapse(coreg_afni_opts, '-COREG_AFNI-', False),
        
        ]
         
    # coreg_afni_opts = [
    #     [sg.Text('Brik File', key='brik_file')]
    #     ]

    # coreg_bsight_opts = [
    #     [sg.Text('Nifti MRI', key='brik_file')]
    #     ]
     
    layout = init_layout
    layout.append(l_button_opts)
    layout.append(standard_opts)
    # if options.ignore_mri_checks != True:
    layout.append(coreg_opts)
    layout.append([sg.Button('EXIT')])
    return layout
        
global size_mult, font_size, x_size, y_size
size_mult=2
font_size=12
x_size = 400*size_mult
y_size = 600*size_mult


def get_window(options=None):
    layout = make_layout(options=options)
    window = sg.Window('MEG BIDS conversion', layout, resizable=True, auto_size_buttons=True, 
                   scaling=True, size=(x_size, y_size))
    del layout
    return window

opts = window_opts()
window = get_window(opts)
coreg_toggle = False

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
            
    if event == '-COREG-':
        if values['-COREG-'] == 'Brainsight':
            window['-COREG_BSIGHT-'].update(visible=True)
        elif values['-COREG-'] == 'Afni':
            window['-COREG_AFNI-'].update(visible=True)
        elif values['-COREG-'] == 'None':
            window['-COREG_BSIGHT-'].update(visible=False)
            window['-COREG_AFNI-'].update(visible=False)
        
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

# while True:
#     event, values = window.read()

#     if event == 'checkbox_key':
#         toggle_bool1 = not toggle_bool1
#         window['section_key'].update(visible=toggle_bool1)
    
#     ...

#     if event == 'Quit' or event == sg.WIN_CLOSED:
#         break


















