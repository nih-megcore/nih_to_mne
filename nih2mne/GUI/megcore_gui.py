#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:41:38 2024

@author: jstout
"""

import PySimpleGUI as sg
import os,os.path as op
import glob
import subprocess
from nih2mne.GUI.make_meg_bids_gui import bids_gui
from nih2mne.GUI.qa_bids_gui import qa_gui

CFG_VERSION = 1.0

global size_mult, font_size, x_size, y_size
size_mult=2
font_size=12  
x_size = 500*size_mult
y_size = 600*size_mult


#%%
maj_font = ('Arial', 150)
min_font = ('Arial', round(maj_font[1] / 2 ))

def meg_gui():
    layout = [
        [sg.Text('Load Config:') ],
        [sg.Button('Make BIDS', key='-MAKE_BIDS-', font=min_font)],
        [sg.Button('QA BIDS', key='-QA_BIDS-', font=min_font)],
        [sg.Button('-----', font=min_font),],
        [sg.Button('Statistics', key='-Stats-', font=min_font) ],
        [sg.Button('EXIT', button_color='red', font=min_font)]
        ]
    
    window = sg.Window('MEG Functions', layout, resizable=True, auto_size_buttons=True, 
                   scaling=True, size=(x_size, y_size))
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'EXIT': # if user closes window or clicks cancel
                break
    
        if event == '-MAKE_BIDS-':
            bids_gui()
        if event == '-QA_BIDS-':
            qa_gui()
    
        
    window.close()



#%%%
        
        
    
    # if event == '-READ_CFG-':
    #     cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', 
    #                                   default_extension='.cfg')
    #     write_opts = read_cfg(cfg_fname)
    #     opts.update_opts(write_opts)
    
    # # Update object options if event triggered
    # if event in value_writedict.keys():
    #     setattr(opts, value_writedict[event], values[event])
    
    # if event == 'anonymize': 
    #     if opts.anonymize == False:
    #         window['anonymize'].update(button_color='green')
    #         window['anonymize'].update(text='Anonymize: Y')
    #         opts.anonymize = not opts.anonymize
    #     else:
    #         window['anonymize'].update(button_color='red')
    #         window['anonymize'].update(text='Anonymize: N')
    #         opts.anonymize = not opts.anonymize
    
    # # Logic for displaying coreg options
    # if event == '-COREG-':
    #     if values['-COREG-'] == 'BrainSight':
    #         window['-COREG_BSIGHT-'].update(visible=True)
    #         window['-COREG_AFNI-'].update(visible=False)
    #     elif values['-COREG-'] == 'Afni':
    #         window['-COREG_AFNI-'].update(visible=True)
    #         window['-COREG_BSIGHT-'].update(visible=False)
    #     elif values['-COREG-'] == 'None':
    #         window['-COREG_BSIGHT-'].update(visible=False)
    #         window['-COREG_AFNI-'].update(visible=False)
    
    # # If input directory is chosen - automatically select subject id and set
    # if event == '-MEG_INPUT_DIR-':
    #     search_dir = values['-MEG_INPUT_DIR-']
    #     tmp = glob.glob(op.join(search_dir, '*.ds'))
    #     tmp = [op.basename(i) for i in tmp]
    #     tmp = list(set([i.split('_')[0] for i in tmp]))
    #     if len(tmp) == 1:
    #         values['-SUBJID_INPUT-']=tmp[0]
    #         opts.subjid_input =tmp[0]
    #         window['-SUBJID_INPUT-'].update(tmp[0])
    #     if len(tmp) > 1:
    #         tmp = subject_selector_POPUP(tmp)
    #         window['-SUBJID_INPUT-'].update(tmp[0])
    #         opts.subjid_input=tmp[0]
            
    # if event == '-PRINT_CMD-':
    #     cmd = format_cmd(opts)
    #     print(cmd)
    
    # if event == '-WRITE_CFG-':
    #     cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', default_path='nih_bids_gui.cfg', 
    #                                   default_extension='.cfg')
    #     write_cfg(opts, fname=cfg_fname)
    
    
    # if event == '-RUN-':
    #     print(f'Running the command: {cmd}')
    #     cmd = format_cmd(opts)
    #     out_txt = subprocess.run(cmd.split(), check=True, capture_output=True)
    #     summary = []
    #     _start = False
    #     for i in str(out_txt.stdout).split('\\n'):
    #         if '########### SUMMARY #################' in i:
    #             _start = True
    #         if _start:
    #             summary.append(i)
    #     sg.popup_get_text('\n'.join(summary), title='SUMMARY')
    #     _tmp = op.dirname(opts.bids_dir)
    #     setattr(opts, 'error_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_err_log.txt'))
    #     setattr(opts, 'full_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_log.txt'))
    #     setattr(opts, 'fids_qa',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_fids_qa.png'))  
    #     window['-CHECK_TRIAX_COREG-'].update(disabled=False)
    #     print('FINISHED')
        
    # if event == '-CHECK_TRIAX_COREG-':
    #     subprocess.run(f'xdg-open {opts.fids_qa}'.split())
    # if event == sg.WIN_CLOSED or event == 'EXIT': # if user closes window or clicks cancel
    #     break











