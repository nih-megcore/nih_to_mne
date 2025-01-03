#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:55:27 2025

@author: jstout

Trigger Channel Coding GUI

Layout:
    File Lookup
    Initial Trigger Procs (Identify transitions):
        UADC1-15  - Analog
        UADC16 - Projector Analog
        PPT001 - Parrallel Port
        --LIST Output Counts per line--
    Code out Line variable names:
        UADC1-15 - what are they named
        UADC16 - Projector
        PPT001:
            Code for each trigger value
    Visual Tasks:
        Correct all to projector?
        Check for projector inversion?
    Auditory Tasks:
        Add static delay to event time
    General Add static delay:
        Didn't code out projector so add 19ms delay
    Parse Marks:
        In1
        In2
        MarkOn
        MarkName
    Display  Totals:
        For each condition write out the pandas value counts
    Check Visual scan of data:
        Invoke MNE display
    WriteScript:
        Create a trigger processing script


TODO: Temporal coding on PPT -- a 1 followed by a 5 is a Y event        

"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, \
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel,  QComboBox, QLineEdit, QCheckBox

import sys
from nih2mne.dataQA.bids_project_interface import subject_bids_info, bids_project
import os, os.path as op
import numpy as np
from nih2mne.utilities.montages import montages
from nih2mne.dataQA.qa_config_reader import qa_dataset, read_yml
import glob
import mne


class trig_tile(QHBoxLayout):
    def __init__(self, chan_name=None, include_polarity=True):
        super(trig_tile, self).__init__()
        
        self.trig_type = chan_name[1:4]
        self.ch_name = chan_name

        self.addWidget(QLabel(f'{chan_name} :'))
        
        # Add Checkboxes for upgoing/Downgoing trigger polarity 
        self.b_upgoing_trigger = QCheckBox()
        self.b_upgoing_trigger.setCheckState(2)
        self.b_upgoing_trigger.clicked.connect(self.set_up_trigger_polarity)
        self.b_downgoing_trigger = QCheckBox()
        self.b_downgoing_trigger.setCheckState(0)
        self.b_downgoing_trigger.clicked.connect(self.set_down_trigger_polarity)
        
        self.addWidget(self.b_upgoing_trigger)
        self.addWidget(QLabel('Up'))
        self.addWidget(self.b_downgoing_trigger)
        self.addWidget(QLabel('Down'))
        
        # 
        self.addWidget(QLabel('   Event Name:'))
        self.event_name = QLineEdit()
        self.addWidget(self.event_name)

    def set_up_trigger_polarity(self):
        self.trigger_polarity = 'up'
        self.b_downgoing_trigger.setCheckState(0)
        print(self.b_upgoing_trigger.checkState())
    
    def set_down_trigger_polarity(self):
        self.trigger_polarity = 'down'
        self.b_upgoing_trigger.setCheckState(0)
        
        
        
        
        

class event_coding_Window(QMainWindow):
    def __init__(self):
        super(event_coding_Window, self).__init__()
        self.setGeometry(100,100, 1000, 1000) #250*gridsize_col, 100*gridsize_row)
        self.setWindowTitle('Event Coding GUI')
        
        self.tile_dict = {}
        
        # Finalize Widget and dispaly
        main_layout = self.setup_full_layout()
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
    
    def setup_full_layout(self):
        main_layout = QVBoxLayout()

        # Setup File Chooser
        meg_choose_layout = QHBoxLayout()
        self.b_choose_meg = QPushButton('Select MEG')
        self.b_choose_meg.clicked.connect(self.select_meg_dset)
        meg_choose_layout.addWidget(self.b_choose_meg)
        self.meg_display_name = QLabel('')
        meg_choose_layout.addWidget(self.meg_display_name)
        main_layout.addLayout(meg_choose_layout)
        
        self.ana_trigger_layout = QVBoxLayout()
        main_layout.addLayout(self.ana_trigger_layout)
        self.dig_trigger_layout = QVBoxLayout()
        main_layout.addLayout(self.dig_trigger_layout)
        return main_layout
    
    def fill_trig_chan_layout(self):
        # Display the trig chans
        self.ana_trigger_layout.addWidget(QLabel('Analogue Channels'))
        self.dig_trigger_layout.addWidget(QLabel('Digital Channels'))
        for i in self.trig_ch_names:
            if i.startswith('UADC'):
                self.tile_dict[i]=trig_tile(chan_name=i,include_polarity=True)
                self.ana_trigger_layout.addLayout(self.tile_dict[i])
            elif i.startswith('UPPT'):
                pass
                # self.tile_dict[i]= ;lkj;lkj ;lj ;l ##########<<<<<<<<<<
                # self.dig_trigger_layout.addLayout(self.    lkjlkj lj )
    
    
                                            
        
    # def trigger_tile(self):
    #     "Each tile has a Type:ChanName:Up/Down:OutputName"
    #     tile = QHBoxLayout()
        
        
        
        # self.b_choose_bids_root = QPushButton('BIDS Directory')
        # self.b_choose_bids_root.clicked.connect(self.select_bids_root)
        # self.b_choose_qa_file = QPushButton('QA file')
        # self.b_choose_qa_file.clicked.connect(self.select_qa_file)
        # self.b_subject_number = QLabel(f'Subject Totals: #{len(self.bids_project.subjects)}')
        # top_buttons_layout.addWidget(self.b_choose_bids_root)
        # top_buttons_layout.addWidget(self.b_choose_qa_file)
        # top_buttons_layout.addWidget(self.b_subject_number)
        # self.b_task_chooser = QComboBox()
        # self.b_task_chooser.addItems(self.task_set)
        # self.b_task_chooser.currentIndexChanged.connect(self.filter_task_qa_vis)
        # top_buttons_layout.addWidget(self.b_task_chooser)
        # self.b_out_project_chooser = QComboBox()
        # #Add Project output directory
        # # tmp_ = glob.glob('*', root_dir=op.join(self.bids_project.bids_root, 'derivatives'))
        # # derivatives_dirs = [i for i in tmp_ if op.isdir(op.join(self.bids_project.bids_root, 'derivatives', i))]
        # # for i in ['freesurfer', 'megQA']: 
        # #     if i in derivatives_dirs: derivatives_dirs.remove(i)
        # # self.b_out_project_chooser.addItems(derivatives_dirs)
        # # top_buttons_layout.addWidget(self.b_out_project_chooser)
        
        
        
        # main_layout.addLayout(top_buttons_layout)
        
        # # Add Subject Chooser Grid Layer
        # subjs_layout = self.init_subjects_layout()
        # main_layout.addLayout(subjs_layout)
        
        # # Add Bottom Row Buttons
        # #-Freesurfer-
        # bottom_buttons_layout = QHBoxLayout()
        # _needs_fs = len(self.bids_project.issues['Freesurfer_notStarted'])
        # self.b_run_freesurfer = QPushButton(f'Run Freesurfer (N={_needs_fs})')
        # self.b_run_freesurfer.clicked.connect(self.proc_freesurfer)
        # bottom_buttons_layout.addWidget(self.b_run_freesurfer)
        # #-MRI Prep-
        # self.b_run_mriprep = QPushButton('Run MRIPrep')
        # self.b_run_mriprep.clicked.connect(self.proc_mriprep)
        # bottom_buttons_layout.addWidget(self.b_run_mriprep)
        # #-MRI Prep Vol/Surf selection
        # self.b_mri_volSurf_selection = QComboBox()
        # self.b_mri_volSurf_selection.addItems(['Surf','Vol'])
        # bottom_buttons_layout.addWidget(self.b_mri_volSurf_selection)
        # #-MEGNet Cleaning-
        # self.b_run_megnet = QPushButton('Run MEGnet')
        # self.b_run_megnet.clicked.connect(self.proc_megnet)
        # bottom_buttons_layout.addWidget(self.b_run_megnet)
        # #-Next / Prev Page buttons
        # self.b_next_page = QPushButton('Next')
        # self.b_next_page.clicked.connect(self.increment_page_idx)
        # self.b_prev_page = QPushButton('Prev')
        # self.b_prev_page.clicked.connect(self.decrement_page_idx)
        # bottom_buttons_layout.addWidget(self.b_prev_page)
        # bottom_buttons_layout.addWidget(self.b_next_page)
        # #-Page Counter-
        # self.b_current_page_idx = QLabel(f'Page: {self.page_idx} / {self.last_page_idx}')
        # bottom_buttons_layout.addWidget(self.b_current_page_idx)
        
        # #Finalize
        # main_layout.addLayout(bottom_buttons_layout)


    def select_meg_dset(self):
        meg_fname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a MEG folder (.ds)')
        self.meg_fname = meg_fname
        meg_display_name = meg_fname.split('/')[-1]
        self.meg_display_name.setText(meg_display_name)
        print(meg_fname)
        if meg_fname != None:
            self.meg_raw = mne.io.read_raw_ctf(meg_fname, clean_names=True, 
                                               system_clock='ignore', preload=False)
        self.trig_ch_names = [i for i in self.meg_raw.ch_names if i.startswith('UADC') or i.startswith('UPPT')]
        self.fill_trig_chan_layout()
        # self.update_subjects_layout()


def window():
    app = QApplication(sys.argv)
    win = event_coding_Window() 
    win.show()
    sys.exit(app.exec_())
    
window()


#%% Testing
meg_fname = '/fast2/BIDS/sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-airpuff_run-01_meg.ds/'
raw = mne.io.read_raw_ctf(meg_fname, clean_names=True, system_clock='ignore')

trig_picks = [i for i in raw.ch_names if i.startswith('UADC') or i.startswith('UPPT')]




    
# def cmdline_main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-bids_root', help='Path to your BIDS folder', 
#                         required=True)
#     parser.add_argument('-num_rows', help='Number of subject rows',
#                         default=6, type=int)
#     parser.add_argument('-num_cols', help='Number of subject columns',
#                         default=4, type=int)
#     args = parser.parse_args()
#     bids_root = args.bids_root
    
#     bids_pro = bids_project(bids_root=bids_root)
#     window(bids_project=bids_pro, num_rows=args.num_rows, num_cols=args.num_cols)

# if __name__ == '__main__':
#     cmdline_main()






