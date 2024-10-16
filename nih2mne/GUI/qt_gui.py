#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:45:31 2024

@author: jstout

TODO -- 
check the MEG datasets and assess bad subjs data (bad chans etc)
launch single subject check from push button 

"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, \
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel,  QComboBox, QLineEdit

import sys
from nih2mne.dataQA.bids_project_interface import subject_bids_info, bids_project
import os, os.path as op
import numpy as np
from nih2mne.utilities.montages import montages



## Create subject tile
class Subject_Tile(QWidget):
    # SubjidButton -- MEG () MRI () FS ()
    def __init__(self,bids_info, task_filter='All'):
        super(Subject_Tile, self).__init__()
        
        self.bids_info = bids_info
        self.meg_count = self.bids_info.meg_count
        # self.meg_status = 'GOOD'
        self.mri_status = self.get_mri_status()
        self.fs_status = self.get_fs_status()
        self.evt_status = 'GOOD'
        
        self.subj_button = QPushButton(self) 
        self.subj_button.setText(bids_info.subject) 
        self.subj_button.clicked.connect(self.clicked)
        
        layout = QHBoxLayout()
        layout.addWidget(self.subj_button)
        
        if task_filter != 'All':
            self.filtered_tasklist = [i for i in self.bids_info.meg_list if i.task==task_filter]
            self.meg_status = self.get_meg_status()
        else:
            self.filtered_tasklist = self.bids_info.meg_list
            self.meg_status = self.get_meg_status()
            
        
        for tag in ['Meg', 'Mri','FS','EVT']:
            color = self.color(tag)
            if tag=='Meg':
                tag+= f' {str(len(self.filtered_tasklist))}'
                    
            tmp = QLabel(tag)
            layout.addWidget(tmp)
            if color == 'red':
                status_txt = '  /  '
            else:
                status_txt = '     '
            tmp = QLabel(status_txt)
            tmp.setAutoFillBackground(True)
            tmp.setStyleSheet(f"background-color: {color}")
            layout.addWidget(tmp)
        self.setLayout(layout)
    
    def clicked(self):
        '''This will eventually open the subject QA'''
        self.subj_button.setText('X_'+self.bids_info.subject+'_X')
        self.subj_button.adjustSize()
        self.w = Subject_GUI(bids_info = self.bids_info)
        self.w.show()
    
    def get_meg_status(self):
        if len(self.filtered_tasklist) == 0:
            return 'BAD'
        else:
            return 'GOOD'
    
    def get_fs_status(self):
        if self.bids_info.fs_recon['fs_success']:
            return 'GOOD'
        else:
            return 'BAD'
    
    def get_mri_status(self):
        if self.bids_info.mri=='Multiple':
            return 'BAD'
        elif self.bids_info.mri==None:
            return 'BAD'
        
        if self.bids_info.mri_json_qa == 'GOOD':
            return 'GOOD'
        else:
            return 'BAD'
    
    def color(self, tag=None):
        if getattr(self, tag.lower()+'_status') == 'GOOD':
            return 'green'
        else:
            return 'red'
        
       
class Subject_GUI(QWidget):        
    '''All the subject level QA at a finer detail'''
    def __init__(self, bids_info):
        super(Subject_GUI, self).__init__()
        self.bids_info = bids_info
        
        ## Save button
        main_layout = QVBoxLayout()
        self.b_save = QPushButton('Save')
        self.b_save.clicked.connect(self.save)
        main_layout.addWidget(self.b_save)
        main_layout.addWidget(QLabel(bids_info.__repr__()))
        
        ## Plotting
        plot_widget_layout = QHBoxLayout()
        self.b_plot_fids = QPushButton('Plot FIDS')
        self.b_plot_fids.clicked.connect(self.plot_fids)
        plot_widget_layout.addWidget(self.b_plot_fids)
                
        self.b_plot_3Dcoreg = QPushButton('Plot 3D Coreg')
        self.b_plot_3Dcoreg.clicked.connect(self.plot_3d_coreg)
        plot_widget_layout.addWidget(self.b_plot_3Dcoreg)
                
        self.b_plot_meg = QPushButton('Plot MEG Data')
        self.b_plot_meg.clicked.connect(self.plot_meg)
        plot_widget_layout.addWidget(self.b_plot_meg)
        main_layout.addLayout(plot_widget_layout)
        
        # MEG chooser and filter parameters
        meg_display_layout = QHBoxLayout()
        self.b_chooser_meg = QComboBox()
        self.b_chooser_meg.addItems(self.get_meg_choices())
        meg_display_layout.addWidget(self.b_chooser_meg)
        meg_display_layout.addWidget(QLabel('fmin'))
        self.b_fmin = QLineEdit()
        meg_display_layout.addWidget(self.b_fmin)
        meg_display_layout.addWidget(QLabel('fmax'))
        self.b_fmax = QLineEdit()
        meg_display_layout.addWidget(self.b_fmax)
        self.b_plot_montage = QComboBox()
        self.b_plot_montage.addItems(montages.keys())
        meg_display_layout.addWidget(self.b_plot_montage)
        main_layout.addLayout(meg_display_layout)
        
        # Add an events display -- SEt this to update after selection
        self.b_meg_events = QLabel(self.get_meg_events())
        self.b_chooser_meg.currentIndexChanged.connect(self.update_chooser_meg_idx)
        main_layout.addWidget(self.b_meg_events)
        
        self.setLayout(main_layout)
        
    def update_chooser_meg_idx(self):
        self.b_meg_events.setText(self.get_meg_events())
        
    def get_meg_events(self):
        megidx_ = self.b_chooser_meg.currentIndex()
        megtmp_ = self.bids_info.meg_list[megidx_]
        try:
            return megtmp_.event_counts.__repr__().replace('description\n','').replace('Name: count, dtype: int64','')
        except:
            return 'No Events to list'
        
    def get_meg_choices(self):
        return [f'{i}: {j.fname}' for i,j in enumerate(self.bids_info.meg_list)]
    
    def get_mri_choices(self):
        return [f'{i}: {op.basename(j)}' for i,j in enumerate(self.bids_info.all_mris)]
        
    def plot_meg(self):
        idx = self.b_chooser_meg.currentIndex()
        fmin = self.b_fmin.text().strip()
        fmax = self.b_fmax.text().strip()
        tmp_idx = self.b_plot_montage.currentIndex()
        tmp_choice = list(montages.keys())[tmp_idx]
        montage_choice = montages[tmp_choice]
        print(montage_choice)
        if (fmin == '') or (fmin.lower() == 'none'):
            fmin = None
        else:
            fmin = float(fmin)
        if (fmax == '') or (fmax.lower() == 'none'):
            fmax = None
        else:
            fmax = float(fmax)
        self.bids_info.plot_meg(idx=idx, hp=fmin, lp=fmax, montage=montage_choice)
        
    def plot_fids(self):
        self.bids_info.plot_mri_fids()
    
    def plot_3d_coreg(self):
        idx = self.b_chooser_meg.currentIndex()
        self.bids_info.plot_3D_coreg(idx=idx)
        
    def save(self):
        self.bids_info.save(overwrite=True)
    
    def override_mri(self):
        pass
        
        
                              
        




## create the window 

class BIDS_Project_Window(QMainWindow):
    def __init__(self, bids_root=os.getcwd(), gridsize_row=8, gridsize_col=5, 
                 bids_project=None):
        super(BIDS_Project_Window, self).__init__()
        self.setGeometry(100,100, 250*gridsize_col, 100*gridsize_row)
        self.setWindowTitle(f'BIDS Folder: {bids_root}')
        self.gridsize_row = gridsize_row
        self.gridsize_col = gridsize_col
        self.bids_project = bids_project
        self.page_idx = 0
        self.subject_start_idx = 0
        self.last_page_idx = len(bids_project.subjects)//(gridsize_col * gridsize_row) -1
        _tmp = len(bids_project.subjects)/(gridsize_col * gridsize_row)
        if _tmp != 0:
            self.last_page_idx += 1  #Add a page for the remaining subjs
        self.make_task_set()
        self.selected_task = 'All'
        
        # Finalize Widget and dispaly
        # main_layout = self.setup_full_layout()
        widget = QWidget()
        widget.setLayout(self.setup_full_layout())
        self.setCentralWidget(widget)
    
    def setup_full_layout(self):
        main_layout = QVBoxLayout()

        # Setup up Top Button Row
        top_buttons_layout = QHBoxLayout()
        self.b_choose_bids_root = QPushButton('BIDS Directory')
        self.b_choose_bids_root.clicked.connect(self.select_bids_root)
        self.b_choose_qa_file = QPushButton('QA file')
        self.b_choose_qa_file.clicked.connect(self.select_qa_file)
        self.b_subject_number = QLabel(f'Subject Totals: #{len(self.bids_project.subjects)}')
        top_buttons_layout.addWidget(self.b_choose_bids_root)
        top_buttons_layout.addWidget(self.b_choose_qa_file)
        top_buttons_layout.addWidget(self.b_subject_number)
        self.b_task_chooser = QComboBox()
        self.b_task_chooser.addItems(self.task_set)
        self.b_task_chooser.currentIndexChanged.connect(self.filter_task_qa_vis)
        top_buttons_layout.addWidget(self.b_task_chooser)
        main_layout.addLayout(top_buttons_layout)
        
        # Add Subject Chooser Grid Layer
        subjs_layout = self.init_subjects_layout()
        main_layout.addLayout(subjs_layout)
        
        # Add Bottom Row Buttons
        #-Freesurfer-
        bottom_buttons_layout = QHBoxLayout()
        _needs_fs = len(self.bids_project.issues['Freesurfer_notStarted'])
        self.b_run_freesurfer = QPushButton(f'Run Freesurfer (N={_needs_fs})')
        self.b_run_freesurfer.clicked.connect(self.proc_freesurfer)
        bottom_buttons_layout.addWidget(self.b_run_freesurfer)
        #-MRI Prep-
        self.b_run_mriprep = QPushButton('Run MRIPrep')
        self.b_run_mriprep.clicked.connect(self.proc_mriprep)
        bottom_buttons_layout.addWidget(self.b_run_mriprep)
        #-MRI Prep Vol/Surf selection
        self.b_mri_volSurf_selection = QComboBox()
        self.b_mri_volSurf_selection.addItems(['Surf','Vol'])
        bottom_buttons_layout.addWidget(self.b_mri_volSurf_selection)
        #-MEGNet Cleaning-
        self.b_run_megnet = QPushButton('Run MEGnet')
        self.b_run_megnet.clicked.connect(self.proc_megnet)
        bottom_buttons_layout.addWidget(self.b_run_megnet)
        #-Next / Prev Page buttons
        self.b_next_page = QPushButton('Next')
        self.b_next_page.clicked.connect(self.increment_page_idx)
        self.b_prev_page = QPushButton('Prev')
        self.b_prev_page.clicked.connect(self.decrement_page_idx)
        bottom_buttons_layout.addWidget(self.b_prev_page)
        bottom_buttons_layout.addWidget(self.b_next_page)
        #-Page Counter-
        self.b_current_page_idx = QLabel(f'Page: {self.page_idx} / {self.last_page_idx}')
        bottom_buttons_layout.addWidget(self.b_current_page_idx)
        
        #Finalize
        main_layout.addLayout(bottom_buttons_layout)
        return main_layout
    
    def make_task_set(self):
        self.task_set = {'All':''}
        for bids_key in self.bids_project.subjects.keys():
            bids_info = self.bids_project.subjects[bids_key]
            for dset in bids_info.meg_list:
                if dset.task not in self.task_set.keys():
                    self.task_set[dset.task] = 1
                else:
                    self.task_set[dset.task] += 1
        _keysort = sorted(list(self.task_set.keys()))
        _tmp = [f'{i} : {self.task_set[i]}' for i in _keysort]
        self.task_set = _tmp
        
    def filter_task_qa_vis(self):
        'Get the task from the chosen task set'
        idx = self.b_task_chooser.currentIndex()
        self.selected_task = self.task_set[idx].split(':')[0].strip()
        self.update_subjects_layout()
        
    
            
    
    def update_page_idx_display(self):
        self.b_current_page_idx.setText(f'Page: {self.page_idx} / {self.last_page_idx}')
    
    def increment_page_idx(self):
        if self.page_idx<self.last_page_idx:
            self.page_idx+=1
            self.update_page_idx_display()
            self.subject_start_idx += (self.gridsize_col * self.gridsize_row)
            self.update_subjects_layout()
        else:
            pass
    
    def decrement_page_idx(self):
        if self.page_idx==0:
            pass
        else:
            self.page_idx-=1
            self.update_page_idx_display()
            self.subject_start_idx -= (self.gridsize_col * self.gridsize_row)
            self.update_subjects_layout()
            
        
    
    def select_qa_file(self):
        self.qa_file, filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Select QA file',
                                                             filter='*.yml')
        
        
    
    def proc_freesurfer(self):
        'Loop over subjects that do not have a fs dir and run fs'
        issues = self.bids_project.issues
        freesurfer_proclist=[] 
        for subject, bids_info in self.bids_project.subjects.items():
            if subject in issues['Freesurfer_notStarted']:
                print(f'Submitting sbatch job for {subject}')
                bids_info.proc_freesurfer()
    
    
    def proc_mriprep(self):
        'Run bem/src/fwd/trans for the datasets'
        'Pre-req checks - Freesurfer and fids'
        issues = self.bids_project.issues
        mriprep_proclist=[]
        if self.b_mri_volSurf_selection.currentText().lower() == 'surf':
            surf = True
        elif self.b_mri_volSurf_selection.currentText().lower() == 'vol':
            surf = False
        for subject, bids_info in self.bids_project.subjects.items():
            if (subject in issues['Freesurfer_failed']) or (subject in issues['Freesurfer_notStarted']):
                continue
            else:
                mriprep_proclist.append(subject)
        task = self.selected_task.split(':')[0].strip()
        for subject in mriprep_proclist:
            self.bids_project.subjects[subject].mri_preproc(surf=surf, fname='all')
    
    def proc_megnet(self):
        pass
        
        
    def select_bids_root(self):
        self.bids_root = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        os.chdir(self.bids_root)
        self.bids_project = bids_project(bids_root=self.bids_root)
        self.page_idx = 0
        self.subject_start_idx = 0
        self.update_subjects_layout()
        
    def update_subjects_layout(self):
        tile_idxs = np.arange(self.gridsize_row * self.gridsize_col)
        tile_idxs_grid = tile_idxs.reshape(self.gridsize_row, self.gridsize_col)
        row_idxs, col_idxs = np.unravel_index(tile_idxs, [self.gridsize_row, self.gridsize_col])
        i=self.subject_start_idx
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            _for_replace = self.subjs_layout.itemAtPosition(row_idx, col_idx)
            widget = _for_replace.widget()
            widget.deleteLater()
            if i+1 > len(self.bids_project.subjects):
                self.subjs_layout.addWidget(QLabel(''), row_idx, col_idx)
            else:
                bids_info = self.bids_project.subjects[self.subject_keys[i]]
                self.subjs_layout.addWidget(Subject_Tile(bids_info, task_filter=self.selected_task), row_idx, col_idx)
            i+=1
        
        
    def init_subjects_layout(self):
        self.subjs_layout = QGridLayout()
        self.subject_keys = sorted(list(self.bids_project.subjects.keys()))
        
        tile_idxs = np.arange(self.gridsize_row * self.gridsize_col)
        tile_idxs_grid = tile_idxs.reshape(self.gridsize_row, self.gridsize_col)
        row_idxs, col_idxs = np.unravel_index(tile_idxs, [self.gridsize_row, self.gridsize_col])
        i=0
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            if i+1 > len(self.bids_project.subjects):
                self.subjs_layout.addWidget(QLabel(''), row_idx, col_idx)
            bids_info = self.bids_project.subjects[self.subject_keys[i]]
            self.subjs_layout.addWidget(Subject_Tile(bids_info), row_idx, col_idx)
            i+=1
        return self.subjs_layout
            



def window():
    os.chdir(bids_pro.bids_root)
    app = QApplication(sys.argv)
    win = BIDS_Project_Window(bids_project = bids_pro)
    win.show()
    sys.exit(app.exec_())

# bids_pro = bids_project(bids_root='/fast2/BIDS')

    
window()


# bids_info = bids_pro.subjects['sub-ON95742']
# window = Subject_Tile(bids_info)
# window.show()


