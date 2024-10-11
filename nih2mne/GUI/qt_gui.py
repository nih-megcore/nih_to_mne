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


## Create subject tile
class Subject_Tile(QWidget):
    # SubjidButton -- MEG () MRI () FS ()
    def __init__(self,bids_info):
        super(Subject_Tile, self).__init__()
        
        self.bids_info = bids_info
        self.meg_count = self.bids_info.meg_count
        self.meg_status = 'GOOD'
        self.mri_status = self.get_mri_status()
        self.fs_status = self.get_fs_status()
        self.evt_status = 'GOOD'
        
        self.subj_button = QPushButton(self) #bids_info.subject)
        self.subj_button.setText(bids_info.subject) 
        self.subj_button.clicked.connect(self.clicked)
        
        layout = QHBoxLayout()
        layout.addWidget(self.subj_button)
        
        for tag in ['Meg', 'Mri','FS','EVT']:
            color = self.color(tag)
            if tag=='Meg':
                tag+=f' {self.meg_count}'
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
        if (fmin == '') or (fmin.lower() == 'none'):
            fmin = None
        else:
            fmin = float(fmin)
        if (fmax == '') or (fmax.lower() == 'none'):
            fmax = None
        else:
            fmax = float(fmax)
        self.bids_info.plot_meg(idx=idx, hp=fmin, lp=fmax)
        
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
    def __init__(self, bids_root=os.getcwd(), gridsize_row=10, gridsize_col=6, 
                 bids_project=None):
        super(BIDS_Project_Window, self).__init__()
        self.setGeometry(100,100, 250*gridsize_col, 100*gridsize_row)
        self.setWindowTitle(f'BIDS Folder: {bids_root}')
        self.gridsize_row = gridsize_row
        self.gridsize_col = gridsize_col
        self.bids_project = bids_project
        self.initUI()
        
    def initUI(self):
        subjs_layout = QGridLayout()
        subject_keys = sorted(list(self.bids_project.subjects.keys()))
        
        tile_idxs = np.arange(self.gridsize_row * self.gridsize_col)
        tile_idxs_grid = tile_idxs.reshape(self.gridsize_row, self.gridsize_col)
        row_idxs, col_idxs = np.unravel_index(tile_idxs, [self.gridsize_row, self.gridsize_col])
        i=0
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            if i+1 > len(self.bids_project.subjects):
                break
            bids_info = self.bids_project.subjects[subject_keys[i]]
            subjs_layout.addWidget(Subject_Tile(bids_info), row_idx, col_idx)
            i+=1
            
        widget = QWidget()
        widget.setLayout(subjs_layout)
        self.setCentralWidget(widget)

    def clicked(self):
        self.label.setText('you pressed the button')
        self.label.adjustSize()


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


