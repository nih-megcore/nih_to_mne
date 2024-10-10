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
    QHBoxLayout, QPushButton, QLabel

import sys
from nih2mne.dataQA.bids_project_interface import subject_bids_info, bids_project
import os
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
            tmp = QLabel('    ')
            tmp.setAutoFillBackground(True)
            tmp.setStyleSheet(f"background-color: {color}")
            layout.addWidget(tmp)
        self.setLayout(layout)
    
    def clicked(self):
        '''This will eventually open the subject QA'''
        self.subj_button.setText('you pressed the button')
        self.subj_button.adjustSize()
    
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
        
        
class Subject_GUI():        
    '''All the subject level QA at a finer detail'''




## create the window 

class BIDS_Project_Window(QMainWindow):
    def __init__(self, bids_root=os.getcwd(), gridsize_row=4, gridsize_col=6, 
                 bids_project=None):
        super(BIDS_Project_Window, self).__init__()
        self.setGeometry(100,100, 1080, 800)
        self.setWindowTitle(f'BIDS Folder: {bids_root}')
        self.gridsize_row = gridsize_row
        self.gridsize_col = gridsize_col
        self.bids_project = bids_project
        self.initUI()
        
    def initUI(self):
        layout = QGridLayout()
        subject_keys = sorted(list(self.bids_project.subjects.keys()))
        
        tile_idxs = np.arange(self.gridsize_row * self.gridsize_col)
        tile_idxs_grid = tile_idxs.reshape(self.gridsize_row, self.gridsize_col)
        row_idxs, col_idxs = np.unravel_index(tile_idxs, [self.gridsize_row, self.gridsize_col])
        i=0
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            bids_info = self.bids_project.subjects[subject_keys[i]]
            layout.addWidget(Subject_Tile(bids_info), row_idx, col_idx)
            i+=1
            
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def clicked(self):
        self.label.setText('you pressed the button')
        self.label.adjustSize()


def window():
    app = QApplication(sys.argv)
    win = BIDS_Project_Window(bids_project = bids_pro)
    win.show()
    sys.exit(app.exec_())

# bids_pro = bids_project(bids_root='/fast2/BIDS')

    
window()


# bids_info = bids_pro.subjects['sub-ON95742']
# window = Subject_Tile(bids_info)
# window.show()


