#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:45:31 2024

@author: jstout
"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, \
    QHBoxLayout, QPushButton, QLabel

import sys
from nih2mne.dataQA.bids_project_interface import subject_bids_info, bids_project
import os
import numpy as np

# class Color(QWidget):

#     def __init__(self, color):
#         super(Color, self).__init__()
#         self.setAutoFillBackground(True)

#         palette = self.palette()
#         palette.setColor(QPalette.Window, QColor(color))
#         self.setPalette(palette)



## Create subject tile
class Subject_Tile(QWidget):
    # SubjidButton -- MEG () MRI () FS ()
    def __init__(self,bids_info):
        super(Subject_Tile, self).__init__()
        
        self.meg_count = bids_info.meg_count
        self.meg_status = 'GOOD'
        self.mri_status = 'GOOD'
        self.fs_status = 'GOOD'
        self.evt_status = 'GOOD'
        
        self.subj_button = QPushButton(self) #bids_info.subject)
        self.subj_button.setText(bids_info.subject) 
        self.subj_button.clicked.connect(self.clicked)
        
        layout = QHBoxLayout()
        layout.addWidget(self.subj_button)
        
        for tag in ['Meg', 'Mri','FS','EVT']:
            tmp = QLabel(tag)
            layout.addWidget(tmp)
            tmp = QLabel('    ')
            tmp.setAutoFillBackground(True)
            tmp.setStyleSheet("background-color: lightgreen")
            layout.addWidget(tmp)
        self.setLayout(layout)
    
    def clicked(self):
        '''This will eventually open the subject QA'''
        self.subj_button.setText('you pressed the button')
        self.subj_button.adjustSize()
    
    def color(self, tag=None):
        if getattr(self, tag) == 'GOOD':
            return 'green'
        else:
            return 'red'
        
        
        




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
        # self.label = QtWidgets.QLabel(self)
        # self.label.setText("test")
        # self.label.move(50,50)
        
        # self.b1 = QtWidgets.QPushButton(self)
        # self.b1.setText('Click')
        # self.b1.clicked.connect(self.clicked)
        layout = QGridLayout()
        
        subject_keys = sorted(list(self.bids_project.subjects.keys()))
        
        tile_idxs = self.gridsize_row * self.gridsize_col
        tile_idxs_grid = tile_idxs.reshape(self.gridsize_row, self.gridsize_col)
        row_idxs, col_idxs = np.unravel_index(tile_idxs, [self.gridsize_row, self.gridsize_col])
        i=0
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            bids_info = self.bids_project.subject[subject_keys[i]]
            layout.addWidget(Subject_Tile(bids_info), row_idx, col_idx)
            
            
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def clicked(self):
        self.label.setText('you pressed the button')
        self.label.adjustSize()


def window():
    app = QApplication(sys.argv)
    win = QMainWindow() #BIDS_Project_Window()
    win = 
    win.show()
    sys.exit(app.exec_())

bids_pro = bids_project(bids_root='/fast2/BIDS')

    
window()
#%%
bids_info = bids_pro.subjects['sub-ON95742']
window = Subject_Tile(bids_info)
window.show()


