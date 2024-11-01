#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:51:20 2024

@author: jstout
"""

import matplotlib.pyplot as plt
import numpy as np
import os, os.path as op
import nibabel as nib
import glob, sys
import pyqtgraph as pg

from PyQt5 import QtWidgets, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, \
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel,  QComboBox, QLineEdit
from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt  


bids_root = '/fast2/BIDS'
os.chdir('/fast2/BIDS')
project = 'nihmeg_nai'
project_root = op.join(bids_root, 'derivatives',project)

nii_files = glob.glob(op.join(project_root, '**','*Enc6o4RatioFS15t35_meg.nii'), recursive=True)
nii_files=nii_files[0:20]

subjs=[]
imgs2=[]
for i in nii_files:
    if op.basename(i).split('_')[0] not in subjs:
        imgs2.append(i)    
        subjs.append(op.basename(i).split('_')[0])



# Try to find the files
img_dict = {}
img_shape = None
for img in imgs2:
    print(img)
    mr_tmp = nib.load(img).get_fdata().squeeze()  #.astype(np.int8) ## << May want to scale this to 16bit
    if img_shape == None:
        img_shape = mr_tmp.shape
    else:
        # continue
        assert img_shape == mr_tmp.shape
    img_dict[op.basename(img).split('_')[0]] = mr_tmp  #assign on subject id
    
def window(img_dict=None, num_rows=6, num_cols=4):
    # os.chdir(bids_project.bids_root)
    app = QApplication(sys.argv)
    win = image_plot_grid(img_dict, gridsize_row=num_rows, gridsize_col=num_cols)
    win.show()
    sys.exit(app.exec_())

#%%
#FIX !!!!!!  Set panel title -- will not update on second panel

class image_plot_grid(QMainWindow):
    def __init__(self, img_dict, gridsize_row=4, gridsize_col=6, cut_axis=2):
        super(image_plot_grid, self).__init__()
        
        self.data_dict = img_dict
        self.subject_keys = sorted(list(img_dict.keys()))
        self.data_array =  np.array([img_dict[key] for key in self.subject_keys])
        self.gridsize_row = gridsize_row
        self.gridsize_col = gridsize_col
        
        #Set the cut plane on the image volumes
        self.axis=cut_axis
        self.slices = self.data_dict[self.subject_keys[0]].shape[cut_axis]
        self.ind = self.slices // 2
        
        main_layout = QVBoxLayout()
        subject_slider_layout = QHBoxLayout()
        self.image_grid_layout = self.init_subject_image_layout()
        subject_slider_layout.addLayout(self.image_grid_layout)
        self.b_slice_adjuster = QSlider(Qt.Vertical)
        self.b_slice_adjuster.valueChanged.connect(self.slice_change)
        self.b_slice_adjuster.setMinimum(0)
        self.b_slice_adjuster.setMaximum(self.data_dict[self.subject_keys[0]].shape[cut_axis]-1)
        self.b_slice_adjuster.setValue(self.ind)
        subject_slider_layout.addWidget(self.b_slice_adjuster)
        main_layout.addLayout(subject_slider_layout)
        
        #Set up final display
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
    def slice_change(self):
        self.ind = self.b_slice_adjuster.value()
        for i in self.subject_keys[0:self.gridsize_row*self.gridsize_col]:
            img_new = self.data_dict[i][:,:, self.ind]
            self.data_setters[i].setImage(img_new, autoRange=False) 
        
    def init_subject_image_layout(self):
        '''Assemble the Volumetric viewer grid'''
        
        self.imgs_layout = QGridLayout()
        tile_idxs = np.arange(self.gridsize_row * self.gridsize_col)
        tile_idxs_grid = tile_idxs.reshape(self.gridsize_row, self.gridsize_col)
        row_idxs, col_idxs = np.unravel_index(tile_idxs, [self.gridsize_row, self.gridsize_col])
        i=0
        self.img_views = {}  #Direct image item for manipulation
        self.data_setters = {} #Widget for adding to grid
        
        #Loop through images and assign GUI outputs
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            if i+1 > len(self.subject_keys): 
                self.imgs_layout.addWidget(QLabel(''), row_idx, col_idx)
            else:
                img2 = np.copy(self.data_array[i][:,:,self.ind]) 
                key = self.subject_keys[i]
                
                imageWidget = pg.GraphicsLayoutWidget()
                vb = imageWidget.addViewBox(row=1, col=1)
                img = pg.ImageItem(image=img2)
                vb.addItem(img)
                vb.setAspectLocked(True)
                cm = pg.colormap.get('CET-L9')
                img.setColorMap(cm)
                self.img_views[key] = imageWidget
                self.data_setters[key] = img
                self.imgs_layout.addWidget(self.img_views[key], row_idx, col_idx)
            i+=1
        return self.imgs_layout




# tmp = image_plot_grid(img_dict)
# test = img_dict[list(img_dict.keys())[0]][:,:,20]
# window(img_dict)


# Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)


if __name__ == "__main__":
    window(img_dict)

