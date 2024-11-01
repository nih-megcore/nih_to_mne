#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:51:20 2024

@author: jstout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import os, os.path as op
import nibabel as nib
import glob

# from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, \
#     QHBoxLayout, QVBoxLayout, QPushButton, QLabel,  QComboBox, QLineEdit


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
imgs = {}
img_shape = None
for img in imgs2:
    print(img)
    mr_tmp = nib.load(img).get_fdata().squeeze()
    if img_shape == None:
        img_shape = mr_tmp.shape
    else:
        # continue
        assert img_shape == mr_tmp.shape
    imgs[op.basename(img).split('_')[0]] = mr_tmp  #assign on subject id
    

imgs.keys()

#FIX !!!!!!  Set panel title -- will not update on second panel

class IndexTracker(object):
    def __init__(self, axes, data_dict):
        # self.ax = axes[0,0]
        # self.ax.set_title('use scroll wheel to navigate images')
        self.axes=axes
        self.data_dict = data_dict
        self.key_list = list(data_dict.keys())
        self.data_array =  np.array([imgs[key] for key in imgs.keys()])
        

        # self.X = data_dict[self.key_list[0]]
        rows, cols, self.slices = self.data_dict[self.key_list[0]].shape #X.shape
        self.ind = self.slices // 2
        
        row_idxs, col_idxs = np.unravel_index(range(axes.shape[0]*axes.shape[1]), axes.shape)
        i=0
        self.im = []
        for row_idx, col_idx in zip(row_idxs,col_idxs):
            self.im.append(self.axes[row_idx, col_idx].imshow(self.data_array[i][:,:,self.ind]))
            self.im[i].axes.xaxis.set_ticks([])
            self.im[i].axes.yaxis.set_ticks([])
            self.im[i].axes.set_title(self.key_list[i])  #FIX !!!!!!  -- will not update on second panel
            plt.tight_layout()
            i+=1
        # self.update()
        self.ani = FuncAnimation(plt.gcf(), self.update, frames=10, interval=None)
        

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        row_idxs, col_idxs = np.unravel_index(range(self.axes.shape[0]*self.axes.shape[1]), self.axes.shape)
        # i=0
        for i in range(self.axes.shape[0]*self.axes.shape[1]): #row_idx, col_idx, in zip (row_idxs, col_idxs):
            self.im[i].set_data(self.data_array[i][:, :, self.ind])
            self.im[i].axes.figure.canvas.draw()
        # return self.im


def plot3d(image):
    fig, axes = plt.subplots(3,4)
    plt.subplots_adjust(0,0,1,1)
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    tracker = IndexTracker(axes, image)
    
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
     #, interval=10, blit=True)
    plt.show()


if __name__ == "__main__":
    img = imgs
    plot3d(img)
    
    
    
#%%
# from matplotlib.animation import FuncAnimation

# im = plt.imshow(np.random.randn(10,10))

# def update(i):
#     A = np.random.randn(10,10)
#     im.set_array(A)
#     return im, text

# ani = FuncAnimation(plt.gcf(), update, frames=range(100), interval=50, blit=False)

# plt.show()
