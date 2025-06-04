#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:44:47 2025

@author: jstout
"""
from mne.fixes import _safe_svd
import numpy as np

# Mod of the pca flip from mne v1.5
def _pca(data):
    U, s, V = _safe_svd(data, full_matrices=False)
    # use average power in label for scaling
    scale = np.linalg.norm(s) / np.sqrt(len(data))
    return scale * V[0]

def get_label_vertex_idxs(label, stc):
    if label.hemi=='lh':
        hemi_idx=0
        hemi_offset = 0
    else:
        hemi_idx=1
        hemi_offset = len(stc.vertices[0])
    label_stc_vertices = label.get_vertices_used(stc.vertices[hemi_idx])
    label_vert_idxs = np.searchsorted(stc.vertices[hemi_idx], label_stc_vertices)
    return label_vert_idxs
    

def get_full_label_ts(label, stcs):
    '''Returns a continuous dataset of the stcs data from the label'''
    label_vert_idxs = get_label_vertex_idxs(label, stcs[0])
    
    #Extract the label vertex data into a list from the stcs
    _tmp = [stcs[i].data[label_vert_idxs,:] for i in range(len(stcs))]
    #Concatenate to get a timeseries of Vertices X epoTime   (epoTime is concatenation along epochs and time)
    label_ts = np.concatenate(_tmp, axis=1) 
    return label_ts

def flip_verts(label, stcs):
    '''Compute the label vertices flips based on correlations with 1st PCA.
    '''
    label_vert_data = get_full_label_ts(label, stcs)
    label_pca = _pca(label_vert_data)
    
    # Identify the in-phase data to determine the flips
    _tmp = np.dot(label_pca, label_vert_data.T)
    flips = _tmp<0
    label_vert_data[flips,:] *= -1
    return label_vert_data


def _compute_label_ts(label, stcs):
    label_vert_data = flip_verts(label,stcs)

    #Get the final PCA after performing the flips
    label_data = _pca(label_vert_data)

    #Reshape the pca back into epochs
    epo_label_pca = label_data.reshape([len(stcs), stcs[0].shape[-1]])
    
    return epo_label_pca

def extract_label_ts(labels, stcs):
    label_dat=[]
    for label in labels:
        print(label.name)
        label_dat.append(_compute_label_ts(label, stcs))
    return np.stack(label_dat)
        



