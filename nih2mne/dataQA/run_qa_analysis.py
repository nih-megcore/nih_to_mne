#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:27:12 2024

@author: jstout
"""
import mne
import os, os.path as op
from mne.preprocessing import ICA
import numpy as np
import copy 
import logging
import pandas as pd
import pyctf
import scipy as sp
import shutil
from sklearn.neighbors import NearestNeighbors
logging.basicConfig(level=logging.INFO)
from scipy.stats import zscore


'''
Load data
Before ICA
    Resample    
    1Hz hp
    60/120/180 notch
    Find jumps
ICA Noisy Chans
    Locate independently operating chans
        Focal source with high gradient - localized to single chan
Noisey Epochs
    Rank based chanXepoch time
    Identify outliers
ICA     
    

'''
def assess_bads(raw, is_eroom=False): # assess MEG data for bad channels
    '''Code sampled from MNE python website
    https://mne.tools/dev/auto_tutorials/preprocessing/\
        plot_60_maxwell_filtering_sss.html'''
    from mne.preprocessing import find_bad_channels_maxwell
    # load data with load_data to ensure correct function is chosen
    raw.info['bads'] = []
    raw_check = raw.copy()

    raw_check.apply_gradient_compensation(0)
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw_check, cross_talk=None, calibration=None, coord_frame='meg',
        return_scores=True, verbose=True, ignore_ref=True)
    
   
    megs = mne.pick_types(raw_check.info, meg=True,ref_meg=False)
    # get the standard deviation for each channel, and the trimmed mean of the stds
    stdraw_megs = np.std(raw_check._data[megs,:],axis=1)
    stdraw_trimmedmean_megs = sp.stats.trim_mean(stdraw_megs,0.1)
    flat_idx_megs = megs[np.where(stdraw_megs < stdraw_trimmedmean_megs/100)[0]]
    flats = []
    for flat in flat_idx_megs:
        flats.append(raw_check.info['ch_names'][flat]) 
    
    auto_flat_chs = auto_flat_chs + flats
    auto_flat_chs = list(set(auto_flat_chs))
    print(auto_noisy_chs, auto_flat_chs)
            
    return auto_noisy_chs, auto_flat_chs  


def get_sensor_locs(raw):
    '''Return sensor coordinates'''
    locs = np.array([i['loc'][0:3] for i in raw.info['chs']])
    return locs

def get_neighbors(raw=None, n_neighbors=6):
    '''Enter the MNE raw object
    Returns neighborhood index matrix with the first column being the index
    of the input channel - and the rest of the columns being the neighbor 
    indices.  n_neighbors defines the number of neighbors found'''
    if raw!=None:
        locs=get_sensor_locs(raw)
    n_neighbors+=1  #Add 1 - because input chan is one of hte "neighbors" 
        
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nbrs.fit(locs)
    distances, neighbor_mat = nbrs.kneighbors(locs) 
    return distances, neighbor_mat

def spatial_deriv(raw, spatial_maps):
    dists, neighbor_mat = get_neighbors(raw) 
    test_mat = np.abs(spatial_maps[neighbor_mat,:])
    diff_mat = np.abs(test_mat[:,1:,:] - test_mat[:,0,:][:,np.newaxis,:])
    
    test = test_mat[:,0,:] / np.abs(test_mat[:,1:,:]).sum(axis=1)
    

def neighborhood_corr(raw, n_neighbors=6):
    dists, neighbor_mat = get_neighbors(raw) 
    corr_vec=np.zeros(neighbor_mat.shape[0])
    for idx,row in enumerate(neighbor_mat):
        tmp = (np.corrcoef(raw._data[row])[0,1:] / dists[idx][1:]) * dists[idx][1:].mean()
        corr_vec[idx] = np.abs(tmp.mean())



OUT_SUFFIXES = ['meg.fif', 'meg1hz.fif', 'epo1hz.fif', 'ica.fif']

class meg_qa():
    def __init__(self, fname=None, clear_results=False):
        self.fname = fname
        self.basename = op.basename(fname).replace('.ds', '')
        logging.info(f'Loading file {fname}')

        self.initialize_dirs(clear_results)
        self.preproc()
        self.bad_chans = dict()
        tmp_flat, tmp_noisy = assess_bads(self.raw, is_eroom=False)
        self.bad_chans['sss'] = dict(flat=tmp_flat, noisy=tmp_noisy)
        self.raw.info['bads'] = tmp_flat + tmp_noisy

        self.do_ica40()
        ica40data = self.ica40.get_sources(self.raw1hp)._data
        # Do the jump detection here:
            
        self.ch_neighbors = get_neighbors(self.raw1hp, 10)
        
        
        
        
        
        #self.raw_noise = None
        #self.hp1_noise = None
    
    def initialize_dirs(self, clear_results):
        parent_dir = op.dirname(self.fname)
        self.outdir = op.join(parent_dir, 'QAdir', op.basename(self.fname).replace('.ds','')) 
        if clear_results==True:
            # Check before deleting directory
            assert self.outdir[-3:]!='.ds'
            assert op.basename(self.outdir) == self.basename
            shutil.rmtree(self.outdir)
        os.makedirs(self.outdir, exist_ok=True)
        
    def preproc(self):
        self.out_raw_fname = op.join(self.outdir, self.basename + '_meg.fif')
        if op.exists(self.out_raw_fname):
            logging.info(f'Loading from file {self.out_raw_fname}')
            self.raw = mne.io.read_raw_fif(self.out_raw_fname)
        else:
            self.raw = mne.io.read_raw_ctf(self.fname, preload=True,
                                           system_clock='ignore')
            if int(self.raw.info['sfreq']) > 600:
                logging.info(f'Resampling from {self.raw.info["sfreq"]} to 600Hz')
                self.raw.resample(600, n_jobs=-1)
            logging.info('Notch filter 60,120,180')
            self.raw.notch_filter([60, 120, 180], n_jobs=-1)
            self.raw.save(self.out_raw_fname)
        
        self.out_raw1hp_fname = op.join(self.outdir, self.basename + '_f1hz_meg.fif')
        if op.exists(self.out_raw1hp_fname):
            logging.info(f'Loading from file {self.out_raw1hp_fname}')
            self.raw1hp = mne.io.read_raw_fif(self.out_raw1hp_fname)
        else:
            self.raw1hp = copy.deepcopy(self.raw).filter(1.0, None, n_jobs=-1)
            self.raw1hp.save(self.out_raw1hp_fname)
        # evts = 
        # self.epo1hp = mne.Epochs
    
    def do_ica40(self):
        out_ica40_fname = op.join(self.outdir, self.basename + '_comp40_ica.fif')
        if op.exists(out_ica40_fname):
            self.ica40 = mne.preprocessing.read_ica(out_ica40_fname)
        else:
            self.ica40 = ICA(n_components=40)
            self.ica40.fit(self.raw1hp) 
            self.ica40.save(out_ica40_fname)
    
    def get_jumps(self, data):
        'Pull the data matrix and do a numpy diff. Input is a numpy matrix'
        tmp_ = np.diff(data, axis=1)
        
    def get_noisy_chans_ica(self):
        'Look for focal topography in the ICA - denoting abberant channels'
        
    
    @property
    def coil_locs_dewar(self):
        return pyctf.getHC.getHC(op.join(self.fname, op.basename(self.fname).replace('.ds','.hc')), 'dewar')

    @property
    def coil_locs_head(self):
        return pyctf.getHC.getHC(op.join(self.fname, op.basename(self.fname).replace('.ds','.hc')), 'head')

    @property    
    def event_counts(self):
        self.events_dframe = pd.DataFrame(self.raw.annotations)
    
    def _get_movement(self):
        "COH2 - COH1"
        
    def _check_jumps(self):
        "Jump artifacts with np.diff"
    
    def _check_eroom(self):
        "Fill in stuff"
    
    def _check_bad_sensors(self):
        "Run ICA to determine single sensor covariance"
    
    def _check_cellphone_artifact(self):
        "Look for the cell bursts"
    
    def _check_bad_epochs(self):
        "Look for epochs that have a substantial deviation from norm"
    

# def test_meg_qa():
fname = ''
qa_fit = meg_qa(fname)

#%%
def test_meg_qa():
    fname = ''
    qa_fit = meg_qa(fname)
    

          

    