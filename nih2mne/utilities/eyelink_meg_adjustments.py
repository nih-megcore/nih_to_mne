#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:30:13 2025

Code to align eyelink data from the MEG acquisition channels: 
    UADC009/10
    
This code does the following:
    Load eyelink and meg data into MNE
    Resample MEG at eyelink sampling freq
    Backfill Nans in eyelink data
    Perform cross correlation between signals to evaluate max corr lag time
    Extract events from eyelink annotations
    Subtract the max lag time from the annotations
    Write to the MEG markerfile
    
@author: jstout
"""
import os, os.path as op
import pandas as pd
from mne.io import read_raw_eyelink
import mne
from nih2mne.utilities.markerfile_write import main as write_markerfile
import numpy as np
from scipy.signal import correlation_lags
from scipy import signal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-eye_fname', 
                    help='Eyelink dataset associated with MEG file (ASC format)', 
                    required=True)
parser.add_argument('-meg_fname', 
                    help='MEG CTF dataset',
                    required=True)
parser.add_argument('-align_time', 
                    help='''Use correllation between MEG and Eylink data to 
                    align the two timeseries.  Subtract the difference before
                    writing the information to the MEG markerfile. 
                    
                    Curently this does not produce perfect alignment - maybe 1/4 second precision
                    ''',
                    default=False, 
                    action='store_true'
                    )
args = parser.parse_args()

eye_fname = args.eye_fname
meg_fname = args.meg_fname
do_align = args.align_time


def impute_locf(arr):
    """
    Imputes NaN values in a 1D NumPy array using the Last Observation Carried Forward (LOCF) method.

    Parameters:
    arr (np.ndarray): The input 1D NumPy array with potential NaN values.

    Returns:
    np.ndarray: The array with NaN values imputed using LOCF.
    """
    # Create a copy to avoid modifying the original array
    imputed_arr = arr.copy()

    # Get the indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(imputed_arr))[0]

    # If the first element is NaN and there are no preceding non-NaN values,
    # you might need a different strategy (e.g., backfill from the first non-NaN).
    # This implementation assumes there will eventually be a non-NaN value to carry forward.
    
    for i in range(len(imputed_arr)):
        if np.isnan(imputed_arr[i]):
            # Find the index of the last non-NaN value before the current position
            previous_non_nan_indices = non_nan_indices[non_nan_indices < i]
            if len(previous_non_nan_indices) > 0:
                last_observed_index = previous_non_nan_indices[-1]
                imputed_arr[i] = imputed_arr[last_observed_index]
            # If no previous non-NaN value exists (e.g., NaNs at the beginning),
            # the NaN will remain unless handled by another strategy.
            
    return imputed_arr

def get_maxcorr_lag(meg_raw, eye_raw):
    'Returns the sample lag with the maximum correlation'
    assert meg_raw.info['sfreq']==eye_raw.info['sfreq'], "Sampling rates are different"
    
    # Get eyelink trace for x-position
    xpos_left = eye_raw.pick('xpos_left').get_data().squeeze()
    xpos_left = impute_locf(xpos_left)
    xpos_left -= np.nanmean(xpos_left) #Mean center
    
    # Get MEG readout of x position of eye data
    meg_xpos = meg_raw.pick('UADC009').get_data().squeeze()
    
    # THIS CAN BE Computationally IMPROVED BY ONLY DOING A SMALL LAG
    # Get the lags 
    lags = correlation_lags(len(xpos_left), len(meg_xpos))
    cross_corr = signal.correlate(xpos_left, meg_xpos, mode='full', method='direct')
    max_corr_index = np.argmax(cross_corr)
    lag_offset = lags[max_corr_index]
    return lag_offset

    

#%% Load data
meg_raw = mne.io.read_raw_ctf(meg_fname, preload=True, system_clock='ignore', 
                          clean_names=True)
eye_raw = read_raw_eyelink(eye_fname, create_annotations = True)

# Make sure sampling rate is same, so correlation can be used
meg_raw.resample(eye_raw.info['sfreq'], n_jobs=-1)

# Generate Dataframes
meg_dframe = pd.DataFrame(meg_raw.annotations)
eye_dframe = pd.DataFrame(eye_raw.annotations)

if not do_align:
    start_time = eye_dframe.query('description=="STARTRUN"')['onset'].values[0]
    # Crop the calibration time
    eye_dframe = eye_dframe[eye_dframe.onset>=start_time]
    # Remove the start time from the eyelink data
    eye_dframe.onset -= start_time
else:
    maxcorr_sample_lag = get_maxcorr_lag(meg_raw, eye_raw)
    maxcorr_time_lag = maxcorr_sample_lag / meg_raw.info['sfreq']
    eye_dframe.onset -= maxcorr_time_lag
    
    
eye_dframe.dropna(inplace=True)
eye_dframe=eye_dframe[eye_dframe.onset >=0]    

final_dframe = pd.concat([meg_dframe, eye_dframe]).reset_index()
final_dframe=final_dframe[['onset','duration','description','orig_time']]

write_markerfile(dframe=final_dframe, 
                 ds_filename=meg_fname, 
                 stim_column = 'description'
                 )







#%%


# pylab.plot(lags, cross_corr)



# np.correlate(xpos_left, meg_xpos)

# from matplotlib import pyplot as plt

# meg_len = meg_xpos.shape[-1]

# fig, axes = plt.subplots(2, 1) 
# axes[0].plot(xpos_left[lag_offset:lag_offset+270000].T)
# axes[1].plot(meg_xpos[:270000].T)


# fig, axes = plt.subplots(2, 1) 
# axes[0].plot(xpos_left)
# axes[0].set_title('MEG')
# axes[1].plot(meg_xpos)
# axes[1].set_title('Eyelink')


#%%
# dframe = pd.DataFrame(raw.annotations)

# eye_dframe = pd.DataFrame(eyelink_dat.annotations)
# # If started correctly
# # start_time = eye_dframe.query('description=="STARTRUN"')['onset'].values[0]
# # eye_dframe = eye_dframe[eye_dframe.onset>=start_time]
# # eye_dframe.onset-=start_time
# lag_time_offset = lag_offset/raw.info['sfreq']
# eye_dframe.onset -= lag_time_offset
# eye_dframe.dropna(inplace=True)
# eye_dframe=eye_dframe[eye_dframe.onset >=0]


# final_dframe = pd.concat([dframe, eye_dframe]).reset_index()


# final_dframe=final_dframe[['onset','duration','description','orig_time']]

# write_markerfile(dframe=final_dframe, 
#                  ds_filename=meg_fname, 
#                  stim_column = 'description'
#                  )

# raw = mne.io.read_raw_ctf(meg_fname, preload=True, system_clock='ignore', 
#                           clean_names=True)
# raw.pick(['UADC009','UADC010'])
