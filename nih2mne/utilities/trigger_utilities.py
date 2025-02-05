#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 08:59:50 2020

@author: stoutjd
"""
import pandas as pd
import sys, os, tempfile
from math import floor
import numpy as np
import pyctf
from pyctf.util import *
import scipy
import copy


'''
From the WIKI:
    



For psychoPy tasks (Sternberg, Hariri, oddball), the text logfile is imported. The first few timepoints representing instruction screens are removed from the file and the rest are added to the dataframe.
Coding Analog stimuli with Parallel port Values

Analog triggers are generally converted to single states (ON/OFF), the parallel port can be used to code the output values
Reset Timing to Optical Trigger

Due to projector frame buffering there can be timing jitter in the presentation of visual stimuli to the subject. The stimuli computer sends a parallel port trigger to the acquisition computer and the stimuli to the projector. The visual tasks have been designed to activate the upper right corner with a specific bit value The projector - Propix (model ????) has an output BNC signal that triggers when the
Response Coding
The correct response is calculated.... 
'''


def check_analog_inverted(fname=None, ch_name='UADC001'):
    '''Checks to determine if the analog channel has been inverted.
    
    Histogram is performed.  The two histogram bins with the most counts will be 
    the on and off trigger values.  
    The function will return True if the counts for the lower voltage are higher
    than the counts for the high voltage.
    
    Usage:
        fname: Filename
        ch_name: Analog channel name default is UADC001
        print_output: If vizualization needed this will graph using matplolib (True/False)
        
    '''
    df_var=pyctf.dsopen(fname)
    ADC_idx=df_var.getChannelIndex(ch_name)
    dat = df_var.getDsData(0, ADC_idx)
    
    bin_counts, bin_volts =np.histogram(dat)
    
    #Determine the two highest counts
    #By default off_idx should have a greater count than on_idx
    # ** This may prove false if the study is designed to have the trigger on most of the time
    off_idx, on_idx=np.argpartition(bin_counts, -2)[-2:]
    if bin_volts[off_idx] < bin_volts[on_idx]:
        return True
    else:
        return False

############## Code from Tom Holroyd's threshold detect command from pyctf    
def round(x):
    return int(floor(x + .5))

def dydx(d):
    "first difference"
    return d[1:] - d[:-1]

def scale_data(d):
    "scale to 0..1"
    return (d - d.min()) / (d.max() - d.min())

def scale_deriv(d):
    "scale postive values to to 0..1, negative values to -1..0"
    return np.where(d >= 0., d / d.max(), d / -(d.min()))

from scipy.signal import butter, lfilter

def butter_bandpass(data, lowcut, highcut, fs, order=5):
    '''
    Credits:
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def threshold_detect(dsname=None, channel=None, mark=None, mark_color=None,
                   invert=False, ampThresh=.5, derivThresh=.1, trial=None, 
                   lo=None, hi=None, deadTime=.1, t0=None, t1=None):
    '''Tom Holroyd's code from threshold_detect.py. This is a functionalized version
    of the code from pyctf'''
    ds = pyctf.dsopen(dsname)
    
    srate = ds.getSampleRate()
    ntrials = ds.getNumberOfTrials()
    nsamp = ds.getNumberOfSamples()
    ch = ds.channel[channel]
    
    deadSamp = round(deadTime * srate) - 1
    if t0 is not None:
        t0 = round(t0 * srate)
        t1 = round(t1 * srate)
    
    # if lo is not None:
    #     try:
    #         # filt = pyctf.mkiir(lo, hi, srate)
    #     except RuntimeError:
    #         filt = pyctf.mkfft(lo, hi, srate, nsamp)
    
    # output mark list, trial and time
    marklist = []
    
    # list of trials to process
    if trial is None:
        trl = range(ntrials)
    else:
        trl = [trial]
    
    for tr in trl:
        x = ds.getDsRawData(tr, ch)
        if t0 is not None:
            x = x[t0 : t1 + 1]  # data to use when scaling
        if lo is not None:
            x =  butter_bandpass(x, lo, hi, srate, order=6) #pyctf.dofilt(x, filt)
        if invert:
            x = -x
        d = dydx(x)
        x = scale_data(x)
        #with open("/tmp/moo", "w") as f:
        #    for v in x: f.write("%g\n" % v)
        d = scale_deriv(d)
        #with open("/tmp/mu", "w") as f:
        #    for v in d: f.write("%g\n" % v)
    
        if derivThresh < 0.:
            d = -d
            derivThresh = -derivThresh
        if t0 is None:
            s = 1
            e = nsamp - 1
        else:
            y = np.zeros(nsamp)
            y[t0 : t1 + 1] = x
            x = y
            s = t0 + 1
            e = t1 - 1
        while s < e:
            if x[s] > ampThresh and d[s] > derivThresh:
                marklist.append((tr, s / srate))
                s += deadSamp
            s += 1
    output = pd.DataFrame(marklist, columns=['trial', 'onset'])
    output['condition'] = mark
    output['channel'] = channel
    return output
    #return marklist

########### End of threshold detect

def return_edge_timing(ppt_vector, positive_edge=True):
    '''Enter a parrallel port numpy array and return an array of the same
    size with only the positive going triggers and everything else is zero'''
    tmp=np.diff(ppt_vector)  #does this need a prepended 0 <<<<<<<<<<<<<<<<<<<<<<<<<<<
    tmp=np.concatenate([np.array([0]),tmp])  #Here it is prepended CHECK ########################3
    if positive_edge==True:
        tmp[tmp<=0]=0
    else:
        tmp[tmp>=0]=0
    return tmp

def samples_to_trig_timing(ds,digital_vector):
    '''Return the times and values of the of parrallel port positive transitions'''
    sample_idx=np.argwhere(digital_vector>0)
    values=digital_vector[digital_vector>0]
    times=[ds.getTimePt(i) for i in sample_idx]
    return np.array([times, values]).T

def return_ch_pattern(ds, pattern):
    '''Return the the channel name and index for channels present with the
    matching pattern'''
    all_ch_names = list(ds.channel.keys())
    pattern_ch_names = list(filter(lambda x: (pattern in x), all_ch_names))
    pattern_ch_idx = [ds.channel[i] for i in pattern_ch_names] 
    return  pattern_ch_names, pattern_ch_idx

def detect_digital(filename, channel='UPPT001', trial=0):
    if 'PPT' not in channel:
        print('''WARNING: This channel is likely not a digitial parrallel port.
              Use threshold detect instead''')
    ds=pyctf.dsopen(filename)
    idx = return_ch_pattern(ds, channel)[1][0]
    tmp=ds.getDsRawData(trial, idx)
    tmp=return_edge_timing(tmp)
    tmp=samples_to_trig_timing(ds, tmp)
    tmp_dframe=pd.DataFrame(tmp, columns=['onset', 'condition'])
    tmp_dframe['channel'] = channel
    tmp_dframe['trial']=trial
    tmp_dframe['condition'] = tmp_dframe['condition'].astype(int).astype(str)
    return tmp_dframe[['trial', 'onset', 'condition', 'channel']]    

######  End digital channel processing


def append_conditions(dframe_list=[]):
    '''Return a concatenated dataframe sorted by onset time'''
    output = pd.concat(dframe_list, ignore_index=True)
    output.sort_values('onset', inplace=True, ignore_index=True)
    return output

def get_window_value(onset=None, window=None, lag_time_vector=[], time_on='lag',
                     negate=False):
    '''From the onset time, find the first lag time within the window'''
    if negate and (time_on=='lag'):
        raise ValueError('Can only perform null windows if time_on is lead')
    lag_window = lag_time_vector
    lag_window = lag_window[(onset + window[0] < lag_window) & (lag_window < onset + window[1])]
    if negate == True:
        if len(lag_window) > 0:
            return np.nan
        else:
            return onset
    if len(lag_window) == 0:
        return np.nan
    if time_on.lower() == 'lag':
        return lag_window[0] #lag_time_vector[boolean_vector][0]
    elif time_on.lower() == 'lead':
        return onset

def parse_marks(dframe=None, lead_condition=None, lag_condition=None, window=[0,0.5],
                marker_on='lead', marker_name=None, append_result=True, trial=0,
                null_window=False):
    '''Parse trigger codes in the dataframe
    Conditions typically take string inputs even integers "1","2","3"...
    Marker name is used to code the output name
    
    The timing returned depends on if maker_on is set to lead or lag condition
    
    If append_result the original dataframe plus new condition marker will be 
    returned.  dframe = parse_marks(dframe, ..... append_result=True)
    
    null window causes parse_marks return values if the condition is not met in
    that window.  e.g. Return A if B not in 500ms
    '''
    
    lead_idxs = dframe[dframe.condition == lead_condition].index
    lag_idxs = dframe[dframe.condition == lag_condition].index

    lag_time_vector = dframe.loc[lag_idxs].onset.values
    new_onsets=dframe.loc[lead_idxs]['onset'].apply(get_window_value,window=window, 
                                         lag_time_vector=lag_time_vector, 
                                         time_on=marker_on, negate=null_window) 
    new_condition = pd.DataFrame(new_onsets.values, columns=['onset'])
    if marker_on=='lead':
        new_condition['channel']='*'+lead_condition+'>'+lag_condition
    else:
        new_condition['channel']=lead_condition+'>*'+lag_condition
    new_condition['condition']=marker_name
    new_condition['trial']=trial
    if append_result==True:
        return append_conditions([dframe, new_condition])
    else:
        return new_condition
    
def crop_logfile_overflow(dframe, end_buffer=0.5):
    '''
    Sometimes the stimulus continues to present when the acquisition has finished.
    This is a utility function to crop off the end of the logfile.

    Parameters
    ----------
    dframe : Event dataframe

    Returns
    -------
    dataframe

    '''
    final_proj = dframe[dframe.condition=='projector'].iloc[-1].onset
    crop_bool = dframe[dframe.channel=='logfile'].onset > final_proj+end_buffer
    drop_idxs = crop_bool[crop_bool==True].index
    return dframe.drop(drop_idxs) 

# =============================================================================
# Higher level functions that rely on the above
# =============================================================================
def correct_to_projector(dframe, projector_eventID='projector', event_list=[],
                         window=[-0.2, 0.2]):
    '''
    Take a list of events and correct them to the projector
    
    The items in the event list must be in the input dataframe.  The dataframe
    is created using digital_detect and threshold_detect. The two dataframes
    can be combined using append_conditions([dframe1, dframe2,...])

    Parameters
    ----------
    dframe : pd.DataFrame
        Dataframe.
    projector_eventID : str
        Event ID in dataframe for projector. The default is 'projector'.
    event_list : list
        Events to correct to projector timing. The default is [].

    Returns
    -------
    out_dframe : pd.DataFrame
        Corrected dataframe

    '''
    corr2proj_dframe_list = []
    dframe = copy.deepcopy(dframe) #Operate on a copy of the dataframe
    for evt_name in event_list:
        if evt_name=='projector':
            continue
        evt_corr_dframe = parse_marks(dframe, lead_condition=projector_eventID, 
                                 lag_condition=evt_name, marker_on='lead', 
                                 marker_name=evt_name, append_result=False, 
                                 window=window)
        evt_corr_dframe = evt_corr_dframe.dropna()
        corr2proj_dframe_list.append(evt_corr_dframe)
    out_dframe = append_conditions(corr2proj_dframe_list)
    
    out_val_counts = out_dframe.condition.value_counts()
    in_val_counts = dframe.query(f'condition in {event_list}')['condition'].value_counts()
    
    # Sanity check that all of your triggers are preserved
    for evt_name in event_list:
        in_count = in_val_counts.loc[evt_name]
        out_count = out_val_counts.loc[evt_name]
        if in_count != out_count:
            print(f'warning:')
            print(f'{evt_name} - IN={in_count} != OUT={out_count}')
            print('It is possible that you may have selected a response channel to correct to projector')
    
    # Sanity check event order preserved
    selection_dframe = dframe.query(f'condition in {event_list}')
    selection_dframe.reset_index(inplace=True, drop=True)
    if np.all(selection_dframe.condition.values == out_dframe.condition.values):
        timing_diff = out_dframe.onset.values - selection_dframe.onset.values 
        timing_diff *= 1000
        t_diff_mean = np.round(np.mean(timing_diff), 2)
        t_diff_std = np.round(np.std(timing_diff), 2)
        print(f'Correction performed: Mean={t_diff_mean} ms  Std={t_diff_std} ms')
    else:
        print('The resulting order has not been preserved')
    
    # Add the conditions that were not in the event list back into new dataframe
    unaffected_conditions = dframe.query(f'condition not in {event_list}')
    out_dframe = append_conditions([out_dframe, unaffected_conditions])
        
    
    return out_dframe     

def add_event_offset(dframe, event_list=[], offset=0.0):
    '''
    Add a static offset to events in the event list
    Offset is in seconds
    

    Parameters
    ----------
    dframe : pd.DataFrame
        dataframe 
    event_list : list, optional
        event name list. The default is [].
    offset : TYPE, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    dataframe

    '''
    corr2proj_dframe_list = []
    out_dframe = copy.deepcopy(dframe)
    selection_dframe = out_dframe.query(f'condition in {event_list}')
    out_dframe.loc[selection_dframe.index, 'onset']+=offset
    return out_dframe
    
    
    


# tmp = correct_to_projector(dframe, event_list=['test3','test2'], window=[-0.3,0.3])
# tmp = add_event_offset(dframe, event_list=['test2','test3'], offset=0.019)

# def correct_events_to_projector(dframe, projector_channel='UADC016', 
#                                 input_channel=None, proc_on_derived=False):
#     '''
    

#     Parameters
#     ----------
#     dframe : pandas.DataFrame
#         Events extracted from .
#     projector_line : TYPE, optional
#         DESCRIPTION. The default is 'UADC016'.
#     input_channel : str or list, required
#         Provide a single channel or a list of channels to correct to the 
#         projector event timing. The default is None.

#     Returns
#     -------
#     None.

#     '''
#     if (('<' in input_channel) or ('>' in input_channel)) and proc_on_derived==False:
#         raise ValueError('Cannot process on derived channels')
#     channel_set = set(dframe.channel.values)
    
#     if type(input_channel) is None:
#         raise ValueError('Input channel is required')
#     elif type(input_channel) is str:
#         input_channel=[input_channel]
    
#     #Check if the channels are actually in the data
#     for inchan in input_channel:
#         assert inchan in channel_set
    
#     #Make a list of the modified data
#     # for inchan in input_channel:
        
        
        
        
        
    
#     # inchan_frame = dframe[dframe
    
#     # dframe.loc[
#     # parse_marks(dframe, 
#     #             lead_condition =     ,
#     #             lag_condition =     , 
#     #             )
    
    
  
    
    
