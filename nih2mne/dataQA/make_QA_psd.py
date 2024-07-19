#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:32:43 2024

@author: jstout
"""

import sys
import glob
import mne
import os.path as op
import scipy
import copy


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dset')
parser.add_argument('outdir')
args = parser.parse_args()

dset = args.dset
outdir = args.outdir

raw = mne.io.read_raw_ctf(dset, preload=True, system_clock='ignore', 
                          clean_names=True)

#Preproc
raw.resample(300, n_jobs=-1)
raw.filter(0.1, None, n_jobs=-1)
raw.notch_filter([60,120])
raw.apply_gradient_compensation(3)

#Make Epochs
evts = mne.make_fixed_length_events(raw, duration=10)
epo_grads = mne.Epochs(raw, evts, preload=True, baseline=None, 
                 tmin=0, tmax=10)
epo_refs = copy.deepcopy(epo_grads)

#Select channels for the epochs 
ch_grads = [i for i in epo_grads.ch_names if i[0]=='M']
epo_grad = epo_grads.pick(ch_grads)

ch_refs = [i for i in epo_refs.ch_names if i[0]!='M']
epo_ref = epo_refs.pick(ch_refs)

#Get the trimmed mean so that outliers are suppressed
dat = scipy.stats.trimboth(epo_grads._data, 0.1, axis=0)
epo_grads = mne.EpochsArray(dat, epo_grads.info)
dat_ref = scipy.stats.trimboth(epo_refs._data, 0.1, axis=0)
epo_refs = mne.EpochsArray(dat_ref, epo_refs.info)

# Compute PSD
psd_grads = epo_grads.compute_psd()
psd_refs = epo_refs.compute_psd()

# Save
basename = op.basename(dset).replace('.ds','.h5')
psd_grads.save(op.join(outdir, 'PSD_grads_'+basename))
psd_refs.save(op.join(outdir, 'PSD_refs_'+basename))
