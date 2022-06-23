#!/usr/bin/env python
import mne
import numpy as np
import sys

fname = sys.argv[1]
raw = mne.io.read_raw_ctf(fname, preload=False, system_clock='ignore', verbose=False)
if 'UPPT001' not in raw.ch_names:
    print('Bad')
    exit(0)
raw.pick_channels(['UPPT001'])  
raw.load_data(verbose=False)
trig_vec = np.unique(raw.get_data())
if trig_vec.__len__() > 1:
    print(f'Good: trigger vals {trig_vec}')
else:
    print('Bad')
