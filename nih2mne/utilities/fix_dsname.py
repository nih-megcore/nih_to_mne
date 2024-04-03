#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:46:52 2022

@author: jstout
"""

import shutil
import os, glob
import os.path as op
import sys

def fixDsName(input_name):
    dsname = op.abspath(input_name)
    if dsname[-1]=='/': dsname=dsname[:-1]  #Remove trailing slash
    
    if dsname[-3:] != '.ds':
        raise ValueError(f'This is not a meg dataset ending in .ds: {dsname}')
    ds_base = op.basename(dsname)[:-3]
    
    fnames = os.listdir(dsname)
    ds_suffixes=['acq','hc', 'hist','infods','meg4','newds','res4','eeg']
    
    fnames_to_change = [i for i in fnames if i.split('.')[-1] in ds_suffixes]
    for f in fnames_to_change:
        ext=op.splitext(f)
        in_name=op.join(dsname, f)
        out_name=op.join(dsname, ds_base+ext[-1])
        print(in_name, out_name)
        shutil.move(in_name, out_name)
    
    hz_fnames=glob.glob(op.join(dsname,'hz_t_*.txt'))
    if len(glob.glob(op.join(dsname,'hz_t_*.txt'))) >=1:
        for f in hz_fnames:
            in_name=f
            out_name = op.join(dsname, 'hz_t_'+ds_base+'.txt')
            # print(out_name)
            shutil.move(in_name, out_name)
        
if __name__=='__main__':
    input_name = sys.argv[1]
    fixDsName(input_name)
    
        
