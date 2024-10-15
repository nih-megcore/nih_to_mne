#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:11:10 2024

@author: jstout

Montages can either be a list or a function

"""

montages = {}

montages['montage_56'] = ['MLC21', 'MLC23','MLC25','MLC53', 'MLC55',
 'MLF12','MLF14','MLF42','MLF44','MLF62','MLF64','MLF66',
 'MLO11','MLO32','MLO53','MLP23','MLP41','MLP54',
 'MLT11','MLT16','MLT23','MLT25','MLT51','MLT53','MLT55','MLT57',
 'MRC21','MRC23','MRC25', 'MRC53','MRC55',
 'MRF12','MRF14','MRF42','MRF44','MRF62','MRF64','MRF66',
 'MRO11','MRO32','MRO53',
 'MRP41', 'MRP54',
 'MRT11','MRT16','MRT23','MRT25','MRT51','MRT53','MRT55','MRT57',
'MZC03','MZF01','MZF03','MZO02','MZP01']

def montage_ALL(raw):
    return raw.ch_names

def montage_TRIG(raw):
    montage = [i for i in raw.ch_names if i[0:4] in ['SCLK', 'UADC','UPPT','trig']]
    return montage

montages['montage_ALL'] = montage_ALL
montages['montage_TRIG'] = montage_TRIG
               
