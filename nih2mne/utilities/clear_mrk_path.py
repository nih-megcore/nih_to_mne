#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:55:29 2020

@author: stoutjd
"""

import glob, os


def clean_filepath_header(mrk_file):
    '''Find the PATH OF DATASET flag and remove the next entry'''
    tmp=open(mrk_file, 'r')
    data=tmp.readlines()
    path_idx=data.index('PATH OF DATASET:\n')+1
    data[path_idx]='#Cleared_Path\n'
    tmp.close()
    
    #Write the augmented data
    tmp = open(mrk_file, 'w')
    tmp.writelines(data)
    tmp.close()
    
    #Try to read a datapath to verify that all Unix style paths have been cleared
    tmp = open(mrk_file, 'r')
    data = tmp.readlines()
    data = [i[:-1] for i in data if i[-1:]=='\n']
    tmp.close()
    for val in data:
        if len(val.split('/')) > 1: 
            print('Could not clear the path or "/" is used in the file')
            raise ValueError

def calc_extra_mark_filelist(filename, mrk_outfile='MarkerFile.mrk'):
    extra_mrk_files = glob.glob(os.path.join(filename,'*.mrk')) + \
        glob.glob(os.path.join(filename, '*.mrk.bak')) + \
        glob.glob(os.path.join(filename, '*.mrkBAK'))
    extra_mrk_files_basename=[os.path.basename(name) for name in extra_mrk_files]
    outfile_idx = extra_mrk_files_basename.index(mrk_outfile)
    _ = extra_mrk_files.pop(outfile_idx) #discard outfile from deletion
    return extra_mrk_files


def remove_extra_mrk_files(clear_filelist):
    '''Remove all backup mark files'''
    print('Removing extraneous mark files')
    for i in clear_filelist:
        if 'mrk' not in i:
            print('Warning: Not removing {}.  It does not have "mrk" in the name')
            continue
        print('Removing mark file {}'.format(i))
        os.remove(i)

if __name__=='__main__':    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-strip_marker_path', help='Remove the marker file path that may have subject info', action='store_true')
    parser.add_argument('-ds_filename', help='CTF filename - generally a folder ending in .ds')
    parser.add_argument('-delete_extra_mrk_files', help='Generally CTF creates backups of the .mrk files.  This option deletes all extra copies', action='store_true')
    parser.description='''Strip path from marker file for anonymous upload'''
    
    args = parser.parse_args()
    if not args.ds_filename:
        raise ValueError('No dataset filename provided')
    else:
        filename = args.ds_filename
    if args.strip_marker_path:
        mrk_outfile = os.path.join(filename, 'MarkerFile.mrk')
        clean_filepath_header(mrk_outfile)
    if args.delete_extra_mrk_files:
        extra_mrk_files = calc_extra_mark_filelist(filename)
        remove_extra_mrk_files(extra_mrk_files)
        
       
        
        
        