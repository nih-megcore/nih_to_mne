#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:45:18 2022

@author: jstout
"""

import pandas as pd
import nih2mne
import os.path as op
import os 
import numpy as np
import subprocess

dirpath = op.join(nih2mne.__path__[0], 'templates', 'bids_entry_template.csv')
csv_cmdline_mapping = dict(bids_dir='bids_dir',
                        meg_dir='meg_input_dir',
                        mri_brik='mri_brik',
                        brainsight_mri='mri_bsight',
                        brainsight_electrodes='mri_bsight_elec',
                        bids_session='bids_session',
                        subjid_input='subjid_input',
                        bids_id = 'bids_id')

mapping_dtypes = dict(bids_dir=str,
                        meg_input_dir=str,
                        mri_brik=str,
                        mri_bsight=str,
                        mri_bsight_elec=str,
                        bids_session='Int16',
                        subjid_input=str, 
                        bids_id=str)

# =============================================================================
# 
# =============================================================================
def find_end_hdr(csvfile):
    '''
    This determines the End of header line - for import into pandas

    Parameters
    ----------
    csvfile : textfile, csv format
        Entries of csv file input for processing.

    Returns
    -------
    i : int
        End of header line - pandas.read_csv(...,skiprows = i+1)

    '''
    i=0
    with open(csvfile) as w:
        lines=w.readlines()
    for row in lines:
        if 'EndOfHeader' in row:
            break
        i+=1
    return i
            
def get_version(csvfile):
    '''Get the template version - for future changes to code'''
    i=0
    with open(csvfile) as w:
        lines=w.readlines()
    for row in lines:
        if 'version' in row:
            break
        i+=1
    return row.split(':')[-1].replace(' ','').replace(',','').replace('\n','')    


def read_csv_entries(csvfile):
    '''
    Read the CSV file.  
    Skip header.
    Assign datatypes.
    Change empty lines to null values.

    Parameters
    ----------
    csvfile : csv text file
        CSV file with data entries.

    Returns
    -------
    dframe : pd.DataFrame
        dataframe with cleaned entries from csv file

    '''
    hdrline = find_end_hdr(csvfile)+1
    dframe = pd.read_csv(csvfile, skiprows=hdrline, dtype=mapping_dtypes)
    dframe = dframe.replace(r'^\s*$', np.nan, regex=True)
    return dframe


def make_cmd(row):
    '''Assemble a make_meg_bids commandline entry'''
    tmp=row.copy()
    tmp.rename(csv_cmdline_mapping, inplace=True) 
    tmp.dropna(inplace=True) #First round cleaning  
    cmd = ['make_meg_bids.py']
    for option,value in tmp.items():
        cmd.append(f'-{option} {value}')
    cmd = ' '.join(cmd)
    return cmd.replace('"','') 

        
def make_swarm_file(csvfile, swarmfile='megbids_swarm.sh', write=False):
    '''Assemble csv file into swarm file'''
    dframe = read_csv_entries(csvfile)
    swarm = []
    for i,row in dframe.iterrows():
        cmd = make_cmd(row)
        swarm.append(cmd)
    swarm = [i+'\n' for i in swarm]
    if write==True:
        with open(swarmfile,'w') as f:
            f.writelines(swarm)
    else:
        return swarm
    
def make_serial_proc(csvfile, run=False, return_cmd=False):
    dframe = read_csv_entries(csvfile)
    cmd_chain = []
    for i,row in dframe.iterrows():
        cmd = make_cmd(row)
        cmd_chain.append(cmd)
    if (run==False) and (return_cmd==False):
        cmd_chain = ';'.join(cmd_chain)
        print(cmd_chain)
        return
    if return_cmd==True:
        cmd_chain = ';'.join(cmd_chain)
        return cmd_chain
    else:
        for current_cmd in cmd_chain:
            subprocess.run(current_cmd.split())
        
def main():
    import argparse
    template = op.join(op.dirname(__file__),'..', 'templates', 'bids_entry_template.csv')
    template = op.abspath(template)
    parser = argparse.ArgumentParser(description=f'''Write bids from a csv file using the template found in {template}.
                                     Edit the template and save as a csv file somewhere, then use as an input to this command.''')
    parser.add_argument('-csvfile', required=True)
    parser.add_argument('-print_bids_loop', required=False,
                        action='store_true',
                        help='''Print out a serial processing of the bids import''')
    parser.add_argument('-run_bids_loop', required=False,
                        action='store_true',
                        help='''Send the bids loop to a subprocess for computing''')
    parser.add_argument('-write_swarmf', required=False,
                        action='store_true', help='''Write a swarm file from
                        the csv.  Default name is megbids_swarm.sh unless set''')
    parser.add_argument('-swarmfile_fname',
                        action='store_true',
                        help='''Used inconjunction with 
                        -write_swarmf flag''', required=False)
    parser.add_argument('-anonymize',
                        required=False, 
                        help='Deface MRI and anonymize MEG', 
                        default=False, action='store_true')
    parser.add_argument('-additional_args', required=False,
                        help='''Add additional arguments to the processing. 
                        Put all flags in '''
                        )
    args = parser.parse_args()
    
    csvfile=args.csvfile
    if args.print_bids_loop:
        make_serial_proc(csvfile, run=False)
    if args.run_bids_loop:
        make_serial_proc(csvfile, run=True)
    if (args.write_swarmf and args.swarmfile_fname):
        make_swarm_file(csvfile, swarmfile=args.swarmfile_fname, write=True)
    elif args.write_swarmf:
        make_swarm_file(csvfile, write=True)
                        
if __name__ == '__main__':
    main()