#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday March 6

@author: stoutjd
"""
import os, sys
import numpy as np
import pandas as pd

usage='''Convert pandas dataframe to CTF formatted mark file'''

def return_header1(dataset_path):
    return 'PATH OF DATASET:\n{}\n\n\n'.format(dataset_path)

def return_header2(stims):
    number_of_stims=len(stims)
    return 'NUMBER OF MARKERS:\n{}\n\n'.format(number_of_stims)

def create_stim_mark_block(classgroupid=None,name=None,color=None,classid=None,
                      number_of_samples=None, comment='Autocomment', editable='No'):
    '''Assemble a trial block'''
    
    #Do not adjust the tab spacing below - it is removed later by
    template_txt='''
        CLASSGROUPID:
        +{}
        NAME:
        {}
        COMMENT:
        {}
        COLOR:
        {}
        EDITABLE:
        {}
        CLASSID:
        +{}
        NUMBER OF SAMPLES:
        {}
        LIST OF SAMPLES:
        TRIAL NUMBER		TIME FROM SYNC POINT (in seconds)\n'''.replace('        ','').format(str(classgroupid),
                name,comment, color, editable,str(classid), str(number_of_samples))
    return template_txt
    
def dframe_to_single_stimtimes(dframe, column_id=None, stim_name=None, 
                                  time_column_id='onset', trial_num=0):
    '''Return the time values for a marker
    Zeros out null values and filters for stimnames'''
    stim_times=dframe[dframe[column_id]==stim_name][time_column_id].values
    return ['\t\t\t'+str(trial_num)+'\t\t\t\t\t+'+str(i) for i in stim_times]

################## Color calculation
def _pad0(value):
    tmp_val=str(value).strip('0x')
    if len(tmp_val)<2:
        return '0'+str(tmp_val)
    return str(tmp_val)

def create_color_tag(red256, green256, blue256):
    color_sub_red=_pad0(hex(red256))
    color_sub_green=_pad0(hex(green256))
    color_sub_blue=_pad0(hex(blue256))
    return '#'+color_sub_red+color_sub_green+color_sub_blue

np.random.seed(0)
def create_next_color(): #(color_idx):
    r_= np.random.randint(0,255)  # r_= list(range(0,255,40))[color_idx]
    g_= np.random.randint(0,255)  # g_= list(range(255,0,-40))[color_idx]
    b_= np.random.randint(0,255) # b_= list(range(0,256,40))[color_idx]
    return create_color_tag(r_ , g_ , b_ )
#################### << end color calculation

def append_file(mrk_output_file, textwrite=''):
    '''Open and append the marker file with the header section'''
    if textwrite=='':
        print('Warning: Nothing to write')
    else:
        with open(mrk_output_file, 'a+') as output_file:
            output_file.writelines(textwrite)

def append_stim_vector(dframe, column_name=None, classid=None, 
                       mrk_output_file=None):
    '''Enter a dataframe and a column name.  All unique values in the column
    will be added to the mrk file'''
    classgroupid=3 
    rfmt = dframe
    stim_names=rfmt[column_name].dropna().unique()
    #Loop over the stim_names to write the stim times into the blocks
    for stim_name in stim_names:
        color=create_next_color() 
        
        #Get stim times as an array to tab prefixed strings
        stim_times=dframe_to_single_stimtimes(rfmt, column_id=column_name, stim_name=stim_name)
        #Generate the header text for each stim
        block_header=create_stim_mark_block(classgroupid=classgroupid, name=stim_name,color=color,
                               classid=classid, number_of_samples=len(stim_times), comment='Autocomment', editable='No')
        #Join the array using newlines
        stim_times_txt='\n'.join(stim_times)
        #Concatenate the header with the stim times
        block_txt=block_header+stim_times_txt + '\n\n'
        
        classid+=1
        append_file(mrk_output_file, textwrite=block_txt)
    return classid
 
    


def main(dframe=None, ds_filename=None, mrk_output_file=None, 
         stim_column='condition'):
    '''Requires data input to be in the form of a pandas dataframe
    
    Converts the dataframe produced by the process_{task}.py files
    The header template will be added to the Markerfile.
    Each stimuli will be appended as a section in the marker file with a 
      psuedo-random color assigned to the code.

    If a marker file is present in the MEG folder, the file will be moved to 
    a backup copy called MarkerFile.mrkBAK_{DATE_TIME}
    '''
    if mrk_output_file==None:
        mrk_output_file=os.path.join(os.path.abspath(ds_filename), 'MarkerFile.mrk')
    if os.path.exists(mrk_output_file):
        import shutil, datetime
        shutil.move(mrk_output_file, mrk_output_file+'BAK_{}'.format(datetime.datetime.today().strftime('%m%d%Y_%H:%M')))
    rfmt=dframe
    
    #classgroupid=3  #Current understanding is that this index must start at 3
    classid=1 #Start the index for the CTF stims
    
    stim_names=rfmt[stim_column].dropna().unique()  ######  This may need to be changed name wise ---------------
    
    #Add filename header
    hdr1=return_header1(ds_filename)
    append_file(mrk_output_file, textwrite=hdr1)
    
    #Add marker number to the header
    hdr2=return_header2(stim_names) 
    append_file(mrk_output_file, textwrite=hdr2)
    
    #Iterate over stimuli and append to mrk file
    if stim_column != None:
        classid = append_stim_vector(dframe, column_name=stim_column, classid=classid, 
                           mrk_output_file=mrk_output_file)
    
    # #Process Targets
    # if target_column != None:
    #     classid = append_stim_vector(dframe, column_name=target_column, classid=classid, 
    #                         mrk_output_file=mrk_output_file)
    
    # #Process Responses
    # if response_column != None:
    #     classid = append_stim_vector(dframe, column_name=response_column, classid=classid, 
    #                         mrk_output_file=mrk_output_file)
    
    append_file(mrk_output_file, textwrite='\n') #Requires two returns at the end of file    
    
    

if __name__=='__main__':    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-logfile', help='Logfile from psychopy - Not required')
    parser.add_argument('-ds_filename', help='CTF filename - generally a folder ending in .ds')
    #parser.add_argument('-mrk_file', help='Markerfile to create - will use the ds_filename/MarkerFile.mrk by default')
    parser.add_argument('-stim_dictionary', help='Csv file with the stimuli name to code')
    parser.add_argument('-processed_dframe', help='Dataframe in .csv format to put into mrk file')
    parser.description='''Used to create a CTF mark file from a pandas dataframe
    The default is to create a MarkerFile.mrk in the ds_filename folder
    '''
    
    args = parser.parse_args()
    if not args.ds_filename:
        raise ValueError('No dataset filename provided')
    else:
        filename = args.ds_filename
    if args.logfile:
        log_file=args.logfile
    if args.stim_dictionary:
        stim_dictionary=args.stim_dictionary
        import csv
        #Check for comma delimited
        try:
            tmp=open(stim_dictionary)
            csv_reader=csv.reader(tmp)
            stim_dictionary=dict(csv_reader)
        except Exception as b:
            #Check for tab delimited
            try:
                tmp=open(stim_dictionary)
                csv_reader=csv.reader(tmp, delimiter='\t')
                stim_dictionary=dict(csv_reader)
            except Exception as e:
                print('The csv file could not be interpretted as comma or tab \
                      delimited ')
                raise ValueError()
        #Convert text to numerical  ---  Checks for hex and converts
        if stim_dictionary[list(stim_dictionary.keys())[0]][0:2]=='0x':
            stim_dictionary={i:int(j,16) for (i,j) in stim_dictionary.items()}
        else:
            stim_dictionary={i:int(j) for (i,j) in stim_dictionary.items()}
    if args.stim_dictionary and args.logfile and args.ds_filename:
        from megblocks.interfaces.trigger_to_dframe import main as create_dframe
        dframe=create_dframe(filename, log_file, stim_dictionary)
        main(dframe=dframe, ds_filename=filename, mrk_output_file=None,
             stim_column='Stims_Time_Aligned', target_column='Targets_Time_Aligned')
    elif args.ds_filename and args.processed_dframe:
        dframe_filename = args.processed_dframe
        dframe = pd.read_csv(dframe_filename, index_col=0)
        if 'Targets_Time_Aligned' in dframe.columns:
            main(dframe=dframe, ds_filename=filename, stim_column='Stims_Time_Aligned', 
                 target_column='Targets_Time_Aligned', response_column='Response')
        else:
            main(dframe=dframe, ds_filename=filename, stim_column='Stims_Time_Aligned', 
                 response_column='Response')
            
      
