#! /usr/bin/env python

import sys, os
import os.path as op
import numpy as np
#from pyctf.thd_atr import afni_header_read
#from pyctf.fid import NASION, LEAR, REAR
import mne
from mne.io.constants import FIFF
from mne.io.meas_info import read_fiducials, write_fiducials
import json
from mne.coreg import _fiducial_coords, fit_matched_points
from mne.transforms import read_trans, write_trans, Transform
from mne.io import read_raw_ctf

from nih2mne.bstags import txt_to_tag

#NASION='Nasion'
#LEAR='Left Ear'
#REAR='Right Ear'

def coords_from_tagfile(tag_fname):
    fid = open(tag_fname)
    lines = fid.readlines()
    lines = [lineval.replace('\n','') for lineval in lines]
    fid.close()
    coord = {}
    for row in lines:
        if 'Nasion' in row:
            keyval, xyz = row.split(sep=' ', maxsplit=1)
        if ('Left Ear' in row) | ('Right Ear' in row):
            keyval_1, keyval_2, xyz = row.split(sep=' ', maxsplit=2)
            keyval=keyval_1+' '+keyval_2
        keyval = keyval.strip("'") #Remove extra quotes
        xyz = [float(i) for i in xyz.split(' ')]
        coord[keyval] = xyz
    return coord

def coords_from_bsight_txt(bsight_txt_fname):
    '''Input the text file from the brainsight file.
    This is exported to a text file using the brainsight software'''
    tags = txt_to_tag(bsight_txt_fname)
    lines = tags.values()
    coord = {}
    for row in lines:
        if 'Nasion' in row:
            keyval, xyz = row.split(sep=' ', maxsplit=1)
        if ('Left Ear' in row) | ('Right Ear' in row):
            keyval_1, keyval_2, xyz = row.split(sep=' ', maxsplit=2)
            keyval=keyval_1+' '+keyval_2
        keyval = keyval.strip("'") #Remove extra quotes
        xyz = [float(i) for i in xyz.split(' ')]
        coord[keyval] = xyz
    return coord

def correct_keys(input_dict):
    '''Change the NIH MEG keys to BIDS formatted keys'''
    if 'Nasion' in input_dict:
        input_dict['NAS'] = input_dict.pop('Nasion')
    if 'Left Ear' in input_dict:
        input_dict['LPA'] = input_dict.pop('Left Ear')
    if 'Right Ear' in input_dict:
        input_dict['RPA'] = input_dict.pop('Right Ear')
    return input_dict
        




def write_mne_fiducials(subject=None, subjects_dir=None, tagfile=None, 
                        bsight_txt_fname=None, output_fid_path=None,
                        t1w_json_path=None):
    '''Pull the LPA,RPA,NAS indices from the T1w json file and correct for the
    freesurfer alignment.  The output is the fiducial file written in .fif format
    written to the (default) freesurfer/bem/"name"-fiducials.fif file
    
    Inputs:
        subject - Freesurfer subject id
        t1w_json_path - the BIDS anatomical json w/ fiducial locations LPA/RPA/NAS
        subjects_dir - Freesurfer subjects dir - defaults to $SUBJECTS_DIR
    
    Currently requires freesurfer on the system to extract the c_ras info'''
    
    if output_fid_path!=None:
        if output_fid_path[-14:]!='-fiducials.fif':
            print('The suffix of the filename must be -fiducials.fif')
            sys.exit(1)
    

    #Load an input for fiducial localizer
    if tagfile!=None:
        mri_coords_dict = coords_from_tagfile(tagfile)
    elif bsight_txt_fname!=None:
        mri_coords_dict = coords_from_bsight_txt(bsight_txt_fname)
    elif t1w_json_path!=None:
        with open(t1w_json_path, 'r') as f:
            t1w_json = json.load(f)        
            mri_coords_dict = t1w_json.get('AnatomicalLandmarkCoordinates', dict())
    else:
        raise(ValueError('Must assign tagfile, bsight_txt_fname, or t1w_json'))
    
    if subjects_dir == None:
        subjects_dir=os.environ['SUBJECTS_DIR']
    Subjdir = subjects_dir
    
    name = op.join(Subjdir, subject)
    if not os.access(name, os.F_OK):
        print("Can't access FS subject", subject)
        sys.exit(1)

    # Get the origin offset of the FS surface.
    c_ras = None
    
    from subprocess import check_output
    import shutil
    if not shutil.which('mri_info'):
        print('Could not find freesurfer mri_info.  If on biowulf - \
              load freesurfer module first')
        sys.exit(1)
    offset_cmd = 'mri_info --cras {}'.format(os.path.join(Subjdir,
                                                          subject, 'mri', 'orig','001.mgz'))
    offset = check_output(offset_cmd.split(' ')).decode()[:-1]
    offset = np.array(offset.split(' '), dtype=float)
    
    c_ras = offset * .001  # mm to m
    
    d={}
    for label, x in mri_coords_dict.items(): 
        x = np.array(x) * .001        # convert from mm to m
        x = np.array((-x[0], -x[1], x[2]))  # convert from RAI to LPI (aka ras)
        d[label] = x - c_ras               # shift to ras origin
        
        
    LEAR, NASION, REAR = 'LPA', 'NAS', 'RPA'  
    mri_coords_dict = correct_keys(mri_coords_dict)
    
    # AFNI tag name to MNE tag ident
    ident = { NASION: FIFF.FIFFV_POINT_NASION,
              LEAR: FIFF.FIFFV_POINT_LPA,
              REAR: FIFF.FIFFV_POINT_RPA }
    frame = FIFF.FIFFV_COORD_MRI
    
    # Create the MNE pts list and write the output .fif file.
    pts = []
    for p in [LEAR, NASION, REAR]:    
        pt = {}
        pt['kind'] = 1
        pt['ident'] = ident[p]
        pt['r'] = d[p].astype(np.float32)
        pt['coord_frame'] = frame
        pts.append(pt)
    
    if output_fid_path==None:
        name = op.join(Subjdir, subject, "bem", "{}-fiducials.fif".format(subject))
    else:
        name = output_fid_path
    
    try:
        write_fiducials(name, pts, frame)
        print()
        print('Created {} fiducial file'.format(name))
        print()
        print('Run the pyctf/mne/mktrans.py file to create the trans file needed for MNE')
        return name
    
    except:
        print("Can't write output file:", name)
        
def write_mne_trans(mne_fids_path=None, dsname=None,
                    output_name=None, subjects_dir=None):
    if output_name==None:
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir=os.environ['SUBJECTS_DIR']
        else:
            print('SUBJECTS_DIR not an environemntal variable.  Set manually\
                  during the function call or set with export SUBJECTS_DIR=...')
            sys.exit(1)
        name = op.join(subjects_dir, subject, "bem", "{}-fiducials.fif".format(subject))
    else:
        name = mne_fids_path
    
    fids = read_fiducials(name)
    fidc = _fiducial_coords(fids[0])

    raw = read_raw_ctf(dsname, clean_names = True, preload = False)
    fidd = _fiducial_coords(raw.info['dig'])

    xform = fit_matched_points(fidd, fidc, weights = [1, 10, 1])
    t = Transform(FIFF.FIFFV_COORD_HEAD, FIFF.FIFFV_COORD_MRI, xform)
    if output_name==None:
        output_name = op.join(subjects_dir, subject, "bem", "{}-trans.fif".format(subject))
    if output_name[-10:]!='-trans.fif':
        print('The suffix to the file must be -trans.fif')
        sys.exit(1)
    write_trans(output_name, t)
    return output_name

def view_coreg(dsname=None, trans_file=None, subjects_dir=None):
    raw = read_raw_ctf(dsname)
    trans = mne.read_trans(trans_file)
       
    mne.viz.plot_alignment(raw.info, trans=trans, subject=subject, src=None,
                       subjects_dir=subjects_dir, dig=False,
                       surfaces=['head', 'white'], coord_frame='meg')
    _ = input('Press enter to close')
    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subjects_dir', help='''Set SUBJECTS_DIR different
                        from the environment variable. If not set this defaults
                        to os.environ['SUBJECTS_DIR]''')
    parser.add_argument('-anat_json', help='''Full path to the BIDS anatomy json
                        file with the NAS,RPA,LPA locations''', required=True)
    parser.add_argument('-subject', help='''The freesurfer subject id.  
                        This folder is expected to be in the freesurfer 
                        SUBJECTS_DIR''', required=True)
    parser.add_argument('-trans_output', help='''The output path for the mne
                        trans.fif file''')
    parser.add_argument('-dsname', help='''CTF dataset to create the transform''',
                        required=True)
    parser.add_argument('-view_coreg', help='''Display the coregistration of 
                        MEG and head surface''', action='store_true')
    
    args = parser.parse_args()
    if not args.subjects_dir:
        subjects_dir=os.environ['SUBJECTS_DIR']
    else:
        subjects_dir=args.subjects_dir
        os.environ['SUBJECTS_DIR']=args.subjects_dir
    
    subject=args.subject
    t1w_json_path = args.anat_json
        
    #Write out the fiducials
    mne_fid_name = write_mne_fiducials(subject=subject, t1w_json_path=t1w_json_path, 
                                       subjects_dir=subjects_dir)
    
    if args.trans_output:
        output_path=args.trans_output
    else:
        output_path=None
    mne_trans_name = write_mne_trans(mne_fids_path=mne_fid_name, dsname=args.dsname,
                    output_name=output_path, subjects_dir=subjects_dir)
    
    if args.view_coreg:
        view_coreg(args.dsname, mne_trans_name, subjects_dir)
        
        
    
                        

