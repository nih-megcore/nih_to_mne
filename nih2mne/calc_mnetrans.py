#! /usr/bin/env python

import copy
import sys, os
import glob
import os.path as op
import numpy as np
import nibabel as nb
import mne
from mne.io.constants import FIFF
from mne.io.meas_info import read_fiducials, write_fiducials
import json
from mne.coreg import _fiducial_coords, fit_matched_points
from mne.transforms import read_trans, write_trans, Transform
from mne.transforms import apply_trans, invert_transform
from mne.io import read_raw_ctf

from nih2mne.bstags import txt_to_tag, txt_to_tag_pd, tags_from_bsight_targetfile


# =============================================================================
# This code is a modification of Tom Holroyd's pyctf/bids code
# The scripts have been modified to run as function calls and dropping 
# some dependencies to run without compilation.
# =============================================================================

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
    if len(tags)==0:
        #Checks if the fiducial labels are in the Electrode Type column
        tags = txt_to_tag_pd(bsight_txt_fname)
    if len(tags)==0:
        raise BaseException(f'''No tags were found in brainsight file:
                            {bsight_txt_fname}''')
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

def json_list_to_dict(tmp):
    '''If json file has a list of values rather than dictionary of key:value\
        pairs - convert to correct format'''
    if type(tmp) is list:
        tmp=dict(tmp)
        if 'NAS:' in tmp.keys():
            tmp['NAS']=tmp.pop('NAS:')
        if 'LPA:' in tmp.keys():
            tmp['LPA']=tmp.pop('LPA:')
        if 'RPA:' in tmp.keys():
            tmp['RPA']=tmp.pop('RPA:')
    return tmp


def correct_keys(input_dict):
    '''Change the NIH MEG keys to BIDS formatted keys'''
    if 'Nasion' in input_dict:
        input_dict['NAS'] = input_dict.pop('Nasion')
    if 'Left Ear' in input_dict:
        input_dict['LPA'] = input_dict.pop('Left Ear')
    if 'Right Ear' in input_dict:
        input_dict['RPA'] = input_dict.pop('Right Ear')
    return input_dict

def coords_from_afni(afni_fname):
    if os.path.splitext(afni_fname)[1] == '.BRIK':
        afni_fname = os.path.splitext(afni_fname)[0]+'.HEAD'
    ## Process afni header  ## >>
    with open(afni_fname) as w:
        header_orig = w.readlines()
    header_orig = [i.replace('\n','') for i in header_orig]
    header = []
    for i in header_orig:
        if i!='' and i[0:4]!='type' and i[0:5]!='count':
            header.append(i)

    name_idxs=[]    
    afni_dict={}
    for idx,line in enumerate(header):
        if 'name' == line[0:4]:
            name_idxs.append(idx)
    for i,idx in enumerate(name_idxs):
        if i != len(name_idxs)-1:
            afni_dict[header[idx][7:].replace(" ","")]=header[idx+1 : name_idxs[i+1]]
        else: 
            afni_dict[header[idx][7:].replace(" ","")]=header[idx+1 : ]
     ## <<   
     ## Done processing Afni header

    if 'TAGSET_NUM' not in afni_dict:
        print("{} has no tags".format(afni_fname), file = sys.stderr)
        sys.exit(1)

    tmp_ = afni_dict['TAGSET_NUM'][0].split(' ')
    afni_dict['TAGSET_NUM']= [int(i) for i in tmp_ if i!='']
    ntags, pertag = afni_dict['TAGSET_NUM']
    if ntags != 3 or pertag != 5:
        print("improperly formatted tags", file = sys.stderr)
        sys.exit(1)

    f = afni_dict['TAGSET_FLOATS']
    lab = afni_dict['TAGSET_LABELS'][0]
    #Remove string garbage
    if lab[0]=='"':
        lab=lab.replace('"','')
    elif lab[0]=="'":
        lab=lab.replace("'","")
    lab = lab.split('~')
    lab = [i for i in lab if i!='']
    
    coords_str = [i.split() for i in f]
    coord ={}
    for label, row in zip(lab,coords_str):
        tmp = row[0:3]
        coord[label] = [float(i) for i in tmp]
    
    return coord

def coords_from_oblique_afni(afni_fname):
    '''Correct for oblique cut afni fiducials:
        The fiducial locations in the TAGSET_FLOATS are relative to a cardinal
        plane and do not reflect the oblique angle.  This function projects the
        data back to the image voxel and uses the IJK_TO_DICOM_REAL / image 
        affine to get the real RAS value.  The output is then converted to 
        LPS to conform to AFNI format.
        The function returns a dictionary of the tag labels with LPS coords'''
    im = nb.load(afni_fname)
    if len(im.header.info['TAGSET_FLOATS'])==15:
        fid_mat = np.array(im.header.info['TAGSET_FLOATS']).reshape(3,5)
    if len(im.header.info['TAGSET_FLOATS'])==25:
        fid_mat = np.array(im.header.info['TAGSET_FLOATS']).reshape(5,5)
        fid_mat = fid_mat[0:3,:]  #!!! HACK to perform 3 point coreg -- Assumes first 3 are LPA/RPA/NAS
    # =============================================================================
    # Correct oblique transform
    # =============================================================================
    tmp_ = mne.transforms.Transform('mri','head')
    tmp_['trans'][0:3,:] = np.array(im.header.info['IJK_TO_DICOM']).reshape(3,4)
    to_ijk = invert_transform(tmp_)
    fid_mat_ijk = apply_trans(to_ijk, fid_mat[:3,:3])
    fid_mat_ras = apply_trans(im.affine, fid_mat_ijk)
    fid_mat_lps = copy.copy(fid_mat_ras)
    fid_mat_lps[:,0:2]*=-1  #Convert to AFNI orientation LPS - assumed downstream
    tag_labels = im.header.info['TAGSET_LABELS'].split('~')
    if len(tag_labels) > 3:
        tag_labels=tag_labels[0:3]  #!!! HACK to perform 3 point coreg
    coord={}
    for idx, label in enumerate(tag_labels):
        if label[0].upper() in ['N','L','R']:
            coord[label]=list(fid_mat_lps[idx, :])
    return coord

        
def write_mne_fiducials(subject=None, subjects_dir=None, tagfile=None, 
                        bsight_txt_fname=None, output_fid_path=None,
                        afni_fname=None, t1w_json_path=None, 
                        searchpath=None, bsight_target_fname=None):
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
    elif bsight_target_fname != None:
        mri_coords_dict = tags_from_bsight_targetfile(bsight_target_fname)
    elif afni_fname!=None:
        mri_coords_dict = coords_from_oblique_afni(afni_fname)
    elif t1w_json_path!=None:
        with open(t1w_json_path, 'r') as f:
            t1w_json = json.load(f)        
            mri_coords_dict = t1w_json.get('AnatomicalLandmarkCoordinates', dict())
        if type(mri_coords_dict) is list:
            mri_coords_dict = json_list_to_dict(mri_coords_dict)
    elif searchpath!=None:
        if not os.path.isdir(searchpath):
            raise(ValueError('Must provide a directory'))
        mri_coords_dict = assess_available_localizers(searchpath)
    else:
        raise(ValueError('''Must assign tagfile, bsight_txt_fname, or t1w_json,
                         or afni_mri'''))
    
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
    if os.path.exists(os.path.join(Subjdir,subject, 'mri', 'orig','001.mgz')):
            offset_cmd = 'mri_info --cras {}'.format(os.path.join(Subjdir,
                                                          subject, 'mri', 'orig','001.mgz'))
    else:
        offset_cmd = 'mri_info --cras {}'.format(os.path.join(Subjdir,
                                                          subject, 'mri', 'T1.mgz'))
    offset = check_output(offset_cmd.split(' ')).decode()[:-1]
    offset = np.array(offset.split(' '), dtype=float)
    
    c_ras = offset * .001  # mm to m
    
    mri_coords_dict = correct_keys(mri_coords_dict)
    
    d={}
    for label, x in mri_coords_dict.items(): 
        x = np.array(x) * .001        # convert from mm to m
        x = np.array((-x[0], -x[1], x[2]))  # convert from RAI to LPI (aka ras)
        d[label] = x - c_ras               # shift to ras origin
        
        
    LEAR, NASION, REAR = 'LPA', 'NAS', 'RPA'  

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
    
    if not op.exists(op.dirname(name)): os.mkdir(op.dirname(name))

    write_fiducials(name, pts, frame, overwrite=True)
    print()
    print('Created {} fiducial file'.format(name))
    return name
        
def write_mne_trans(mne_fids_path=None, dsname=None,
                    output_name=None, subject=None, subjects_dir=None):
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
    
    if dsname.endswith('.ds'):
        raw = read_raw_ctf(dsname, clean_names = True, preload = False, 
                           system_clock='ignore')
    elif dsname.endswith('.con'):
        raw = mne.io.read_raw_kit(dsname, preload=False)
        
    fidd = _fiducial_coords(raw.info['dig'])

    xform = fit_matched_points(fidd, fidc, weights = [1, 10, 1])
    t = Transform(FIFF.FIFFV_COORD_HEAD, FIFF.FIFFV_COORD_MRI, xform)
    if output_name==None:
        output_name = op.join(subjects_dir, subject, "bem", "{}-trans.fif".format(subject))
    if output_name[-10:]!='-trans.fif':
        print('The suffix to the file must be -trans.fif')
        sys.exit(1)
    write_trans(output_name, t, overwrite=True)
    return output_name

def assess_available_localizers(pathvar):
    '''Search through toplevel path and find candidate files with anatomical
    landmarks
    
    Returns list of files with fiducial landmarks
    '''
    # Create candidate lists of entries
    afni_list_tmp = glob.glob(op.join(pathvar, '*+orig.HEAD'))
    txt_list_tmp = glob.glob(op.join(pathvar, '*.txt'))
    tag_list_tmp = glob.glob(op.join(pathvar, '*.tag'))
    bsight_raw_list = glob.glob(op.join(pathvar, '*.bsproj'))
    
    # Assess candidates using screening criteria
    afni_list = [i for i in afni_list_tmp if _afni_tags_present(i)]
    bsight_txt_list = [i for i in txt_list_tmp if _is_exported_bsight(i)]
    tag_list = [i for i in tag_list_tmp if _is_exported_tag(i)]
    
    # Check to make sure that the outputs have the same value
    if len(afni_list)==0 and len(bsight_txt_list)==0 and len(tag_list)==0:
        if len(bsight_raw_list) > 0:
            print('Raw brainsight project found.  Must be exported as text file in \
              brainsight prior to conversion.')
        return []
    
    coords_funcs = []
    # for coord_type, curr_list in zip(['afni','bsight','tag'],[afni_list, bsight_txt_list, tag_list]):
    for coord_type, curr_list in zip(['bsight','tag','afni'],[bsight_txt_list, tag_list,afni_list]):
        if len(curr_list)==0:
            continue
        else:
            if coord_type=='afni':
                coords_funcs.append(zip(afni_list, [coords_from_oblique_afni]))
            elif coord_type=='bsight':
                coords_funcs.append(zip(bsight_txt_list, [coords_from_bsight_txt]))
            elif coord_type=='tag':
                coords_funcs.append(zip(tag_list, [coords_from_tagfile]))
    
    import itertools
    full_list = itertools.chain.from_iterable(coords_funcs)
    coords_out = []
    for fname, coord_func in full_list:
        print(fname)
        coords_out.append(coord_func(fname))
    return coords_out[0]
        
    #Do a data check to verify consistency across coords
            # print(i)
            #coord_func(fname)
    # =============================================================================
    #     Must extract cras from AFNI TO match    
    # =============================================================================
    # offset_cmd = 'mri_info --cras {}'.format(os.path.join(Subjdir,
    #                                                       subject, 'mri', 'orig','001.mgz'))
    # offset = check_output(offset_cmd.split(' ')).decode()[:-1]
    # offset = np.array(offset.split(' '), dtype=float)
    
    # c_ras = offset * .001  # mm to m               
    # print(afni_list_)
    # print(txt_list_)
    # print(tag_list_)
    # print(bsight_raw_list_)
              
def _afni_tags_present(afni_fname):
    '''Verify that the TAGSET is present and not all zeros in the afni HEAD file'''
    tmp_ = nb.load(afni_fname)
    hdr_ = tmp_.header.info
    if ('TAGSET_LABELS' in hdr_) and ('TAGSET_FLOATS' in hdr_):
        tagset = np.array(hdr_['TAGSET_FLOATS'])
        #Test to see if all the values in the tagset are zero - could be null.tag
        if sum(tagset==0.0) > 0:
            return True
        else:
            print('Tagset values are zero')
            return False
    else:
        print('TAGSET values not found in header')
        return False
    
def _is_exported_bsight(txt_fname):
    '''Evaluate text file to determine if an exported brainsight electrode
    file
    
    Returns True if brainsight electrode file
    '''
    import linecache
    bsight_line = linecache.getline(txt_fname, 3)
    if 'Brainsight' in bsight_line.split():
        return True
    else:
        return False
    
def _is_exported_tag(tag_fname):
    '''Evaluate tag to determine if in the correct fortmat'''
    with open(tag_fname) as fid:
        tmp_ = fid.readlines()
    lines=[i.replace('\n','') for i in tmp_]
    nas = [i for i in lines if 'Nasion' in i]
    lpa = [i for i in lines if 'Left Ear' in i]
    rpa = [i for i in lines if 'Right Ear' in i]
    if len(nas)>0 and len(lpa)>0 and len(rpa)>0:
        return True
    else:
        print('tag file not formatted correctly')
        return False
    
    

# =============================================================================
# TESTS for datasets    
# =============================================================================

testpath='/home/jstout/src/nih_to_mne/nih2mne/tests/calc_mne_trans_testfiles'
def test_afni_tags_present():
    neg_fname = op.join(testpath, 's1+orig.HEAD')
    assert not _afni_tags_present(neg_fname)
    pos_fname = op.join(testpath, 's2+orig.HEAD')
    assert _afni_tags_present(pos_fname)
              
def test_assess_available_localizers():
    testpath = '/home/jstout/src/nih_to_mne/nih2mne/tests/calc_mne_trans_testfiles'
    assess_available_localizers(testpath)
    
def test_is_exported_bsight():
    neg_fname = op.join(testpath,'README.txt') 
    assert not _is_exported_bsight(neg_fname)
    pos_fname = op.join(testpath, 's1.txt')
    assert _is_exported_bsight(pos_fname)

def test_is_tagfile():
    neg_fname = op.join(testpath, 's1_mod.tag')
    assert not _is_exported_tag(neg_fname)
    pos_fname = op.join(testpath, 's1.tag')
    assert _is_exported_tag(pos_fname)
    
    
    


def view_coreg(dsname=None, trans_file=None, subjects_dir=None, subject=None):
    raw = read_raw_ctf(dsname, system_clock='ignore')
    trans = mne.read_trans(trans_file)
       
    mne.viz.plot_alignment(raw.info, trans=trans, subject=subject, src=None,
                       subjects_dir=subjects_dir, dig=False,
                       surfaces=['head', 'white'], coord_frame='meg')
    _ = input('Press enter to close')
    
def main():    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject', help='''The freesurfer subject id.  
                        This folder is expected to be in the freesurfer 
                        SUBJECTS_DIR''', required=True)
    parser.add_argument('-subjects_dir', help='''Set SUBJECTS_DIR different
                        from the environment variable. If not set this defaults
                        to os.environ['SUBJECTS_DIR]''')
    parser.add_argument('-dsname', help='''CTF dataset to create the transform''',
                        required=True)
    parser.add_argument('-elec_txt', help='''Electrode text file exported from
                        brainsight''', required=False, default=None)
    parser.add_argument('-target_txt', help='''Non-default brainsight export!!  
                        Target file export from brainsight.''', 
                        required=False, default=None)
    parser.add_argument('-afni_mri', help='''Provide a BRIK or HEAD file as input.
                        Data must have the tags assigned to the header.''')
    parser.add_argument('-trans_output', help='''The output path for the mne
                        trans.fif file''')
    parser.add_argument('-anat_json', help='''Full path to the BIDS anatomy json
                        file with the NAS,RPA,LPA locations''', required=False,
                        default=None)
    parser.add_argument('-tagfile', help='''Tagfile generated by bstags.py''',
                        required=False, default=None)
    parser.add_argument('-view_coreg', help='''Display the coregistration of 
                        MEG and head surface''', action='store_true')
    
    args = parser.parse_args()
    if not args.subjects_dir:
        subjects_dir=os.environ['SUBJECTS_DIR']
    else:
        subjects_dir=args.subjects_dir
        os.environ['SUBJECTS_DIR']=args.subjects_dir
    
    subject = args.subject
    t1w_json_path = args.anat_json
    tagfile = args.tagfile
    elec_txt = args.elec_txt
    target_txt = args.target_txt
    afni_fname = args.afni_mri
        
    #Write out the fiducials
    mne_fid_name = write_mne_fiducials(subject=subject, t1w_json_path=t1w_json_path, 
                                       subjects_dir=subjects_dir, tagfile=tagfile,
                                       bsight_txt_fname=elec_txt, afni_fname=afni_fname, 
                                       bsight_target_fname=target_txt)
    
    if args.trans_output:
        output_path=args.trans_output
    else:
        output_path=None
    mne_trans_name = write_mne_trans(mne_fids_path=mne_fid_name, dsname=args.dsname,
                    output_name=output_path, subject=subject, 
                    subjects_dir=subjects_dir)#,
                    # tagfile=tagfile, bsight_txt_fname=elec_txt)
    
    if args.view_coreg:
        view_coreg(args.dsname, mne_trans_name, subjects_dir, subject=subject)
        
if __name__=='__main__':
    main()        
    
                        

