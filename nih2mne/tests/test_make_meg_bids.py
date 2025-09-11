#!/usr/bin/env python3

from ..make_meg_bids import sessdir2taskrundict
from ..make_meg_bids import _check_multiple_subjects
from ..make_meg_bids import get_subj_logger, _input_checks, process_mri_bids, process_mri_json
from ..make_meg_bids import convert_brik, make_bids  
import logging 
from ..calc_mnetrans import coords_from_afni
import glob

import nibabel as nib
import pytest 
import nih2mne
import os.path as op
from nih2mne.make_meg_bids import process_meg_bids
code_path = nih2mne.__path__[0]
data_path = op.join(code_path, 'test_data')
import numpy as np 
import json
import shutil

from ..make_meg_bids import _read_electrodes_file
from ..make_meg_bids import _extract_fidname

# Check for the test data
assert nih2mne.test_data().is_present()

global logger
logger = get_subj_logger('TEST', log_dir='/tmp', loglevel=logging.WARN)


#def test_check_multiple_subjects():
    #Make this extensible - currently limited
    #Make a temp dir and link together the files from two different folders
#    indir=''
#    _check_multiple_subjects(indir)

    
def test_sessdir2taskrundict():
    #Test for single subject folder
    input_list=\
        ['TEST_ASSR_20001010_002.ds',
         'TEST_MMFAU_20001010_009.ds',
         'TEST_M100_20001010_007.ds',
         'TEST_rest_20001010_005.ds',
         'TEST_MMFAU_20001010_003.ds',
         'TEST_rest_20001010_012.ds',
         'TEST_MMFUA_20001010_004.ds',
         'TEST_rest_20001010_011.ds',
         'TEST_M100_20001010_006.ds',
         'TEST_MMFUA_20001010_010.ds',
         'TEST_ASSR_20001010_008.ds',
         'TEST_M100_20001010_001.ds']
    out_dict = sessdir2taskrundict(input_list, subject_in='TEST')
    
    g_truth = \
    {'rest': ['TEST_rest_20001010_005.ds',
              'TEST_rest_20001010_011.ds',
              'TEST_rest_20001010_012.ds'],
     'M100': ['TEST_M100_20001010_001.ds',
              'TEST_M100_20001010_006.ds',
              'TEST_M100_20001010_007.ds'],
     'MMFAU': ['TEST_MMFAU_20001010_003.ds', 'TEST_MMFAU_20001010_009.ds'],
     'MMFUA': ['TEST_MMFUA_20001010_004.ds', 'TEST_MMFUA_20001010_010.ds'],
     'ASSR': ['TEST_ASSR_20001010_002.ds', 'TEST_ASSR_20001010_008.ds']}
    
    assert out_dict == g_truth
    
    #Test for multiple acq in single folder
    input_list2=\
        ['TEST_ASSR_20001010_002.ds',
         'TEST_MMFAU_20001010_009.ds',
         'TEST_M100_20001010_007.ds',
         'TEST_rest_20001010_005.ds',
         'TEST_MMFAU_20001010_003.ds',
         'TEST_rest_20001010_012.ds',
         'TEST_MMFUA_20001010_004.ds',
         'TEST_rest_20001010_011.ds',
         'TEST_M100_20001010_006.ds',
         'TEST_MMFUA_20001010_010.ds',
         'TEST_ASSR_20001010_008.ds',
         'TEST_M100_20001010_001.ds',
         'SUBJ2_MMFAU_20001010_009.ds',
         'SUBJ2_M100_20001010_007.ds',
         'SUBJ2_rest_20001010_005.ds',
         'SUBJ2_MMFAU_20001010_003.ds',
         'SUBJ2_rest_20001010_012.ds',
         'SUBJ2_MMFUA_20001010_004.ds',
         'SUBJ2_rest_20001010_011.ds',
         'SUBJ2_M100_20001010_006.ds']
    out_dict2 = sessdir2taskrundict(input_list2, subject_in='TEST')
    assert out_dict2 == g_truth



test_data = nih2mne.test_data()
good_updated_electrodes_file = op.join(nih2mne.__path__[0], 'tests','bsight_test_updated_good_coordsys.txt')
class test_args():
    def __init__(self, meg_input_dir=None, mri_bsight=None, bsight_elec=None):
        test_data = nih2mne.test_data()
        self.meg_input_dir = meg_input_dir
        self.mri_bsight = mri_bsight
        self.mri_bsight_elec = bsight_elec
        
def test_input_checks_valid():
    args = test_args(meg_input_dir = str(test_data.meg_data_dir),
                     mri_bsight = str(test_data.mri_nii),
                     bsight_elec = str(test_data.bsight_elec))
    assert _input_checks(args) == None

    args = test_args(meg_input_dir = str(test_data.meg_data_dir),
                     mri_bsight = str(test_data.mri_nii),
                     bsight_elec = good_updated_electrodes_file
                     )
    

bad_electrodes_file = op.join(nih2mne.__path__[0], 'tests', 'bsight_test_bad_coordsys.txt')
@pytest.mark.parametrize("meg_input_dir, mri_bsight, bsight_elec", [ 
    ('/test/ 2000000', 'test.nii', 'electrodes.txt'), 
    ('/test/2000000', 'test.ni', 'electrodes.txt'), 
    ('/test/2000000', 'test.nii', 'Subj electrodes.txt'), 
    (str(test_data.meg_data_dir), str(test_data.mri_nii), bad_electrodes_file),
    ])
def test_input_check_invalid(meg_input_dir, mri_bsight, bsight_elec):
    args = test_args(meg_input_dir, mri_bsight, bsight_elec)
    try:
        _input_checks(args)
        success = True
    except:
        success = False
    if success==True:
        raise ValueError
    
    
    


def test_process_meg_bids(tmp_path):
    out_dir = tmp_path / "bids_test_dir"
    out_dir.mkdir()
    
    #Input setup
    meg_dir = op.join(data_path, '20010101')
    subject_in = 'ABABABAB'
    bids_dir = out_dir / 'bids_dir'
    bids_id = 'S01'
    session = '1'
    
    process_meg_bids(input_path = meg_dir, 
                     subject_in = subject_in,
                     bids_dir = bids_dir,
                     bids_id = bids_id, 
                     session = session, 
                     anonymize = False,
                     ignore_eroom=True, 
                     crop_trailing_zeros= False
                     )
    
    assert op.exists(bids_dir)
    for i in ['dataset_description.json','participants.json','participants.tsv','README']:
        assert op.exists(bids_dir / i)
    assert op.exists(bids_dir / f'sub-{bids_id}')
    assert op.exists(bids_dir / f'sub-{bids_id}' / 'ses-1' /'meg')
    dset_checklist = ['sub-S01_ses-1_task-haririhammer_run-01_meg.json',
                     'sub-S01_ses-1_task-airpuff_run-01_events.tsv',
                     'sub-S01_ses-1_task-airpuff_run-01_events.json',
                     'sub-S01_ses-1_task-airpuff_run-01_meg.ds',
                     'sub-S01_ses-1_coordsystem.json',
                     'sub-S01_ses-1_task-haririhammer_run-01_meg.ds',
                     'sub-S01_ses-1_task-haririhammer_run-01_channels.tsv',
                     'sub-S01_ses-1_task-airpuff_run-01_channels.tsv',
                     'sub-S01_ses-1_task-airpuff_run-01_meg.json',
                     'sub-S01_ses-1_task-haririhammer_run-01_events.tsv',
                     'sub-S01_ses-1_task-haririhammer_run-01_events.json']
    for i in dset_checklist:
        assert op.exists(bids_dir / f'sub-{bids_id}' / 'ses-1' /'meg' / i)
        
def test_process_mri_bids(tmp_path):
    out_dir = tmp_path / "bids_test_dir"
    out_dir.mkdir()
    
    #Input setup
    mri_path  = str(test_data.mri_nii)
    bids_dir = out_dir / 'bids_dir'
    bids_id = 'S01'
    session = '1'
    
    process_mri_bids(bids_dir=bids_dir,
                     nii_mri = mri_path,
                         bids_id=bids_id, 
                         session=session)
    out_bids_mri = out_dir / 'bids_dir' / 'sub-S01' / 'ses-1' / 'anat' / 'sub-S01_ses-1_T1w.nii.gz'
    assert op.exists(out_bids_mri)
    gtruth_mri_load = nib.load(mri_path)
    bids_mri_load = nib.load(out_bids_mri)
    #Confirm nothing has been done to the transform
    assert np.allclose(bids_mri_load.affine,gtruth_mri_load.affine)

def test_process_mri_json(tmp_path):
    tmp_path_mri = tmp_path.parent / 'test_process_mri_bidscurrent' / 'bids_test_dir'
    out_dir = tmp_path / "bids_test_dir"
    # out_dir.mkdir(exist_ok=True)
    
    
    elec_fname = str(test_data.mri_data_dir / 'ABABABAB_elec.txt')
    mri_fname = str(tmp_path_mri / 'bids_dir' / 'sub-S01' / 'ses-1' / 'anat' / 'sub-S01_ses-1_T1w.nii.gz')
    process_mri_json(elec_fname=elec_fname, mri_fname=mri_fname)
    
    out_json_fname = mri_fname.replace('.nii.gz','.json')
    with open(out_json_fname) as f:
        _json_vals = json.load(f)
    assert 'AnatomicalLandmarkCoordinates' in  _json_vals.keys()
    fids = _json_vals['AnatomicalLandmarkCoordinates']
    assert 'NAS' in fids.keys() and 'LPA' in fids.keys() and 'RPA' in fids.keys()
    assert np.allclose(fids['NAS'], [111.79899853515624,216.0946962158203,125.91931025390625])
    assert np.allclose(fids['LPA'], [36.48359853515625, 138.14889621582032, 92.27751025390626])
    assert np.allclose(fids['RPA'], [175.88069853515626, 122.28609621582031, 92.83361025390624])

def test_extract_fidname():
    from nih2mne.make_meg_bids import _extract_fidname
    elec_names = ['left ear', 'right ear' , 'nasion']
    assert _extract_fidname('lpa', elec_names)==0
    assert _extract_fidname('rpa', elec_names)==1
    assert _extract_fidname('nas', elec_names)==2
    
    elec_names = ['lpa','rpa','nas']
    assert _extract_fidname('lpa', elec_names)==0
    assert _extract_fidname('rpa', elec_names)==1
    assert _extract_fidname('nas', elec_names)==2
    
    elec_names = ['LPA','Right Ear','NAsion']
    assert _extract_fidname('lpa', elec_names)==0
    assert _extract_fidname('rpa', elec_names)==1
    assert _extract_fidname('nas', elec_names)==2
    
def test_read_electrodes_file():
    dirname = op.dirname(__file__)
    elec_fname = op.join(dirname, 'bsight_test_updated_good_coordsys.txt')
    locs_ras = _read_electrodes_file(elec_fname)
    assert np.all(locs_ras['Nasion']==np.array([100.7006, 34.9441, -117.9930]))
    assert np.all(locs_ras['Left Ear']==np.array([173.2163, 110.0248, -146.5442]))
    assert np.all(locs_ras['Right Ear']==np.array([35.4196, 113.4793, -152.5658]))
    
    elec_fname = op.join(dirname, 'Exported_Electrodes.txt')
    locs_ras = _read_electrodes_file(elec_fname)   
    assert np.all(locs_ras['Nasion']==np.array([-0.6248, 108.7596,   2.0581]))
    assert np.all(locs_ras['Left Ear']==np.array([-81.9025,  11.953 , -26.8693]))
    assert np.all(locs_ras['Right Ear']==np.array([79.3304,   6.8787, -28.0673]))
    
    #Use the downloaded CTF example data
    elec_fname = str(test_data.mri_data_dir / 'ABABABAB_elec.txt')
    locs_ras = _read_electrodes_file(elec_fname)
    assert np.all(locs_ras['Nasion']==np.array([  8.838 , 124.6017, -11.2287]))
    assert np.all(locs_ras['Left Ear']==np.array([ -66.4774,  46.6559, -44.8705 ]))
    assert np.all(locs_ras['Right Ear']==np.array([  72.9197,  30.7931, -44.3144]))
    
# def test_process_mri_json2(tmp_path):
#     tmp_path_mri = tmp_path.parent / 'test_process_mri_bidscurrent' / 'bids_test_dir'
#     out_dir = tmp_path / "bids_test_dir"
    
#     elec_fname = str(test_data.mri_data_dir / 'ABABABAB_elec.txt')
#     mri_fname = str(tmp_path_mri / 'bids_dir' / 'sub-S01' / 'ses-1' / 'anat' / 'sub-S01_ses-1_T1w.nii.gz')
#     process_mri_json(elec_fname=elec_fname, mri_fname=mri_fname)
    
#     out_json_fname = mri_fname.replace('.nii.gz','.json')
#     with open(out_json_fname) as f:
#         _json_vals = json.load(f)
#     assert 'AnatomicalLandmarkCoordinates' in  _json_vals.keys()
#     fids = _json_vals['AnatomicalLandmarkCoordinates']
#     assert 'NAS' in fids.keys() and 'LPA' in fids.keys() and 'RPA' in fids.keys()
#     assert np.allclose(fids['NAS'], [111.79899853515624,216.0946962158203,125.91931025390625])
#     assert np.allclose(fids['LPA'], [36.48359853515625, 138.14889621582032, 92.27751025390626])
#     assert np.allclose(fids['RPA'], [175.88069853515626, 122.28609621582031, 92.83361025390624])
    


def test_process_mri_json_afni_input(tmp_path):
    head_fname = str(test_data.mri_head)
    #Using the nifti - which is the same in the data matrix/header
    mri_fname = str(test_data.mri_nii)  
    test_mri_fname = op.join(tmp_path, op.basename(mri_fname))
    shutil.copy(mri_fname, test_mri_fname)
    
    coords_lps = coords_from_afni(head_fname)
    coords_ras = {}
    for key in coords_lps.keys():
        tmp = np.array(coords_lps[key])
        tmp[0:2]*=-1
        coords_ras[key]=tmp
    process_mri_json(mri_fname = test_mri_fname, ras_coords = coords_ras)

    out_json_fname = test_mri_fname.replace('.nii.gz','.json')
    with open(out_json_fname) as f:
        _json_vals = json.load(f)
    assert 'AnatomicalLandmarkCoordinates' in  _json_vals.keys()
    fids = _json_vals['AnatomicalLandmarkCoordinates']
    assert 'NAS' in fids.keys() and 'LPA' in fids.keys() and 'RPA' in fids.keys()
    #NOTE: The afni localizers are slightly off from the NII/Brainsight due to 
    #not having subvoxel precision on the coil placement when creating the tagset
    assert np.allclose(fids['NAS'], [111.99999953515625, 215.99999621582032, 126.00002025390626])
    assert np.allclose(fids['LPA'], [35.99999853515625, 137.99999621582032, 92.00002025390626])
    assert np.allclose(fids['RPA'], [175.99999853515624, 121.99999621582032, 93.00002025390626])
    
    
def test_convert_afni(tmp_path):
    import nibabel as nib
    brik_fname = str(test_data.mri_brik)
    nii_fname = str(test_data.mri_nii)
    g_truth_brik = nib.load(brik_fname) 
    g_truth_nii = nib.load(nii_fname)
    
    
    out_nii_fname = convert_brik(brik_fname, outdir=tmp_path)
    out_nii_dat = nib.load(out_nii_fname)
    
    assert np.allclose(g_truth_brik.affine, out_nii_dat.affine)
    assert np.allclose(g_truth_brik.get_fdata().squeeze(), out_nii_dat.get_fdata().squeeze())
    
    assert np.allclose(g_truth_nii.affine, out_nii_dat.affine)
    assert np.allclose(g_truth_nii.get_fdata().squeeze(), out_nii_dat.get_fdata().squeeze())
    

class make_args():
    def __init__(self, bids_dir, meg_input_dir, subjid_input, bids_id, mri_bsight, bsight_elec, mri_brik):
        self.bids_dir = str(bids_dir)
        self.meg_input_dir = str(meg_input_dir)
        self.subjid_input = subjid_input 
        self.bids_id = bids_id 
        self.bids_session = '1'
        
        self.mri_bsight = str(mri_bsight) if mri_bsight != None else None
        self.mri_bsight_elec = str(bsight_elec) if bsight_elec != None else None
        self.mri_brik = str(mri_brik) if mri_brik != None else None
        self.anonymize = False
        self.ignore_mri_checks = False
        self.autocrop_zeros = False
        self.ignore_eroom = True
        self.eventID_csv = None
        self.freesurfer = False
        self.mri_prep_s = False
        self.mri_prep_v = False


@pytest.mark.parametrize("bids_dir, meg_input_dir, subjid_input, bids_id, mri_bsight, bsight_elec, mri_brik", [ 
    ('bsight_test', test_data.meg_data_dir, 'ABABABAB', 'BSIGHT1', test_data.mri_nii, test_data.bsight_elec, None), 
    ('afni_test', test_data.meg_data_dir, 'ABABABAB', 'AFNI1', None, None, test_data.mri_brik), 
    ])
def test_make_meg_bids_fullpipeline(bids_dir,meg_input_dir, subjid_input, bids_id, mri_bsight, bsight_elec, mri_brik, tmpdir):
    out_dir = op.join(tmpdir, bids_dir)
    cmd = f'make_meg_bids.py -bids_dir {out_dir} -subjid_input ABABABAB -meg_input_dir {meg_input_dir} -bids_id {bids_id} -mri_bsight {mri_bsight} -mri_bsight_elec {bsight_elec} -mri_brik {mri_brik} -ignore_eroom'
    args = make_args(out_dir, meg_input_dir, subjid_input, bids_id, mri_bsight, bsight_elec, mri_brik)
    make_bids(args)
    
    anats = glob.glob(op.join(out_dir, f'sub-{bids_id}','ses-1','anat', '*'))
    assert len([i for i in anats if i.endswith('.nii.gz')])==1
    assert len([i for i in anats if i.endswith('.json')])==1
    
    dsets = glob.glob(op.join(out_dir, f'sub-{bids_id}', 'ses-1','meg','*.ds'))
    for dset in dsets:
        print(dset)
        assert op.basename(dset).split('_task-')[-1].split('_')[0] in ['airpuff','haririhammer']
                      

                
# args = make_args('bsight_test', test_data.meg_data_dir, None, 'BSIGHT1', test_data.mri_nii, test_data.bsight_elec, None)
    
    
    
    
                         
        
    
    
    
    
    
    
