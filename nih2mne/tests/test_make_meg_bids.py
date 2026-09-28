#!/usr/bin/env python3

from ..make_meg_bids import sessdir2taskrundict
from ..make_meg_bids import _check_multiple_subjects
from ..make_meg_bids import (
    _format_bids_entity,
    _get_bids_zfill,
    _get_conversion_dict,
)
from ..make_meg_bids import get_subj_logger, _input_checks, process_mri_bids, process_mri_json
from ..make_meg_bids import convert_brik, make_bids, _gen_taskrundict  
import logging 
from ..calc_mnetrans import coords_from_afni
import glob
from mne_bids import BIDSPath

import nibabel as nib
import pandas as pd
import pytest 
import nih2mne
import os.path as op
from nih2mne.make_meg_bids import process_meg_bids
code_path = nih2mne.__path__[0]
data_path = op.join(code_path, 'test_data')
import numpy as np 
import json
import shutil
from types import SimpleNamespace

import nih2mne.make_meg_bids as make_meg_bids_module

from ..make_meg_bids import _read_electrodes_file
from ..make_meg_bids import _extract_fidname

# Check for the test data
assert nih2mne.test_data().is_present()

global logger
logger = get_subj_logger('TEST', log_dir='/tmp', loglevel=logging.WARN)


def test_get_bids_zfill_reads_configured_widths(monkeypatch):
    config = SimpleNamespace(DEFAULTS={
        'BIDS_gen': {'zfill_run': '4', 'zfill_ses': 3},
    })
    monkeypatch.setattr(
        make_meg_bids_module.importlib,
        'import_module',
        lambda _module_name: config,
    )

    assert _get_bids_zfill() == {'zfill_run': 4, 'zfill_ses': 3}


@pytest.mark.parametrize(
    'bids_defaults',
    [
        {},
        {'zfill_run': 0, 'zfill_ses': -1},
        {'zfill_run': True, 'zfill_ses': 'invalid'},
    ],
)
def test_get_bids_zfill_falls_back_for_missing_or_invalid_values(
        monkeypatch, bids_defaults):
    config = SimpleNamespace(DEFAULTS={'BIDS_gen': bids_defaults})
    monkeypatch.setattr(
        make_meg_bids_module.importlib,
        'import_module',
        lambda _module_name: config,
    )

    assert _get_bids_zfill() == {'zfill_run': 2, 'zfill_ses': 2}


def test_get_bids_zfill_falls_back_independently(monkeypatch):
    config = SimpleNamespace(DEFAULTS={
        'BIDS_gen': {'zfill_run': 4},
    })
    monkeypatch.setattr(
        make_meg_bids_module.importlib,
        'import_module',
        lambda _module_name: config,
    )

    assert _get_bids_zfill() == {'zfill_run': 4, 'zfill_ses': 2}


def test_get_bids_zfill_falls_back_when_config_cannot_be_read(monkeypatch):
    def fail_import(_module_name):
        raise OSError('defaults.yml is unreadable')

    monkeypatch.setattr(
        make_meg_bids_module.importlib,
        'import_module',
        fail_import,
    )

    assert _get_bids_zfill() == {'zfill_run': 2, 'zfill_ses': 2}


@pytest.mark.parametrize(
    ('value', 'width', 'expected'),
    [
        (1, 3, '001'),
        ('001', 2, '01'),
        (1234, 2, '1234'),
        ('Pre', 5, 'Pre'),
        (None, 2, None),
    ],
)
def test_format_bids_entity(value, width, expected):
    assert _format_bids_entity(value, width) == expected


def _movement_record():
    return {
        'locations': {
            'hz.ds': {
                'NAS': (1.0, 2.0, 3.0),
                'LPA': (4.0, 5.0, 6.0),
                'RPA': (7.0, 8.0, 9.0),
            },
            'hz2.ds': {
                'NAS': (1.1, 2.1, 3.1),
                'LPA': (4.1, 5.1, 6.1),
                'RPA': (7.1, 8.1, 9.1),
            },
        },
        'movement': {
            'N': 0.1234,
            'L': 0.2,
            'R': 0.3456,
            'Max': 0.3456,
        },
    }


def test_format_movement_includes_commented_locs_and_movement():
    movement = _movement_record()

    assert make_meg_bids_module._format_movement(movement) == (
        '#hz.ds\n'
        '#NAS  1.0000,2.0000,3.0000\n'
        '#LPA  4.0000,5.0000,6.0000\n'
        '#RPA  7.0000,8.0000,9.0000\n'
        '\n'
        '#hz2.ds\n'
        '#NAS  1.1000,2.1000,3.1000\n'
        '#LPA  4.1000,5.1000,6.1000\n'
        '#RPA  7.1000,8.1000,9.1000\n'
        '\n'
        '#Movement\n'
        'NAS: 0.12 cm\n'
        'LPA: 0.20 cm\n'
        'RPA: 0.35 cm\n'
        'Max: 0.35 cm\n'
    )


def test_write_movement_file_writes_and_removes_stale_file(tmp_path):
    bids_path = BIDSPath(
        subject='TEST', session='01', task='rest', run='01',
        datatype='meg', suffix='meg', extension='.ds', root=tmp_path,
    )
    bids_path.fpath.mkdir(parents=True)
    movement = _movement_record()

    movement_path = make_meg_bids_module._write_movement_file(
        bids_path,
        movement,
    )

    assert movement_path == bids_path.fpath / 'movement.txt'
    assert movement_path.read_text(encoding='utf-8') == (
        '#hz.ds\n'
        '#NAS  1.0000,2.0000,3.0000\n'
        '#LPA  4.0000,5.0000,6.0000\n'
        '#RPA  7.0000,8.0000,9.0000\n'
        '\n'
        '#hz2.ds\n'
        '#NAS  1.1000,2.1000,3.1000\n'
        '#LPA  4.1000,5.1000,6.1000\n'
        '#RPA  7.1000,8.1000,9.1000\n'
        '\n'
        '#Movement\n'
        'NAS: 0.12 cm\n'
        'LPA: 0.20 cm\n'
        'RPA: 0.35 cm\n'
        'Max: 0.35 cm\n'
    )

    make_meg_bids_module._write_movement_file(bids_path, None)
    assert not movement_path.exists()


def test_calculate_movement_returns_none_without_ctf_tools(
        tmp_path, monkeypatch):
    monkeypatch.setattr(make_meg_bids_module.shutil, 'which', lambda _name: None)

    assert make_meg_bids_module._calculate_movement(tmp_path) is None


def test_calculate_movement_returns_none_without_localizers(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        make_meg_bids_module.shutil,
        'which',
        lambda _name: '/opt/ctf/bin/calcHeadPos',
    )

    assert make_meg_bids_module._calculate_movement(tmp_path) is None


def test_get_movement_rows_selects_last_hz_and_first_hz2_trials():
    dframe = pd.DataFrame([
        {'hz_val': 'hz', 'trial': '1', 'value': 'initial'},
        {'hz_val': 'hz', 'trial': '3', 'value': 'last'},
        {'hz_val': 'hz2', 'trial': '1', 'value': 'post'},
    ])

    initial_row, final_row = make_meg_bids_module.get_movement_rows(dframe)

    assert initial_row['value'] == 'last'
    assert final_row['value'] == 'post'


def test_calculate_movement_uses_original_localizers(tmp_path, monkeypatch):
    (tmp_path / 'hz.ds').mkdir()
    (tmp_path / 'hz2.ds').mkdir()
    dframe = object()
    monkeypatch.setattr(
        make_meg_bids_module.shutil,
        'which',
        lambda _name: '/opt/ctf/bin/calcHeadPos',
    )
    monkeypatch.setattr(
        make_meg_bids_module,
        'get_localizer_dframe',
        lambda fname: dframe if fname == tmp_path else None,
    )

    def compute(input_dframe, verbose):
        assert input_dframe is dframe
        assert verbose is False
        return {'N': 0.1, 'L': 0.2, 'R': 0.3, 'Max': 0.3}

    monkeypatch.setattr(make_meg_bids_module, 'compute_movement', compute)
    initial_row = {
        'nas_x': 1, 'nas_y': 2, 'nas_z': 3,
        'lpa_x': 4, 'lpa_y': 5, 'lpa_z': 6,
        'rpa_x': 7, 'rpa_y': 8, 'rpa_z': 9,
    }
    final_row = {
        'nas_x': 1.1, 'nas_y': 2.1, 'nas_z': 3.1,
        'lpa_x': 4.1, 'lpa_y': 5.1, 'lpa_z': 6.1,
        'rpa_x': 7.1, 'rpa_y': 8.1, 'rpa_z': 9.1,
    }
    monkeypatch.setattr(
        make_meg_bids_module,
        'get_movement_rows',
        lambda input_dframe: (
            (initial_row, final_row) if input_dframe is dframe else (None, None)
        ),
    )

    assert make_meg_bids_module._calculate_movement(tmp_path) == {
        'locations': {
            'hz.ds': {
                'NAS': (1.0, 2.0, 3.0),
                'LPA': (4.0, 5.0, 6.0),
                'RPA': (7.0, 8.0, 9.0),
            },
            'hz2.ds': {
                'NAS': (1.1, 2.1, 3.1),
                'LPA': (4.1, 5.1, 6.1),
                'RPA': (7.1, 8.1, 9.1),
            },
        },
        'movement': {'N': 0.1, 'L': 0.2, 'R': 0.3, 'Max': 0.3},
    }


def test_calculate_movement_returns_none_for_invalid_result(
        tmp_path, monkeypatch):
    (tmp_path / 'hz.ds').mkdir()
    (tmp_path / 'hz2.ds').mkdir()
    monkeypatch.setattr(
        make_meg_bids_module.shutil,
        'which',
        lambda _name: '/opt/ctf/bin/calcHeadPos',
    )
    monkeypatch.setattr(
        make_meg_bids_module,
        'get_localizer_dframe',
        lambda _fname: object(),
    )
    monkeypatch.setattr(
        make_meg_bids_module,
        'compute_movement',
        lambda _dframe, verbose: {
            'N': 0.1, 'L': 0.2, 'R': 0.3, 'Max': None,
        },
    )

    assert make_meg_bids_module._calculate_movement(tmp_path) is None


def test_calculate_movement_returns_none_on_calculation_error(
        tmp_path, monkeypatch):
    (tmp_path / 'hz.ds').mkdir()
    (tmp_path / 'hz2.ds').mkdir()
    monkeypatch.setattr(
        make_meg_bids_module.shutil,
        'which',
        lambda _name: '/opt/ctf/bin/calcHeadPos',
    )

    def fail_calculation(_fname):
        raise RuntimeError('test calculation error')

    monkeypatch.setattr(
        make_meg_bids_module,
        'get_localizer_dframe',
        fail_calculation,
    )

    assert make_meg_bids_module._calculate_movement(tmp_path) is None


def test_conversion_dict_uses_configured_padding(tmp_path, monkeypatch):
    meg_fname = tmp_path / 'TEST_rest_20001010_001.ds'
    monkeypatch.setattr(
        make_meg_bids_module,
        '_get_bids_zfill',
        lambda: {'zfill_run': 4, 'zfill_ses': 3},
    )

    conversion = _get_conversion_dict(
        bids_id='S01',
        bids_dir=tmp_path / 'bids',
        session='03',
        meg_dataset_list=[str(meg_fname)],
    )

    assert str(conversion[str(meg_fname)]).endswith(
        'sub-S01/ses-003/meg/'
        'sub-S01_ses-003_task-rest_run-0001_meg.ds'
    )


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
    
    dset_dict = sessdir2taskrundict(session_dir=meg_dir, subject_in=subject_in)
    dset_dict = {i:[op.join(meg_dir, j[0])] for i,j in dset_dict.items()}
    
    
    process_meg_bids(dset_dict = dset_dict,
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
    assert op.exists(bids_dir / f'sub-{bids_id}' / 'ses-01' /'meg')
    dset_checklist = ['sub-S01_ses-01_task-haririhammer_run-01_meg.json',
                     'sub-S01_ses-01_task-airpuff_run-01_events.tsv',
                     'sub-S01_ses-01_task-airpuff_run-01_events.json',
                     'sub-S01_ses-01_task-airpuff_run-01_meg.ds',
                     'sub-S01_ses-01_coordsystem.json',
                     'sub-S01_ses-01_task-haririhammer_run-01_meg.ds',
                     'sub-S01_ses-01_task-haririhammer_run-01_channels.tsv',
                     'sub-S01_ses-01_task-airpuff_run-01_channels.tsv',
                     'sub-S01_ses-01_task-airpuff_run-01_meg.json',
                     'sub-S01_ses-01_task-haririhammer_run-01_events.tsv',
                     'sub-S01_ses-01_task-haririhammer_run-01_events.json']
    for i in dset_checklist:
        print(i)
        assert op.exists(bids_dir / f'sub-{bids_id}' / 'ses-01' /'meg' / i)
        
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
    out_bids_mri = out_dir / 'bids_dir' / 'sub-S01' / 'ses-01' / 'anat' / 'sub-S01_ses-01_T1w.nii.gz'
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
    mri_fname = str(tmp_path_mri / 'bids_dir' / 'sub-S01' / 'ses-01' / 'anat' / 'sub-S01_ses-01_T1w.nii.gz')
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
    args = make_args(out_dir, str(meg_input_dir), subjid_input, bids_id, mri_bsight, bsight_elec, mri_brik)
    make_bids(args)
    
    anats = glob.glob(op.join(out_dir, f'sub-{bids_id}','ses-01','anat', '*'))
    assert len([i for i in anats if i.endswith('.nii.gz')])==1
    assert len([i for i in anats if i.endswith('.json')])==1
    
    dsets = glob.glob(op.join(out_dir, f'sub-{bids_id}', 'ses-01','meg','*.ds'))
    for dset in dsets:
        print(dset)
        assert op.basename(dset).split('_task-')[-1].split('_')[0] in ['airpuff','haririhammer']
                      


def test_gen_taskrundict():
    tmp_meglist =glob.glob(op.join(nih2mne.__path__[0], 'test_data', '20010101', '*.ds'))
    dset_dict = _gen_taskrundict(meg_list = tmp_meglist)
    assert 'haririhammer' in dset_dict
    assert 'airpuff' in dset_dict
    
    _tmp_topdir = '/home/User/test/datasets'
    test_files =     ['TEST_ASSR_20001010_002.ds',
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
    test_files = [op.join(_tmp_topdir, i) for i in test_files]
    dset_dict = _gen_taskrundict(meg_list = test_files)
    assert len(dset_dict['MMFAU'])==2
    assert len(dset_dict['M100'])==3
    
#%% New interface testing
from nih2mne.make_meg_bids import _proc_meg_bids, _proc_mri_bids

def test_proc_meg_bids(tmpdir):
    out_bids_path = op.join(tmpdir, 'BIDS')
    _bids_path = BIDSPath(subject='TEST', session='1', task='airpuff', 
                          datatype='meg', suffix='meg', run='01',
                          root = out_bids_path, extension='.ds'
                          )
    
    meg_fname = str(test_data.meg_airpuff_fname)
    _proc_meg_bids(meg_fname=meg_fname, bids_path=_bids_path,
                        anonymize=False, tmpdir=None, ignore_eroom=True, 
                        crop_trailing_zeros=False, 
                       )
    assert _bids_path.fpath.exists()


def test_proc_meg_bids_preserves_movement_through_anonymization(
        tmp_path, monkeypatch):
    source_path = tmp_path / 'input.ds'
    source_path.mkdir()
    bids_path = BIDSPath(
        subject='TEST', session='01', task='rest', run='01',
        datatype='meg', suffix='meg', extension='.ds',
        root=tmp_path / 'BIDS',
    )
    movement = _movement_record()
    call_order = []

    def calculate_movement(meg_fname):
        assert meg_fname == str(source_path)
        call_order.append('movement')
        return movement

    def anonymize_meg(meg_fname, tmpdir):
        assert meg_fname == str(source_path)
        call_order.append('anonymize')
        return meg_fname

    class DummyRaw:
        info = {}

    def write_raw_bids(_raw, output_path, **_kwargs):
        call_order.append('write_bids')
        output_path.fpath.mkdir(parents=True)

    monkeypatch.setattr(
        make_meg_bids_module, '_calculate_movement', calculate_movement,
    )
    monkeypatch.setattr(make_meg_bids_module, '_clear_ClassFile', lambda _f: None)
    monkeypatch.setattr(make_meg_bids_module, '_check_markerfile', lambda _f: None)
    monkeypatch.setattr(make_meg_bids_module, 'anonymize_meg', anonymize_meg)
    monkeypatch.setattr(
        make_meg_bids_module, 'anonymize_finalize', lambda _f: None,
    )
    monkeypatch.setattr(
        make_meg_bids_module.mne.io,
        'read_raw_ctf',
        lambda *_args, **_kwargs: DummyRaw(),
    )
    monkeypatch.setattr(
        make_meg_bids_module, 'write_raw_bids', write_raw_bids,
    )

    make_meg_bids_module._proc_meg_bids(
        meg_fname=str(source_path),
        bids_path=bids_path,
        anonymize=True,
        tmpdir=tmp_path / 'anonymized',
        ignore_eroom=True,
    )

    assert call_order == ['movement', 'anonymize', 'write_bids']
    assert (bids_path.fpath / 'movement.txt').read_text(
        encoding='utf-8',
    ) == (
        '#hz.ds\n'
        '#NAS  1.0000,2.0000,3.0000\n'
        '#LPA  4.0000,5.0000,6.0000\n'
        '#RPA  7.0000,8.0000,9.0000\n'
        '\n'
        '#hz2.ds\n'
        '#NAS  1.1000,2.1000,3.1000\n'
        '#LPA  4.1000,5.1000,6.1000\n'
        '#RPA  7.1000,8.1000,9.1000\n'
        '\n'
        '#Movement\n'
        'NAS: 0.12 cm\n'
        'LPA: 0.20 cm\n'
        'RPA: 0.35 cm\n'
        'Max: 0.35 cm\n'
    )


def test_proc_mri_bids(tmpdir):
    out_bids_path = op.join(tmpdir, 'BIDS')
    _bids_path = BIDSPath(subject='TEST', session='1',  
                          datatype='anat', extension='.nii.gz',
                          root = out_bids_path, suffix='T1w') #, extension='.ds'
    _bids_path.fpath
    
    _proc_mri_bids(t1_bids_path= _bids_path, temp_dir=tmpdir, 
                   anonymize=False, 
                   mri_bsight=str(test_data.mri_nii), 
                   mri_bsight_elec=str(test_data.bsight_elec), 
                   mri_brik=False, input_id='test')
    
    assert op.exists(_bids_path.fpath)
    json_fname = str(_bids_path.fpath).replace('.nii.gz','.json')
    assert op.exists(json_fname)


# def test_textfill(tmpdir):
#     'Confirm that anat textline gets filled out'
#     from nih2mne.GUI.templates.bids_creator_gui_control_functions import BIDS_MainWindow
#     dsets = ['/fast2/20250205_hv2set/JACTZJMA_rest_20250205_004.ds', '/fast2/20250205_hv2set/JACTZJMA_rest_20250205_007.ds','/fast2/20250205_hv2set/JACTZJMA_MID_20250205_008.ds','/fast2/20250205_hv2set/JACTZJMA_movie_20250205_005.ds']
    
#     bmw = BIDS_MainWindow(meg_dsets=dsets)
#     bmw.ui.list_fname_conversion.addItems(dsets)
#     bmw._make_task_dict()
#     bmw.ui.list_fname_conversion.count()





## << -- testing
# from nih2mne.make_meg_bids import (make_bids, process_meg_bids, _gen_taskrundict, 
#                                    sessdir2taskrundict)
# from nih2mne.GUI.templates.bids_creator_gui_control_functions import Args #, test_Args
# def test_Args():
#     opts = {'anonymize': True, 'meghash': 'None', 'bids_id': 'S01', 
#             'bids_dir': '/tmp/BIDS', 'bids_session': '1', 
#             'meg_dataset_list': ['/tmp/20251203/DUJGWRKZ_EmptyRoom6m_20251203_001.ds',
#                                  '/tmp/20251203/DUJGWRKZ_flanker_20251203_007.ds'], 
#             'mri_none': False, 
#             'mri_bsight': '/tmp/Uploaded_cohort3/DUJGWRKZ/DUJGWRKZ.nii', 
#             'mri_elec': '/tmp/Uploaded_cohort3/DUJGWRKZ/Exported_Electrodes.txt', 
#             'mri_brik': False, 'crop_zeros': True, 'include_empty_room': False, 
#             'subjid_input':'TEST'}
     
#             #'subjid_input' : False}
#     return Args(opts)
    
# args = test_Args()
# make_bids(args)
    
    
    
    
                         
        
    
    
    
    
    
    
