#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os.path as op

import dill
import numpy as np
import pytest
import yaml

from nih2mne.dataQA.bids_project_interface import (
    _subject_bids_info,
    load_megqa_file,
    reinitialize_megqa_files,
    reinitialize_megqa_pickles,
    subject_bids_info,
)


def test_update_bids_root_after_project_move(tmp_path):
    old_root = tmp_path / 'old_project'
    new_root = tmp_path / 'new_project'
    anat_dir = old_root / 'sub-01' / 'anat'
    meg_dir = old_root / 'sub-01' / 'meg'
    anat_dir.mkdir(parents=True)
    meg_dir.mkdir(parents=True)

    mri = anat_dir / 'sub-01_T1w.nii'
    mri.touch()
    mri.with_suffix('.json').write_text(json.dumps({
        'AnatomicalLandmarkCoordinates': {
            'LPA': [0, 0, 0],
            'NAS': [0, 0, 0],
            'RPA': [0, 0, 0],
        },
    }))
    (meg_dir / 'sub-01_task-rest_meg.ds').mkdir()

    bids_info = _subject_bids_info('01', bids_root=old_root)
    bids_info.current_meg_dset = object()
    bids_info.save(overwrite=True)
    old_root.rename(new_root)

    bids_info = subject_bids_info('01', bids_root=new_root)

    assert bids_info.bids_root == str(new_root)
    assert bids_info.deriv_root == op.join(
        new_root, 'derivatives', bids_info.deriv_project)
    assert bids_info.qa_default_fname == op.join(
        new_root, 'derivatives', 'megQA', 'sub-01.yml')
    assert bids_info.subjects_dir == op.join(
        new_root, 'derivatives', 'freesurfer', 'subjects')
    assert bids_info.mri == op.join(
        new_root, 'sub-01', 'anat', 'sub-01_T1w.nii')
    assert bids_info.mri_json == op.join(
        new_root, 'sub-01', 'anat', 'sub-01_T1w.json')
    assert bids_info.mri_json_qa == 'GOOD'
    assert bids_info.meg_list[0].rel_path == op.join(
        new_root, 'sub-01', 'meg', 'sub-01_task-rest_meg.ds')
    assert not hasattr(bids_info, 'current_meg_dset')


def test_reinitialize_megqa_files_rescans_bids_tree(tmp_path):
    bids_root = tmp_path / 'bids'
    anat_dir = bids_root / 'sub-01' / 'anat'
    meg_dir = bids_root / 'sub-01' / 'meg'
    anat_dir.mkdir(parents=True)
    meg_dir.mkdir(parents=True)

    mri = anat_dir / 'sub-01_T1w.nii'
    mri.touch()
    mri.with_suffix('.json').write_text(json.dumps({
        'AnatomicalLandmarkCoordinates': {
            'LPA': [0, 0, 0],
            'NAS': [0, 0, 0],
            'RPA': [0, 0, 0],
        },
    }))
    (meg_dir / 'sub-01_task-rest_meg.ds').mkdir()

    stale_info = _subject_bids_info('01', bids_root=bids_root)
    stale_info.save(overwrite=True)
    (meg_dir / 'sub-01_task-noise_meg.ds').mkdir()

    rebuilt = reinitialize_megqa_files(bids_root)

    yaml_path = bids_root / 'derivatives' / 'megQA' / 'sub-01.yml'
    saved_info = load_megqa_file('sub-01', bids_root)

    assert list(rebuilt) == ['sub-01']
    assert saved_info.bids_root == str(bids_root)
    assert saved_info.qa_default_fname == str(yaml_path)
    assert sorted(dset.fname for dset in saved_info.meg_list) == [
        'sub-01_task-noise_meg.ds',
        'sub-01_task-rest_meg.ds',
    ]


def test_yaml_round_trip_preserves_editable_and_extension_fields(tmp_path):
    bids_root = tmp_path / 'bids'
    meg_dir = bids_root / 'sub-01' / 'meg'
    meg_dir.mkdir(parents=True)
    (meg_dir / 'sub-01_task-rest_meg.ds').mkdir()

    bids_info = _subject_bids_info('01', bids_root=bids_root)
    bids_info.review_note = 'check movement'
    bids_info.current_meg_dset = object()
    bids_info.meg_list[0].chan_power = np.array([1.5, 2.5])
    bids_info.meg_list[0].BADS = {
        'JUMPS': {'CHANS': {'MZC01': np.int64(12)},
                  'TSTEP': np.array([12, 24])},
    }
    bids_info.meg_list[0].psd = object()
    bids_info.save(overwrite=True)

    yaml_path = bids_root / 'derivatives' / 'megQA' / 'sub-01.yml'
    data = yaml.safe_load(yaml_path.read_text())
    assert data['review_note'] == 'check movement'
    assert data['meg_list'][0]['chan_power'] == [1.5, 2.5]
    assert 'current_meg_dset' not in data
    assert 'psd' not in data['meg_list'][0]

    data['review_note'] = 'approved by hand'
    data['future_field'] = {'enabled': True}
    del data['deriv_project']
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    loaded = subject_bids_info('01', bids_root=bids_root)

    assert loaded.review_note == 'approved by hand'
    assert loaded.future_field == {'enabled': True}
    assert np.array_equal(loaded.meg_list[0].chan_power, [1.5, 2.5])
    assert np.array_equal(
        loaded.meg_list[0].BADS['JUMPS']['TSTEP'], [12, 24])
    assert not hasattr(loaded, 'current_meg_dset')
    assert not hasattr(loaded.meg_list[0], 'psd')

    loaded.save(overwrite=True)
    resaved = yaml.safe_load(yaml_path.read_text())
    assert resaved['future_field'] == {'enabled': True}
    assert resaved['deriv_project'] == 'nihmeg'


def test_hand_edited_mri_selection_refreshes_derived_fields(tmp_path):
    bids_root = tmp_path / 'bids'
    anat_dir = bids_root / 'sub-01' / 'anat'
    anat_dir.mkdir(parents=True)
    for name in ('sub-01_acq-a_T1w.nii', 'sub-01_acq-b_T1w.nii'):
        mri = anat_dir / name
        mri.touch()
        mri.with_suffix('.json').write_text(json.dumps({
            'AnatomicalLandmarkCoordinates': {
                'LPA': [0, 0, 0],
                'NAS': [0, 0, 0],
                'RPA': [0, 0, 0],
            },
        }))

    subject_bids_info('01', bids_root=bids_root)
    yaml_path = bids_root / 'derivatives' / 'megQA' / 'sub-01.yml'
    data = yaml.safe_load(yaml_path.read_text())
    selected = anat_dir / 'sub-01_acq-b_T1w.nii'
    data['mri'] = str(selected)
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))

    loaded = subject_bids_info('01', bids_root=bids_root)

    assert loaded.mri == str(selected)
    assert loaded.mri_json == str(selected.with_suffix('.json'))
    assert loaded.mri_json_qa == 'GOOD'


def test_legacy_pickle_is_migrated_and_retained(tmp_path):
    bids_root = tmp_path / 'bids'
    meg_dir = bids_root / 'sub-01' / 'meg'
    meg_dir.mkdir(parents=True)
    (meg_dir / 'sub-01_task-rest_meg.ds').mkdir()

    bids_info = _subject_bids_info('01', bids_root=bids_root)
    bids_info.mri = 'legacy-selection.nii'
    pickle_path = bids_root / 'derivatives' / 'megQA' / 'sub-01.pkl'
    pickle_path.parent.mkdir(parents=True)
    with pickle_path.open('wb') as fid:
        dill.dump(bids_info, fid)

    loaded = subject_bids_info('01', bids_root=bids_root)
    yaml_path = pickle_path.with_suffix('.yml')

    assert loaded.mri == 'legacy-selection.nii'
    assert yaml_path.is_file()
    assert pickle_path.is_file()
    assert yaml.safe_load(yaml_path.read_text())['mri'] == 'legacy-selection.nii'

    yaml_data = yaml.safe_load(yaml_path.read_text())
    yaml_data['mri'] = 'yaml-selection.nii'
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False))
    assert subject_bids_info('01', bids_root=bids_root).mri == 'yaml-selection.nii'


def test_pickle_named_reinitialize_alias_writes_yaml(tmp_path):
    bids_root = tmp_path / 'bids'
    (bids_root / 'sub-01').mkdir(parents=True)

    rebuilt = reinitialize_megqa_pickles(bids_root)

    assert list(rebuilt) == ['sub-01']
    assert (bids_root / 'derivatives' / 'megQA' / 'sub-01.yml').is_file()


def test_invalid_yaml_reports_its_filename(tmp_path):
    bids_root = tmp_path / 'bids'
    (bids_root / 'sub-01').mkdir(parents=True)
    yaml_path = bids_root / 'derivatives' / 'megQA' / 'sub-01.yml'
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text('- not\n- a mapping\n')

    with pytest.raises(ValueError, match=r'sub-01\.yml'):
        subject_bids_info('01', bids_root=bids_root)
