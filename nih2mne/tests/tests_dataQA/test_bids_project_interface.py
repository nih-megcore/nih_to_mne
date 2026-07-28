#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os.path as op

from nih2mne.dataQA.bids_project_interface import (
    _subject_bids_info,
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
        new_root, 'derivatives', 'megQA', 'sub-01.pkl')
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
