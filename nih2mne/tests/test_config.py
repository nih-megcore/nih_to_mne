#!/usr/bin/env python3

import importlib
import sys
from pathlib import Path

import yaml


def test_initialize_defaults_creates_megcore_directories(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv('MEGCORE_DEFAULTS_FNAME', raising=False)
    sys.modules.pop('nih2mne.config', None)

    config = importlib.import_module('nih2mne.config')

    megcore_dir = tmp_path / 'megcore'
    assert (megcore_dir / 'trigproc').is_dir()
    assert (megcore_dir / 'datproc').is_dir()
    assert (megcore_dir / 'defaults.yml').is_file()
    assert Path(config.TRIG_FILE_LOC) == megcore_dir / 'trigproc'
    assert Path(config.DATPROC_FILE_LOC) == megcore_dir / 'datproc'
    saved_defaults = yaml.safe_load(
        (megcore_dir / 'defaults.yml').read_text(encoding='utf-8')
    )
    assert saved_defaults['BIDS_gen']['zfill_run'] == 2
    assert saved_defaults['BIDS_gen']['zfill_ses'] == 2
    assert saved_defaults['logging'] == {'meg_dataset_gui': None}


def test_existing_defaults_are_synchronized_with_padding_options(
        tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    config_path.write_text(
        yaml.safe_dump({'BIDS_gen': {'bids_root': '/tmp/bids'}}),
        encoding='utf-8',
    )
    monkeypatch.setenv('MEGCORE_DEFAULTS_FNAME', str(config_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    sys.modules.pop('nih2mne.config', None)

    config = importlib.import_module('nih2mne.config')

    assert config.DEFAULTS['BIDS_gen']['zfill_run'] == 2
    assert config.DEFAULTS['BIDS_gen']['zfill_ses'] == 2
    saved_defaults = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert saved_defaults['BIDS_gen']['zfill_run'] == 2
    assert saved_defaults['BIDS_gen']['zfill_ses'] == 2
