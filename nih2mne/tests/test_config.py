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
    assert saved_defaults['logging'] == {'meg_dataset_gui': None}
