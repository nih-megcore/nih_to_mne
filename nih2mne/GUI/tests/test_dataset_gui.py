#!/usr/bin/env python3

import importlib
import os
import logging
import sys
from pathlib import Path

import nih2mne
import pytest
import yaml

from nih2mne.GUI import dataset_gui


@pytest.fixture(autouse=True)
def isolate_config_module():
    missing = object()
    original_module = sys.modules.pop('nih2mne.config', missing)
    original_attribute = nih2mne.__dict__.pop('config', missing)
    original_log_level = logging.getLogger().level
    yield
    logging.getLogger().setLevel(original_log_level)
    sys.modules.pop('nih2mne.config', None)
    nih2mne.__dict__.pop('config', None)
    if original_module is not missing:
        sys.modules['nih2mne.config'] = original_module
    if original_attribute is not missing:
        nih2mne.config = original_attribute


def _write_defaults(path, bids_root):
    defaults = {
        'BIDS_gen': {
            'bids_root': bids_root,
            'bids_session': None,
            'bids_session_list': [None],
            'meg_dir': None,
            'mri_dir': None,
            'coreg_type': 'Brainsight',
            'anonymize': 'N',
            'crop_zeros': 'N',
            'emptyroom': 'N',
            'run_rank_reorder': 'Y',
        }
    }
    path.write_text(yaml.safe_dump(defaults), encoding='utf-8')


def test_parser_accepts_config_option():
    args = dataset_gui._get_parser().parse_args(
        [
            '-config', 'custom.yml',
            '-bids_root', '/data/bids',
            '-log', '/tmp/bids.log',
        ]
    )

    assert args.config == 'custom.yml'
    assert args.bids_root == '/data/bids'
    assert args.log == '/tmp/bids.log'


def test_cli_config_takes_precedence_over_environment(tmp_path, monkeypatch):
    env_config = tmp_path / 'environment.yml'
    cli_config = tmp_path / 'command-line.yml'
    _write_defaults(env_config, '/environment/bids')
    _write_defaults(cli_config, '/command-line/bids')
    monkeypatch.setenv('MEGCORE_DEFAULTS_FNAME', str(env_config))
    monkeypatch.setenv('HOME', str(tmp_path))

    config = dataset_gui._initialize_config(cli_config)

    assert Path(config._get_defaults_fname()) == cli_config
    assert config.DEFAULTS['BIDS_gen']['bids_root'] == '/command-line/bids'
    assert Path(os.environ['MEGCORE_DEFAULTS_FNAME']) == cli_config


def test_environment_config_is_used_without_cli_option(tmp_path, monkeypatch):
    env_config = tmp_path / 'environment.yml'
    _write_defaults(env_config, '/environment/bids')
    monkeypatch.setenv('MEGCORE_DEFAULTS_FNAME', str(env_config))
    monkeypatch.setenv('HOME', str(tmp_path))

    config = dataset_gui._initialize_config()

    assert Path(config._get_defaults_fname()) == env_config
    assert config.DEFAULTS['BIDS_gen']['bids_root'] == '/environment/bids'


def test_missing_config_can_be_created(tmp_path, monkeypatch):
    config_path = tmp_path / 'new' / 'defaults.yml'
    monkeypatch.setenv('HOME', str(tmp_path))

    config = dataset_gui._initialize_config(
        config_path, input_func=lambda _prompt: 'yes'
    )

    assert config_path.is_file()
    assert yaml.safe_load(config_path.read_text(encoding='utf-8')) == config.DEFAULTS


def test_missing_config_decline_leaves_filesystem_unchanged(tmp_path):
    config_path = tmp_path / 'new' / 'defaults.yml'

    with pytest.raises(dataset_gui._ConfigCreationDeclined):
        dataset_gui._initialize_config(
            config_path, input_func=lambda _prompt: ''
        )

    assert not config_path.parent.exists()


def test_config_path_must_be_a_file(tmp_path):
    with pytest.raises(ValueError, match='is not a file'):
        dataset_gui._initialize_config(tmp_path)


def test_invalid_confirmation_response_reprompts(tmp_path, capsys):
    config_path = tmp_path / 'defaults.yml'
    responses = iter(['maybe', 'no'])

    with pytest.raises(dataset_gui._ConfigCreationDeclined):
        dataset_gui._initialize_config(
            config_path, input_func=lambda _prompt: next(responses)
        )

    assert "Please answer 'yes' or 'no'." in capsys.readouterr().out


def test_bids_root_without_explicit_config_is_session_override(
        tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    bids_root = tmp_path / 'command-line-bids'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('MEGCORE_DEFAULTS_FNAME', str(config_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config()

    dataset_gui._configure_bids_root(config, bids_root=bids_root)

    assert config.DEFAULTS['BIDS_gen']['bids_root'] == str(bids_root)
    saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert saved['BIDS_gen']['bids_root'] == '/config/bids'


def test_bids_root_can_be_written_to_explicit_config(tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    bids_root = tmp_path / 'command-line-bids'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config(config_path)

    dataset_gui._configure_bids_root(
        config,
        bids_root=bids_root,
        config_fname=config_path,
        input_func=lambda _prompt: 'write',
    )

    saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert saved['BIDS_gen']['bids_root'] == str(bids_root)
    assert config.DEFAULTS['BIDS_gen']['bids_root'] == str(bids_root)


def test_bids_root_can_defer_to_explicit_config(tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config(config_path)

    dataset_gui._configure_bids_root(
        config,
        bids_root=tmp_path / 'command-line-bids',
        config_fname=config_path,
        input_func=lambda _prompt: 'config',
    )

    saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert saved['BIDS_gen']['bids_root'] == '/config/bids'
    assert config.DEFAULTS['BIDS_gen']['bids_root'] == '/config/bids'


def test_invalid_bids_root_choice_reprompts(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / 'defaults.yml'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config(config_path)
    responses = iter(['maybe', 'config'])

    dataset_gui._configure_bids_root(
        config,
        bids_root=tmp_path / 'command-line-bids',
        config_fname=config_path,
        input_func=lambda _prompt: next(responses),
    )

    assert "Please answer 'write' or 'config'." in capsys.readouterr().out


def _remove_log_handler(log_path):
    resolved_path = Path(log_path).expanduser().resolve()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if not isinstance(handler, logging.FileHandler):
            continue
        if Path(handler.baseFilename).resolve() == resolved_path:
            root_logger.removeHandler(handler)
            handler.close()


def test_logging_config_is_synchronized_into_existing_defaults(
        tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))

    config = dataset_gui._initialize_config(config_path)

    assert config.DEFAULTS['logging'] == {'meg_dataset_gui': None}
    saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert saved['logging'] == {'meg_dataset_gui': None}


def test_prompted_log_path_can_be_persisted(tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    log_path = '~/project-logs/bids_conversion.log'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config(config_path)
    responses = iter([log_path, 'yes'])

    try:
        resolved = dataset_gui._configure_logging(
            config,
            input_func=lambda _prompt: next(responses),
        )

        assert resolved == tmp_path / 'project-logs' / 'bids_conversion.log'
        saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert saved['logging']['meg_dataset_gui'] == log_path
    finally:
        _remove_log_handler(tmp_path / 'project-logs' / 'bids_conversion.log')


def test_prompted_default_log_path_can_remain_session_only(
        tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config(config_path)
    responses = iter(['', 'no'])
    expected = tmp_path / 'megcore' / 'logging' / 'bids_conversion.log'

    try:
        resolved = dataset_gui._configure_logging(
            config,
            input_func=lambda _prompt: next(responses),
        )

        assert resolved == expected
        assert expected.is_file()
        saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert saved['logging']['meg_dataset_gui'] is None
    finally:
        _remove_log_handler(expected)


def test_command_log_overrides_config_without_prompting(tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    configured_log = tmp_path / 'configured.log'
    command_log = tmp_path / 'command.log'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config(config_path)
    config.DEFAULTS['logging']['meg_dataset_gui'] = str(configured_log)

    try:
        resolved = dataset_gui._configure_logging(
            config,
            command_log=command_log,
            input_func=lambda _prompt: pytest.fail('unexpected prompt'),
        )

        assert resolved == command_log
        saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert saved['logging']['meg_dataset_gui'] is None
    finally:
        _remove_log_handler(command_log)


def test_configured_log_path_suppresses_prompt_and_records_messages(
        tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    log_path = tmp_path / 'configured.log'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    config = dataset_gui._initialize_config(config_path)
    config.DEFAULTS['logging']['meg_dataset_gui'] = str(log_path)

    try:
        dataset_gui._configure_logging(
            config,
            input_func=lambda _prompt: pytest.fail('unexpected prompt'),
        )
        logging.getLogger('nih2mne.test').info('test logging message')
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert 'test logging message' in log_path.read_text(encoding='utf-8')
    finally:
        _remove_log_handler(log_path)


def test_logging_initialization_deduplicates_shared_path(tmp_path):
    log_path = tmp_path / 'shared.log'

    try:
        dataset_gui._initialize_file_logging(log_path)
        dataset_gui._initialize_file_logging(log_path)

        matching_handlers = [
            handler for handler in logging.getLogger().handlers
            if isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == log_path
        ]
        assert len(matching_handlers) == 1
    finally:
        _remove_log_handler(log_path)


def test_logging_rejects_directory_as_logfile(tmp_path):
    with pytest.raises(ValueError, match='is not a file'):
        dataset_gui._initialize_file_logging(tmp_path)


def test_bids_creator_dialogs_use_qt_default_options(
        tmp_path, monkeypatch):
    config_path = tmp_path / 'defaults.yml'
    _write_defaults(config_path, '/config/bids')
    monkeypatch.setenv('HOME', str(tmp_path))
    dataset_gui._initialize_config(config_path)

    module_name = (
        'nih2mne.GUI.templates.bids_creator_gui_control_functions'
    )
    sys.modules.pop(module_name, None)
    bids_creator = importlib.import_module(module_name)
    calls = {}

    def get_open_file_name(*args, **kwargs):
        calls['file'] = (args, kwargs)
        return '/data/electrodes.txt', '*.txt'

    def get_existing_directory(*args, **kwargs):
        calls['directory'] = (args, kwargs)
        return '/data/bids'

    monkeypatch.setattr(
        bids_creator.QtWidgets.QFileDialog,
        'getOpenFileName',
        get_open_file_name,
    )
    monkeypatch.setattr(
        bids_creator.QtWidgets.QFileDialog,
        'getExistingDirectory',
        get_existing_directory,
    )

    file_name = bids_creator.BIDS_MainWindow.open_file_dialog(
        object(), file_filters='*.txt', default_dir='/data'
    )
    directory = bids_creator.BIDS_MainWindow.open_folder_dialog(
        object(), default_dir='/data'
    )

    assert file_name == '/data/electrodes.txt'
    assert directory == '/data/bids'
    assert calls['file'][1] == {}
    assert calls['directory'][1] == {}
