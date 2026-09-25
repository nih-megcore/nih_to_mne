#!/usr/bin/env python3

import importlib
import os
import logging
import sys
from pathlib import Path

import nih2mne
import pytest
import yaml

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from nih2mne.GUI import dataset_gui


BIDS_CREATOR_MODULE = (
    'nih2mne.GUI.templates.bids_creator_gui_control_functions'
)


@pytest.fixture(autouse=True)
def isolate_config_module():
    missing = object()
    original_module = sys.modules.pop('nih2mne.config', missing)
    original_attribute = nih2mne.__dict__.pop('config', missing)
    original_bids_creator = sys.modules.pop(BIDS_CREATOR_MODULE, missing)
    original_log_level = logging.getLogger().level
    yield
    logging.getLogger().setLevel(original_log_level)
    sys.modules.pop('nih2mne.config', None)
    nih2mne.__dict__.pop('config', None)
    if original_module is not missing:
        sys.modules['nih2mne.config'] = original_module
    if original_attribute is not missing:
        nih2mne.config = original_attribute
    sys.modules.pop(BIDS_CREATOR_MODULE, None)
    if original_bids_creator is not missing:
        sys.modules[BIDS_CREATOR_MODULE] = original_bids_creator


@pytest.fixture(scope='module')
def qapp():
    app = dataset_gui.QtWidgets.QApplication.instance()
    if app is None:
        app = dataset_gui.QtWidgets.QApplication([])
    return app


def _write_defaults(path, bids_root, zfill_run=2, zfill_ses=2):
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
            'zfill_run': zfill_run,
            'zfill_ses': zfill_ses,
        }
    }
    path.write_text(yaml.safe_dump(defaults), encoding='utf-8')


def _run_dict():
    return {
        'anonymize': False,
        'subjid_input': 'MEGHASH',
        'bids_id': '01',
        'bids_dir': '/tmp/bids',
        'bids_session': '02',
        'meg_dataset_list': [
            '/tmp/MEGHASH_task_20260101_01.ds',
            '/tmp/MEGHASH_task_20260101_02.ds',
        ],
        'mri_none': False,
        'mri_bsight': '/tmp/mri.nii.gz',
        'mri_elec': '/tmp/electrodes.txt',
        'mri_brik': False,
        'crop_zeros': True,
        'include_empty_room': False,
        'run_rank_reorder': False,
    }


def _load_bids_creator(
        tmp_path, monkeypatch, zfill_run=2, zfill_ses=2):
    config_path = tmp_path / 'defaults.yml'
    _write_defaults(
        config_path,
        '/config/bids',
        zfill_run=zfill_run,
        zfill_ses=zfill_ses,
    )
    monkeypatch.setenv('HOME', str(tmp_path))
    dataset_gui._initialize_config(config_path)
    sys.modules.pop(BIDS_CREATOR_MODULE, None)
    return importlib.import_module(BIDS_CREATOR_MODULE)


def test_parser_accepts_config_option():
    args = dataset_gui._get_parser().parse_args(
        [
            '-config', 'custom.yml',
            '-bids_root', '/data/bids',
            '-log', '/tmp/bids.log',
            '-rundict', 'RUNDICT: {}',
        ]
    )

    assert args.config == 'custom.yml'
    assert args.bids_root == '/data/bids'
    assert args.log == '/tmp/bids.log'
    assert args.rundict == 'RUNDICT: {}'


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
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
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


def test_bids_creator_uses_configured_session_and_run_padding(
        tmp_path, monkeypatch, qapp):
    bids_creator = _load_bids_creator(
        tmp_path,
        monkeypatch,
        zfill_run=4,
        zfill_ses=3,
    )
    meg_fname = '/tmp/MEGHASH_test_20260101_01.ds'
    window = bids_creator.BIDS_MainWindow(meg_dsets=[meg_fname])
    window.opts.update(
        bids_id='01',
        bids_dir='/tmp/bids',
        bids_session='03',
        mri_bsight='/tmp/mri.nii.gz',
        mri_brik=False,
    )

    try:
        window._make_task_dict(run_rank_reorder=True)
        window._make_anat_dict()

        assert window.io_mapping[meg_fname]['run'] == '0001'
        assert str(window.io_mapping[meg_fname]['bidspath'].fpath) == (
            '/tmp/bids/sub-01/ses-003/meg/'
            'sub-01_ses-003_task-test_run-0001_meg.ds'
        )
        assert str(
            window.anat_io_mapping['/tmp/mri.nii.gz']['bidspath'].fpath
        ) == (
            '/tmp/bids/sub-01/ses-003/anat/'
            'sub-01_ses-003_run-0001_T1w.nii.gz'
        )
        assert window.opts['bids_session'] == '03'
    finally:
        window.close()


def test_rundict_round_trip_accepts_full_log_line(tmp_path, monkeypatch):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    run_dict = _run_dict()

    serialized = bids_creator.serialize_run_dict(run_dict)
    restored = bids_creator.parse_run_dict(
        '2026-09-25 10:30:00 - INFO - bids.gui - RUNDICT: '
        f'{serialized}'
    )

    assert '\n' not in serialized
    assert restored == run_dict
    assert bids_creator.parse_run_dict(serialized) == run_dict


@pytest.mark.parametrize(
    'mutation, match',
    [
        (lambda value: value.pop('bids_id'), 'missing required keys'),
        (lambda value: value.update(extra='value'), 'unknown keys'),
        (lambda value: value.update(anonymize='yes'), 'must be true or false'),
        (lambda value: value.update(meg_dataset_list='dataset.ds'),
         'must be a list of strings'),
    ],
)
def test_rundict_validation_rejects_invalid_records(
        tmp_path, monkeypatch, mutation, match):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    run_dict = _run_dict()
    mutation(run_dict)

    with pytest.raises(ValueError, match=match):
        bids_creator.validate_run_dict(run_dict)


def test_rundict_restores_bids_creator_controls(
        tmp_path, monkeypatch, qapp):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    window = bids_creator.BIDS_MainWindow(run_dict=_run_dict())

    try:
        assert window.opts == _run_dict()
        assert window.ui.pb_LoadRunDict.text() == 'Load RUNDICT'
        assert window.ui.te_meghash.toPlainText() == 'MEGHASH'
        assert window.ui.te_BIDS_id.toPlainText() == '01'
        assert window.ui.te_bids_dir.toPlainText() == '/tmp/bids'
        assert window.ui.cb_Bids_Session.currentText() == '02'
        assert window.ui.list_fname_conversion.count() == 2
        assert window.ui.te_brainsight_mri.toPlainText() == '/tmp/mri.nii.gz'
        assert window.ui.te_brainsight_elec.toPlainText() == '/tmp/electrodes.txt'
        assert window.ui.tab_Coreg.currentWidget() is window.ui.tab_BSight
        assert window.ui.cb_crop_zeros.isChecked()
        assert not window.ui.cb_emptyroom.isChecked()
    finally:
        window.close()


def test_load_rundict_action_populates_and_checks_outputs(
        tmp_path, monkeypatch, qapp):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    record = f'RUNDICT: {bids_creator.serialize_run_dict(_run_dict())}'
    window = bids_creator.BIDS_MainWindow()
    checked = []
    monkeypatch.setattr(
        bids_creator.QtWidgets.QInputDialog,
        'getMultiLineText',
        lambda *_args, **_kwargs: (record, True),
    )
    monkeypatch.setattr(
        window,
        'check_restored_outputs',
        lambda: checked.append(True),
    )

    try:
        window._action_load_rundict()

        assert window.opts == _run_dict()
        assert checked == [True]
    finally:
        window.close()


def test_invalid_pasted_rundict_does_not_change_controls(
        tmp_path, monkeypatch, qapp):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    window = bids_creator.BIDS_MainWindow()
    original_opts = dict(window.opts)
    warnings = []
    monkeypatch.setattr(
        bids_creator.QtWidgets.QInputDialog,
        'getMultiLineText',
        lambda *_args, **_kwargs: ('RUNDICT: {invalid}', True),
    )
    monkeypatch.setattr(
        bids_creator.QtWidgets.QMessageBox,
        'warning',
        lambda *args: warnings.append(args),
    )

    try:
        window._action_load_rundict()

        assert window.opts == original_opts
        assert len(warnings) == 1
    finally:
        window.close()


def test_run_logs_rundict_before_output_check_failure(
        tmp_path, monkeypatch, qapp, caplog):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    log_path = tmp_path / 'bids_conversion.log'
    dataset_gui._initialize_file_logging(log_path)
    window = bids_creator.BIDS_MainWindow(run_dict=_run_dict())
    monkeypatch.setattr(
        window,
        '_action_pb_CheckOutputs',
        lambda: (_ for _ in ()).throw(RuntimeError('check failed')),
    )

    try:
        with caplog.at_level(logging.INFO, logger=bids_creator.logger.name):
            with pytest.raises(RuntimeError, match='check failed'):
                window._action_pb_run()

        messages = [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith('RUNDICT: {')
        ]
        assert len(messages) == 1
        assert bids_creator.parse_run_dict(messages[0]) == _run_dict()
        for handler in logging.getLogger().handlers:
            handler.flush()
        log_lines = log_path.read_text(encoding='utf-8').splitlines()
        rundict_lines = [line for line in log_lines if 'RUNDICT:' in line]
        assert len(rundict_lines) == 1
        assert bids_creator.parse_run_dict(rundict_lines[0]) == _run_dict()
    finally:
        window.close()
        _remove_log_handler(log_path)


def test_run_reports_success_in_status_bar_and_final_terminal_line(
        tmp_path, monkeypatch, qapp, caplog, capsys):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    run_dict = _run_dict()
    run_dict.update(mri_none=True, mri_bsight=False, mri_elec=False)
    window = bids_creator.BIDS_MainWindow(run_dict=run_dict)

    def prepare_outputs():
        window.io_mapping = {
            '/tmp/input.ds': {'bidspath': '/tmp/bids/output_meg.ds'}
        }

    monkeypatch.setattr(window, '_action_pb_CheckOutputs', prepare_outputs)
    monkeypatch.setattr(window, '_set_single_filelist_text', lambda **_kwargs: None)
    monkeypatch.setattr(bids_creator, '_proc_meg_bids', lambda **_kwargs: None)

    try:
        with caplog.at_level(logging.INFO, logger=bids_creator.logger.name):
            window._action_pb_run()

        message = 'BIDS conversion finished successfully.'
        assert window.ui.statusbar.currentMessage() == message
        assert window.ui.pb_review_errors.isHidden()
        assert capsys.readouterr().out.rstrip().splitlines()[-1] == message
        completion_records = [
            record for record in caplog.records
            if record.getMessage() == message
        ]
        assert len(completion_records) == 1
        assert completion_records[0].levelno == logging.INFO
    finally:
        window.close()


@pytest.mark.parametrize('failed_stage', ['meg', 'mri'])
def test_run_reports_caught_conversion_errors(
        tmp_path, monkeypatch, qapp, caplog, capsys, failed_stage):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    run_dict = _run_dict()
    if failed_stage == 'meg':
        run_dict.update(mri_none=True, mri_bsight=False, mri_elec=False)
    else:
        run_dict.update(meg_dataset_list=[])
    window = bids_creator.BIDS_MainWindow(run_dict=run_dict)

    def prepare_outputs():
        if failed_stage == 'meg':
            window.io_mapping = {
                '/tmp/input.ds': {'bidspath': '/tmp/bids/output_meg.ds'}
            }
        else:
            window.io_mapping = {}
            window.anat_io_mapping = {
                '/tmp/mri.nii.gz': {'bidspath': '/tmp/bids/output_T1w.nii.gz'}
            }
            window._anat_idx = 0

    def fail_conversion(**_kwargs):
        raise RuntimeError(f'{failed_stage} failed')

    monkeypatch.setattr(window, '_action_pb_CheckOutputs', prepare_outputs)
    monkeypatch.setattr(window, '_set_single_filelist_text', lambda **_kwargs: None)
    monkeypatch.setattr(bids_creator, '_proc_meg_bids', (
        fail_conversion if failed_stage == 'meg' else lambda **_kwargs: None
    ))
    monkeypatch.setattr(bids_creator, '_proc_mri_bids', (
        fail_conversion if failed_stage == 'mri' else lambda **_kwargs: None
    ))

    try:
        with caplog.at_level(logging.INFO, logger=bids_creator.logger.name):
            window._action_pb_run()

        message = 'BIDS conversion finished with errors.'
        assert window.ui.statusbar.currentMessage() == message
        assert capsys.readouterr().out.rstrip().splitlines()[-1] == message
        completion_records = [
            record for record in caplog.records
            if record.getMessage() == message
        ]
        assert len(completion_records) == 1
        assert completion_records[0].levelno == logging.WARNING
    finally:
        window.close()


def test_error_log_filter_includes_tracebacks_but_not_warnings(
        tmp_path, monkeypatch):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    log_text = (
        '2026-09-25 10:00:00,000 - INFO - test - starting\n'
        '2026-09-25 10:00:01,000 - WARNING - test - warning\n'
        '2026-09-25 10:00:02,000 - ERROR - test - failed\n'
        'Traceback (most recent call last):\n'
        '  File "conversion.py", line 1, in run\n'
        'RuntimeError: failed\n'
        '2026-09-25 10:00:03,000 - CRITICAL - test - stopped\n'
        'critical detail\n'
        '2026-09-25 10:00:04,000 - INFO - test - done\n'
    )

    errors = bids_creator._extract_error_log_blocks(log_text)

    assert 'WARNING' not in errors
    assert 'starting' not in errors
    assert 'done' not in errors
    assert 'ERROR - test - failed' in errors
    assert 'RuntimeError: failed' in errors
    assert 'CRITICAL - test - stopped' in errors
    assert 'critical detail' in errors


def test_log_path_is_propagated_to_bids_creator(
        tmp_path, monkeypatch, qapp):
    _load_bids_creator(tmp_path, monkeypatch)
    log_path = tmp_path / 'bids_conversion.log'
    parent = dataset_gui.GUI_MainWindow(log_path=log_path)

    try:
        parent._bids_window_open()

        assert parent.bids_gui.log_path == log_path.resolve()
    finally:
        if hasattr(parent, 'bids_gui'):
            parent.bids_gui.close()
        parent.close()


def test_encode_and_qa_all_processes_tiles_in_order_and_skips_errors(
        qapp, caplog):
    parent = dataset_gui.GUI_MainWindow()
    calls = []

    def add_tile(name, processor=None):
        tile = dataset_gui.QtWidgets.QWidget()
        tile.fname = name
        if processor is not None:
            tile.trigprocess = processor
        item = dataset_gui.QtWidgets.QListWidgetItem()
        parent.ui.scrollAreaWidgetContents.addItem(item)
        parent.ui.scrollAreaWidgetContents.setItemWidget(item, tile)

    add_tile('first.ds', lambda: calls.append('first'))
    add_tile('error.ds')

    def fail_processing():
        calls.append('failed')
        raise RuntimeError('trigger processing failed')

    add_tile('failed.ds', fail_processing)
    add_tile('last.ds', lambda: calls.append('last'))

    try:
        with caplog.at_level(logging.ERROR, logger=dataset_gui.logger.name):
            parent.ui.pb_CheckData.click()

        assert parent.ui.pb_CheckData.text() == 'Encode+QA All'
        assert parent.ui.pb_CheckData.toolTip() == (
            'Run trigger processing and refresh QA status for all loaded '
            'datasets'
        )
        assert calls == ['first', 'failed', 'last']
        assert 'Encode+QA failed for dataset failed.ds' in caplog.text
    finally:
        parent.close()


def test_encode_and_qa_all_accepts_empty_tile_list(qapp):
    parent = dataset_gui.GUI_MainWindow()

    try:
        parent.ui.pb_CheckData.click()
        assert parent.ui.scrollAreaWidgetContents.count() == 0
    finally:
        parent.close()


def test_review_errors_uses_only_latest_failed_run_log_range(
        tmp_path, monkeypatch, qapp):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    log_path = tmp_path / 'bids_conversion.log'
    dataset_gui._initialize_file_logging(log_path)
    run_dict = _run_dict()
    run_dict.update(mri_none=True, mri_bsight=False, mri_elec=False)
    window = bids_creator.BIDS_MainWindow(
        run_dict=run_dict,
        log_path=log_path,
    )

    def prepare_outputs():
        window.io_mapping = {
            '/tmp/input.ds': {'bidspath': '/tmp/bids/output_meg.ds'}
        }

    def fail_conversion(**_kwargs):
        logging.getLogger('conversion.library').warning('CURRENT WARNING')
        raise RuntimeError('CURRENT FAILURE')

    monkeypatch.setattr(window, '_action_pb_CheckOutputs', prepare_outputs)
    monkeypatch.setattr(window, '_set_single_filelist_text', lambda **_kwargs: None)
    monkeypatch.setattr(bids_creator, '_proc_meg_bids', fail_conversion)
    logging.getLogger('conversion.library').error('OLDER FAILURE')

    try:
        assert window.ui.pb_review_errors.isHidden()
        window._action_pb_run()
        logging.getLogger('conversion.library').error('LATER FAILURE')
        for handler in logging.getLogger().handlers:
            handler.flush()

        review_text = window._error_review_text()
        assert not window.ui.pb_review_errors.isHidden()
        assert review_text.splitlines()[0] == (
            f'The full log is located {log_path.resolve()}'
        )
        assert 'MEG conversion failed for /tmp/input.ds' in review_text
        assert 'RuntimeError: CURRENT FAILURE' in review_text
        assert 'CURRENT WARNING' not in review_text
        assert 'OLDER FAILURE' not in review_text
        assert 'LATER FAILURE' not in review_text

        dialogs = []

        def capture_dialog(dialog):
            dialogs.append(dialog)
            return bids_creator.QtWidgets.QDialog.DialogCode.Rejected

        monkeypatch.setattr(
            bids_creator.QtWidgets.QDialog,
            'exec',
            capture_dialog,
        )
        window._show_run_errors()

        assert len(dialogs) == 1
        assert dialogs[0].windowTitle() == 'BIDS Conversion Errors'
        text_widget = dialogs[0].findChild(
            bids_creator.QtWidgets.QPlainTextEdit,
            'bids_conversion_error_text',
        )
        assert text_widget.isReadOnly()
        assert text_widget.toPlainText() == review_text

        def successful_conversion(**_kwargs):
            assert window.ui.pb_review_errors.isHidden()

        monkeypatch.setattr(
            bids_creator, '_proc_meg_bids', successful_conversion
        )
        window._action_pb_run()

        assert window.ui.pb_review_errors.isHidden()
        assert window._last_run_log_range is None
    finally:
        window.close()
        _remove_log_handler(log_path)


def test_library_error_exposes_review_button_without_raised_exception(
        tmp_path, monkeypatch, qapp):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    log_path = tmp_path / 'bids_conversion.log'
    dataset_gui._initialize_file_logging(log_path)
    run_dict = _run_dict()
    run_dict.update(mri_none=True, mri_bsight=False, mri_elec=False)
    window = bids_creator.BIDS_MainWindow(
        run_dict=run_dict,
        log_path=log_path,
    )

    def prepare_outputs():
        window.io_mapping = {
            '/tmp/input.ds': {'bidspath': '/tmp/bids/output_meg.ds'}
        }

    def log_error(**_kwargs):
        logging.getLogger('conversion.library').error('LIBRARY ERROR')

    monkeypatch.setattr(window, '_action_pb_CheckOutputs', prepare_outputs)
    monkeypatch.setattr(window, '_set_single_filelist_text', lambda **_kwargs: None)
    monkeypatch.setattr(bids_creator, '_proc_meg_bids', log_error)

    try:
        window._action_pb_run()

        assert not window.ui.pb_review_errors.isHidden()
        assert window.ui.statusbar.currentMessage() == (
            'BIDS conversion finished with errors.'
        )
        assert 'LIBRARY ERROR' in window._error_review_text()
    finally:
        window.close()
        _remove_log_handler(log_path)


def test_error_review_reports_missing_logfile(tmp_path, monkeypatch, qapp):
    bids_creator = _load_bids_creator(tmp_path, monkeypatch)
    log_path = tmp_path / 'missing.log'
    window = bids_creator.BIDS_MainWindow(log_path=log_path)
    window._last_run_log_range = (0, 10)

    try:
        review_text = window._error_review_text()

        assert review_text.startswith(
            f'The full log is located {log_path.resolve()}\n\n'
        )
        assert 'Unable to read the RUN errors:' in review_text
    finally:
        window.close()


def test_restore_bids_creator_checks_without_running_conversion():
    calls = []

    class FakeBidsWindow:
        def check_restored_outputs(self):
            calls.append('check')

    class FakeParent:
        def _bids_window_open(self, meg_dsets=None, run_dict=None):
            calls.append(('open', meg_dsets, run_dict))
            self.bids_gui = FakeBidsWindow()

    parent = FakeParent()

    dataset_gui.GUI_MainWindow.restore_bids_creator(parent, _run_dict())

    assert calls == [('open', None, _run_dict()), 'check']
