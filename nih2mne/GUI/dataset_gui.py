#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:39:56 2025

@author: jstout

Ui_MainWindow - From QT designer
Ui_InputDatasetTile - From Qt designer (converted without -x)

This code implements a drag/drop GUI for quick data QA and event marking
Each file gets its own tile that has independent information and processing buttons
The tile also presents some status information on the file - movement / missing markerfile
Once the files have been event tagged and their MarkerFile.mrk produced, 
the bids gui can be opened to generate the correct output + Anat association

#make glob case insensitive -- for the task match on trigproc files
Make the trigger file locations environmental overidable

"""

from nih2mne.GUI.templates.file_staging_meg import Ui_MainWindow
from nih2mne.GUI.templates.input_meg_dset_tile_listWidgetBase import \
    Ui_InputDatasetTile
from nih2mne.GUI.templates.input_error_dset_tile_listWidgetBase import \
    Ui_ErrorDatasetTile

from nih2mne.GUI.qt_compat import QtCore, QtGui, QtWidgets

pyqtSignal = QtCore.pyqtSignal
import argparse
import importlib
import logging
from numbers import Number
import os, os.path as op
from pathlib import Path
import sys
import mne
import glob
import pandas as pd
from nih2mne.utilities.calc_hm import get_localizer_dframe, compute_movement
from nih2mne.utilities.data_crop_wrapper import get_term_time
from nih2mne.utilities.qa_runner import (
    filename_version,
    find_qa_file,
    run_qa_file,
)
import shutil
import copy
import numpy as np


logger = logging.getLogger(__name__)
DEFAULT_LOG_PATH = '~/megcore/logging/bids_conversion.log'
LOGGING_CATEGORY = 'meg_dataset_gui'


class _ConfigCreationDeclined(Exception):
    """Raised when the user declines to create a requested config file."""


def _procfile_sort_key(filename):
    """Sort processing files by numeric version, then name."""
    version = filename_version(filename)
    return (version is not None, version if version is not None else -1,
            filename.lower())


def _task_procfiles(directory, task):
    """Return task processing files, excluding QA definitions."""
    if not directory or not op.isdir(directory):
        return []
    task_prefix = f'{task}_'.lower()
    filenames = [
        filename
        for filename in os.listdir(directory)
        if filename.lower().startswith(task_prefix)
        and not filename.lower().endswith('.qa')
        and op.isfile(op.join(directory, filename))
    ]
    return sorted(filenames, key=_procfile_sort_key, reverse=True)


def _display_qa_value(value):
    """Format QA values without unnecessary decimal zeros."""
    if value is None:
        return '—'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _aligned_table(headers, rows):
    """Return a fixed-width plain-text table."""
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row):
        return '  '.join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        ).rstrip()

    lines = [
        format_row(headers),
        format_row(tuple('-' * width for width in widths)),
    ]
    lines.extend(format_row(row) for row in rows)
    return '\n'.join(lines)


def _qa_table_text(result):
    """Format detailed QA conditions with failures first."""
    status_order = {'fail': 0, 'pass': 1, 'skipped': 2}
    checks = sorted(
        result['checks'],
        key=lambda check: status_order[check['status']],
    )
    rows = [
        (
            check['term'],
            f"{check['operator']} {_display_qa_value(check['expected'])}",
            _display_qa_value(check['actual']),
            f"<{check['status'].upper()}>",
        )
        for check in checks
    ]
    return _aligned_table(
        ('Condition', 'Expected', 'Actual', 'Result'),
        rows,
    )


def _qa_review_text(result):
    """Format one detailed QA result as an aligned text block."""
    overall = result['overall'].upper()
    task = (result.get('task') or 'TASK').upper()
    lines = [
        f'{task}  <{overall}>',
        f"QA file: {op.basename(result['qa_file'])}",
        '',
        _qa_table_text(result),
    ]
    return '\n'.join(lines)


def _event_counts(raw):
    """Return complete deterministic annotation counts for a loaded dataset."""
    event_frame = pd.DataFrame(raw.annotations)
    if 'description' not in event_frame.columns:
        return ()
    counts = event_frame.description.value_counts().to_dict()
    return tuple(sorted(
        ((str(name), int(count)) for name, count in counts.items()),
        key=lambda item: (-item[1], item[0].lower(), item[0]),
    ))


def _batch_review_block_text(block):
    """Format one dataset snapshot for the aggregate batch review."""
    kind = block['kind']
    if kind == 'qa':
        outcome = block['qa_result']['overall'].upper()
    elif kind == 'error':
        outcome = 'ERROR'
    elif block.get('note', '').startswith('QA ERROR'):
        outcome = 'QA ERROR'
    else:
        outcome = 'EVENT COUNTS'

    lines = [
        f"{block['task'].upper()} — Run {block['run']}  <{outcome}>",
        f"Movement max: {block['movement']}  |  "
        f"Duration: {block['duration']}",
        f"Dataset: {block['dataset']}",
    ]

    if kind == 'qa':
        result = block['qa_result']
        lines.extend([
            f"QA file: {op.basename(result['qa_file'])}",
            '',
            _qa_table_text(result),
        ])
    elif kind == 'error':
        lines.extend(['', f"Error: {block['error']}"])
    else:
        if block.get('note'):
            lines.append(f"Note: {block['note']}")
        lines.append('')
        if block['event_counts']:
            rows = tuple(
                (name, str(count)) for name, count in block['event_counts']
            )
            lines.extend([
                'Events:',
                _aligned_table(('Event', 'Count'), rows),
            ])
        else:
            lines.append('Events: NONE')
    return '\n'.join(lines)


def _batch_review_text(blocks):
    """Format the immutable snapshot from the latest Encode+QA All run."""
    divider = '=' * 80
    body = f'\n\n{divider}\n\n'.join(
        _batch_review_block_text(block) for block in blocks
    )
    return f'ENCODE+QA RESULTS\n\n{body}'


def _get_parser():
    parser = argparse.ArgumentParser(
        description='Open the NIH MEG dataset staging GUI.'
    )
    parser.add_argument(
        '-config',
        metavar='PATH',
        help=(
            'Defaults YAML file. This takes precedence over '
            'MEGCORE_DEFAULTS_FNAME.'
        ),
    )
    parser.add_argument(
        '-bids_root',
        metavar='PATH',
        help='BIDS root to use when opening the BIDS creator.',
    )
    parser.add_argument(
        '-log',
        metavar='PATH',
        help=(
            'Logfile for the dataset GUI and BIDS creator. This is a '
            'session-only override of logging.meg_dataset_gui.'
        ),
    )
    parser.add_argument(
        '-rundict',
        metavar='JSON',
        help=(
            'Restore the BIDS Creator from a RUNDICT log line or its JSON '
            'payload. Quote the value for the shell.'
        ),
    )
    return parser


def _confirm_config_creation(config_path, input_func=None):
    if input_func is None:
        input_func = input

    prompt = (
        f'No config file exists at {config_path}. '
        'Create a default config file there? [y/N] '
    )
    while True:
        try:
            response = input_func(prompt).strip().lower()
        except EOFError:
            response = ''

        if response in ('y', 'yes'):
            return True
        if response in ('', 'n', 'no'):
            return False
        print("Please answer 'yes' or 'no'.")


def _initialize_config(config_fname=None, input_func=None):
    """Select and initialize the defaults file before dependent imports."""
    if config_fname is not None:
        config_path = Path(config_fname).expanduser().resolve()
        if config_path.exists():
            if not config_path.is_file():
                raise ValueError(
                    f'Config path exists but is not a file: {config_path}'
                )
        else:
            if not _confirm_config_creation(config_path, input_func=input_func):
                raise _ConfigCreationDeclined
            config_path.parent.mkdir(parents=True, exist_ok=True)

        # nih2mne.config resolves this variable during import. Setting it here
        # makes the command-line value take precedence over an existing value.
        os.environ['MEGCORE_DEFAULTS_FNAME'] = str(config_path)

    return importlib.import_module('nih2mne.config')


def _confirm_bids_root_action(config_bids_root, command_bids_root,
                              input_func=None):
    if input_func is None:
        input_func = input

    prompt = (
        'Both -config and -bids_root were provided.\n'
        f'Config bids_root: {config_bids_root}\n'
        f'Command-line bids_root: {command_bids_root}\n'
        'Write the command-line value into the config, or use the config '
        'value? [w/C] '
    )
    while True:
        try:
            response = input_func(prompt).strip().lower()
        except EOFError:
            response = ''

        if response in ('w', 'write'):
            return True
        if response in ('', 'c', 'config'):
            return False
        print("Please answer 'write' or 'config'.")


def _configure_bids_root(config, bids_root=None, config_fname=None,
                         input_func=None):
    """Apply a session BIDS root or persist it when explicitly requested."""
    if bids_root is None:
        return

    command_bids_root = str(Path(bids_root).expanduser().resolve())
    bids_defaults = config.DEFAULTS['BIDS_gen']

    if config_fname is None:
        bids_defaults['bids_root'] = command_bids_root
        return

    should_write = _confirm_bids_root_action(
        config_bids_root=bids_defaults['bids_root'],
        command_bids_root=command_bids_root,
        input_func=input_func,
    )
    if not should_write:
        return

    import yaml

    bids_defaults['bids_root'] = command_bids_root
    defaults_fname = config._get_defaults_fname()
    with open(defaults_fname, 'w') as defaults_file:
        yaml.dump(
            config.DEFAULTS,
            defaults_file,
            sort_keys=False,
            default_flow_style=False,
        )


def _prompt_for_log_path(input_func=None):
    """Request a logfile, using the standard path for an empty response."""
    if input_func is None:
        input_func = input

    try:
        response = input_func(
            f'Logging file [{DEFAULT_LOG_PATH}]: '
        ).strip()
    except EOFError:
        response = ''
    return response or DEFAULT_LOG_PATH


def _confirm_log_path_persistence(log_path, input_func=None):
    """Ask whether a prompted logfile should be saved to defaults.yml."""
    if input_func is None:
        input_func = input

    prompt = (
        f'Save logging path {log_path} to defaults.yml? [y/N] '
    )
    while True:
        try:
            response = input_func(prompt).strip().lower()
        except EOFError:
            response = ''

        if response in ('y', 'yes'):
            return True
        if response in ('', 'n', 'no'):
            return False
        print("Please answer 'yes' or 'no'.")


def _persist_log_path(config, log_path):
    """Save the user-facing path representation to the active defaults file."""
    import yaml

    config.DEFAULTS['logging'][LOGGING_CATEGORY] = log_path
    with open(config._get_defaults_fname(), 'w') as defaults_file:
        yaml.dump(
            config.DEFAULTS,
            defaults_file,
            sort_keys=False,
            default_flow_style=False,
        )


def _initialize_file_logging(log_path):
    """Attach one append-mode root handler for the normalized logfile path."""
    resolved_path = Path(log_path).expanduser().resolve()
    if resolved_path.exists() and not resolved_path.is_file():
        raise ValueError(f'Log path exists but is not a file: {resolved_path}')
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        handler_path = Path(handler.baseFilename).expanduser().resolve()
        if handler_path == resolved_path:
            return resolved_path

    handler = logging.FileHandler(resolved_path, mode='a', encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    ))
    root_logger.addHandler(handler)
    return resolved_path


def _configure_logging(config, command_log=None, input_func=None):
    """Resolve, initialize, and optionally persist the workflow logfile."""
    configured_log = config.DEFAULTS['logging'][LOGGING_CATEGORY]
    prompted = command_log is None and configured_log in (None, '')

    if command_log is not None:
        selected_log = command_log
    elif not prompted:
        selected_log = configured_log
    else:
        selected_log = _prompt_for_log_path(input_func=input_func)

    resolved_path = _initialize_file_logging(selected_log)

    if prompted and _confirm_log_path_persistence(
            selected_log, input_func=input_func):
        _persist_log_path(config, selected_log)

    return resolved_path


class GUI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, log_path=None):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.log_path = log_path
        
        ####  Setup addon features >>>
        self.ui.FileDrop.dragEnterEvent = self.dragEnterEvent
        self.ui.FileDrop.dropEvent = self.dropEvent
        self.ui.pb_DeleteAllEntries.clicked.connect(self.clear_all_entries)
        self.ui.pb_CheckData.clicked.connect(self.encode_and_qa_all)
        self.ui.pb_ReviewResults.clicked.connect(self.show_batch_results)
        self.ui.pb_ReviewResults.hide()
        self.ui.pb_LaunchBidsCreator.clicked.connect(self.open_bids_creator)
        self._last_batch_results = ()

        #### <<< 
        
        # This should have been part of UI design, but posthoc added
        self.ui.scrollAreaWidgetContents = QtWidgets.QListWidget()
        self.ui.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 853, 303))
        self.ui.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.ui.scrollArea.setWidget(self.ui.scrollAreaWidgetContents)
        
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()        
        
    def dropEvent(self, event):
        self.meg_files = [url.toLocalFile() for url in event.mimeData().urls()]
        print(f"Dropped: {', '.join(self.meg_files)}")
        self.populate_file_tiles()
        
    def populate_file_tiles(self):
        self._invalidate_batch_results()
        for i in self.meg_files: 
            try:
                # Instantiate a filename tile
                _tmp_tile = InputDatasetTile(fname=i)
            except BaseException as e:
                logger.exception('Could not prepare dataset %s', i)
                # Instantiate a NULL/ERROR tile
                _tmp_tile = ErrorDatasetTile(fname=i, 
                                             error_type=type(e), 
                                             error_code=str(e))
            
            # Add signal propagation from tile to MainWindow class
            _tmp_tile.close_clicked.connect(self.handle_close_request)
            
            # Do the weird Qt List Item/ItemWidget add
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(_tmp_tile.sizeHint())
            self.ui.scrollAreaWidgetContents.addItem(item)
            self.ui.scrollAreaWidgetContents.setItemWidget(item, _tmp_tile)
            
    def clear_all_entries(self):
        '''Delete all of the entries in the dataset list'''
        self._invalidate_batch_results()
        contents = self.ui.scrollAreaWidgetContents
        while contents.count():
            item = contents.takeItem(0)
    
    def get_fnames_from_list(self):
        '''Return a list of all filenames that have been dropped into list'''
        fname_list = []
        listwidget = self.ui.scrollAreaWidgetContents
        for i in range(listwidget.count()):
            item = listwidget.item(i)
            itemWidget = listwidget.itemWidget(item)
            fname_list.append(itemWidget.fname)
        return fname_list

    def encode_and_qa_all(self):
        '''Run trigger processing for every processable dataset tile.'''
        self._invalidate_batch_results()
        batch_results = []
        listwidget = self.ui.scrollAreaWidgetContents
        for i in range(listwidget.count()):
            item = listwidget.item(i)
            tile = listwidget.itemWidget(item)
            trigprocess = getattr(tile, 'trigprocess', None)
            if not callable(trigprocess):
                error = getattr(
                    tile,
                    'error_code',
                    'Dataset tile could not be initialized.',
                )
                batch_results.append(
                    self._batch_result_for_tile(tile, error=str(error))
                )
                continue
            try:
                processed = trigprocess()
            except Exception as error:
                logger.exception(
                    'Encode+QA failed for dataset %s',
                    getattr(tile, 'fname', '<unknown>'),
                )
                batch_results.append(
                    self._batch_result_for_tile(tile, error=str(error))
                )
                continue

            if processed is False or getattr(tile, '_trigproc_error', False):
                batch_results.append(self._batch_result_for_tile(
                    tile,
                    error='Trigger processing failed; event QA was not run.',
                ))
                continue

            qa_events = getattr(tile, 'qa_events', None)
            if callable(qa_events):
                try:
                    qa_events(show_error_dialog=False)
                except Exception as error:
                    logger.exception(
                        'Event QA failed for dataset %s',
                        getattr(tile, 'fname', '<unknown>'),
                    )
                    batch_results.append(self._batch_result_for_tile(
                        tile,
                        note=f'QA ERROR: {error}',
                    ))
                    continue
            batch_results.append(self._batch_result_for_tile(tile))

        self._last_batch_results = tuple(copy.deepcopy(batch_results))
        self.ui.pb_ReviewResults.setVisible(bool(self._last_batch_results))
        return self._last_batch_results

    def _invalidate_batch_results(self):
        """Clear the snapshot whenever the loaded tile collection changes."""
        self._last_batch_results = ()
        self.ui.pb_ReviewResults.hide()

    def _batch_result_for_tile(self, tile, error=None, note=None):
        """Capture one tile without retaining live widget or Raw references."""
        fname = getattr(tile, 'fname', '<unknown>')
        task = getattr(tile, 'taskname', None)
        run = getattr(tile, 'run', None)
        if task is None or run is None:
            basename = op.basename(str(fname).rstrip('/'))
            if '_task-' in basename:
                task = task or basename.split('_task-')[-1].split('_')[0]
                run = run or (
                    basename.split('_run-')[-1].split('_')[0]
                    if '_run-' in basename else 'NA'
                )
            else:
                parts = basename.split('_')
                task = task or (parts[1] if len(parts) >= 4 else 'DATASET')
                run = run or (
                    parts[-1].replace('.ds', '')
                    if len(parts) >= 4 else 'NA'
                )

        movement = getattr(tile, 'head_movement', None)
        if isinstance(movement, Number):
            movement_text = f'{movement:.2f} cm'
        elif movement not in (None, ''):
            movement_text = str(movement)
        else:
            movement_text = 'unavailable'

        duration = getattr(tile, 'duration_seconds', None)
        duration_text = f'{duration} s' if duration is not None else 'unavailable'
        base = {
            'task': str(task),
            'run': str(run),
            'dataset': op.basename(str(fname).rstrip('/')),
            'movement': movement_text,
            'duration': duration_text,
        }

        if error is not None:
            return {**base, 'kind': 'error', 'error': str(error)}

        qa_result = getattr(tile, '_qa_result', None)
        if qa_result is not None:
            return {
                **base,
                'kind': 'qa',
                'qa_result': copy.deepcopy(qa_result),
            }

        if note is None:
            qa_status = getattr(tile, '_qa_status', '')
            if qa_status.startswith('QA ERROR'):
                note = qa_status
            elif qa_status == 'No QA File':
                note = 'No compatible QA file; event counts shown.'
            else:
                note = 'No QA results; event counts shown.'
        raw = getattr(tile, 'raw', None)
        counts = _event_counts(raw) if raw is not None else ()
        return {
            **base,
            'kind': 'events',
            'note': note,
            'event_counts': counts,
        }

    def show_batch_results(self):
        """Display the immutable snapshot from the latest batch run."""
        if not self._last_batch_results:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Encode+QA Results')
        dialog.resize(900, 650)
        layout = QtWidgets.QVBoxLayout(dialog)

        result_text = QtWidgets.QPlainTextEdit(dialog)
        result_text.setObjectName('batch_qa_review_text')
        result_text.setReadOnly(True)
        result_text.setPlainText(_batch_review_text(
            self._last_batch_results
        ))
        fixed_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )
        result_text.setFont(fixed_font)
        layout.addWidget(result_text)

        close_button = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=dialog,
        )
        close_button.rejected.connect(dialog.reject)
        layout.addWidget(close_button)
        dialog.exec()

    def open_bids_creator(self):
        '''Open second window and populate the dataset list'''
        fnames = self.get_fnames_from_list()
        print('Opening bids app')
        logger.info('Opening BIDS Creator with %d dataset(s)', len(fnames))
        self._bids_window_open(meg_dsets = fnames)
        self.bids_gui.ui.list_fname_conversion.addItems(fnames)
        
        _meghash = self._assess_meghash(fnames)
        self.bids_gui.ui.te_meghash.setPlainText(_meghash)

    def restore_bids_creator(self, run_dict):
        """Open the BIDS Creator from recovery state and check its outputs."""
        logger.info('Opening BIDS Creator from RUNDICT')
        self._bids_window_open(run_dict=run_dict)
        self.bids_gui.check_restored_outputs()
    
    def _assess_meghash(self, fnames):
        try:
            _tmp = [op.basename(i).split('_')[0] for i in fnames]
            _tmp = set(_tmp)
            if (len(_tmp) > 1) or (len(_tmp)==0):
                return 'None'
            else:
                return list(_tmp)[0]
        except BaseException:
            logger.exception('Could not assess the MEG dataset identifier')
            print('Could not assess meghash')
            return 'None'
        
    
    def _bids_window_open(self, meg_dsets=None, run_dict=None):
        '''Implement the logic to create and maintain a second main window'''
        from nih2mne.GUI.templates.bids_creator_gui_control_functions import \
            BIDS_MainWindow as BIDS_Ui_MainWindow

        self.bids_gui = BIDS_Ui_MainWindow(
            meg_dsets=meg_dsets,
            run_dict=run_dict,
            log_path=self.log_path,
        )
        self.bids_gui.show()
        
    def handle_close_request(self, widget):
        '''If file tile "emits" a close signal, this will trigger a loop over
        filenames to identify the widget that produced the close signal'''
        self._invalidate_batch_results()
        item_count = self.ui.scrollAreaWidgetContents.count()
        for i in range(item_count):
            item = self.ui.scrollAreaWidgetContents.item(i)
            print(type(item))
            if self.ui.scrollAreaWidgetContents.itemWidget(item) == widget:
                self.ui.scrollAreaWidgetContents.takeItem(i)
                print(f"Parent removed item at row {i}")
                break
        
class InputDatasetTile(QtWidgets.QWidget):
    close_clicked = pyqtSignal(object)
    
    def __init__(self, parent=None, fname=None):
        super().__init__(parent)
        
        # Create instance of the UI class and set it up
        self.ui = Ui_InputDatasetTile()
        self.ui.setupUi(self)
        
        # Determine filename info
        if fname.endswith('/'): fname=fname[:-1] #Remove trailing slash - has caused problems on Mac
        self.fname = fname
        base_fname = op.basename(fname)
        _splits = base_fname.split('_')
        if len(_splits)==4:
            subjid = _splits[0]
            taskname = _splits[1]
            date = _splits[2]
            run = _splits[3].replace('.ds','')
        elif _splits[0].startswith('sub-'):
            subjid = _splits[0].replace('sub-','')
            taskname = fname.split('task-')[-1].split('_')[0]
            date = 'NA'
            if 'run-' in fname:
                run = fname.split('run-')[-1].split('_')[0]
            else:
                run = 'NA'
        else:
            subjid = 'NA'
            taskname = 'NA'
            date = 'NA'
            run = 'NA'
            
        self.load_meg()
        self.subjid = subjid
        self.taskname = taskname
        self.date = date
        self.run = run
        self.duration_seconds = round(self.raw.times[-1])
        self._trigproc_error = False  #Initialize, because referenced in info section
        self._qa_file = None
        self._qa_result = None
        self._qa_status = ''
        self._tile_initialized = False
        
        self.ui.ReadoutFilename.setText(f'File: {base_fname}')
        self.ui.ReadoutSubjid.setText(f'  Subjid: {subjid}')
        self.ui.ReadoutTaskname.setText(f'Task: {taskname}')
        self.ui.label_Duration.setText(f'Duration: {self.duration_seconds}s')
        
        ## Process Triggers
        self.ui.ProcFileComboBox.currentTextChanged.connect(
            self._on_procfile_changed
        )
        self.fill_procfile_list()
        self.ui.pb_TrigProcess.clicked.connect(self.trigprocess)
        self.ui.pb_QAEvts.clicked.connect(lambda: self.qa_events())
        self.ui.pb_ReviewQAFails.clicked.connect(self.show_qa_failures)
        self.ui.pb_ReviewQAFails.hide()
        
        ## Plotting
        self.ui.pb_PlotTrig.clicked.connect(self.plot_trig)
        self.ui.pb_PlotData.clicked.connect(self.plot_data)
        self.ui.pb_FFT.clicked.connect(self.plot_fft)
        
        ## Compute status info
        self._compute_movement()
        
        ## Info 
        self.set_events_label()
        self.set_status_label() 
        self._tile_initialized = True
        
        ## Remove tile
        self.ui.pb_DeleteTile.clicked.connect(lambda: self.close_clicked.emit(self))
        
        
        
    def load_meg(self):
        ''' Added as a method, so reloading data (updating annotation) can be done easily'''
        self.raw = mne.io.read_raw_ctf(self.fname, preload=False, 
                                  system_clock='ignore', clean_names=True)
    
    def _compute_movement(self):
        if shutil.which('calcHeadPos') == None:
            self.head_movement = 'NoCTFCode'
            return
        _has_hz = op.exists(op.join(self.fname, 'hz.ds'))
        _has_hz2 = op.exists(op.join(self.fname, 'hz2.ds'))
        if _has_hz and _has_hz2:
            try:
                dframe = get_localizer_dframe(self.fname)
                move_dict = compute_movement(dframe)
                self.head_movement = move_dict['Max']
            except:
                self.head_movement = 'Mvt - Hz Read Error'
        elif not _has_hz:
            self.head_movement = 'No hz.ds'
        elif not _has_hz2:
            self.head_movement = 'No hz2.ds'
    
    def _check_trailing_zeros(self):
        tmp_ = self.raw.copy().pick('meg')
        max_time_ = tmp_.times[-1]
        if max_time_ < 10:
            return None
        tmp_.crop(max_time_ - 10, None) #Pull just the last 10 seconds
        tmp_.load_data()
        test_vals = np.ones(tmp_._data.shape) * tmp_._data
        if (test_vals).sum() == 0.0:
            self.early_termination = True
        else:
            self.early_termination = False
        del tmp_
        
        
    def set_status_label(self):
        '''
        Set information on the status bar
        # check for default head transform
        '''
        status_text = ''
        if glob.glob(op.join(self.fname, 'MarkerFile.mrk')).__len__()==0:
            status_text+='No MrkFile! : '
        
        # Add movement info    
        if isinstance(self.head_movement, Number):
            _mvt_text = f'MVT={self.head_movement:.2f}cm'
        else:
            _mvt_text = self.head_movement
        status_text+=_mvt_text + ': '
        
        # Check early termination
        self._check_trailing_zeros()
        if self.early_termination:
            _term_text = f'Early Termination Detected: '
        else:
            _term_text = ''
        status_text+=_term_text
        
        # Check trigproc errors
        _trigproc_text=''
        if not self._trigproc_script_present:
            _trigproc_text = 'NoTrigProc Script: '
        else:
            if self._trigproc_error not in [None, False, '']:
                _trigproc_text = 'Trigproc Error (check terminal): ' # Eventually pipe in error text
        status_text += _trigproc_text

        if self._qa_status:
            status_text += self._qa_status + ': '
        
        self.ui.lbl_Status.setText(f'STATUS: {status_text}')
        
    def set_events_label(self):
        'Extract the annotations and list the counts in EventInfo'
        evt_dframe = pd.DataFrame(self.raw.annotations)
        if len(evt_dframe) != 0:
            val_counts = evt_dframe.description.value_counts()
            evt_text = ' '.join([f"{k}:({v})" for k, v in val_counts.items()])
            if len(evt_text) > 120:
                evt_text = evt_text[:120] + '...(trucated to fit)'
            self.ui.lbl_EventInfo.setText(f'EVENTS: {evt_text}')
        else:
            self.ui.lbl_EventInfo.setText(f'EVENTS: NONE')
    
    def plot_trig(self):
        tmp_ = self.raw.copy()
        tmp_.pick_types(misc=True, meg=False, eeg=False)
        if 'SCLK01' in tmp_.ch_names:
            _chs = copy.copy(tmp_.ch_names)
            _chs.remove('SCLK01')
            tmp_.pick(_chs)
        tmp_.load_data()
        tmp_.plot(scalings=10) #10 was empirically determined
        del tmp_
    
    def plot_data(self):
        if not _can_load(self.fname): 
            self.ui.pb_PlotData.setText('PlotData: Not enough RAM')
            return 
        tmp_ = self.raw.copy()
        tmp_.pick_types(meg=True)
        tmp_.load_data()
        tmp_.plot()
        del tmp_
        
    def plot_fft(self):
        'Generate and plot fft - remove early termination zeros if present'
        # Check if data can fit in RAM
        if not _can_load(self.fname): 
            self.ui.pb_FFT.setText('FFT: Not enough RAM')
            return 
        tmp_ = self.raw.copy()
        tmp_.pick_types(meg=True)
        tmp_.load_data()
        # Remove the excess zeros produced by early termination
        if self.early_termination:
            data = tmp_._data
            sfreq = tmp_.info['sfreq']
            test_channel_idx = 100
            _, term_time = get_term_time(data[test_channel_idx,:], sfreq)
            tmp_.crop(0, term_time)
        psd_ = tmp_.compute_psd(fmin=0, fmax=100)
        psd_.plot()
        del tmp_, psd_
    
    def fill_procfile_list(self):
        from nih2mne.config import TRIG_FILE_LOC

        if op.isdir(TRIG_FILE_LOC):
            self.trigfile_dir = TRIG_FILE_LOC
        else:
            self.trigfile_dir = None

        task_trig_files = _task_procfiles(
            self.trigfile_dir,
            self.taskname,
        )
        if task_trig_files:
            self.ui.ProcFileComboBox.addItems(task_trig_files)
            self._trigproc_script_present = True
        else:
            self._trigproc_script_present = False
        self._select_qa_file()

    def _on_procfile_changed(self, _procfile=None):
        """Refresh the compatible QA file when processing version changes."""
        self._select_qa_file()
        if self._tile_initialized:
            self.set_status_label()

    def _select_qa_file(self):
        """Select the newest QA version compatible with the ProcFile."""
        proc_file = self.ui.ProcFileComboBox.currentText()
        self._qa_file = find_qa_file(
            self.trigfile_dir,
            self.taskname,
            proc_file=proc_file,
        )
        self._qa_result = None
        self.ui.pb_ReviewQAFails.hide()
        if self._qa_file is None:
            self._qa_status = 'No QA File'
            self.ui.pb_QAEvts.setEnabled(False)
            self.ui.pb_QAEvts.setToolTip(
                'No compatible task QA file was found.'
            )
        else:
            self._qa_status = ''
            self.ui.pb_QAEvts.setEnabled(True)
            self.ui.pb_QAEvts.setToolTip(
                f'Run event QA using {op.basename(self._qa_file)}'
            )

    def _clear_qa_result(self):
        """Invalidate QA output after event processing changes the dataset."""
        self._qa_result = None
        self.ui.pb_ReviewQAFails.hide()
        self._qa_status = '' if self._qa_file else 'No QA File'

    def qa_events(self, show_error_dialog=True):
        """Run the selected event QA file and refresh tile status."""
        if self._qa_file is None:
            self._qa_status = 'No QA File'
            self.set_status_label()
            return None

        try:
            result = run_qa_file(
                self.fname,
                self._qa_file,
                task=self.taskname,
                logger=logger,
            )
        except Exception as error:
            self._qa_result = None
            self._qa_status = f'QA ERROR {op.basename(self._qa_file)}'
            self.ui.pb_ReviewQAFails.hide()
            logger.exception('Event QA failed for %s', self.fname)
            self.set_status_label()
            if show_error_dialog:
                QtWidgets.QMessageBox.critical(
                    self,
                    'Event QA Error',
                    str(error),
                )
            return None

        self._qa_result = result
        summary = result['summary']
        outcome = result['overall'].upper()
        self._qa_status = (
            f'QA {op.basename(result["qa_file"])}: {outcome} '
            f'(P={summary["pass"]} F={summary["fail"]} '
            f'S={summary["skipped"]})'
        )
        self.ui.pb_ReviewQAFails.setVisible(summary['fail'] > 0)
        self.set_status_label()
        return result

    def show_qa_failures(self):
        """Show all conditions from the most recent failed QA run."""
        if self._qa_result is None:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Event QA Results')
        dialog.resize(760, 480)
        layout = QtWidgets.QVBoxLayout(dialog)

        result_text = QtWidgets.QPlainTextEdit(dialog)
        result_text.setObjectName('qa_failure_review_text')
        result_text.setReadOnly(True)
        result_text.setPlainText(_qa_review_text(self._qa_result))
        fixed_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )
        result_text.setFont(fixed_font)
        layout.addWidget(result_text)

        close_button = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=dialog,
        )
        close_button.rejected.connect(dialog.reject)
        layout.addWidget(close_button)
        dialog.exec()
    
    def trigprocess(self):
        import subprocess
        import sys
        self._clear_qa_result()
        current_trigfile = self.ui.ProcFileComboBox.currentText()
        if current_trigfile.strip() == '':
            print(f'No associated trigger processing file for task: {self.taskname.lower()}')
        if (current_trigfile == None) or (current_trigfile == ""):
            self.set_status_label()
            return False
        if current_trigfile.endswith('.py'):
            _python_path = sys.executable
            cmd = f'{_python_path} {self.trigfile_dir}/{current_trigfile} {self.fname}'
        else:  #All non python files require executable permissions
            print('This file does not end with .py - It is required to have executable permissions')
            print(f'If an error occurs, this can be accomplished on the commandline with:  chmod +x {self.trigfile_dir}/{current_trigfile}')
            cmd = f'{self.trigfile_dir}/{current_trigfile} {self.fname}'
        _subproc = subprocess.run(cmd.split())  #Add errror processing
        if _subproc.returncode !=0:
            self._trigproc_error = True
        else:
            self._trigproc_error = False
        self.load_meg() #Reload to get the newly created annotations
        self.set_events_label()
        self.set_status_label() 
        return not self._trigproc_error

        
class ErrorDatasetTile(QtWidgets.QWidget):
    '''If an error occurs in making a dataset tile, this is an override to list 
    the error'''
    close_clicked = pyqtSignal(object)
    
    def __init__(self, parent=None, fname=None, error_type=None, 
                 error_code=None):
        super().__init__(parent)
        
        # Create instance of the UI class and set it up
        self.ui = Ui_ErrorDatasetTile()
        self.ui.setupUi(self)
        
        # Determine filename info
        self.fname = fname
        base_fname = op.basename(self.fname)
        self.ui.ReadoutFilename.setText(f'(!ERROR!) File: {base_fname}')
        
        ## Info 
        self.error_code = error_code
        self.error_type = error_type
        self.set_status_label() 
        
        ## Launch Error Code
        self.ui.pb_ReviewError.clicked.connect(self.show_error)
        
        ## Remove tile
        self.ui.pb_DeleteTile.clicked.connect(lambda: self.close_clicked.emit(self))
    
    def show_error(self):
        QtWidgets.QMessageBox.critical(self, 'Error', self.error_code)
    
    def set_status_label(self):
        '''
        Set information on the status bar
        '''
        status_text = self.error_type
        self.ui.lbl_Status.setText(f'STATUS: {status_text}')
        
        
    
        
# Helper functions        
  
def _assess_ram():
    if 'SLURM_JOB_ID' in os.environ:
        _slurm = True
    else:
        _slurm = False
        
    if _slurm: 
        _jobid = os.environ.get('SLURM_JOBID')
        _uid = os.environ.get('SLURM_JOB_UID')
        usage_f = f'/sys/fs/cgroup/memory/slurm/uid_{_uid}/job_{_jobid}/memory.memsw.usage_in_bytes'
        lim_f = f'/sys/fs/cgroup/memory/slurm/uid_{_uid}/job_{_jobid}/memory.limit_in_bytes'
        
        with open(usage_f) as f: _usage=f.readline().replace("\n","")
        with open(lim_f) as f: _lim=f.readline().replace("\n","")
        
        avail_mem = int(_lim) - int(_usage)
    else:
        import psutil
        mem = psutil.virtual_memory()
        avail_mem = mem.available
    return avail_mem

def _get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip if it's a symbolic link
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def _can_load(fname):
    f_size = _get_folder_size(fname) #.ds meg files are directories
    if _assess_ram() > f_size:
        return True
    else:
        print('Cannot load due to RAM size limitations')
        return False
        
        


def main(argv=None):
    parser = _get_parser()
    args = parser.parse_args(argv)

    try:
        config = _initialize_config(args.config)
    except _ConfigCreationDeclined:
        parser.exit(1, 'Config file was not created; exiting.\n')
    except ValueError as error:
        parser.error(str(error))

    _configure_bids_root(
        config=config,
        bids_root=args.bids_root,
        config_fname=args.config,
    )

    try:
        log_path = _configure_logging(config=config, command_log=args.log)
    except (OSError, ValueError) as error:
        parser.error(f'Could not initialize logging: {error}')
    logger.info('Starting meg_dataset_gui; logging to %s', log_path)

    run_dict = None
    if args.rundict is not None:
        from nih2mne.GUI.templates.bids_creator_gui_control_functions import \
            parse_run_dict
        try:
            run_dict = parse_run_dict(args.rundict)
        except ValueError as error:
            parser.error(str(error))

    # All supported command-line arguments have already been consumed. Avoid
    # forwarding ``-config`` to Qt's independent argument parser.
    app = QtWidgets.QApplication([sys.argv[0]])
    # Add App Icon
    icon_img = op.join(op.dirname(__file__), 'templates', 'opposum_squid_icon.png')
    if op.exists(icon_img): app.setWindowIcon(QtGui.QIcon(icon_img))
    
    MainWindow = GUI_MainWindow(log_path=log_path)
    MainWindow.show()
    if run_dict is not None:
        MainWindow.restore_bids_creator(run_dict)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
