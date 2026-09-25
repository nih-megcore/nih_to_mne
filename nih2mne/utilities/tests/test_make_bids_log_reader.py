#!/usr/bin/env python3

from datetime import datetime
from io import StringIO
from pathlib import Path

import pytest
import yaml

from nih2mne.utilities import make_bids_log_reader as reader


def _run_dict(
        bids_id='01', meghash='HASH01', bids_root='/data/bids',
        include_mri=True):
    return {
        'anonymize': False,
        'subjid_input': meghash,
        'bids_id': bids_id,
        'bids_dir': bids_root,
        'bids_session': '1',
        'meg_dataset_list': [
            f'/input/{meghash}_task_20260925_01.ds',
            f'/input/{meghash}_rest_20260925_02.ds',
        ],
        'mri_none': not include_mri,
        'mri_bsight': '/input/anat.nii.gz' if include_mri else False,
        'mri_elec': '/input/electrodes.txt' if include_mri else False,
        'mri_brik': False,
        'crop_zeros': True,
        'include_empty_room': False,
        'run_rank_reorder': True,
    }


def _log_line(timestamp, message, logger='bids.gui'):
    return f'{timestamp} - INFO - {logger} - {message}\n'


def _entry_lines(timestamp, run_dict, include_outputs=True):
    lines = [
        _log_line(
            timestamp,
            f'RUNDICT: {reader.serialize_run_dict(run_dict)}',
        )
    ]
    if include_outputs:
        for index, input_path in enumerate(run_dict['meg_dataset_list'], start=1):
            output = (
                f'{run_dict["bids_dir"]}/sub-{run_dict["bids_id"]}/ses-01/meg/'
                f'sub-{run_dict["bids_id"]}_ses-01_task-test_run-0{index}_meg.ds'
            )
            lines.append(_log_line(
                timestamp,
                f'Converting MEG dataset {input_path} to {output}',
            ))
        if run_dict['mri_bsight']:
            output = (
                f'{run_dict["bids_dir"]}/sub-{run_dict["bids_id"]}/ses-01/anat/'
                f'sub-{run_dict["bids_id"]}_ses-01_run-01_T1w.nii.gz'
            )
            lines.append(_log_line(
                timestamp,
                f'Converting MRI dataset {run_dict["mri_bsight"]} to {output}',
            ))
    return lines


def test_parse_log_groups_runs_and_all_referenced_files():
    first = _run_dict(bids_id='01', meghash='HASH01')
    second = _run_dict(bids_id='02', meghash='HASH02', include_mri=False)
    lines = _entry_lines('2026-09-20 09:00:00,000', first)
    lines += ['Traceback (most recent call last):\n', '  ignored\n']
    lines += _entry_lines('2026-09-25 10:00:00,000', second, False)

    result = reader.parse_log_lines(lines)

    assert result.warnings == []
    assert len(result.entries) == 2
    mappings = result.entries[0].file_mappings()
    assert [mapping.input_path for mapping in mappings] == [
        *first['meg_dataset_list'],
        first['mri_bsight'],
        first['mri_elec'],
    ]
    assert all(mapping.output_path for mapping in mappings[:3])
    assert mappings[-1].auxiliary is True
    assert all(
        mapping.output_path is None
        for mapping in result.entries[1].file_mappings()
    )


def test_malformed_rundict_warns_and_does_not_capture_following_mapping():
    valid = _run_dict()
    lines = [
        _log_line('2026-09-20 09:00:00,000', 'RUNDICT: {invalid}'),
        _log_line(
            '2026-09-20 09:00:01,000',
            'Converting MEG dataset /wrong.ds to /wrong/output.ds',
        ),
        *_entry_lines('2026-09-21 09:00:00,000', valid, False),
    ]

    result = reader.parse_log_lines(lines)

    assert len(result.warnings) == 1
    assert result.warnings[0].startswith('Line 1:')
    assert len(result.entries) == 1
    assert result.entries[0].outputs == {}


@pytest.mark.parametrize(
    'text, expected',
    [
        ('09/01/2026-09/25/2026', (2026, 9, 1, 2026, 9, 25)),
        ('"09/01/2026-09/25/2026"', (2026, 9, 1, 2026, 9, 25)),
        ("'9/1/2026-9/25/2026'", (2026, 9, 1, 2026, 9, 25)),
    ],
)
def test_parse_date_range_accepts_quoted_and_unquoted_values(text, expected):
    start, end = reader.parse_date_range(text)
    assert (start.year, start.month, start.day,
            end.year, end.month, end.day) == expected


@pytest.mark.parametrize(
    'text, match',
    [
        ('09/01/2026', 'must use'),
        ('13/01/2026-09/25/2026', 'Invalid date'),
        ('09/25/2026-09/01/2026', 'cannot be before'),
    ],
)
def test_parse_date_range_rejects_invalid_values(text, match):
    with pytest.raises(ValueError, match=match):
        reader.parse_date_range(text)


@pytest.mark.parametrize(
    'text, expected',
    [
        ('all', None),
        ('', None),
        ('01', {'01'}),
        ('sub-01', {'01'}),
        ('"sub-01,02"', {'01', '02'}),
    ],
)
def test_parse_bids_id_filter_accepts_all_prefixes_and_lists(text, expected):
    assert reader.parse_bids_id_filter(text) == expected


@pytest.mark.parametrize('text', ['all,01', '01,,02', 'sub-'])
def test_parse_bids_id_filter_rejects_invalid_lists(text):
    with pytest.raises(ValueError):
        reader.parse_bids_id_filter(text)


@pytest.mark.parametrize(
    'text, expected',
    [
        ('all', None),
        ('', None),
        ('HASH01', {'hash01'}),
        ('"HASH01,hash02"', {'hash01', 'hash02'}),
    ],
)
def test_parse_hash_id_filter_accepts_all_and_lists(text, expected):
    assert reader.parse_hash_id_filter(text) == expected


@pytest.mark.parametrize('text', ['all,HASH01', 'HASH01,,HASH02'])
def test_parse_hash_id_filter_rejects_invalid_lists(text):
    with pytest.raises(ValueError):
        reader.parse_hash_id_filter(text)


def test_bids_id_filter_accepts_prefixed_logged_ids():
    entries = [
        reader.ConversionEntry(
            datetime(2026, 9, 20), _run_dict('sub-01', 'A'), 0
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 21), _run_dict('02', 'B'), 1
        ),
    ]

    selected = reader.filter_entries_by_bids_id(entries, {'01'})

    assert selected == [entries[0]]


def test_hash_id_filter_is_case_insensitive_and_intersects_subject_filter():
    entries = [
        reader.ConversionEntry(
            datetime(2026, 9, 20), _run_dict('01', 'HASH01'), 0
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 21), _run_dict('01', 'HASH02'), 1
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 22), _run_dict('02', 'HASH01'), 2
        ),
    ]

    selected = reader.filter_entries_by_bids_id(entries, {'01'})
    selected = reader.filter_entries_by_hash_id(selected, {'hash02'})

    assert selected == [entries[1]]


def test_default_week_is_anchored_to_latest_log_and_sorted_oldest_first():
    entries = [
        reader.ConversionEntry(
            datetime(2026, 8, 1, 12), _run_dict('03', 'OLD'), 0
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 19, 12), _run_dict('02', 'START'), 1
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 24, 12), _run_dict('01', 'MIDDLE'), 2
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 25, 12), _run_dict('04', 'LATEST'), 3
        ),
    ]

    selected = reader.select_entries(entries, view='week')
    selected = reader.sort_entries(selected, sort_by='date')

    assert [entry.meghash for entry in selected] == [
        'START', 'MIDDLE', 'LATEST'
    ]
    assert selected[-1].timestamp == max(entry.timestamp for entry in entries)


def test_explicit_range_is_inclusive_and_last_uses_log_order():
    entries = [
        reader.ConversionEntry(
            datetime(2026, 9, day, 12), _run_dict(str(day), f'HASH{day}'), index
        )
        for index, day in enumerate((1, 10, 20, 25))
    ]

    selected = reader.select_entries(
        entries,
        view='range',
        date_range=reader.parse_date_range('09/10/2026-09/20/2026'),
    )

    assert [entry.timestamp.day for entry in selected] == [10, 20]
    assert reader.select_entries(entries, view='last') == [entries[-1]]


def test_subject_sorts_group_then_order_each_group_by_date():
    entries = [
        reader.ConversionEntry(
            datetime(2026, 9, 25), _run_dict('02', 'A'), 0
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 20), _run_dict('01', 'Z'), 1
        ),
        reader.ConversionEntry(
            datetime(2026, 9, 24), _run_dict('01', 'B'), 2
        ),
    ]

    by_bids = reader.sort_entries(entries, 'bids_id')
    by_hash = reader.sort_entries(entries, 'meghash')

    assert [(entry.bids_id, entry.timestamp.day) for entry in by_bids] == [
        ('01', 20), ('01', 24), ('02', 25)
    ]
    assert [entry.meghash for entry in by_hash] == ['A', 'B', 'Z']


def test_format_entry_uses_safe_bids_placeholder_and_auxiliary_label():
    run_dict = _run_dict()
    entry = reader.ConversionEntry(
        datetime(2026, 9, 25, 10, 30), run_dict, 0,
        outputs={
            run_dict['meg_dataset_list'][0]: '/data/bids/sub-01/meg/run.ds',
            run_dict['meg_dataset_list'][1]: '/data/bids-other/run.ds',
            run_dict['mri_bsight']: '/data/bids/sub-01/anat/T1w.nii.gz',
        },
    )

    output = reader.format_entry(entry, terminal_width=80)

    assert 'Conversion date: 09/25/2026 10:30:00' in output
    assert 'MEGHash -> BIDS ID: HASH01 -> 01' in output
    assert 'BIDS_DIR/sub-01/meg/run.ds' in output
    assert '/data/bids-other/run.ds' in output
    assert '— (auxiliary input)' in output
    assert max(len(line) for line in output.splitlines()[4:]) <= 80


def test_resolve_log_path_precedence(tmp_path, monkeypatch):
    configured_log = tmp_path / 'configured.log'
    cli_log = tmp_path / 'cli.log'
    config_path = tmp_path / 'defaults.yml'
    config_path.write_text(yaml.safe_dump({
        'logging': {'meg_dataset_gui': str(configured_log)}
    }), encoding='utf-8')
    monkeypatch.setenv('MEGCORE_DEFAULTS_FNAME', str(config_path))

    assert reader.resolve_log_path() == configured_log
    assert reader.resolve_log_path(cli_log) == cli_log


def test_main_defaults_to_latest_week_and_date_sort(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    lines = []
    lines += _entry_lines(
        '2026-08-01 09:00:00,000', _run_dict('99', 'OLD'), False
    )
    lines += _entry_lines(
        '2026-09-20 09:00:00,000', _run_dict('02', 'EARLIER'), False
    )
    lines += _entry_lines(
        '2026-09-25 09:00:00,000', _run_dict('01', 'LATEST'), False
    )
    log_path.write_text(''.join(lines), encoding='utf-8')
    responses = iter(['', '', '', ''])
    stdout = StringIO()

    status = reader.main(
        ['-log', str(log_path)],
        input_func=lambda _prompt: next(responses),
        stdout=stdout,
    )

    rendered = stdout.getvalue()
    assert status == 0
    assert 'OLD' not in rendered
    assert rendered.index('EARLIER') < rendered.index('LATEST')


def test_main_cli_range_and_sort_match_interactive_features(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    lines = []
    lines += _entry_lines(
        '2026-09-10 09:00:00,000', _run_dict('02', 'A'), False
    )
    lines += _entry_lines(
        '2026-09-20 09:00:00,000', _run_dict('01', 'Z'), False
    )
    lines += _entry_lines(
        '2026-09-25 09:00:00,000', _run_dict('03', 'B'), False
    )
    log_path.write_text(''.join(lines), encoding='utf-8')
    stdout = StringIO()

    status = reader.main(
        [
            '-log', str(log_path),
            '-date_range', '"09/01/2026-09/20/2026"',
            '-bids_id', 'all',
            '-hash_id', 'all',
            '-sort', 'bids_id',
        ],
        input_func=lambda _prompt: pytest.fail('unexpected prompt'),
        stdout=stdout,
    )

    rendered = stdout.getvalue()
    assert status == 0
    assert 'MEGHash -> BIDS ID: B -> 03' not in rendered
    assert (
        rendered.index('MEGHash -> BIDS ID: Z -> 01')
        < rendered.index('MEGHash -> BIDS ID: A -> 02')
    )


def test_main_tui_range_reprompts_and_applies_selected_sort(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    lines = []
    lines += _entry_lines(
        '2026-09-10 09:00:00,000', _run_dict('02', 'Z'), False
    )
    lines += _entry_lines(
        '2026-09-20 09:00:00,000', _run_dict('01', 'A'), False
    )
    lines += _entry_lines(
        '2026-09-25 09:00:00,000', _run_dict('03', 'OUTSIDE'), False
    )
    log_path.write_text(''.join(lines), encoding='utf-8')
    responses = iter([
        'range',
        'not-a-range',
        '09/01/2026-09/20/2026',
        '',
        '',
        'meghash',
    ])
    prompts = []
    stdout = StringIO()

    def respond(prompt):
        prompts.append(prompt)
        return next(responses)

    status = reader.main(
        ['-log', str(log_path)],
        input_func=respond,
        stdout=stdout,
    )

    rendered = stdout.getvalue()
    assert status == 0
    assert prompts[0].startswith('View ')
    assert prompts[1].startswith('Date range ')
    assert prompts[2].startswith('Date range ')
    assert prompts[3].startswith('BIDS subject ID')
    assert prompts[4].startswith('MEGHash ID')
    assert prompts[5].startswith('Sort ')
    assert 'Date range must use MM/DD/YYYY-MM/DD/YYYY format' in rendered
    assert 'OUTSIDE' not in rendered
    assert (
        rendered.index('MEGHash -> BIDS ID: A -> 01')
        < rendered.index('MEGHash -> BIDS ID: Z -> 02')
    )


def test_main_returns_one_for_empty_selected_range(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    log_path.write_text(''.join(_entry_lines(
        '2026-09-25 09:00:00,000', _run_dict(), False
    )), encoding='utf-8')
    stdout = StringIO()

    status = reader.main(
        [
            '-log', str(log_path),
            '-date_range', '01/01/2020-01/02/2020',
            '-bids_id', 'all',
            '-hash_id', 'all',
            '-sort', 'date',
        ],
        stdout=stdout,
    )

    assert status == 1
    assert 'No conversion entries found' in stdout.getvalue()


def test_main_subject_filter_precedes_latest_week_selection(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    lines = []
    lines += _entry_lines(
        '2026-09-01 09:00:00,000', _run_dict('01', 'SUBJECT1_OLD'), False
    )
    lines += _entry_lines(
        '2026-09-10 09:00:00,000', _run_dict('01', 'SUBJECT1_NEW'), False
    )
    lines += _entry_lines(
        '2026-09-25 09:00:00,000', _run_dict('02', 'GLOBAL_NEW'), False
    )
    log_path.write_text(''.join(lines), encoding='utf-8')
    stdout = StringIO()

    status = reader.main(
        [
            '-log', str(log_path),
            '-view', 'week',
            '-bids_id', 'sub-01',
            '-hash_id', 'all',
            '-sort', 'date',
        ],
        input_func=lambda _prompt: pytest.fail('unexpected prompt'),
        stdout=stdout,
    )

    rendered = stdout.getvalue()
    assert status == 0
    assert 'SUBJECT1_NEW' in rendered
    assert 'SUBJECT1_OLD' not in rendered
    assert 'GLOBAL_NEW' not in rendered


def test_main_tui_subject_filter_defaults_to_all_or_accepts_one(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    lines = []
    lines += _entry_lines(
        '2026-09-24 09:00:00,000', _run_dict('01', 'FIRST'), False
    )
    lines += _entry_lines(
        '2026-09-25 09:00:00,000', _run_dict('02', 'SECOND'), False
    )
    log_path.write_text(''.join(lines), encoding='utf-8')
    responses = iter(['', 'sub-02', '', ''])
    stdout = StringIO()

    status = reader.main(
        ['-log', str(log_path)],
        input_func=lambda _prompt: next(responses),
        stdout=stdout,
    )

    rendered = stdout.getvalue()
    assert status == 0
    assert 'MEGHash -> BIDS ID: SECOND -> 02' in rendered
    assert 'MEGHash -> BIDS ID: FIRST -> 01' not in rendered


def test_main_hash_filter_precedes_latest_week_selection(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    lines = []
    lines += _entry_lines(
        '2026-09-01 09:00:00,000', _run_dict('01', 'HASH01'), False
    )
    lines += _entry_lines(
        '2026-09-10 09:00:00,000', _run_dict('02', 'HASH01'), False
    )
    lines += _entry_lines(
        '2026-09-25 09:00:00,000', _run_dict('03', 'OTHER'), False
    )
    log_path.write_text(''.join(lines), encoding='utf-8')
    stdout = StringIO()

    status = reader.main(
        [
            '-log', str(log_path),
            '-view', 'week',
            '-bids_id', 'all',
            '-hash_id', 'hash01',
            '-sort', 'date',
        ],
        input_func=lambda _prompt: pytest.fail('unexpected prompt'),
        stdout=stdout,
    )

    rendered = stdout.getvalue()
    assert status == 0
    assert 'MEGHash -> BIDS ID: HASH01 -> 02' in rendered
    assert 'MEGHash -> BIDS ID: HASH01 -> 01' not in rendered
    assert 'OTHER' not in rendered


def test_main_tui_hash_filter_defaults_to_all_or_accepts_one(tmp_path):
    log_path = tmp_path / 'bids_conversion.log'
    lines = []
    lines += _entry_lines(
        '2026-09-24 09:00:00,000', _run_dict('01', 'FIRST'), False
    )
    lines += _entry_lines(
        '2026-09-25 09:00:00,000', _run_dict('02', 'SECOND'), False
    )
    log_path.write_text(''.join(lines), encoding='utf-8')
    responses = iter(['', '', 'first', ''])
    stdout = StringIO()

    status = reader.main(
        ['-log', str(log_path)],
        input_func=lambda _prompt: next(responses),
        stdout=stdout,
    )

    rendered = stdout.getvalue()
    assert status == 0
    assert 'MEGHash -> BIDS ID: FIRST -> 01' in rendered
    assert 'MEGHash -> BIDS ID: SECOND -> 02' not in rendered
