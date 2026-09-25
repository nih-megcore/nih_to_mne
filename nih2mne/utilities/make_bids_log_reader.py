#!/usr/bin/env python3
"""Review BIDS conversions recorded in the dataset GUI logfile."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap

import yaml


DEFAULT_LOG_PATH = '~/megcore/logging/bids_conversion.log'
RUNDICT_MARKER = 'RUNDICT:'
RUNDICT_KEYS = {
    'anonymize',
    'subjid_input',
    'bids_id',
    'bids_dir',
    'bids_session',
    'meg_dataset_list',
    'mri_none',
    'mri_bsight',
    'mri_elec',
    'mri_brik',
    'crop_zeros',
    'include_empty_room',
    'run_rank_reorder',
}
LOG_LINE_PATTERN = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
    r' - (?P<level>[^ ]+) - (?P<logger>.*?) - (?P<message>.*)$'
)
LOG_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S,%f'
DISPLAY_TIMESTAMP_FORMAT = '%m/%d/%Y %H:%M:%S'
DATE_FORMAT = '%m/%d/%Y'


@dataclass
class FileMapping:
    """One input referenced by a conversion and its logged output."""

    input_path: str
    output_path: str | None = None
    auxiliary: bool = False


@dataclass
class ConversionEntry:
    """A RUNDICT and the conversion messages associated with it."""

    timestamp: datetime
    run_dict: dict
    sequence: int
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def bids_root(self) -> str:
        return self.run_dict['bids_dir']

    @property
    def bids_id(self) -> str:
        return self.run_dict['bids_id']

    @property
    def meghash(self) -> str:
        return self.run_dict['subjid_input']

    def file_mappings(self) -> list[FileMapping]:
        """Return converted and auxiliary inputs in GUI processing order."""
        mappings = [
            FileMapping(path, self._find_output(path))
            for path in self.run_dict['meg_dataset_list']
        ]
        for key in ('mri_bsight', 'mri_brik'):
            path = self.run_dict[key]
            if path:
                mappings.append(FileMapping(path, self._find_output(path)))
        electrode_path = self.run_dict['mri_elec']
        if electrode_path:
            mappings.append(FileMapping(electrode_path, auxiliary=True))
        return mappings

    def _find_output(self, input_path: str) -> str | None:
        if input_path in self.outputs:
            return self.outputs[input_path]
        normalized_input = os.path.normpath(input_path)
        for logged_input, output in self.outputs.items():
            if os.path.normpath(logged_input) == normalized_input:
                return output
        return None


@dataclass
class LogParseResult:
    """Valid conversion entries plus recoverable parsing warnings."""

    entries: list[ConversionEntry]
    warnings: list[str]


def serialize_run_dict(run_dict):
    """Serialize BIDS Creator options as deterministic, single-line JSON."""
    validated = validate_run_dict(run_dict)
    return json.dumps(validated, sort_keys=True, separators=(',', ':'))


def parse_run_dict(record):
    """Parse JSON alone or extract it from a complete RUNDICT log line."""
    if not isinstance(record, str) or not record.strip():
        raise ValueError('RUNDICT input is empty')

    payload = record.strip()
    if RUNDICT_MARKER in payload:
        payload = payload.rsplit(RUNDICT_MARKER, 1)[1].strip()

    try:
        run_dict = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f'RUNDICT is not valid JSON: {error.msg}') from error
    return validate_run_dict(run_dict)


def validate_run_dict(run_dict):
    """Validate and copy the complete BIDS Creator recovery state."""
    if not isinstance(run_dict, dict):
        raise ValueError('RUNDICT must be a JSON object')

    missing = RUNDICT_KEYS.difference(run_dict)
    unknown = set(run_dict).difference(RUNDICT_KEYS)
    if missing:
        raise ValueError(
            f'RUNDICT is missing required keys: {", ".join(sorted(missing))}'
        )
    if unknown:
        raise ValueError(
            f'RUNDICT contains unknown keys: {", ".join(sorted(unknown))}'
        )

    for key in (
        'anonymize', 'mri_none', 'crop_zeros', 'include_empty_room',
        'run_rank_reorder',
    ):
        if not isinstance(run_dict[key], bool):
            raise ValueError(f'RUNDICT {key} must be true or false')

    for key in ('subjid_input', 'bids_id', 'bids_dir', 'bids_session'):
        if not isinstance(run_dict[key], str):
            raise ValueError(f'RUNDICT {key} must be a string')

    datasets = run_dict['meg_dataset_list']
    if not isinstance(datasets, list) or not all(
            isinstance(dataset, str) for dataset in datasets):
        raise ValueError('RUNDICT meg_dataset_list must be a list of strings')

    for key in ('mri_bsight', 'mri_elec', 'mri_brik'):
        value = run_dict[key]
        if value is not None and value is not False and not isinstance(value, str):
            raise ValueError(f'RUNDICT {key} must be a path, false, or null')

    return json.loads(json.dumps(run_dict))


def parse_log_lines(lines) -> LogParseResult:
    """Parse conversion entries from an iterable of logfile lines."""
    entries = []
    warnings = []
    current_entry = None

    for line_number, line in enumerate(lines, start=1):
        match = LOG_LINE_PATTERN.match(line.rstrip('\n'))
        if match is None:
            continue
        message = match.group('message')

        if RUNDICT_MARKER in message:
            current_entry = None
            try:
                run_dict = parse_run_dict(message)
                timestamp = datetime.strptime(
                    match.group('timestamp'), LOG_TIMESTAMP_FORMAT
                )
            except ValueError as error:
                warnings.append(f'Line {line_number}: {error}')
                continue
            current_entry = ConversionEntry(
                timestamp=timestamp,
                run_dict=run_dict,
                sequence=len(entries),
            )
            entries.append(current_entry)
            continue

        if current_entry is None:
            continue
        for prefix in ('Converting MEG dataset ', 'Converting MRI dataset '):
            if not message.startswith(prefix):
                continue
            mapping = message[len(prefix):]
            if ' to ' not in mapping:
                warnings.append(
                    f'Line {line_number}: conversion mapping has no output'
                )
                break
            input_path, output_path = mapping.rsplit(' to ', 1)
            current_entry.outputs[input_path] = output_path
            break

    return LogParseResult(entries=entries, warnings=warnings)


def parse_log_file(log_path) -> LogParseResult:
    """Read and parse a BIDS conversion logfile."""
    with Path(log_path).open('r', encoding='utf-8') as log_file:
        return parse_log_lines(log_file)


def parse_date_range(value: str) -> tuple[date, date]:
    """Parse an inclusive MM/DD/YYYY-MM/DD/YYYY range."""
    if not isinstance(value, str):
        raise ValueError('Date range must be text')
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    match = re.fullmatch(
        r'(?P<start>\d{1,2}/\d{1,2}/\d{4})\s*-\s*'
        r'(?P<end>\d{1,2}/\d{1,2}/\d{4})',
        value,
    )
    if match is None:
        raise ValueError(
            'Date range must use MM/DD/YYYY-MM/DD/YYYY format'
        )
    try:
        start = datetime.strptime(match.group('start'), DATE_FORMAT).date()
        end = datetime.strptime(match.group('end'), DATE_FORMAT).date()
    except ValueError as error:
        raise ValueError(f'Invalid date range: {error}') from error
    if end < start:
        raise ValueError('Date range end cannot be before its start')
    return start, end


def _parse_id_filter(value: str, label: str, strip_sub_prefix=False):
    """Parse ``all`` or a comma-separated collection of identifiers."""
    if not isinstance(value, str):
        raise ValueError(f'{label} filter must be text')
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    if not value or value.casefold() == 'all':
        return None

    identifiers = set()
    for item in value.split(','):
        item = item.strip()
        if not item:
            raise ValueError(f'{label} IDs cannot be empty')
        if item.casefold() == 'all':
            raise ValueError(
                f'Use all by itself, not with individual {label} IDs'
            )
        if strip_sub_prefix and item.casefold().startswith('sub-'):
            item = item[4:]
        if not item:
            raise ValueError(f'{label} IDs cannot be empty')
        identifiers.add(item.casefold())
    return identifiers


def parse_bids_id_filter(value: str) -> set[str] | None:
    """Parse ``all`` or a comma-separated collection of BIDS subject IDs."""
    return _parse_id_filter(value, 'BIDS subject', strip_sub_prefix=True)


def parse_hash_id_filter(value: str) -> set[str] | None:
    """Parse ``all`` or a comma-separated collection of MEGHash IDs."""
    return _parse_id_filter(value, 'MEGHash')


def filter_entries_by_bids_id(entries, bids_ids=None):
    """Return every entry or only entries matching normalized BIDS IDs."""
    entries = list(entries)
    if bids_ids is None:
        return entries
    return [
        entry for entry in entries
        if entry.bids_id.casefold().removeprefix('sub-') in bids_ids
    ]


def filter_entries_by_hash_id(entries, hash_ids=None):
    """Return every entry or only entries matching MEGHash IDs."""
    entries = list(entries)
    if hash_ids is None:
        return entries
    return [
        entry for entry in entries
        if entry.meghash.casefold() in hash_ids
    ]


def latest_week_range(entries) -> tuple[date, date]:
    """Return seven calendar dates ending on the newest conversion date."""
    if not entries:
        raise ValueError('No conversion entries are available')
    end = max(entry.timestamp for entry in entries).date()
    return end - timedelta(days=6), end


def select_entries(entries, view='week', date_range=None):
    """Apply the selected time view while preserving logfile order."""
    entries = list(entries)
    if not entries:
        return []
    if view == 'last':
        return [entries[-1]]
    if view == 'all':
        return entries
    if view == 'week':
        start, end = latest_week_range(entries)
    elif view == 'range':
        if date_range is None:
            raise ValueError('A date range is required for the range view')
        start, end = date_range
    else:
        raise ValueError(f'Unknown view: {view}')
    return [entry for entry in entries if start <= entry.timestamp.date() <= end]


def sort_entries(entries, sort_by='date'):
    """Sort chronologically or group by a source/destination subject ID."""
    if sort_by == 'date':
        key = lambda entry: (entry.timestamp, entry.sequence)
    elif sort_by == 'bids_id':
        key = lambda entry: (
            entry.bids_id.casefold(), entry.timestamp, entry.sequence
        )
    elif sort_by == 'meghash':
        key = lambda entry: (
            entry.meghash.casefold(), entry.timestamp, entry.sequence
        )
    else:
        raise ValueError(f'Unknown sort: {sort_by}')
    return sorted(entries, key=key)


def _display_output(mapping: FileMapping, bids_root: str) -> str:
    if mapping.auxiliary:
        return '— (auxiliary input)'
    if mapping.output_path is None:
        return '— (output not logged)'

    output = Path(mapping.output_path)
    root = Path(bids_root)
    try:
        relative = output.relative_to(root)
    except ValueError:
        return mapping.output_path
    return f'BIDS_DIR/{relative.as_posix()}'


def _wrap_cell(value: str, width: int) -> list[str]:
    return textwrap.wrap(
        value,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or ['']


def format_entry(entry: ConversionEntry, terminal_width=120) -> str:
    """Format one conversion as metadata and a two-column table."""
    terminal_width = max(49, terminal_width)
    column_width = max(22, (terminal_width - 3) // 2)
    header = f'{"Input file":<{column_width}} | Output file'
    separator = f'{"-" * column_width}-+-{"-" * column_width}'
    table_lines = [header, separator]

    for mapping in entry.file_mappings():
        input_lines = _wrap_cell(mapping.input_path, column_width)
        output_lines = _wrap_cell(
            _display_output(mapping, entry.bids_root), column_width
        )
        for index in range(max(len(input_lines), len(output_lines))):
            input_text = input_lines[index] if index < len(input_lines) else ''
            output_text = output_lines[index] if index < len(output_lines) else ''
            table_lines.append(
                f'{input_text:<{column_width}} | {output_text}'
            )

    metadata = [
        f'Conversion date: {entry.timestamp.strftime(DISPLAY_TIMESTAMP_FORMAT)}',
        f'BIDS root: {entry.bids_root}',
        f'MEGHash -> BIDS ID: {entry.meghash} -> {entry.bids_id}',
        '',
    ]
    return '\n'.join(metadata + table_lines)


def format_entries(entries, terminal_width=None) -> str:
    """Format multiple conversions with a visible separator."""
    if terminal_width is None:
        terminal_width = shutil.get_terminal_size((120, 24)).columns
    divider = '=' * min(max(49, terminal_width), 120)
    return f'\n{divider}\n\n'.join(
        format_entry(entry, terminal_width=terminal_width) for entry in entries
    )


def resolve_log_path(log_path=None, config_path=None) -> Path:
    """Resolve CLI, defaults.yml, and standard logfile precedence."""
    if log_path is not None:
        return Path(log_path).expanduser().resolve()

    explicit_config = config_path is not None
    if config_path is None:
        config_path = os.environ.get(
            'MEGCORE_DEFAULTS_FNAME', '~/megcore/defaults.yml'
        )
    config_path = Path(config_path).expanduser().resolve()
    if explicit_config and not config_path.is_file():
        raise ValueError(f'Config file does not exist: {config_path}')

    configured_log = None
    if config_path.is_file():
        try:
            defaults = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            configured_log = defaults.get('logging', {}).get('meg_dataset_gui')
        except (AttributeError, OSError, yaml.YAMLError) as error:
            raise ValueError(f'Could not read config file {config_path}: {error}') from error
    return Path(configured_log or DEFAULT_LOG_PATH).expanduser().resolve()


def _prompt_choice(prompt, choices, default, input_func=input, output_func=print):
    while True:
        try:
            response = input_func(prompt).strip().lower()
        except EOFError:
            response = ''
        response = response or default
        aliases = {choice[0]: choice for choice in choices}
        response = aliases.get(response, response)
        if response in choices:
            return response
        output_func(f'Choose one of: {", ".join(choices)}.')


def _prompt_date_range(default_range, input_func=input, output_func=print):
    default_text = (
        f'{default_range[0].strftime(DATE_FORMAT)}-'
        f'{default_range[1].strftime(DATE_FORMAT)}'
    )
    while True:
        try:
            response = input_func(
                f'Date range MM/DD/YYYY-MM/DD/YYYY [{default_text}]: '
            ).strip()
        except EOFError:
            response = ''
        try:
            return parse_date_range(response or default_text)
        except ValueError as error:
            output_func(str(error))


def _prompt_bids_id_filter(input_func=input, output_func=print):
    while True:
        try:
            response = input_func(
                'BIDS subject ID(s), comma-separated [all]: '
            ).strip()
        except EOFError:
            response = ''
        try:
            return parse_bids_id_filter(response or 'all')
        except ValueError as error:
            output_func(str(error))


def _prompt_hash_id_filter(input_func=input, output_func=print):
    while True:
        try:
            response = input_func(
                'MEGHash ID(s), comma-separated [all]: '
            ).strip()
        except EOFError:
            response = ''
        try:
            return parse_hash_id_filter(response or 'all')
        except ValueError as error:
            output_func(str(error))


def _get_parser():
    parser = argparse.ArgumentParser(
        description='Review conversions recorded in a BIDS conversion logfile.'
    )
    parser.add_argument('-log', metavar='PATH', help='BIDS conversion logfile.')
    parser.add_argument(
        '-config', metavar='PATH', help='defaults.yml used to locate the logfile.'
    )
    parser.add_argument(
        '-view', choices=('week', 'last', 'all', 'range'),
        help='Entries to display. The interactive default is week.',
    )
    parser.add_argument(
        '-date_range', metavar='MM/DD/YYYY-MM/DD/YYYY',
        help='Inclusive date range; providing it implies -view range.',
    )
    parser.add_argument(
        '-bids_id', metavar='ID[,ID...]',
        help=(
            'BIDS subject ID filter, optionally comma-separated. Accepts '
            '01 or sub-01; the default is all.'
        ),
    )
    parser.add_argument(
        '-hash_id', metavar='ID[,ID...]',
        help=(
            'MEGHash ID filter, optionally comma-separated; the default is '
            'all.'
        ),
    )
    parser.add_argument(
        '-sort', choices=('date', 'bids_id', 'meghash'),
        help='Ordering for multi-entry views. The default is date.',
    )
    return parser


def main(argv=None, input_func=input, stdout=None, stderr=None):
    """Run the prompt-driven terminal log reviewer."""
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr
    parser = _get_parser()
    args = parser.parse_args(argv)

    try:
        log_path = resolve_log_path(args.log, args.config)
        result = parse_log_file(log_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for warning in result.warnings:
        print(f'Warning: {warning}', file=stderr)
    if not result.entries:
        parser.error(f'No valid RUNDICT entries found in {log_path}')

    if args.date_range is not None:
        view = 'range'
    elif args.view is not None:
        view = args.view
    else:
        view = _prompt_choice(
            'View [week/last/all/range] (week): ',
            ('week', 'last', 'all', 'range'),
            'week',
            input_func=input_func,
            output_func=lambda message: print(message, file=stdout),
        )

    selected_range = None
    if view == 'range':
        if args.date_range is not None:
            try:
                selected_range = parse_date_range(args.date_range)
            except ValueError as error:
                parser.error(str(error))
        elif sys.stdin.isatty() or input_func is not input:
            selected_range = _prompt_date_range(
                latest_week_range(result.entries),
                input_func=input_func,
                output_func=lambda message: print(message, file=stdout),
            )
        else:
            parser.error('-view range requires -date_range outside a terminal')

    if args.bids_id is not None:
        try:
            bids_ids = parse_bids_id_filter(args.bids_id)
        except ValueError as error:
            parser.error(str(error))
    else:
        bids_ids = _prompt_bids_id_filter(
            input_func=input_func,
            output_func=lambda message: print(message, file=stdout),
        )
    filtered_entries = filter_entries_by_bids_id(result.entries, bids_ids)

    if args.hash_id is not None:
        try:
            hash_ids = parse_hash_id_filter(args.hash_id)
        except ValueError as error:
            parser.error(str(error))
    else:
        hash_ids = _prompt_hash_id_filter(
            input_func=input_func,
            output_func=lambda message: print(message, file=stdout),
        )
    filtered_entries = filter_entries_by_hash_id(filtered_entries, hash_ids)
    if not filtered_entries:
        print('No conversion entries match the selected subject filters.',
              file=stdout)
        return 1

    if view == 'last':
        sort_by = 'date'
    elif args.sort is not None:
        sort_by = args.sort
    else:
        sort_by = _prompt_choice(
            'Sort [date/bids_id/meghash] (date): ',
            ('date', 'bids_id', 'meghash'),
            'date',
            input_func=input_func,
            output_func=lambda message: print(message, file=stdout),
        )

    entries = select_entries(
        filtered_entries,
        view=view,
        date_range=selected_range,
    )
    entries = sort_entries(entries, sort_by=sort_by)
    if not entries:
        print('No conversion entries found in the selected date range.', file=stdout)
        return 1

    print(format_entries(entries), file=stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
