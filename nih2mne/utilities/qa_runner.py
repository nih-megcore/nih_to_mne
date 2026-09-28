#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config-driven QA of MEG trigger/marker counts.

A "QA file" is a plain-text file describing the expected marker counts for one
task, one check per line.  Grammar:

    <term> <op> <number> [optional]

    term      := marker[ + marker ...]      # sum of one or more marker counts
    op        := == | != | > | >= | < | <=
    number    := integer or float
    optional  := literal keyword; if present, the check is skipped when NONE
                 of the markers in <term> appear in the data

Blank lines and lines beginning with '#' are ignored.  Trailing '#' comments
are allowed.

Example (gonogo.qa):

    go + nogo                   >= 295
    go                          >  175
    nogo                        >  75
    response_hit                >  145
    response_correct_rejection  >  70
    response_false_alarm        <  30   optional

@author: (adapted from stoutjd's marker_quality_assurance.py)
"""

import argparse
import glob
import logging
import operator
import os
import re
import sys

import mne
import pandas as pd


# --------------------------------------------------------------------------- #
# Logging (kept compatible with the original per-subject logger)
# --------------------------------------------------------------------------- #

def get_subj_logger(subjid, log_dir=None):
    '''Return a subject-specific logger writing to <log_dir>/<subjid>_log.txt.
    Safe to call repeatedly; the file handler is only added once.'''
    fmt = '%(asctime)s :: %(levelname)s :: %(message)s'
    subj_logger = logging.getLogger(str(subjid))
    subj_logger.setLevel(logging.INFO)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.abspath(
            os.path.join(log_dir, f'{subjid}_log.txt')
        )
        file_paths = {
            os.path.abspath(handler.baseFilename)
            for handler in subj_logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if log_path not in file_paths:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(fmt))
            subj_logger.addHandler(fh)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in subj_logger.handlers
    )
    if not has_stream_handler:
        subj_logger.addHandler(logging.StreamHandler(sys.stdout))
        subj_logger.info('Initializing subject-level QA log')
    return subj_logger


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def annot_counts(filename):
    '''Load a CTF dataset and return a Series: marker description -> count.'''
    raw = mne.io.read_raw_ctf(filename, verbose=False, system_clock='ignore')
    desc = pd.Series(raw.annotations.description)
    return desc.value_counts()


# --------------------------------------------------------------------------- #
# The "vocabulary": operators and the check parser
# --------------------------------------------------------------------------- #

OPS = {
    '==': operator.eq,
    '!=': operator.ne,
    '>=': operator.ge,
    '<=': operator.le,
    '>':  operator.gt,
    '<':  operator.lt,
}

# Longest operators first so '>=' is matched before '>'.
_OP_TOKENS = sorted(OPS, key=len, reverse=True)


class Check:
    '''One line of a QA file, parsed into an executable comparison.'''

    __slots__ = ('markers', 'op_str', 'op', 'threshold', 'optional', 'raw')

    def __init__(self, markers, op_str, threshold, optional, raw):
        self.markers = markers          # list[str]
        self.op_str = op_str            # e.g. '>='
        self.op = OPS[op_str]           # callable
        self.threshold = threshold      # float
        self.optional = optional        # bool
        self.raw = raw                  # original line, for messages

    @property
    def term(self):
        return ' + '.join(self.markers)

    def evaluate(self, counts):
        '''Return (status, actual) where status is one of:
        'pass', 'fail', or 'skipped'.  counts is a pd.Series.'''
        present = [m for m in self.markers if m in counts.index]

        if self.optional and not present:
            return 'skipped', None

        # Missing (non-optional) markers count as 0 — a hard-coded expectation
        # that a required marker never appeared is itself a failure worth seeing.
        actual = sum(int(counts.get(m, 0)) for m in self.markers)
        status = 'pass' if self.op(actual, self.threshold) else 'fail'
        return status, actual


def parse_line(line):
    '''Parse a single non-comment line into a Check, or None for blanks.'''
    # Strip trailing comments and whitespace.
    line = line.split('#', 1)[0].strip()
    if not line:
        return None

    optional = False
    tokens = line.split()
    if tokens and tokens[-1].lower() == 'optional':
        optional = True
        tokens = tokens[:-1]

    text = ' '.join(tokens)

    # Find the operator.
    op_found = None
    for op in _OP_TOKENS:
        if op in text:
            op_found = op
            break
    if op_found is None:
        raise ValueError(f'No comparison operator found in line: {line!r}')

    left, right = text.split(op_found, 1)

    markers = [m.strip() for m in left.split('+') if m.strip()]
    if not markers:
        raise ValueError(f'No marker term before operator in line: {line!r}')

    right = right.strip()
    try:
        threshold = float(right)
        if threshold.is_integer():
            threshold = int(threshold)
    except ValueError:
        raise ValueError(f'Right-hand side is not a number in line: {line!r}')

    return Check(markers, op_found, threshold, optional, line)


def load_qa_file(path):
    '''Read a QA file and return a list[Check].'''
    checks = []
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            try:
                chk = parse_line(raw)
            except ValueError as e:
                raise ValueError(f'{path}:{lineno}: {e}')
            if chk is not None:
                checks.append(chk)
    if not checks:
        raise ValueError(f'{path}: no QA checks found')
    return checks


# --------------------------------------------------------------------------- #
# Running checks against a dataset
# --------------------------------------------------------------------------- #

def run_checks_detailed(counts, checks, task=None, logger=None):
    '''Evaluate checks and return aggregate and per-condition results.'''
    log = logger or logging.getLogger(__name__)
    summary = {'pass': 0, 'fail': 0, 'skipped': 0}
    details = []
    tag = f'{task}: ' if task else ''

    for chk in checks:
        status, actual = chk.evaluate(counts)
        summary[status] += 1
        details.append({
            'term': chk.term,
            'operator': chk.op_str,
            'expected': chk.threshold,
            'actual': actual,
            'status': status,
            'optional': chk.optional,
        })
        if status == 'fail':
            log.warning(
                f'{tag}FAIL  {chk.term} {chk.op_str} {chk.threshold} '
                f'(actual={actual})'
            )
        elif status == 'skipped':
            log.info(f'{tag}skip  {chk.term} (optional, no markers present)')
        else:
            log.info(f'{tag}pass  {chk.term} {chk.op_str} {chk.threshold} '
                     f'(actual={actual})')
    return {
        'overall': 'fail' if summary['fail'] else 'pass',
        'summary': summary,
        'checks': details,
    }


def run_checks(counts, checks, task=None, logger=None):
    '''Evaluate all checks and return the backward-compatible summary dict.'''
    return run_checks_detailed(
        counts,
        checks,
        task=task,
        logger=logger,
    )['summary']


def run_qa_file(filename, qa_path, task=None, logger=None):
    '''Run one QA file against one dataset and return detailed results.'''
    checks = load_qa_file(qa_path)
    counts = annot_counts(filename)
    result = run_checks_detailed(
        counts,
        checks,
        task=task,
        logger=logger,
    )
    return {
        'task': task,
        'qa_file': os.path.abspath(qa_path),
        **result,
    }


_VERSION_RE = re.compile(r'_v(?P<version>\d+)(?=\.[^.]+$)', re.IGNORECASE)


def filename_version(path):
    '''Return an integer `_vN` suffix, or None when no version is present.'''
    match = _VERSION_RE.search(os.path.basename(os.fspath(path)))
    return int(match.group('version')) if match else None


def _qa_filename_version(filename, task):
    '''Return a QA version for an exact task match, otherwise None.'''
    escaped_task = re.escape(task)
    match = re.fullmatch(
        rf'{escaped_task}(?:_v(?P<version>\d+))?\.qa',
        filename,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    version = match.group('version')
    return int(version) if version is not None else 0


def find_qa_file(qa_dir, task, proc_file=None):
    '''Select the newest compatible task QA file for a ProcFile.'''
    if not qa_dir or not os.path.isdir(qa_dir):
        return None

    proc_version = filename_version(proc_file) if proc_file else None
    candidates = []
    for filename in os.listdir(qa_dir):
        version = _qa_filename_version(filename, task)
        if version is None:
            continue
        path = os.path.join(qa_dir, filename)
        if not os.path.isfile(path):
            continue
        if proc_version is not None and version > proc_version:
            continue
        is_versioned = filename_version(filename) is not None
        candidates.append((version, is_versioned, filename.lower(), path))

    if not candidates:
        return None
    return max(candidates)[-1]


def _task_from_qa_filename(filename):
    '''Return the task encoded by a supported QA filename.'''
    match = re.fullmatch(
        r'(?P<task>.+?)(?:_v\d+)?\.qa',
        filename,
        flags=re.IGNORECASE,
    )
    return match.group('task') if match else None


def get_all_filenames(topdir, tasks):
    '''Glob dataset paths per task from topdir.'''
    return {task: glob.glob(os.path.join(topdir, f'*{task}*')) for task in tasks}


def qa_datasets(topdir, qa_dir, tasks=None, ignore=(), test_only=(),
                log_dir=None):
    '''Loop over datasets in topdir, running each task's QA file.'''
    if tasks is None:
        # Infer task set from available .qa files.
        tasks = sorted({
            task
            for filename in os.listdir(qa_dir)
            if (task := _task_from_qa_filename(filename)) is not None
        })

    if ignore:
        tasks = [t for t in tasks if t not in ignore]
    if test_only:
        tasks = [t for t in tasks if t in test_only]

    filenames = get_all_filenames(topdir, tasks)

    for task in tasks:
        qa_path = find_qa_file(qa_dir, task)
        if qa_path is None:
            print(f'>>> No QA file for task {task!r} in {qa_dir}; skipping')
            continue

        for fname in filenames.get(task, []):
            subjid = os.path.basename(fname.rstrip('/'))
            logger = get_subj_logger(subjid, log_dir=log_dir)
            logger.info(f'{task.upper()} :: {subjid}')
            print(f'Testing: {fname}')
            try:
                run_qa_file(fname, qa_path, task=task, logger=logger)
            except Exception as e:  # noqa: BLE001 — QA must not abort the batch
                logger.exception(f'{task}: FAILED TO PROCESS {fname}: {e}')
                print(f'>>> FAILED TEST: {fname} <<<')


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description='Config-driven QA of MEG trigger counts using per-task '
                    'QA files.')
    p.add_argument('-data_folder',
                   help='Top subject directory (typically dated).')
    p.add_argument('-qa_dir',
                   help='Directory containing <task>.qa files.')
    p.add_argument('-log_dir', default=None,
                   help='Directory for per-subject log files.')
    p.add_argument('-ignore', action='append', default=[],
                   help='Task to skip (repeatable).')
    p.add_argument('-test_only', action='append', default=[],
                   help='Task to test exclusively (repeatable).')
    p.add_argument('-print_counts',
                   help='Path to a single dataset; print its marker counts '
                        'and exit (useful for writing/debugging QA files).')
    p.add_argument('-check_file',
                   help='Path to a single dataset to QA against -single_qa.')
    p.add_argument('-single_qa',
                   help='Path to a QA file, used with -check_file.')
    args = p.parse_args()

    if args.print_counts:
        print(annot_counts(args.print_counts).to_string())
        return

    if args.check_file:
        if not args.single_qa:
            raise SystemExit('-check_file requires -single_qa')
        counts = annot_counts(args.check_file)
        checks = load_qa_file(args.single_qa)
        logger = get_subj_logger(os.path.basename(args.check_file),
                                 log_dir=args.log_dir)
        res = run_checks(counts, checks, logger=logger)
        print(f'\nSummary: {res}')
        return

    if args.ignore and args.test_only:
        raise SystemExit('Use only one of -ignore / -test_only')
    if not args.data_folder:
        raise SystemExit('No -data_folder provided')
    if not args.qa_dir:
        raise SystemExit('-data_folder requires -qa_dir')

    qa_datasets(args.data_folder, args.qa_dir,
                ignore=args.ignore, test_only=args.test_only,
                log_dir=args.log_dir)


if __name__ == '__main__':
    main()
