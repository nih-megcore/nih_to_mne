import logging
import sys

import pandas as pd
import pytest

from nih2mne.utilities import qa_runner


HARIRI_QA = """\
# hariri.qa
probe_face + probe_shape     == 150
response_hit                 >  120     # ~80% accuracy
probe_face                   == 90
encode_face                  == 90
encode_shape                 == 60
probe_shape                  == 60
probe_match_sad              >  35
probe_match_happy            >  35
encode_male + encode_female  == 90
response_r + response_l      >  120
"""


def test_hariri_example_parses_and_evaluates(tmp_path):
    qa_path = tmp_path / 'hariri.qa'
    qa_path.write_text(HARIRI_QA, encoding='utf-8')
    counts = pd.Series({
        'probe_face': 90,
        'probe_shape': 60,
        'response_hit': 121,
        'encode_face': 90,
        'encode_shape': 60,
        'probe_match_sad': 36,
        'probe_match_happy': 36,
        'encode_male': 45,
        'encode_female': 45,
        'response_r': 61,
        'response_l': 60,
    })

    checks = qa_runner.load_qa_file(qa_path)
    result = qa_runner.run_checks_detailed(counts, checks, task='hariri')

    assert len(checks) == 10
    assert result['overall'] == 'pass'
    assert result['summary'] == {'pass': 10, 'fail': 0, 'skipped': 0}
    assert result['checks'][0] == {
        'term': 'probe_face + probe_shape',
        'operator': '==',
        'expected': 150,
        'actual': 150,
        'status': 'pass',
        'optional': False,
    }


@pytest.mark.parametrize(
    ('operator', 'actual', 'expected', 'status'),
    [
        ('==', 2, 2, 'pass'),
        ('!=', 2, 3, 'pass'),
        ('>=', 2, 2, 'pass'),
        ('<=', 2, 2, 'pass'),
        ('>', 2, 3, 'fail'),
        ('<', 3, 2, 'fail'),
    ],
)
def test_all_comparison_operators(operator, actual, expected, status):
    check = qa_runner.parse_line(f'marker {operator} {expected}')

    assert check.evaluate(pd.Series({'marker': actual})) == (status, actual)


def test_optional_missing_marker_skips_but_required_marker_fails():
    counts = pd.Series(dtype='int64')

    optional = qa_runner.parse_line('response > 0 optional')
    required = qa_runner.parse_line('response > 0')

    assert optional.evaluate(counts) == ('skipped', None)
    assert required.evaluate(counts) == ('fail', 0)


def test_load_qa_file_rejects_empty_file(tmp_path):
    qa_path = tmp_path / 'empty.qa'
    qa_path.write_text('# comments only\n', encoding='utf-8')

    with pytest.raises(ValueError, match='no QA checks found'):
        qa_runner.load_qa_file(qa_path)


def test_load_qa_file_reports_line_number(tmp_path):
    qa_path = tmp_path / 'invalid.qa'
    qa_path.write_text('valid == 1\nnot a check\n', encoding='utf-8')

    with pytest.raises(ValueError, match=r'invalid\.qa:2:'):
        qa_runner.load_qa_file(qa_path)


def test_run_checks_keeps_summary_only_return():
    checks = [qa_runner.parse_line('marker == 1')]

    result = qa_runner.run_checks(pd.Series({'marker': 1}), checks)

    assert result == {'pass': 1, 'fail': 0, 'skipped': 0}


def test_run_qa_file_returns_source_and_details(tmp_path, monkeypatch):
    qa_path = tmp_path / 'task.qa'
    qa_path.write_text('marker == 2\n', encoding='utf-8')
    monkeypatch.setattr(
        qa_runner,
        'annot_counts',
        lambda _filename: pd.Series({'marker': 1}),
    )

    result = qa_runner.run_qa_file(
        '/data/input.ds', qa_path, task='task'
    )

    assert result['task'] == 'task'
    assert result['qa_file'] == str(qa_path.resolve())
    assert result['overall'] == 'fail'
    assert result['summary']['fail'] == 1
    assert result['checks'][0]['actual'] == 1


def test_filename_version_handles_multidigit_and_unversioned_names():
    assert qa_runner.filename_version('hariri_v12.py') == 12
    assert qa_runner.filename_version('hariri_v2.qa') == 2
    assert qa_runner.filename_version('hariri.py') is None


def test_find_qa_file_uses_latest_version_not_newer_than_proc(tmp_path):
    for filename in (
        'hariri.qa',
        'HARIRI_v2.qa',
        'hariri_v10.qa',
        'hariri_v11.qa',
        'hariri_v9.txt',
        'hariri_notes.qa',
        'other_v2.qa',
    ):
        (tmp_path / filename).write_text('marker == 1\n', encoding='utf-8')

    assert qa_runner.find_qa_file(
        tmp_path, 'HaRiRi', proc_file='hariri_v3.py'
    ).endswith('HARIRI_v2.qa')
    assert qa_runner.find_qa_file(
        tmp_path, 'hariri', proc_file='hariri_v10.py'
    ).endswith('hariri_v10.qa')
    assert qa_runner.find_qa_file(
        tmp_path, 'hariri', proc_file='hariri.py'
    ).endswith('hariri_v11.qa')


def test_find_qa_file_falls_back_to_baseline(tmp_path):
    (tmp_path / 'hariri.qa').write_text('marker == 1\n', encoding='utf-8')
    (tmp_path / 'hariri_v2.qa').write_text('marker == 1\n', encoding='utf-8')

    selected = qa_runner.find_qa_file(
        tmp_path, 'hariri', proc_file='hariri_v1.py'
    )

    assert selected == str(tmp_path / 'hariri.qa')


def test_get_subj_logger_does_not_duplicate_handlers(tmp_path):
    logger = logging.getLogger('qa-runner-handler-test')
    original_handlers = list(logger.handlers)
    logger.handlers.clear()
    try:
        first = qa_runner.get_subj_logger(logger.name, log_dir=tmp_path)
        second = qa_runner.get_subj_logger(logger.name, log_dir=tmp_path)

        assert first is second
        assert len([
            handler for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]) == 1
        assert len([
            handler for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ]) == 1
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = original_handlers


def test_print_counts_cli_does_not_require_qa_dir(monkeypatch, capsys):
    monkeypatch.setattr(
        qa_runner,
        'annot_counts',
        lambda _filename: pd.Series({'marker': 2}),
    )
    monkeypatch.setattr(
        sys,
        'argv',
        ['qa_runner.py', '-print_counts', '/data/input.ds'],
    )

    qa_runner.main()

    assert 'marker' in capsys.readouterr().out


def test_batch_cli_requires_qa_dir(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['qa_runner.py', '-data_folder', '/data'],
    )

    with pytest.raises(SystemExit, match='requires -qa_dir'):
        qa_runner.main()
