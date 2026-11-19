"""Tests for logfile lookup, timing alignment, and standard event conversion."""

import pytest
import pandas as pd
import yaml

from nih2mne.utilities import log_utilities


def _config(tmp_path, monkeypatch, sources):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / "megcore" / "trigproc"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "MID_logfind.yml"
    config_path.write_text(yaml.safe_dump({"log_sources": sources}))
    return config_path


def _source(root, pattern="{subject_hash}/{subject_hash}_MID*.csv", recursive=False):
    return {
        "directories": [str(root)],
        "patterns": [pattern],
        "recursive": recursive,
    }


def test_find_one_logfile_and_ignore_duplicate_source_matches(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    subject_dir = root / "ADHCVYAX"
    subject_dir.mkdir(parents=True)
    logfile = subject_dir / "ADHCVYAX_MID_001.csv"
    logfile.write_text("events")
    source = _source(root)
    _config(tmp_path, monkeypatch, [source, source])

    assert log_utilities.logfind(task="MID", subject="ADHCVYAX") == str(logfile)


def test_recursive_search_and_multiple_patterns(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    nested = root / "archive" / "ADHCVYAX"
    nested.mkdir(parents=True)
    logfile = nested / "ADHCVYAX_MID_001.csv"
    logfile.write_text("events")
    source = _source(root, recursive=True)
    source["patterns"].append("{subject_hash}/missing*.txt")
    _config(tmp_path, monkeypatch, [source])

    assert log_utilities.logfind("MID", "ADHCVYAX") == str(logfile)


def test_zero_and_multiple_matches_raise(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    subject_dir = root / "ADHCVYAX"
    subject_dir.mkdir(parents=True)
    _config(tmp_path, monkeypatch, [_source(root)])
    with pytest.raises(FileNotFoundError, match="No logfile found"):
        log_utilities.logfind("MID", "ADHCVYAX")

    (subject_dir / "ADHCVYAX_MID_001.csv").touch()
    (subject_dir / "ADHCVYAX_MID_002.csv").touch()
    with pytest.raises(ValueError, match="Multiple logfiles found"):
        log_utilities.logfind("MID", "ADHCVYAX")


def test_missing_or_invalid_configuration_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="MID_logfind.yml"):
        log_utilities.logfind("MID", "ADHCVYAX")

    config = _config(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match="log_sources"):
        log_utilities.logfind("MID", "ADHCVYAX")

    config.write_text(yaml.safe_dump({"log_sources": [{"directories": ["/tmp"]}]}))
    with pytest.raises(ValueError, match="patterns"):
        log_utilities.logfind("MID", "ADHCVYAX")

    config.write_text("log_sources: [invalid\n")
    with pytest.raises(ValueError, match="Invalid logfile configuration.*MID_logfind.yml"):
        log_utilities.logfind("MID", "ADHCVYAX")


@pytest.mark.parametrize("task,subject", [("../MID", "ADHCVYAX"), ("MID", "A*"), ("", "A")])
def test_invalid_identifiers_raise(task, subject):
    with pytest.raises(ValueError, match="identifier"):
        log_utilities.logfind(task, subject)


def _trigger_events(monkeypatch, onsets, codes):
    from nih2mne.utilities import trigger_utilities

    def detect(filename, channel="UPPT001"):
        assert filename == "recording.ds"
        assert channel == "UPPT001"
        return pd.DataFrame({"onset": onsets, "condition": codes})

    monkeypatch.setattr(trigger_utilities, "detect_digital", detect)


def test_time_offset_and_drift(tmp_path, monkeypatch):
    log_times = [0, 30, 60, 90, 120]
    codes = [1, 2, 1, 3, 2]
    logfile = tmp_path / "events.csv"
    pd.DataFrame({"event_code": codes, "event_time": log_times}).to_csv(logfile, index=False)
    # Offset starts at 2 s and grows by 3 ms per minute.
    trigger_times = [time + 2 + 0.003 * time / 60 for time in log_times]
    _trigger_events(monkeypatch, trigger_times, [str(code) for code in codes])

    result = log_utilities.find_time_offset(logfile, "recording.ds", "event_code", "event_time")

    assert result.offset_seconds == pytest.approx(2.003)
    assert result.drift_ms_per_minute == pytest.approx(3)
    assert result.matches["residual_seconds"].tolist() == pytest.approx(
        [-0.003, -0.0015, 0, 0.0015, 0.003]
    )
    assert result.unmatched_log.empty
    assert result.unmatched_trigger.empty


def test_time_offset_allows_gaps_and_repeated_codes(tmp_path, monkeypatch):
    logfile = tmp_path / "events.csv"
    pd.DataFrame({
        "code": [1, 2, 1, 3, 1, 4],
        "seconds": [0, 1, 2, 3, 4, 5],
    }).to_csv(logfile, index=False)
    _trigger_events(monkeypatch, [10, 11, 12, 13, 14, 15], ["1", "2", "1", "9", "1", "4"])

    result = log_utilities.find_time_offset(logfile, "recording.ds", "code", "seconds")

    assert result.offset_seconds == pytest.approx(10)
    assert result.drift_ms_per_minute == pytest.approx(0, abs=1e-9)
    assert result.matches["code"].tolist() == ["1", "2", "1", "1", "4"]
    assert result.unmatched_log["code"].tolist() == ["3"]
    assert result.unmatched_trigger["code"].tolist() == ["9"]


def test_time_offset_rejects_missing_columns_and_too_few_matches(tmp_path, monkeypatch):
    logfile = tmp_path / "events.csv"
    pd.DataFrame({"code": [1, 2], "seconds": [0, 1]}).to_csv(logfile, index=False)
    _trigger_events(monkeypatch, [10, 11], ["1", "2"])

    with pytest.raises(ValueError, match="Missing logfile column"):
        log_utilities.find_time_offset(logfile, "recording.ds", "code", "wrong")
    with pytest.raises(ValueError, match="At least three"):
        log_utilities.find_time_offset(logfile, "recording.ds", "code", "seconds")


@pytest.fixture
def correction_log(tmp_path, monkeypatch):
    logfile = tmp_path / 'timing.csv'
    pd.DataFrame({
        'Code': [3, 1, 2, None],
        'time': [20., 0., 10., None],
        'cue.started': [25., -5., 5., None],
        'cue.stopped': [26., -4., 6., None],
        'rt': [.1, .2, .3, None],
        'label': ['last', 'first', 'middle', 'footer'],
    }).to_csv(logfile, index=False)
    _trigger_events(monkeypatch, [2., 12.01, 22.02], [1, 2, 3])
    return logfile


@pytest.mark.parametrize('method,expected', [
    ('first', [22., 2., 12., None]),
    ('middle', [22.01, 2.01, 12.01, None]),
    ('local', [22.02, 2., 12.01, None]),
])
def test_correction_methods(correction_log, method, expected):
    result = log_utilities.find_time_offset(
        correction_log, 'recording.ds', 'Code', 'time', method=method,
    )
    assert result.corrected_log.time.iloc[:3].tolist() == pytest.approx(expected[:3])
    assert pd.isna(result.corrected_log.time.iloc[3])
    assert result.correction_method == method
    original = pd.read_csv(correction_log)
    pd.testing.assert_series_equal(result.corrected_log.rt, original.rt)
    pd.testing.assert_series_equal(result.corrected_log.label, original.label)
    assert result.matches.residual_seconds.tolist() == pytest.approx([-.01, 0, .01])
    if method == 'local':
        assert result.corrected_log['cue.started'].iloc[:3].tolist() == pytest.approx(
            [27.02, -3., 7.005])
        assert result.matches.corrected_residual_seconds.tolist() == pytest.approx([0, 0, 0])


def test_explicit_columns_and_csv(correction_log, tmp_path):
    output = tmp_path / 'corrected.csv'
    result = log_utilities.find_time_offset(
        correction_log, 'recording.ds', 'Code', 'time', time_columns=['cue.started'],
        output_file=output,
    )
    original = pd.read_csv(correction_log)
    pd.testing.assert_series_equal(result.corrected_log['cue.stopped'], original['cue.stopped'])
    pd.testing.assert_frame_equal(pd.read_csv(output), result.corrected_log)
    with pytest.raises(FileExistsError):
        log_utilities.find_time_offset(correction_log, 'recording.ds', 'Code', 'time', output_file=output)
    with pytest.raises(ValueError, match='source logfile'):
        log_utilities.find_time_offset(correction_log, 'recording.ds', 'Code', 'time', output_file=correction_log)
    pd.testing.assert_frame_equal(pd.read_csv(correction_log), original)


@pytest.mark.parametrize('method', ['first', 'local'])
def test_offset_limit_lists_all_trials_and_does_not_write(correction_log, tmp_path, method):
    output = tmp_path / 'invalid.csv'
    with pytest.raises(ValueError) as error:
        log_utilities.find_time_offset(
            correction_log, 'recording.ds', 'Code', 'time', method=method,
            offset_error=.009, output_file=output,
        )
    message = str(error.value)
    assert 'row 0, code' in message
    assert 'row 1, code' in message
    assert 'deviation=10.000000 ms' in message
    assert not output.exists()


@pytest.mark.parametrize('limit', [.020, .021])
def test_offset_limit_boundary(correction_log, monkeypatch, limit):
    _trigger_events(monkeypatch, [2., 12.02, 22.04], [1, 2, 3])
    log_utilities.find_time_offset(
        correction_log, 'recording.ds', 'Code', 'time', method='local', offset_error=limit,
    )
    with pytest.raises(ValueError, match='offset_error'):
        log_utilities.find_time_offset(
            correction_log, 'recording.ds', 'Code', 'time', method='local', offset_error=.019,
        )
    # Middle always has zero deviation from itself.
    log_utilities.find_time_offset(correction_log, 'recording.ds', 'Code', 'time', offset_error=0)


@pytest.mark.parametrize('limit', [-1, float('nan'), float('inf'), None, 'bad'])
def test_invalid_offset_limit(correction_log, limit):
    with pytest.raises(ValueError, match='offset_error'):
        log_utilities.find_time_offset(correction_log, 'recording.ds', 'Code', 'time', offset_error=limit)


@pytest.mark.parametrize('bad', ['bad', float('inf')])
def test_invalid_timestamp(correction_log, bad):
    data = pd.read_csv(correction_log)
    data['cue.started'] = data['cue.started'].astype(object)
    data.loc[0, 'cue.started'] = bad
    data.to_csv(correction_log, index=False)
    with pytest.raises(ValueError, match='finite numeric'):
        log_utilities.find_time_offset(correction_log, 'recording.ds', 'Code', 'time')


def test_local_gaps(correction_log, monkeypatch):
    data = pd.read_csv(correction_log)
    data.loc[3, ['Code', 'time']] = [9, 5.]
    data.to_csv(correction_log, index=False)
    _trigger_events(monkeypatch, [2., 4., 12.01, 22.02], [1, 8, 2, 3])
    with pytest.warns(UserWarning, match='unmatched'):
        result = log_utilities.find_time_offset(correction_log, 'recording.ds', 'Code', 'time', method='local')
    assert result.corrected_log.time.iloc[3] == pytest.approx(7.005)
    assert result.unmatched_log.code.tolist() == ['9']
    assert result.unmatched_trigger.code.tolist() == ['8']


@pytest.mark.parametrize('conflict', [True, False])
def test_duplicate_local_anchors(tmp_path, monkeypatch, conflict):
    logfile = tmp_path / 'duplicate.csv'
    pd.DataFrame({'code': [1, 2, 3], 'time': [0., 0., 10.]}).to_csv(logfile, index=False)
    _trigger_events(monkeypatch, [2., 2.001 if conflict else 2., 12.], [1, 2, 3])
    if conflict:
        with pytest.raises(ValueError, match='Conflicting offsets'):
            log_utilities.find_time_offset(logfile, 'recording.ds', 'code', 'time', method='local')
    else:
        result = log_utilities.find_time_offset(logfile, 'recording.ds', 'code', 'time', method='local')
        assert result.corrected_log.time.tolist() == pytest.approx([2, 2, 12])


def test_corrected_events_defaults_and_order():
    source = pd.DataFrame({
        'cue.started': ['2', None, '-1'], 'image.start': [2, 2, None],
        'cue.stopped': [3, 3, 3], 'response.rt': [.2, .3, .4],
    }, index=[10, 10, 12])
    original = source.copy(deep=True)
    events = log_utilities.log_to_longform_standard(source)
    assert list(events.columns) == ['trial', 'onset', 'condition', 'channel']
    assert events.onset.tolist() == [-1, 2, 2, 2]
    assert events.condition.tolist() == ['cue', 'cue', 'image', 'image']
    assert events.trial.tolist() == [0] * 4
    assert events.channel.tolist() == ['LOGFILE'] * 4
    pd.testing.assert_frame_equal(source, original)


def test_corrected_events_mapping_adds_and_overrides():
    source = pd.DataFrame({
        'cue.started': [1, 2, 3, 4], 'image.start': [1, 2, 3, 4],
        'custom_time': [1.5, 2.5, None, 4.5],
        'label': ['win', None, '  ', 'loss'], 'code': [8., 16., 32., 8.],
    })
    events = log_utilities.log_to_longform_standard(source, condition_columns={
        'cue.started': 'label', 'custom_time': 'code',
    })
    assert events.condition.tolist() == ['win', 'image', '8', 'image', '16', 'image', 'loss', 'image', '8']
    assert events.onset.tolist() == [1, 1, 1.5, 2, 2.5, 3, 4, 4, 4.5]


@pytest.mark.parametrize('value', ['bad', float('inf'), float('-inf')])
def test_corrected_events_invalid_time(value):
    with pytest.raises(ValueError, match=r"cue.started.*rows \[1\]"):
        log_utilities.log_to_longform_standard(pd.DataFrame({'cue.started': [1, value]}))


@pytest.mark.parametrize('mapping', [{'missing': 'label'}, {'cue.started': 'missing'}])
def test_corrected_events_missing_mapping_columns(mapping):
    with pytest.raises(ValueError, match='Missing logfile column: missing'):
        log_utilities.log_to_longform_standard(
            pd.DataFrame({'cue.started': [1], 'label': ['a']}), condition_columns=mapping,
        )


@pytest.mark.parametrize('source', [pd.DataFrame(), pd.DataFrame({'rt': [1]}),
                                  pd.DataFrame({'cue.started': [None]})])
def test_corrected_events_empty(source):
    events = log_utilities.log_to_longform_standard(source)
    assert events.empty
    assert list(events.columns) == ['trial', 'onset', 'condition', 'channel']


def test_corrected_events_offset_workflow(tmp_path, monkeypatch):
    logfile = tmp_path / 'events.csv'
    pd.DataFrame({'Code': [8, 16, 32], 'cue.started': [10, 20, 30],
                  'TrialType': ['win', 'lose', 'neutral']}).to_csv(logfile, index=False)
    _trigger_events(monkeypatch, [11, 21, 31], ['8', '16', '32'])
    result = log_utilities.find_time_offset(logfile, 'recording.ds', 'Code', 'cue.started')
    events = log_utilities.log_to_longform_standard(
        result.corrected_log, condition_columns={'cue.started': 'TrialType'},
    )
    assert events.onset.tolist() == [11, 21, 31]
    assert events.condition.tolist() == ['win', 'lose', 'neutral']
