"""Tests for task logfile lookup."""

import pytest
import yaml

from nih2mne.utilities import logfinder


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

    assert logfinder.logfind(task="MID", subject="ADHCVYAX") == str(logfile)


def test_recursive_search_and_multiple_patterns(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    nested = root / "archive" / "ADHCVYAX"
    nested.mkdir(parents=True)
    logfile = nested / "ADHCVYAX_MID_001.csv"
    logfile.write_text("events")
    source = _source(root, recursive=True)
    source["patterns"].append("{subject_hash}/missing*.txt")
    _config(tmp_path, monkeypatch, [source])

    assert logfinder.logfind("MID", "ADHCVYAX") == str(logfile)


def test_zero_and_multiple_matches_raise(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    subject_dir = root / "ADHCVYAX"
    subject_dir.mkdir(parents=True)
    _config(tmp_path, monkeypatch, [_source(root)])
    with pytest.raises(FileNotFoundError, match="No logfile found"):
        logfinder.logfind("MID", "ADHCVYAX")

    (subject_dir / "ADHCVYAX_MID_001.csv").touch()
    (subject_dir / "ADHCVYAX_MID_002.csv").touch()
    with pytest.raises(ValueError, match="Multiple logfiles found"):
        logfinder.logfind("MID", "ADHCVYAX")


def test_missing_or_invalid_configuration_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="MID_logfind.yml"):
        logfinder.logfind("MID", "ADHCVYAX")

    config = _config(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match="log_sources"):
        logfinder.logfind("MID", "ADHCVYAX")

    config.write_text(yaml.safe_dump({"log_sources": [{"directories": ["/tmp"]}]}))
    with pytest.raises(ValueError, match="patterns"):
        logfinder.logfind("MID", "ADHCVYAX")

    config.write_text("log_sources: [invalid\n")
    with pytest.raises(ValueError, match="Invalid logfile configuration.*MID_logfind.yml"):
        logfinder.logfind("MID", "ADHCVYAX")


@pytest.mark.parametrize("task,subject", [("../MID", "ADHCVYAX"), ("MID", "A*"), ("", "A")])
def test_invalid_identifiers_raise(task, subject):
    with pytest.raises(ValueError, match="identifier"):
        logfinder.logfind(task, subject)
