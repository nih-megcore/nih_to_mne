import hashlib
from pathlib import Path

import pytest

from nih2mne.proc.derivative_review import (
    ProcessingScriptError,
    analyze_processing_script,
    evaluate_dataset,
    find_processing_failure,
)


SCRIPT = '''
project = "review"
low, high = 1, 40
conditions = ["face", "house"]
noise_run = "01"
recipe = FilterSpec(l_freq=low, h_freq=high, notch_freqs=(60,))
primary = get_bids_path_from_fname(dataset)
noise = primary.copy().update(task="EmptyRoom", run=noise_run)
template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe, noise=noise)
artifacts = template.evaluate()
ensure_dataset_description(artifacts["project_dataset_description"], project, logger)
result = load_or_make(artifacts["bem"], maker, saver, logger)
save_filtered(raw, artifacts["filtered_raw"], artifacts["filtered_raw_json"], False, logger)
for condition in conditions:
    native = template.artifact("stc", description=condition)
    save_if_requested(value, native, saver, False, logger)
    fsaverage = template.artifact("stc", description=condition, space="fsaverage")
    save_if_requested(value, fsaverage, saver, False, logger)
raise RuntimeError("this file must never execute")
'''


def test_analyzer_is_static_and_preserves_output_order(tmp_path):
    script = tmp_path / "task_v1.py"
    script.write_text(SCRIPT)

    spec = analyze_processing_script(script)

    assert spec.project == "review"
    assert spec.filter_kwargs == {"l_freq": 1, "h_freq": 40, "notch_freqs": (60,)}
    assert spec.noise_entities == {"task": "EmptyRoom", "run": "01"}
    assert spec.discover_emptyroom is False
    assert [request.label for request in spec.outputs] == [
        "project_dataset_description",
        "bem",
        "filtered_raw",
        "filtered_raw_json",
        "stc: face",
        "stc: face / space-fsaverage",
        "stc: house",
        "stc: house / space-fsaverage",
    ]


def test_dynamic_emptyroom_finder_is_reviewed_without_executing_script(
    tmp_path, monkeypatch
):
    script = tmp_path / "task_v1.py"
    script.write_text(
        "project = 'review'\n"
        "recipe = FilterSpec(l_freq=1, h_freq=40)\n"
        "noise = find_emptyroom_dataset(primary)\n"
        "template = MEGDerivativeTemplate(primary, project=project, "
        "filter_spec=recipe, noise=noise)\n"
        "artifacts = template.evaluate()\n"
        "save_if_requested(value, artifacts['noise_covariance'], saver, False, logger)\n"
    )
    raw = tmp_path / "bids" / "sub-01" / "meg" / "sub-01_task-task_meg.ds"
    raw.mkdir(parents=True)
    spec = analyze_processing_script(script)
    monkeypatch.setattr(
        "nih2mne.proc.derivative_review.find_emptyroom_dataset",
        lambda primary: primary.copy().update(task="NoIsE", run="03"),
    )

    evaluated = evaluate_dataset(spec, raw)

    assert spec.noise_entities is None
    assert spec.discover_emptyroom is True
    assert "task-NoIsE_run-03" in evaluated[0].record.path.name


def test_analyzer_rejects_legacy_and_dynamic_scripts(tmp_path):
    legacy = tmp_path / "legacy.py"
    legacy.write_text("output = BIDSPath(root='/tmp').fpath\n")
    with pytest.raises(ProcessingScriptError, match="FilterSpec"):
        analyze_processing_script(legacy)

    dynamic = tmp_path / "dynamic.py"
    dynamic.write_text(
        "project = get_project()\n"
        "recipe = FilterSpec(l_freq=1, h_freq=40)\n"
        "template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe)\n"
        "artifacts = template.evaluate()\n"
        "save_if_requested(value, artifacts['epochs'], saver, False, logger)\n"
    )
    with pytest.raises(ProcessingScriptError, match="project"):
        analyze_processing_script(dynamic)


def test_evaluate_dataset_uses_template_validation_for_surface_stc(tmp_path):
    script = tmp_path / "task_v1.py"
    script.write_text(
        "project = 'review'\n"
        "recipe = FilterSpec(l_freq=1, h_freq=40)\n"
        "template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe)\n"
        "record = template.artifact('stc', description='face')\n"
        "save_if_requested(value, record, saver, False, logger)\n"
    )
    raw = tmp_path / "sub-01" / "meg" / "sub-01_task-task_run-01_meg.ds"
    raw.mkdir(parents=True)
    spec = analyze_processing_script(script)

    first = evaluate_dataset(spec, raw)[0]
    assert first.record.status == "missing"
    first.record.component_paths[0].parent.mkdir(parents=True, exist_ok=True)
    first.record.component_paths[0].touch()

    partial = evaluate_dataset(spec, raw)[0]
    assert partial.record.status == "invalid"
    assert not partial.record.exists

    partial.record.component_paths[1].touch()
    complete = evaluate_dataset(spec, raw)[0]
    assert complete.record.exists


def _git_hash(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _failure_fixture(tmp_path):
    script = tmp_path / "task_v1.py"
    script.write_text(
        "project = 'review'\n"
        "recipe = FilterSpec(l_freq=1, h_freq=40)\n"
        "template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe)\n"
        "save_if_requested(value, artifacts['epochs'], saver, False, logger)\n"
    )
    dataset = (
        tmp_path / "bids" / "sub-01" / "meg" /
        "sub-01_task-task_run-01_meg.ds"
    )
    dataset.mkdir(parents=True)
    log_dir = tmp_path / "bids" / "derivatives" / "review" / "logging" / "sub-01"
    log_dir.mkdir(parents=True)
    return analyze_processing_script(script), script, dataset, log_dir


def test_find_processing_failure_follows_matching_slurm_error(tmp_path):
    spec, script, dataset, log_dir = _failure_fixture(tmp_path)
    process_hash = _git_hash(script)
    stdout = tmp_path / "logdir" / "job.o"
    stderr = stdout.with_suffix(".e")
    stderr.parent.mkdir()
    stderr.write_text(
        "scheduler preamble\nTraceback (most recent call last):\n"
        "  File 'task_v1.py', line 12\nValueError: no events\n"
    )
    (log_dir / "task_v1.log").write_text(
        f"INFO START name=task_v1 version=1 hash={process_hash}\n"
        f"INFO Primary input: {dataset}\n"
        f"INFO SLURM_LOGFILE_PATH :: {stdout}\n"
    )

    report = find_processing_failure(spec, dataset)

    assert report.error_found
    assert report.execution_log == stderr
    assert report.application_log == log_dir / "task_v1.log"
    assert report.details.startswith("Traceback")
    assert "ValueError: no events" in report.display_text


def test_find_processing_failure_preserves_complete_chained_traceback(tmp_path):
    spec, script, dataset, log_dir = _failure_fixture(tmp_path)
    process_hash = _git_hash(script)
    stdout = tmp_path / "logdir" / "job.o"
    stderr = stdout.with_suffix(".e")
    stderr.parent.mkdir()
    stderr.write_text(
        "Traceback (most recent call last):\n"
        "  File 'mne/source_space.py', line 10, in setup_source_space\n"
        "ValueError: source space creation failed\n\n"
        "The above exception was the direct cause of the following exception:\n\n"
        "Traceback (most recent call last):\n"
        "  File 'task_v1.py', line 12, in <module>\n"
        "RuntimeError: Failed to create src\n",
        encoding="utf-8",
    )
    (log_dir / "task_v1.log").write_text(
        f"INFO START name=task_v1 version=1 hash={process_hash}\n"
        f"INFO Primary input: {dataset}\n"
        f"INFO SLURM_LOGFILE_PATH :: {stdout}\n",
        encoding="utf-8",
    )

    report = find_processing_failure(spec, dataset)

    assert report.details.startswith("Traceback")
    assert "mne/source_space.py" in report.details
    assert "ValueError: source space creation failed" in report.details
    assert "RuntimeError: Failed to create src" in report.details


def test_find_processing_failure_uses_legacy_log_only_when_unambiguous(tmp_path):
    spec, script, dataset, log_dir = _failure_fixture(tmp_path)
    process_hash = _git_hash(script)
    logfile = log_dir / "old.log"
    logfile.write_text(
        f"INFO START :: {process_hash}\n"
        "ERROR Failed to create epochs\n"
        "Traceback (most recent call last):\nRuntimeError: autoreject failed\n"
    )

    ambiguous = find_processing_failure(spec, dataset)
    report = find_processing_failure(spec, dataset, allow_legacy_fallback=True)

    assert not ambiguous.error_found
    assert "does not identify the input run" in ambiguous.summary
    assert report.error_found
    assert report.application_log == logfile
    assert "RuntimeError: autoreject failed" in report.details


def test_find_processing_failure_reports_completed_but_missing_artifact(tmp_path):
    spec, script, dataset, log_dir = _failure_fixture(tmp_path)
    process_hash = _git_hash(script)
    (log_dir / "task_v1.log").write_text(
        f"INFO START name=task_v1 version=1 hash={process_hash}\n"
        f"INFO Primary input: {dataset}\n"
        f"INFO FINISHED name=task_v1 version=1 hash={process_hash}\n"
    )

    report = find_processing_failure(spec, dataset)

    assert not report.error_found
    assert "finished successfully" in report.summary

