import hashlib
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from mne_bids import BIDSPath

from nih2mne.proc import megcore_utils


def _raw_bids_path(root, *, subject="01", session="01", task="test", run="01"):
    return BIDSPath(
        root=root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="meg",
        suffix="meg",
        extension=".ds",
    )


def _make_dataset(bids_path):
    Path(bids_path.fpath).mkdir(parents=True)
    return bids_path


def test_find_emptyroom_dataset_matches_case_and_selects_longest_local(
    tmp_path, monkeypatch
):
    root = tmp_path / "bids"
    primary = _make_dataset(_raw_bids_path(root))
    short = _make_dataset(
        _raw_bids_path(root, task="NoIsE", run="02")
    )
    longest = _make_dataset(
        _raw_bids_path(root, task="EMPTYroom", run="03")
    )
    durations = {Path(short.fpath): 100.0, Path(longest.fpath): 360.0}
    monkeypatch.setattr(
        megcore_utils,
        "_ctf_recording_duration",
        lambda path: durations[path],
    )

    selected = megcore_utils.find_emptyroom_dataset(
        primary, task_keywords=["empty", "noise"]
    )

    assert Path(selected.fpath) == Path(longest.fpath)


def test_find_emptyroom_dataset_reads_keywords_from_defaults(tmp_path, monkeypatch):
    root = tmp_path / "bids"
    primary = _make_dataset(_raw_bids_path(root))
    expected = _make_dataset(_raw_bids_path(root, task="Ambient", run="02"))
    fake_config = ModuleType("nih2mne.config")
    fake_config.DEFAULTS = {
        "processing": {"emptyroom_task_keywords": ["ambient"]}
    }
    monkeypatch.setitem(sys.modules, "nih2mne.config", fake_config)
    monkeypatch.setattr(
        megcore_utils,
        "_ctf_recording_duration",
        lambda path: 100.0,
    )

    selected = megcore_utils.find_emptyroom_dataset(primary)

    assert Path(selected.fpath) == Path(expected.fpath)


def test_find_emptyroom_dataset_skips_unreadable_candidate(tmp_path, monkeypatch):
    root = tmp_path / "bids"
    primary = _make_dataset(_raw_bids_path(root))
    broken = _make_dataset(_raw_bids_path(root, task="noise", run="02"))
    expected = _make_dataset(_raw_bids_path(root, task="noise", run="03"))

    def duration(path):
        if path == Path(broken.fpath):
            raise OSError("bad CTF header")
        return 100.0

    monkeypatch.setattr(megcore_utils, "_ctf_recording_duration", duration)

    selected = megcore_utils.find_emptyroom_dataset(primary, ["noise"])

    assert Path(selected.fpath) == Path(expected.fpath)


def test_find_emptyroom_dataset_prefers_any_readable_local_match(
    tmp_path, monkeypatch
):
    root = tmp_path / "bids"
    primary = _make_dataset(_raw_bids_path(root))
    local = _make_dataset(_raw_bids_path(root, task="noise", run="02"))
    fallback = _make_dataset(
        _raw_bids_path(
            root,
            subject="EmptyRoom",
            session="different",
            task="noise",
            run="99",
        )
    )
    durations = {Path(local.fpath): 50.0, Path(fallback.fpath): 500.0}
    monkeypatch.setattr(
        megcore_utils,
        "_ctf_recording_duration",
        lambda path: durations[path],
    )

    selected = megcore_utils.find_emptyroom_dataset(primary, ["noise"])

    assert Path(selected.fpath) == Path(local.fpath)


def test_find_emptyroom_dataset_uses_longest_fallback_across_sessions(
    tmp_path, monkeypatch
):
    root = tmp_path / "bids"
    primary = _make_dataset(_raw_bids_path(root))
    first = _make_dataset(
        _raw_bids_path(
            root, subject="EmptyRoom", session="A", task="noise", run="01"
        )
    )
    longest = _make_dataset(
        _raw_bids_path(
            root, subject="EmptyRoom", session="B", task="EmptyRoom", run="07"
        )
    )
    durations = {Path(first.fpath): 100.0, Path(longest.fpath): 300.0}
    monkeypatch.setattr(
        megcore_utils,
        "_ctf_recording_duration",
        lambda path: durations[path],
    )

    selected = megcore_utils.find_emptyroom_dataset(primary, ["empty", "noise"])

    assert Path(selected.fpath) == Path(longest.fpath)


def test_find_emptyroom_dataset_skips_unreadable_and_reports_total_failure(
    tmp_path, monkeypatch
):
    root = tmp_path / "bids"
    primary = _make_dataset(_raw_bids_path(root))
    broken = _make_dataset(_raw_bids_path(root, task="noise", run="02"))

    def fail(path):
        raise OSError("bad CTF header")

    monkeypatch.setattr(megcore_utils, "_ctf_recording_duration", fail)

    with pytest.raises(FileNotFoundError, match="bad CTF header") as error:
        megcore_utils.find_emptyroom_dataset(primary, ["noise"])

    assert str(Path(broken.fpath)) in str(error.value)
    assert str(root / "sub-EmptyRoom") in str(error.value)


def _record(path: Path, *, exists=False, status="missing", errors=(), key="artifact"):
    return SimpleNamespace(
        path=path,
        exists=exists,
        status=status,
        key=key,
        validation_errors=list(errors),
    )


def _provenance(bids_root, *, inputs):
    context = megcore_utils.ProvenanceContext(
        bids_root=Path(bids_root).resolve(),
        script_filename="task_v1.py",
        script_hash="abc123",
        packages={"mne": "1.2.3", "mne-bids": "0.16.0", "nih2mne": "0.6.2"},
    )
    return context.derivative(
        calls=["raw.filter(l_freq=10, h_freq=25, n_jobs=4)"],
        inputs=inputs,
    )


def test_git_blob_hash(tmp_path):
    script = tmp_path / "script.py"
    data = b"print('hello')\n"
    script.write_bytes(data)

    expected = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    assert megcore_utils.git_blob_hash(script) == expected


def test_portable_path_preserves_linked_directory_below_bids_root(tmp_path):
    bids_root = tmp_path / "bids"
    external_subjects = tmp_path / "shared" / "subjects"
    external_subjects.mkdir(parents=True)
    freesurfer = bids_root / "derivatives" / "freesurfer"
    freesurfer.parent.mkdir(parents=True)
    freesurfer.symlink_to(external_subjects, target_is_directory=True)
    subject = freesurfer / "sub-01"

    context = megcore_utils.ProvenanceContext(
        bids_root=bids_root,
        script_filename="task_v1.py",
        script_hash="abc123",
    )

    assert context.portable_path(subject) == (
        "{bids_root}/derivatives/freesurfer/sub-01"
    )


def test_portable_path_supports_symlinked_bids_root(tmp_path):
    actual_root = tmp_path / "actual-bids"
    subject = actual_root / "sub-01" / "meg"
    subject.mkdir(parents=True)
    linked_root = tmp_path / "linked-bids"
    linked_root.symlink_to(actual_root, target_is_directory=True)

    context = megcore_utils.ProvenanceContext(
        bids_root=linked_root,
        script_filename="task_v1.py",
        script_hash="abc123",
    )

    assert context.portable_path(linked_root / "sub-01" / "meg") == (
        "{bids_root}/sub-01/meg"
    )


def test_slurm_context_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert megcore_utils.slurm_context() == (-1, None)


def test_slurm_context_reads_worker_count_and_output(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    run_calls = []

    def fake_run(args, **kwargs):
        run_calls.append((args, kwargs))
        return SimpleNamespace(stdout="JobId=123 StdOut=/tmp/slurm-123.out")

    monkeypatch.setattr(megcore_utils.subprocess, "run", fake_run)

    assert megcore_utils.slurm_context() == (8, "/tmp/slurm-123.out")
    assert run_calls == [
        (["scontrol", "show", "job", "123"], {"capture_output": True, "text": True})
    ]


def test_slurm_context_tolerates_missing_scontrol(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")

    def fail(*args, **kwargs):
        raise FileNotFoundError("scontrol")

    monkeypatch.setattr(megcore_utils.subprocess, "run", fail)
    assert megcore_utils.slurm_context() == (4, None)


def test_make_logger_uses_caller_identity_and_replaces_handlers(tmp_path):
    logger_name = "nih2mne.tests.megcore_utils"
    logger = logging.getLogger(logger_name)
    old_handler = logging.NullHandler()
    logger.addHandler(old_handler)

    result = megcore_utils.make_logger(
        tmp_path / "logs" / "pipeline.log", logger_name=logger_name
    )
    try:
        assert result is logger
        assert result.name == logger_name
        assert result.level == logging.INFO
        assert result.propagate is False
        assert len(result.handlers) == 1
        assert isinstance(result.handlers[0], logging.FileHandler)
        assert old_handler not in result.handlers
    finally:
        for handler in result.handlers:
            handler.close()
        result.handlers.clear()


def test_load_or_make_loads_valid_artifact(tmp_path, caplog):
    record = _record(tmp_path / "artifact.fif", exists=True, status="valid")
    record.load_file = lambda: "loaded"

    assert megcore_utils.load_or_make(
        record,
        lambda: pytest.fail("make should not be called"),
        lambda value, path: pytest.fail("save should not be called"),
        logging.getLogger("test_load"),
    ) == "loaded"


def test_load_or_make_logs_and_reraises_original_load_error(tmp_path):
    record = _record(
        tmp_path / "artifact.fif", exists=True, status="valid"
    )
    original = OSError("MNE read failed")
    record.load_file = lambda: (_ for _ in ()).throw(original)
    log_path = tmp_path / "pipeline.log"
    logger = megcore_utils.make_logger(log_path, logger_name="test_load_error")

    try:
        with pytest.raises(OSError, match="MNE read failed") as error:
            megcore_utils.load_or_make(
                record,
                lambda: pytest.fail("make should not be called"),
                lambda value, path: pytest.fail("save should not be called"),
                logger,
            )
        assert error.value is original
        logged = log_path.read_text(encoding="utf-8")
        assert f"Failed to load artifact: {record.path}" in logged
        assert "Traceback (most recent call last):" in logged
        assert "OSError: MNE read failed" in logged
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


def test_load_or_make_creates_missing_artifact(tmp_path):
    record = _record(tmp_path / "nested" / "artifact.fif")
    saved = []

    result = megcore_utils.load_or_make(
        record,
        lambda: "created",
        lambda value, path: saved.append((value, path)),
        logging.getLogger("test_create"),
    )

    assert result == "created"
    assert saved == [("created", record.path)]
    assert record.path.parent.is_dir()


def test_load_or_make_overwrites_valid_artifact(tmp_path):
    record = _record(tmp_path / "artifact.fif", exists=True, status="present")
    record.load_file = lambda: pytest.fail("load should not be called")
    saved = []

    result = megcore_utils.load_or_make(
        record,
        lambda: "replacement",
        lambda value, path: saved.append((value, path)),
        logging.getLogger("test_overwrite"),
        overwrite=True,
    )

    assert result == "replacement"
    assert saved == [("replacement", record.path)]


def test_load_or_make_rejects_invalid_artifact(tmp_path):
    path = tmp_path / "artifact.fif"
    path.touch()
    record = _record(path, status="invalid", errors=["metadata mismatch"])

    with pytest.raises(RuntimeError, match="metadata mismatch"):
        megcore_utils.load_or_make(
            record, lambda: None, lambda value, target: None, logging.getLogger("test")
        )


def test_load_or_make_logs_and_reraises_original_creation_error(tmp_path):
    record = _record(tmp_path / "artifact.fif")
    original = ValueError("creation failed")
    log_path = tmp_path / "pipeline.log"
    logger = megcore_utils.make_logger(log_path, logger_name="test_creation_error")

    def fail():
        raise original

    try:
        with pytest.raises(ValueError, match="creation failed") as error:
            megcore_utils.load_or_make(
                record, fail, lambda value, target: None, logger
            )
        assert error.value is original
        logged = log_path.read_text(encoding="utf-8")
        assert f"Failed to create artifact: {record.path}" in logged
        assert "Traceback (most recent call last):" in logged
        assert "ValueError: creation failed" in logged
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


def test_load_or_make_writes_portable_provenance_for_created_artifact(tmp_path):
    bids_root = tmp_path / "bids"
    raw_path = bids_root / "sub-01" / "meg" / "sub-01_task-test_meg.ds"
    raw_path.mkdir(parents=True)
    output = bids_root / "derivatives" / "project" / "sub-01_task-test_epo.fif"
    record = _record(output, key="epochs")
    provenance = _provenance(bids_root, inputs={"raw": raw_path})

    megcore_utils.load_or_make(
        record,
        lambda: "epochs",
        lambda value, path: path.write_text(value),
        logging.getLogger("test_provenance"),
        provenance=provenance,
    )

    sidecar = output.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())["NIH2MNEProvenance"]
    assert metadata["Inputs"] == {
        "raw": "{bids_root}/sub-01/meg/sub-01_task-test_meg.ds"
    }
    assert metadata["Script"] == {
        "Filename": "task_v1.py",
        "Hash": "abc123",
        "HashType": "git-blob-sha1",
    }
    assert metadata["Calls"] == ["raw.filter(l_freq=10, h_freq=25, n_jobs=4)"]
    assert metadata["Packages"]["nih2mne"] == "0.6.2"
    assert metadata["ProcessingDateTime"].endswith("Z")
    assert str(bids_root) not in sidecar.read_text()


def test_provenance_rejects_external_input_before_derivative_creation(tmp_path):
    bids_root = tmp_path / "bids"
    record = _record(
        bids_root / "derivatives" / "project" / "artifact.fif",
        key="forward",
    )
    provenance = _provenance(
        bids_root,
        inputs={"raw": tmp_path / "external" / "raw.ds"},
    )
    made = []

    with pytest.raises(ValueError, match="outside the BIDS root") as error:
        megcore_utils.load_or_make(
            record,
            lambda: made.append(True),
            lambda value, path: None,
            logging.getLogger("test_external_provenance"),
            provenance=provenance,
        )

    assert made == []
    assert isinstance(error.value, ValueError)
    assert "outside the BIDS root" in str(error.value)


def test_load_or_make_does_not_backfill_reused_artifact(tmp_path):
    bids_root = tmp_path / "bids"
    output = bids_root / "derivatives" / "project" / "artifact.fif"
    output.parent.mkdir(parents=True)
    output.touch()
    record = _record(output, exists=True, status="present", key="forward")
    record.load_file = lambda: "cached"
    provenance = _provenance(bids_root, inputs={"raw": bids_root / "raw.ds"})

    result = megcore_utils.load_or_make(
        record,
        lambda: pytest.fail("make should not be called"),
        lambda value, path: pytest.fail("save should not be called"),
        logging.getLogger("test_no_backfill"),
        provenance=provenance,
    )

    assert result == "cached"
    assert not output.with_suffix(".json").exists()


def test_surface_stc_provenance_uses_base_json_path(tmp_path):
    record = _record(tmp_path / "sub-01_desc-face_stc", key="stc")

    assert megcore_utils.provenance_sidecar_path(record) == (
        tmp_path / "sub-01_desc-face_stc.json"
    )


def test_save_filtered_writes_data_and_metadata(tmp_path):
    data_record = _record(tmp_path / "nested" / "raw.fif")
    json_record = _record(tmp_path / "nested" / "raw.json")
    json_record.expected_metadata = {"SoftwareFilters": {"HighpassCutoffHz": 1}}
    calls = []
    raw = SimpleNamespace(
        save=lambda path, overwrite: calls.append((path, overwrite))
    )

    megcore_utils.save_filtered(
        raw, data_record, json_record, False, logging.getLogger("test")
    )

    assert calls == [(data_record.path, False)]
    assert json.loads(json_record.path.read_text()) == json_record.expected_metadata


def test_save_filtered_rejects_invalid_pair_without_overwrite(tmp_path):
    data_record = _record(
        tmp_path / "raw.fif", status="invalid", errors=["expected a regular file"]
    )
    json_record = _record(tmp_path / "raw.json")
    json_record.expected_metadata = {}

    with pytest.raises(RuntimeError, match="Invalid filtered artifact pair"):
        megcore_utils.save_filtered(
            SimpleNamespace(save=lambda *args, **kwargs: None),
            data_record,
            json_record,
            False,
            logging.getLogger("test"),
        )


def test_save_filtered_repairs_invalid_pair_with_overwrite(tmp_path):
    data_record = _record(tmp_path / "raw.fif", status="invalid", errors=["bad data"])
    json_record = _record(
        tmp_path / "raw.json", status="invalid", errors=["bad metadata"]
    )
    json_record.expected_metadata = {"SoftwareFilters": {"HighpassCutoffHz": 1}}
    calls = []

    megcore_utils.save_filtered(
        SimpleNamespace(save=lambda path, overwrite: calls.append((path, overwrite))),
        data_record,
        json_record,
        True,
        logging.getLogger("test"),
    )

    assert calls == [(data_record.path, True)]
    assert json.loads(json_record.path.read_text()) == json_record.expected_metadata


def test_save_filtered_merges_provenance_into_derivative_json(tmp_path):
    bids_root = tmp_path / "bids"
    raw_input = bids_root / "sub-01" / "meg" / "sub-01_task-test_meg.ds"
    raw_input.mkdir(parents=True)
    data_record = _record(
        bids_root / "derivatives" / "project" / "sub-01_proc-filt_meg.fif",
        key="filtered_raw",
    )
    json_record = _record(
        data_record.path.with_suffix(".json"), key="filtered_raw_json"
    )
    json_record.expected_metadata = {
        "Sources": ["bids::sub-01/meg/sub-01_task-test_meg.ds"],
        "SoftwareFilters": {"MNE-Python": {"HighpassCutoffHz": 10}},
    }
    provenance = _provenance(bids_root, inputs={"raw": raw_input})
    raw_sidecar = raw_input.parent / "sub-01_task-test_meg.json"
    raw_sidecar.write_text('{"RawMetadata": true}\n')

    megcore_utils.save_filtered(
        SimpleNamespace(save=lambda path, overwrite: path.write_text("filtered")),
        data_record,
        json_record,
        False,
        logging.getLogger("test_filtered_provenance"),
        provenance=provenance,
    )

    metadata = json.loads(json_record.path.read_text())
    assert metadata["Sources"] == json_record.expected_metadata["Sources"]
    assert metadata["SoftwareFilters"] == json_record.expected_metadata["SoftwareFilters"]
    assert metadata["NIH2MNEProvenance"]["Inputs"]["raw"].startswith(
        "{bids_root}/"
    )
    assert json.loads(raw_sidecar.read_text()) == {"RawMetadata": True}


def test_save_if_requested_preserves_valid_and_rejects_invalid(tmp_path):
    saved = []
    valid = _record(tmp_path / "valid.fif", exists=True, status="valid")
    megcore_utils.save_if_requested(
        "value", valid, lambda value, path: saved.append(path), False,
        logging.getLogger("test"),
    )
    assert saved == []

    invalid = _record(
        tmp_path / "invalid.fif", status="invalid", errors=["partial artifact"]
    )
    with pytest.raises(RuntimeError, match="partial artifact"):
        megcore_utils.save_if_requested(
            "value", invalid, lambda value, path: saved.append(path), False,
            logging.getLogger("test"),
        )


def test_save_if_requested_logs_and_reraises_original_save_error(tmp_path):
    record = _record(tmp_path / "artifact.fif")
    original = OSError("MNE save failed")
    log_path = tmp_path / "pipeline.log"
    logger = megcore_utils.make_logger(log_path, logger_name="test_save_error")

    def fail(value, path):
        raise original

    try:
        with pytest.raises(OSError, match="MNE save failed") as error:
            megcore_utils.save_if_requested(
                "value", record, fail, False, logger
            )
        assert error.value is original
        logged = log_path.read_text(encoding="utf-8")
        assert f"Failed to save artifact: {record.path}" in logged
        assert "Traceback (most recent call last):" in logged
        assert "OSError: MNE save failed" in logged
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


def test_save_if_requested_writes_h5_provenance_sidecar(tmp_path):
    bids_root = tmp_path / "bids"
    epochs = bids_root / "derivatives" / "project" / "sub-01_epo.fif"
    output = bids_root / "derivatives" / "project" / "sub-01_lcmv.h5"
    record = _record(output, key="lcmv")
    provenance = _provenance(bids_root, inputs={"epochs": epochs})

    megcore_utils.save_if_requested(
        "filters",
        record,
        lambda value, path: path.write_text(value),
        False,
        logging.getLogger("test_save_provenance"),
        provenance=provenance,
    )

    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["NIH2MNEProvenance"]["Inputs"] == {
        "epochs": "{bids_root}/derivatives/project/sub-01_epo.fif"
    }


@pytest.mark.parametrize(
    "diagonal, warning",
    [
        ([1.0, 1e-7], "moderate conditioning"),
        ([1.0, 1e-9], "poorly conditioned"),
    ],
)
def test_check_covariance_warns(diagonal, warning, caplog):
    covariance = SimpleNamespace(data=np.diag(diagonal))
    with caplog.at_level(logging.WARNING):
        megcore_utils.check_covariance("Data", covariance, logging.getLogger("cov"))
    assert warning in caplog.text


def test_check_covariance_rejects_ill_conditioned_matrix():
    covariance = SimpleNamespace(data=np.diag([1.0, 1e-12]))
    with pytest.raises(ValueError, match="Data covariance is ill-conditioned"):
        megcore_utils.check_covariance(
            "Data", covariance, logging.getLogger("cov")
        )


def test_normalize_beamfilt_ori_flips_opposing_orientations():
    forward = {
        "src": [
            {
                "nn": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                "vertno": np.array([0, 1]),
            }
        ]
    }
    filters = {
        "weights": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "max_power_ori": np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }

    megcore_utils.normalize_beamfilt_ori(forward, filters)

    np.testing.assert_array_equal(filters["weights"], [[-1.0, -2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        filters["max_power_ori"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )


def test_ensure_dataset_description_uses_caller_identity(tmp_path, monkeypatch):
    record = _record(tmp_path / "derivatives" / "dataset_description.json")
    calls = []
    monkeypatch.setattr(
        megcore_utils.mne_bids,
        "make_dataset_description",
        lambda **kwargs: calls.append(kwargs),
    )

    megcore_utils.ensure_dataset_description(
        record,
        "preprocessing",
        logging.getLogger("test"),
        generated_by="airpuff_v7",
        generated_by_version="7.0",
    )

    assert calls == [
        {
            "path": record.path.parent,
            "name": "preprocessing",
            "dataset_type": "derivative",
            "generated_by": [{"Name": "airpuff_v7", "Version": "7.0"}],
            "overwrite": False,
        }
    ]


def test_ensure_dataset_description_rejects_invalid_without_overwrite(tmp_path):
    record = _record(
        tmp_path / "dataset_description.json",
        status="invalid",
        errors=["DatasetType must be derivative"],
    )

    with pytest.raises(RuntimeError, match="DatasetType must be derivative"):
        megcore_utils.ensure_dataset_description(
            record,
            "project",
            logging.getLogger("test"),
            generated_by="routine",
        )


def test_ensure_dataset_description_repairs_invalid_with_overwrite(
    tmp_path, monkeypatch
):
    record = _record(
        tmp_path / "dataset_description.json",
        status="invalid",
        errors=["invalid JSON"],
    )
    calls = []
    monkeypatch.setattr(
        megcore_utils.mne_bids,
        "make_dataset_description",
        lambda **kwargs: calls.append(kwargs),
    )

    megcore_utils.ensure_dataset_description(
        record,
        "project",
        logging.getLogger("test"),
        generated_by="routine",
        overwrite=True,
    )

    assert calls[0]["overwrite"] is True
