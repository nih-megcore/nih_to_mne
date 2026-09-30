import json
from pathlib import Path

import pytest
from mne_bids import BIDSPath

from nih2mne import bids_templates
from nih2mne.bids_templates import (
    ArtifactSpec,
    FilterSpec,
    MEGDerivativeTemplate,
)


def _raw_path(
    root: Path,
    *,
    subject="ON08710",
    session="1",
    task="rest",
    run="01",
) -> BIDSPath:
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


def _template(tmp_path: Path, **kwargs) -> MEGDerivativeTemplate:
    return MEGDerivativeTemplate(
        _raw_path(tmp_path / "bids"),
        project="ENIGMA_MEG",
        filter_spec=FilterSpec(
            l_freq=1,
            h_freq=40,
            notch_freqs=(120, 60, 60),
            resample_sfreq=250,
            method="fir",
            phase="zero",
        ),
        **kwargs,
    )


def test_filter_spec_builds_stable_bids_labels_and_metadata():
    spec = FilterSpec(
        l_freq=0.5,
        h_freq=100,
        notch_freqs=(120, 60, 60),
        resample_sfreq=250,
        method="fir",
        phase="zero",
        variant="analysis1",
    )
    assert spec.processing_label == "filt0p5to100n60x120r250analysis1"
    assert spec.notch_freqs == (60.0, 120.0)
    assert spec.software_filter_metadata() == {
        "MNE-Python": {
            "HighpassCutoffHz": 0.5,
            "LowpassCutoffHz": 100.0,
            "NotchFrequenciesHz": [60.0, 120.0],
            "ResampleFrequencyHz": 250.0,
            "Method": "fir",
            "Phase": "zero",
        }
    }

    assert FilterSpec(l_freq=1).processing_label == "filt1toinf"
    assert FilterSpec(h_freq=40).processing_label == "filt0to40"
    assert FilterSpec(notch_freqs=(60, 120)).processing_label == "filtn60x120"
    assert FilterSpec(resample_sfreq=250).processing_label == "filtr250"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({}, "filtering or resampling"),
        ({"l_freq": 40, "h_freq": 1}, "less than"),
        ({"l_freq": -1}, "greater than zero"),
        ({"h_freq": float("inf")}, "finite"),
        ({"notch_freqs": (0,)}, "greater than zero"),
        ({"resample_sfreq": True}, "real number"),
        ({"l_freq": 1, "variant": "bad-label"}, "letters and numbers"),
    ],
)
def test_filter_spec_rejects_invalid_recipes(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        FilterSpec(**kwargs)


def test_tree_aligned_paths_and_filter_propagation(tmp_path):
    artifacts = _template(tmp_path, space="fsaverage").evaluate()
    bids_root = tmp_path / "bids"
    proc = "proc-filt1to40n60x120r250"
    primary_prefix = f"sub-ON08710_ses-1_task-rest_run-01_{proc}"

    assert artifacts["raw"].path == (
        bids_root
        / "sub-ON08710/ses-1/meg/sub-ON08710_ses-1_task-rest_run-01_meg.ds"
    )
    assert artifacts["bem"].path == (
        bids_root
        / "derivatives/preprocessing/sub-ON08710/ses-1/meg/sub-ON08710_ses-1_bem.fif"
    )
    assert artifacts["src"].path.name == "sub-ON08710_ses-1_space-fsaverage_src.fif"
    assert artifacts["trans"].path.name == (
        "sub-ON08710_ses-1_task-rest_run-01_trans.fif"
    )
    assert artifacts["fwd"].path.name == (
        "sub-ON08710_ses-1_task-rest_run-01_space-fsaverage_fwd.fif"
    )
    assert artifacts["filtered_raw"].path.name == f"{primary_prefix}_meg.fif"
    assert artifacts["filtered_raw_json"].path.name == f"{primary_prefix}_meg.json"
    assert artifacts["epochs"].path.name == f"{primary_prefix}_epo.fif"
    assert artifacts["covariance"].path.name == f"{primary_prefix}_cov.fif"
    assert artifacts["ica_dir"].path.name == f"{primary_prefix}_ica"
    assert artifacts["ica"].path == (
        artifacts["ica_dir"].path / f"{primary_prefix}_ica.fif"
    )
    assert artifacts["lcmv"].path.name == (
        f"{primary_prefix}_space-fsaverage_lcmv.h5"
    )
    assert "proc-" not in artifacts["bem"].path.name
    assert "proc-" not in artifacts["fwd"].path.name


def test_optional_bids_entities_are_omitted(tmp_path):
    primary = _raw_path(tmp_path / "bids", session=None, run=None)
    template = MEGDerivativeTemplate(
        primary,
        project="project",
        filter_spec=FilterSpec(l_freq=1, h_freq=40),
    )
    artifacts = template.evaluate()
    assert artifacts["epochs"].path.name == (
        "sub-ON08710_task-rest_proc-filt1to40_epo.fif"
    )
    assert artifacts["bem"].path.name == "sub-ON08710_bem.fif"


def test_noise_bidspath_uses_primary_subject_and_noise_entities(tmp_path):
    noise = _raw_path(
        tmp_path / "external",
        subject="emptyroom",
        session="20200101",
        task="noise",
        run="03",
    )
    artifacts = _template(tmp_path, noise=noise).evaluate()

    assert artifacts["noise_raw"].path == Path(noise.fpath)
    assert artifacts["noise_filtered_raw"].path.name == (
        "sub-ON08710_ses-1_task-noise_run-03_"
        "proc-filt1to40n60x120r250_meg.fif"
    )
    assert artifacts["noise_covariance"].path.parent == artifacts["filtered_raw"].path.parent


def test_external_noise_path_defaults_and_entity_overrides(tmp_path):
    external = tmp_path / "noise" / "room-noise.fif"
    default = _template(tmp_path, noise=external).evaluate()
    assert default["noise_raw"].path == external.resolve()
    assert default["noise_epochs"].path.name.startswith(
        "sub-ON08710_ses-1_task-noise_proc-"
    )

    configured = _template(
        tmp_path / "configured",
        noise=external,
        noise_entities={"task": "emptyroom", "run": "04"},
    ).evaluate()
    assert "task-emptyroom_run-04_proc-" in configured["noise_epochs"].path.name


def test_evaluation_reports_files_directories_and_metadata(tmp_path):
    template = _template(tmp_path)
    initial = template.evaluate()
    assert all(record.status == "missing" for record in initial.values())

    Path(template.primary.fpath).mkdir(parents=True)
    initial["ica_dir"].path.mkdir(parents=True)
    initial["filtered_raw"].path.touch()
    initial["filtered_raw_json"].path.write_text(
        json.dumps(initial["filtered_raw_json"].expected_metadata)
    )
    for key, generated_by in (
        ("preprocessing_dataset_description", "preprocessing"),
        ("project_dataset_description", "ENIGMA_MEG"),
    ):
        initial[key].path.parent.mkdir(parents=True, exist_ok=True)
        initial[key].path.write_text(json.dumps({
            "Name": generated_by,
            "BIDSVersion": "1.11.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": generated_by}],
        }))

    evaluated = template.evaluate()
    assert evaluated["raw"].status == "present"
    assert evaluated["ica_dir"].status == "present"
    assert evaluated["filtered_raw"].status == "present"
    assert evaluated["filtered_raw_json"].status == "present"
    assert evaluated["project_dataset_description"].status == "present"

    evaluated["filtered_raw_json"].path.write_text(json.dumps({
        "Description": "wrong recipe",
        "Sources": evaluated["filtered_raw_json"].expected_metadata["Sources"],
        "SoftwareFilters": {"MNE-Python": {"HighpassCutoffHz": 2}},
    }))
    invalid = template.evaluate()["filtered_raw_json"]
    assert invalid.status == "invalid"
    assert any("HighpassCutoffHz" in error for error in invalid.validation_errors)


@pytest.mark.parametrize(
    "contents, expected_error",
    [
        ("not json", "invalid JSON"),
        (json.dumps([]), "JSON object"),
        (json.dumps({"DatasetType": "raw"}), "Name"),
        (
            json.dumps({
                "Name": "project",
                "BIDSVersion": "1.11.0",
                "DatasetType": "derivative",
                "GeneratedBy": [{}],
            }),
            "GeneratedBy",
        ),
    ],
)
def test_dataset_description_validation(tmp_path, contents, expected_error):
    template = _template(tmp_path)
    record = template.evaluate()["project_dataset_description"]
    record.path.parent.mkdir(parents=True)
    record.path.write_text(contents)
    record = template.evaluate()["project_dataset_description"]
    assert record.status == "invalid"
    assert any(expected_error in error for error in record.validation_errors)


def test_load_file_dispatches_and_rejects_missing_or_directory(tmp_path, monkeypatch):
    template = _template(tmp_path)
    records = template.evaluate()
    with pytest.raises(FileNotFoundError):
        records["epochs"].load_file()

    records["epochs"].path.parent.mkdir(parents=True)
    records["epochs"].path.touch()
    records["ica_dir"].path.mkdir()
    records = template.evaluate()
    calls = []

    def fake_loader(loader, target, kwargs):
        calls.append((loader, target, kwargs))
        return "loaded"

    monkeypatch.setattr(bids_templates, "_run_loader", fake_loader)
    assert records["epochs"].load_file(preload=False) == "loaded"
    assert calls == [("epochs", records["epochs"].path, {"preload": False})]
    with pytest.raises(TypeError, match="does not have"):
        records["ica_dir"].load_file()


def test_project_validation_and_raw_input_validation(tmp_path):
    raw = _raw_path(tmp_path / "bids")
    filter_spec = FilterSpec(l_freq=1)
    with pytest.raises(ValueError, match="project"):
        MEGDerivativeTemplate(raw, project="../bad", filter_spec=filter_spec)
    with pytest.raises(TypeError, match="BIDSPath"):
        MEGDerivativeTemplate(Path(raw.fpath), project="good", filter_spec=filter_spec)

    incomplete = BIDSPath(
        root=tmp_path / "bids", subject="01", datatype="meg", suffix="meg",
        extension=".fif",
    )
    with pytest.raises(ValueError, match="task"):
        MEGDerivativeTemplate(incomplete, project="good", filter_spec=filter_spec)


def test_schema_can_be_extended_and_duplicate_keys_are_rejected(tmp_path):
    class ExtendedTemplate(MEGDerivativeTemplate):
        ARTIFACT_SPECS = MEGDerivativeTemplate.ARTIFACT_SPECS + (
            ArtifactSpec(
                "report",
                "project",
                "primary",
                "file",
                suffix="report",
                extension=".html",
                entities=("subject", "session", "task", "run"),
                uses_filter=True,
            ),
        )

    base = _template(tmp_path)
    extended = ExtendedTemplate(
        base.primary,
        project=base.project,
        filter_spec=base.filter_spec,
    )
    assert extended.evaluate()["report"].path.name.endswith(
        "_proc-filt1to40n60x120r250_report.html"
    )

    class InvalidTemplate(MEGDerivativeTemplate):
        ARTIFACT_SPECS = MEGDerivativeTemplate.ARTIFACT_SPECS + (
            ArtifactSpec("bem", "project", "shared", "file", "bem", ".fif"),
        )

    with pytest.raises(ValueError, match="duplicate artifact keys"):
        InvalidTemplate(
            base.primary,
            project=base.project,
            filter_spec=base.filter_spec,
        )
