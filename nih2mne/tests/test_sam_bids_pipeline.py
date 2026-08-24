import json
from pathlib import Path
from types import SimpleNamespace

import dill
import pytest
import yaml

from nih2mne.proc import sam_bids_pipeline as pipeline


def _dataset(tmp_path: Path, *, session="01", run="02") -> tuple[Path, Path]:
    bids_root = tmp_path / "bids"
    parent = bids_root / "sub-01"
    tokens = ["sub-01"]
    if session is not None:
        parent /= f"ses-{session}"
        tokens.append(f"ses-{session}")
    parent /= "meg"
    tokens.append("task-rest")
    if run is not None:
        tokens.append(f"run-{run}")
    tokens.append("meg")
    dataset = parent / ("_".join(tokens) + ".ds")
    dataset.mkdir(parents=True)
    return bids_root, dataset


def _mri_and_megqa(bids_root: Path) -> Path:
    mri = bids_root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    mri.parent.mkdir(parents=True, exist_ok=True)
    mri.touch()
    mri.with_name("sub-01_T1w.json").write_text(
        json.dumps(
            {
                "AnatomicalLandmarkCoordinates": {
                    "NAS": [1, 2, 3],
                    "LPA": [4, 5, 6],
                    "RPA": [7, 8, 9],
                }
            }
        )
    )
    qa_path = bids_root / "derivatives" / "megQA" / "sub-01.yml"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(yaml.safe_dump({
        "schema_version": 1.0,
        "subject": "sub-01",
        "bids_root": str(bids_root),
        "mri": str(mri),
        "meg_list": [],
    }, sort_keys=False))
    return mri


def test_parse_bids_dataset_with_optional_entities(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    context = pipeline.parse_bids_dataset(dataset)
    assert context.bids_root == bids_root
    assert (context.subject, context.session, context.run) == (
        "sub-01",
        "ses-01",
        "run-02",
    )

    bids_root, dataset = _dataset(tmp_path / "sessionless", session=None, run=None)
    context = pipeline.parse_bids_dataset(dataset)
    assert context.bids_root == bids_root
    assert context.session is None
    assert context.run is None


def test_parse_bids_dataset_rejects_relative_and_inconsistent_paths(tmp_path):
    with pytest.raises(pipeline.PipelineError, match="fully qualified"):
        pipeline.parse_bids_dataset("sub-01_task-rest_meg.ds")

    _root, dataset = _dataset(tmp_path)
    inconsistent = dataset.with_name(dataset.name.replace("sub-01", "sub-02", 1))
    dataset.rename(inconsistent)
    with pytest.raises(pipeline.PipelineError, match="does not match directory"):
        pipeline.parse_bids_dataset(inconsistent)


def test_pipeline_runs_expected_commands_and_paths(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    mri = _mri_and_megqa(bids_root)
    parameter = tmp_path / "analysis.param"
    parameter.write_text("CovBand 5 70\n")
    calls = []

    def runner(command, *, cwd, check):
        calls.append((command, cwd, check))
        if command[0] == "orthohull":
            (cwd / "hull.shape").write_text("1\n0 0 0 0 0 1\n")

    paths = pipeline.run_pipeline(
        dataset,
        parameter,
        runner=runner,
        which=lambda command: f"/commands/{command}",
    )

    ortho = bids_root / "derivatives" / "preprocessing" / "sub-01" / "orthoMRI"
    output = bids_root / "derivatives" / "SAMsrc" / "sub-01" / "ses-01" / "run-02"
    assert paths.mri == mri
    assert paths.ortho_mri == ortho
    assert paths.sam_output == output
    staged_mri = ortho / mri.name
    assert calls[0] == (["orthohull", str(staged_mri)], ortho, True)
    assert staged_mri.is_file()
    assert (ortho / "sub-01_T1w.json").read_text() == (
        mri.with_name("sub-01_T1w.json").read_text()
    )
    assert not (mri.parent / "MRI").exists()
    assert not (mri.parent / "MRI2").exists()
    assert [call[0][0] for call in calls] == [
        "orthohull",
        "sam_cov",
        "sam_wts",
        "sam_3d",
    ]
    assert calls[1][0][-2:] == ["-o_SAMdir", str(output)]
    assert ["--MRIDirectory", str(ortho)] == calls[2][0][5:7]
    assert ["--MRIPattern", "%M/%s"] == calls[2][0][7:9]
    assert ["--ImageDirectory", str(output)] == calls[3][0][5:7]


def test_pipeline_reuses_hull_and_supports_custom_project(tmp_path):
    bids_root, dataset = _dataset(tmp_path, session=None, run=None)
    _mri_and_megqa(bids_root)
    parameter = tmp_path / "analysis.param"
    parameter.touch()
    hull = (
        bids_root
        / "derivatives"
        / "preprocessing"
        / "sub-01"
        / "orthoMRI"
        / "hull.shape"
    )
    hull.parent.mkdir(parents=True)
    hull.write_text("existing")
    calls = []

    def runner(command, *, cwd, check):
        calls.append(command)

    paths = pipeline.run_pipeline(
        dataset,
        parameter,
        project="experiment",
        runner=runner,
        which=lambda command: f"/commands/{command}",
    )
    assert [command[0] for command in calls] == ["sam_cov", "sam_wts", "sam_3d"]
    assert paths.sam_output == bids_root / "derivatives" / "experiment" / "sub-01"


def test_force_orthohull_regenerates_existing_hull(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    _mri_and_megqa(bids_root)
    parameter = tmp_path / "analysis.param"
    parameter.touch()
    hull = (
        bids_root
        / "derivatives"
        / "preprocessing"
        / "sub-01"
        / "orthoMRI"
        / "hull.shape"
    )
    hull.parent.mkdir(parents=True)
    hull.write_text("existing")
    calls = []

    def runner(command, *, cwd, check):
        calls.append(command)
        if command[0] == "orthohull":
            hull.write_text("regenerated")

    pipeline.run_pipeline(
        dataset,
        parameter,
        force_orthohull=True,
        runner=runner,
        which=lambda command: f"/commands/{command}",
    )
    assert calls[0][0] == "orthohull"
    assert hull.read_text() == "regenerated"


def test_pipeline_preflights_required_commands(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    _mri_and_megqa(bids_root)
    parameter = tmp_path / "analysis.param"
    parameter.touch()

    with pytest.raises(pipeline.PipelineError, match="sam_wts"):
        pipeline.run_pipeline(
            dataset,
            parameter,
            runner=lambda *args, **kwargs: pytest.fail("runner was called"),
            which=lambda command: None if command == "sam_wts" else command,
        )


def test_invalid_project_fails_before_creating_ortho_directory(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    _mri_and_megqa(bids_root)
    parameter = tmp_path / "analysis.param"
    parameter.touch()

    with pytest.raises(pipeline.PipelineError, match="project"):
        pipeline.run_pipeline(dataset, parameter, project="../elsewhere")
    assert not (bids_root / "derivatives" / "preprocessing").exists()


def test_pipeline_stops_when_orthohull_does_not_create_hull(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    _mri_and_megqa(bids_root)
    parameter = tmp_path / "analysis.param"
    parameter.touch()
    calls = []

    def runner(command, *, cwd, check):
        calls.append(command)

    with pytest.raises(pipeline.PipelineError, match="did not create"):
        pipeline.run_pipeline(
            dataset,
            parameter,
            runner=runner,
            which=lambda command: f"/commands/{command}",
        )
    assert [command[0] for command in calls] == ["orthohull"]


@pytest.mark.parametrize("selected", [None, "Multiple"])
def test_selected_mri_must_be_resolved(tmp_path, selected):
    bids_root, dataset = _dataset(tmp_path)
    qa_path = bids_root / "derivatives" / "megQA" / "sub-01.yml"
    qa_path.parent.mkdir(parents=True)
    qa_path.write_text(yaml.safe_dump({
        "schema_version": 1.0,
        "subject": "sub-01",
        "bids_root": str(bids_root),
        "mri": selected,
        "meg_list": [],
    }, sort_keys=False))

    with pytest.raises(pipeline.PipelineError, match="MRI|MRIs"):
        pipeline.load_selected_mri(pipeline.parse_bids_dataset(dataset))


def test_selected_mri_is_relocated_with_moved_bids_project(tmp_path):
    old_root, dataset = _dataset(tmp_path / "old")
    _mri_and_megqa(old_root)
    new_root = tmp_path / "new" / "bids"
    new_root.parent.mkdir(parents=True)
    old_root.rename(new_root)
    moved_dataset = new_root / dataset.relative_to(old_root)

    mri = pipeline.load_selected_mri(pipeline.parse_bids_dataset(moved_dataset))
    assert mri == new_root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"


def test_pipeline_migrates_legacy_megqa_pickle(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    mri = _mri_and_megqa(bids_root)
    yaml_path = bids_root / "derivatives" / "megQA" / "sub-01.yml"
    yaml_path.unlink()
    pickle_path = yaml_path.with_suffix(".pkl")
    with pickle_path.open("wb") as stream:
        dill.dump(
            SimpleNamespace(mri=str(mri), bids_root=str(bids_root)),
            stream,
        )

    selected = pipeline.load_selected_mri(pipeline.parse_bids_dataset(dataset))

    assert selected == mri
    assert yaml_path.is_file()
    assert pickle_path.is_file()


def test_selected_mri_requires_valid_fiducial_coordinates(tmp_path):
    bids_root, dataset = _dataset(tmp_path)
    mri = _mri_and_megqa(bids_root)
    mri.with_name("sub-01_T1w.json").write_text(
        json.dumps({"AnatomicalLandmarkCoordinates": {"NAS": [1, 2, 3]}})
    )

    with pytest.raises(pipeline.PipelineError, match="NAS, LPA, and RPA"):
        pipeline.load_selected_mri(pipeline.parse_bids_dataset(dataset))
