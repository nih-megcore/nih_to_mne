#!/usr/bin/env python3

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from nih2mne import __version__ as NIH2MNE_VERSION

from nih2mne.GUI.beamformer_form_entries import (
    BeamformerFormEntries,
    BeamformerFormWindow,
    DEFAULT_PROJECT_NAME,
    SCRIPT_DIALOG_START_DIR,
    _default_script_name,
    _next_script_version,
    _task_id_from_dataset_path,
    build_gui_component_block,
    render_beamformer_script,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_entries():
    return BeamformerFormEntries(
        fname="/tmp/sub-TEST/ses-1/meg/sub-TEST_ses-1_task-rest_run-01_meg.ds",
        project_name="beamformer_test",
        tmin="-0.25",
        tmax="0.25",
        fmin="1",
        fmax="110",
        threshold_rejection="3e-12",
        beamformer_regularization="5",
        megnet=True,
        muscle_detect=True,
        conditions_of_interest="rest, task",
        contrast_type="Percent Change",
        anat_overwrite=True,
        beamformer_overwrite=False,
        contrasts_overwrite=True,
    )


def test_beamformer_form_entries_capture_gui_values(qapp):
    window = BeamformerFormWindow()
    ui = window.ui

    ui.lineEdit_projectName.setText("beamformer_test")
    ui.lineEdit_tmin.setText("-0.25")
    ui.lineEdit_tmax.setText("0.25")
    ui.lineEdit_fmin.setText("1")
    ui.lineEdit_fmax.setText("110")
    ui.lineEdit_BeamformerRegularization.setText("5")
    ui.cb_MuscleDetect.setChecked(True)

    entries = BeamformerFormEntries.from_ui(ui)

    assert entries.project_name == "beamformer_test"
    assert entries.tmin == "-0.25"
    assert entries.tmax == "0.25"
    assert entries.fmin == "1"
    assert entries.fmax == "110"
    assert entries.beamformer_regularization == "5"
    assert entries.muscle_detect is True

    assert entries.to_dict()["project_name"] == "beamformer_test"
    assert entries.to_dict()["tmin"] == "-0.25"
    assert entries.to_dict()["tmax"] == "0.25"
    assert entries.to_dict()["fmin"] == "1"
    assert entries.to_dict()["fmax"] == "110"
    assert entries.to_dict()["beamformer_regularization"] == "5"
    assert entries.to_dict()["muscle_detect"] is True


def test_task_id_from_dataset_path_prefers_bids_task_token():
    dataset_path = "/tmp/sub-TEST/ses-1/meg/sub-TEST_ses-1_task-rest_run-01_meg.ds"
    assert _task_id_from_dataset_path(dataset_path) == "rest"


def test_task_id_from_dataset_path_uses_second_underscore_token_for_non_bids():
    dataset_path = "/tmp/scan_flanker_block1.ds"
    assert _task_id_from_dataset_path(dataset_path) == "flanker"


def test_next_script_version_increments_from_existing_files(tmp_path):
    (tmp_path / "rest_v1.py").write_text("", encoding="utf-8")
    (tmp_path / "rest_v3.py").write_text("", encoding="utf-8")
    assert _next_script_version("rest", output_dir=tmp_path) == 4


def test_default_script_name_uses_task_id_and_version_for_bids_dataset(tmp_path):
    (tmp_path / "rest_v1.py").write_text("", encoding="utf-8")
    assert _default_script_name(
        "/tmp/sub-TEST_ses-1_task-rest_run-01_meg.ds",
        output_dir=tmp_path,
    ) == "rest_v2.py"


def test_default_script_name_uses_second_token_for_non_bids_dataset(tmp_path):
    assert _default_script_name(
        "/tmp/scan_flanker_block1.ds",
        output_dir=tmp_path,
    ) == "flanker_v1.py"


def test_build_gui_component_block_contains_expected_template_values(sample_entries):
    block = build_gui_component_block(sample_entries)

    assert "# GUI entries >>" in block
    assert "dataset_path = pathlib.Path(sys.argv[1])" in block
    assert "# GUI entries <<" in block
    assert "project = 'beamformer_test'" in block
    assert "epo_tmin = -0.25" in block
    assert "epo_tmax = 0.25" in block
    assert "f_min = 1" in block
    assert "f_max = 110" in block
    assert "beam_reg = 0.05" in block
    assert "conds_OI = ['rest', 'task']" in block
    assert "contrasts_type = 'percent'" in block
    assert "use_muscle_detection = True" in block
    assert "overwrite_anats = True" in block
    assert "overwrite_beam = False" in block
    assert "overwrite_contrasts = True" in block


def test_render_beamformer_script_inserts_generated_block_at_marker(sample_entries, tmp_path):
    template_path = tmp_path / "beamformer_template.py"
    template_path.write_text(
        "before\nnih2mne version: <<VER>>\n#%% << INSERT GUI COMPONENTS HERE >>\nafter\n",
        encoding="utf-8",
    )

    script_text = render_beamformer_script(sample_entries, template_path=template_path)

    assert "#%% << INSERT GUI COMPONENTS HERE >>" not in script_text
    assert f"before\nnih2mne version: {NIH2MNE_VERSION}\n\n# GUI entries >>\n#%% GUI Components\n" in script_text
    assert "# GUI entries <<\n\nafter\n" in script_text
    assert f"nih2mne version: {NIH2MNE_VERSION}" in script_text
    assert "epo_tmin = -0.25" in script_text
    assert script_text.endswith("after\n")


def test_render_beamformer_script_preserves_template_outside_marker(sample_entries, tmp_path):
    template_path = tmp_path / "beamformer_template.py"
    template_path.write_text(
        "before\n#%% << INSERT GUI COMPONENTS HERE >>\nif use_muscle_detection:\n    keep_me()\nafter\n",
        encoding="utf-8",
    )

    entries = BeamformerFormEntries(**{**sample_entries.to_dict(), "muscle_detect": False})
    script_text = render_beamformer_script(entries, template_path=template_path)

    assert "use_muscle_detection = False" in script_text
    assert "if use_muscle_detection:\n    keep_me()" in script_text


def test_open_dataset_directory_sets_fname_for_ds_directory(qapp, monkeypatch, tmp_path):
    window = BeamformerFormWindow()
    dataset_dir = tmp_path / "subject01.ds"
    dataset_dir.mkdir()

    monkeypatch.setattr(
        "nih2mne.GUI.beamformer_form_entries.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(dataset_dir),
    )

    selected = window.open_dataset_directory()

    assert selected == str(dataset_dir)
    assert window.ui.lineEdit_fname.text() == str(dataset_dir)


def test_open_dataset_directory_rejects_non_ds_directory(qapp, monkeypatch, tmp_path):
    window = BeamformerFormWindow()
    invalid_dir = tmp_path / "subject01"
    invalid_dir.mkdir()
    warnings = []

    monkeypatch.setattr(
        "nih2mne.GUI.beamformer_form_entries.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(invalid_dir),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    selected = window.open_dataset_directory()

    assert selected is None
    assert window.ui.lineEdit_fname.text() == ""
    assert len(warnings) == 1


def test_open_conditions_selector_sets_selected_events(qapp, monkeypatch):
    window = BeamformerFormWindow()
    window.ui.lineEdit_fname.setText("/tmp/example.ds")

    monkeypatch.setattr(
        "nih2mne.GUI.beamformer_form_entries.mne.io.read_raw_ctf",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "nih2mne.GUI.beamformer_form_entries.mne.events_from_annotations",
        lambda raw: (None, {"rest": 1, "task": 2, "noise": 3}),
    )

    class FakeSelector:
        def __init__(self, input_list, title=None, parent=None, gridsize_row=None, gridsize_col=None):
            self.input_list = input_list

        def exec_(self):
            return QDialog.Accepted

        def selected_items(self):
            return ["rest", "task"]

    monkeypatch.setattr(
        "nih2mne.GUI.beamformer_form_entries.grid_selector",
        FakeSelector,
    )

    selected = window.open_conditions_selector()

    assert selected == ["rest", "task"]
    assert window.ui.lineEdit_ConditionsOfInterest.text() == "rest, task"


def test_open_conditions_selector_requires_dataset(qapp, monkeypatch):
    window = BeamformerFormWindow()
    warnings = []

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    selected = window.open_conditions_selector()

    assert selected is None
    assert len(warnings) == 1


def test_write_script_via_dialog_writes_template_output(qapp, monkeypatch, tmp_path):
    window = BeamformerFormWindow()
    save_path = tmp_path / "generated_beamformer.py"

    window.ui.lineEdit_fname.setText("/tmp/sub-TEST/ses-1/meg/sub-TEST_ses-1_task-rest_run-01_meg.ds")
    window.ui.lineEdit_projectName.setText("beamformer_test")
    window.ui.lineEdit_tmin.setText("-0.25")
    window.ui.lineEdit_tmax.setText("0.25")
    window.ui.lineEdit_fmin.setText("1")
    window.ui.lineEdit_fmax.setText("110")
    window.ui.lineEdit_ThresholdRejection.setText("3e-12")
    window.ui.lineEdit_BeamformerRegularization.setText("5")
    window.ui.lineEdit_ConditionsOfInterest.setText("rest, task")
    window.ui.comboBox_ContrastType.setCurrentText("Percent Change")
    window.ui.cb_MEGNET.setChecked(True)
    window.ui.cb_MuscleDetect.setChecked(True)
    window.ui.cb_AnatOverwrite.setChecked(True)
    window.ui.cb_BeamformerOverwrite.setChecked(False)
    window.ui.cb_ContrastsOverwrite.setChecked(True)

    monkeypatch.setattr(
        "nih2mne.GUI.beamformer_form_entries.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(save_path), "Python files (*.py)"),
    )

    written_path = window.write_script_via_dialog()
    written_text = save_path.read_text(encoding="utf-8")

    assert written_path == str(save_path)
    assert "#%% << INSERT GUI COMPONENTS HERE >>" not in written_text
    assert "# GUI entries >>" in written_text
    assert "dataset_path = pathlib.Path(sys.argv[1])" in written_text
    assert "project = 'beamformer_test'" in written_text
    assert "# GUI entries <<" in written_text
    assert "beam_reg = 0.05" in written_text
    assert "conds_OI = ['rest', 'task']" in written_text
    assert "use_muscle_detection = True" in written_text
    assert "if use_muscle_detection:" in written_text
    assert "_musc_annot = annotate_muscle_zscore" in written_text


def test_script_save_path_uses_megcore_datproc_start_dir_and_bids_task(qapp):
    window = BeamformerFormWindow()
    window.ui.lineEdit_fname.setText("/tmp/sub-TEST_ses-1_task-rest_run-01_meg.ds")

    script_path = Path(window._script_save_path())

    assert script_path.parent == SCRIPT_DIALOG_START_DIR
    assert script_path.name == "rest_v1.py"


def test_script_save_path_uses_second_token_for_non_bids(qapp):
    window = BeamformerFormWindow()
    window.ui.lineEdit_fname.setText("/tmp/scan_flanker_block1.ds")

    script_path = Path(window._script_save_path())

    assert script_path.parent == SCRIPT_DIALOG_START_DIR
    assert script_path.name == "flanker_v1.py"


def test_build_gui_component_block_defaults_blank_project_name(sample_entries):
    entries = BeamformerFormEntries(**{**sample_entries.to_dict(), "project_name": ""})

    block = build_gui_component_block(entries)

    assert f"project = {DEFAULT_PROJECT_NAME!r}" in block


def test_render_beamformer_script_replaces_version_marker(sample_entries, tmp_path):
    template_path = tmp_path / "beamformer_template.py"
    template_path.write_text(
        "nih2mne version: <<VER>>\n#%% << INSERT GUI COMPONENTS HERE >>\n",
        encoding="utf-8",
    )

    script_text = render_beamformer_script(sample_entries, template_path=template_path)

    assert "<<VER>>" not in script_text
    assert f"nih2mne version: {NIH2MNE_VERSION}" in script_text
