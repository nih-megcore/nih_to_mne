#!/usr/bin/env python3

import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nih2mne.GUI.qt_compat import QtWidgets

QApplication = QtWidgets.QApplication

_MODULE_NAMES = [
    "nih2mne.dataQA.bids_project_interface",
    "nih2mne.utilities.montages",
    "nih2mne.dataQA.qa_config_reader",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _MODULE_NAMES}

fake_bids_project_interface = types.ModuleType("nih2mne.dataQA.bids_project_interface")
fake_bids_project_interface.subject_bids_info = object
fake_bids_project_interface.bids_project = object
fake_bids_project_interface.run_sbatch = lambda *args, **kwargs: "Submitted batch job 123"
sys.modules["nih2mne.dataQA.bids_project_interface"] = fake_bids_project_interface

fake_montages_module = types.ModuleType("nih2mne.utilities.montages")
fake_montages_module.montages = {"test_montage": "MONTAGE_SENTINEL"}
sys.modules["nih2mne.utilities.montages"] = fake_montages_module

fake_qa_module = types.ModuleType("nih2mne.dataQA.qa_config_reader")
fake_qa_module.qa_dataset = lambda *args, **kwargs: "QA_DATASET"
fake_qa_module.read_yml = lambda *args, **kwargs: {"qa": "ok"}
sys.modules["nih2mne.dataQA.qa_config_reader"] = fake_qa_module

import nih2mne.GUI.qt_gui as qt_gui_module

for _module_name, _original_module in _ORIGINAL_MODULES.items():
    if _original_module is None:
        sys.modules.pop(_module_name, None)
    else:
        sys.modules[_module_name] = _original_module

BIDS_Project_Window = qt_gui_module.BIDS_Project_Window
Subject_GUI = qt_gui_module.Subject_GUI
Subject_Tile = qt_gui_module.Subject_Tile
build_datproc_command = qt_gui_module.build_datproc_command
collect_task_datasets = qt_gui_module.collect_task_datasets
extract_processing_log_info = qt_gui_module.extract_processing_log_info
get_task_datproc_files = qt_gui_module.get_task_datproc_files
get_subject_processing_logfile = qt_gui_module.get_subject_processing_logfile
get_subject_processing_status = qt_gui_module.get_subject_processing_status
git_blob_hash = qt_gui_module.git_blob_hash
get_datproc_mri = qt_gui_module.get_datproc_mri
SubjectSelectionDialog = qt_gui_module.SubjectSelectionDialog
ProcReviewDialog = qt_gui_module.ProcReviewDialog
DerivativeFailureButton = qt_gui_module.DerivativeFailureButton


class FakeEventCounts:
    def __repr__(self):
        return "description\nstim    2\nName: count, dtype: int64"


class FakeMegDataset:
    def __init__(self, task, fname, rel_path):
        self.task = task
        self.fname = fname
        self.rel_path = rel_path
        self.raw = object()
        self._orig_bads = ["MEG 002"]
        self.event_counts = FakeEventCounts()

    def load(self):
        return None


class FakeBidsInfo:
    def __init__(self, subject="sub-01", mri="mri.nii.gz", mri_json_qa="GOOD", fs_success=True):
        self.subject = subject
        self.meg_list = [
            FakeMegDataset("rest", "rest.ds", f"/tmp/{subject}/rest.ds"),
            FakeMegDataset("task", "task.ds", f"/tmp/{subject}/task.ds"),
        ]
        self.meg_count = len(self.meg_list)
        self.fs_recon = {"fs_success": fs_success}
        self.mri = mri
        self.mri_json_qa = mri_json_qa
        self.bids_root = "/tmp/fake_bids"
        self.subjects_dir = "/tmp/fake_bids/derivatives/freesurfer/subjects"
        self.all_mris = ["/tmp/mri_a.nii.gz", "/tmp/mri_b.nii.gz"]
        self.current_meg_dset = SimpleNamespace(
            info={"bads": ["MEG 001"]},
            annotations=[
                {"onset": 1.25, "duration": 0.5, "description": "bad_blink"},
                {"onset": 3.0, "duration": 0.25, "description": "EDGE"},
            ],
        )
        self.plot_meg_calls = []
        self.plot_mri_fids_called = False
        self.plot_3d_coreg_calls = []
        self.saved = False
        self.override_calls = []

    def __repr__(self):
        return f"FakeBidsInfo<{self.subject}>"

    def plot_meg(self, **kwargs):
        self.plot_meg_calls.append(kwargs)
        return object()

    def plot_mri_fids(self):
        self.plot_mri_fids_called = True

    def plot_3D_coreg(self, idx):
        self.plot_3d_coreg_calls.append(idx)

    def save(self, overwrite=True):
        self.saved = overwrite

    def mri_selection_override(self, override_mri):
        self.override_calls.append(override_mri)


class FakeProject:
    def __init__(self, subjects):
        self.subjects = subjects
        self.bids_root = "/tmp/fake_bids"
        self.issues = {"Freesurfer_notStarted": [], "Freesurfer_failed": []}


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_subject_tile_status_and_task_filter(qapp):
    bids_info = FakeBidsInfo(subject="sub-filter")
    tile = Subject_Tile(bids_info, task_filter="rest")

    assert tile.filtered_tasklist == [bids_info.meg_list[0]]
    assert tile.meg_status == "GOOD"
    assert tile.mri_status == "GOOD"
    assert tile.fs_status == "GOOD"
    assert tile.color("Meg") == "green"
    assert tile.color("Mri") == "green"
    assert tile.color("FS") == "green"
    assert tile.mri == bids_info.mri

    bids_info.mri = "/tmp/reselected_T1w.nii.gz"

    assert tile.mri == "/tmp/reselected_T1w.nii.gz"


def test_subject_tile_marks_missing_data_bad(qapp):
    bids_info = FakeBidsInfo(subject="sub-bad", mri="Multiple", fs_success=False)
    bids_info.meg_list = []
    bids_info.meg_count = 0

    tile = Subject_Tile(bids_info, task_filter="rest")

    assert tile.meg_status == "BAD"
    assert tile.mri_status == "BAD"
    assert tile.fs_status == "BAD"
    assert tile.color("Meg") == "red"
    assert tile.color("Mri") == "red"
    assert tile.color("FS") == "red"


def test_subject_gui_get_meg_events_formats_series_repr(qapp):
    gui = Subject_GUI(FakeBidsInfo())

    assert gui.get_meg_choices() == ["0: rest.ds", "1: task.ds"]
    assert gui.get_mri_choices() == ["0: mri_a.nii.gz", "1: mri_b.nii.gz"]
    assert gui.get_meg_events() == "stim    2\n"


def test_subject_gui_plot_save_and_override_actions(qapp, monkeypatch):
    bids_info = FakeBidsInfo(mri="Multiple")
    gui = Subject_GUI(bids_info)
    monkeypatch.setattr(
        qt_gui_module,
        "find_head_surface",
        lambda subject, subjects_dir: "/tmp/outer_skin.surf",
    )

    gui.b_fmin.setText("1.5")
    gui.b_fmax.setText("40")
    gui.b_f_mains.setChecked(True)
    gui.plot_meg()
    gui.plot_fids()
    gui.plot_3d_coreg()
    gui.save()
    gui.b_mri_override_selection.setCurrentIndex(1)
    gui.override_mri()

    assert bids_info.plot_meg_calls == [
        {
            "idx": 0,
            "hp": 1.5,
            "lp": 40.0,
            "montage": "MONTAGE_SENTINEL",
            "f_mains": 60.0,
            "browser_backend": "qt",
            "block": False,
        }
    ]
    assert bids_info.plot_mri_fids_called is True
    assert bids_info.plot_3d_coreg_calls == [0]
    assert bids_info.saved is True
    assert bids_info.override_calls == ["/tmp/mri_b.nii.gz"]


def test_subject_gui_reports_plot_failure(qapp, monkeypatch):
    bids_info = FakeBidsInfo()
    gui = Subject_GUI(bids_info)
    critical_messages = []

    def fail_plot(**_kwargs):
        raise RuntimeError("Qt browser unavailable")

    bids_info.plot_meg = fail_plot
    monkeypatch.setattr(
        qt_gui_module.QMessageBox,
        "critical",
        lambda *args: critical_messages.append(args),
    )

    gui.plot_meg()

    assert len(critical_messages) == 1
    assert critical_messages[0][1:] == (
        "MEG Plot Failed",
        "Qt browser unavailable",
    )


def test_subject_gui_reports_fids_plot_failure(qapp, monkeypatch):
    bids_info = FakeBidsInfo()
    gui = Subject_GUI(bids_info)
    critical_messages = []

    def fail_plot():
        raise RuntimeError("FIDS rendering unavailable")

    bids_info.plot_mri_fids = fail_plot
    monkeypatch.setattr(
        qt_gui_module.QMessageBox,
        "critical",
        lambda *args: critical_messages.append(args),
    )

    gui.plot_fids()

    assert len(critical_messages) == 1
    assert critical_messages[0][1:] == (
        "FIDS Plot Failed",
        "FIDS rendering unavailable",
    )


class FakeStatusMessage:
    class StandardButton:
        NoButton = 0

    events = []

    def __init__(self, parent=None):
        self.parent = parent
        self.events.append(("created", parent))

    def setWindowTitle(self, title):
        self.events.append(("title", title))

    def setText(self, message):
        self.events.append(("text", message))

    def setStandardButtons(self, buttons):
        self.events.append(("buttons", buttons))

    def setModal(self, modal):
        self.events.append(("modal", modal))

    def show(self):
        self.events.append(("show", None))

    def accept(self):
        self.events.append(("accept", None))

    def deleteLater(self):
        self.events.append(("deleteLater", None))

    @classmethod
    def critical(cls, parent, title, message):
        cls.events.append(("critical", title, message))


def test_plot_3d_coreg_generates_missing_head_surface(qapp, monkeypatch):
    bids_info = FakeBidsInfo()
    gui = Subject_GUI(bids_info)
    generation_calls = []
    FakeStatusMessage.events = []
    monkeypatch.setattr(qt_gui_module, "QMessageBox", FakeStatusMessage)
    monkeypatch.setattr(
        qt_gui_module, "find_head_surface", lambda subject, subjects_dir: None
    )
    monkeypatch.setattr(
        qt_gui_module,
        "make_fast_head_surface",
        lambda subject, subjects_dir: generation_calls.append((subject, subjects_dir)),
    )

    gui.plot_3d_coreg()

    assert generation_calls == [(bids_info.subject, bids_info.subjects_dir)]
    assert ("text", "Generating head surface, may take a minute.") in (
        FakeStatusMessage.events
    )
    assert ("show", None) in FakeStatusMessage.events
    assert ("accept", None) in FakeStatusMessage.events
    assert ("deleteLater", None) in FakeStatusMessage.events
    assert bids_info.plot_3d_coreg_calls == [0]


def test_plot_3d_coreg_reports_surface_generation_failure(qapp, monkeypatch):
    bids_info = FakeBidsInfo()
    gui = Subject_GUI(bids_info)
    FakeStatusMessage.events = []
    monkeypatch.setattr(qt_gui_module, "QMessageBox", FakeStatusMessage)
    monkeypatch.setattr(
        qt_gui_module, "find_head_surface", lambda subject, subjects_dir: None
    )

    def fail_generation(subject, subjects_dir):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(qt_gui_module, "make_fast_head_surface", fail_generation)

    gui.plot_3d_coreg()

    assert bids_info.plot_3d_coreg_calls == []
    assert (
        "critical",
        "Head Surface Generation Failed",
        "synthetic generation failure",
    ) in FakeStatusMessage.events
    assert ("accept", None) in FakeStatusMessage.events
    assert ("deleteLater", None) in FakeStatusMessage.events


def test_plot_3d_coreg_reports_plotting_failure(qapp, monkeypatch):
    bids_info = FakeBidsInfo()
    gui = Subject_GUI(bids_info)
    FakeStatusMessage.events = []
    monkeypatch.setattr(qt_gui_module, "QMessageBox", FakeStatusMessage)
    monkeypatch.setattr(
        qt_gui_module,
        "find_head_surface",
        lambda subject, subjects_dir: "/tmp/outer_skin.surf",
    )

    def fail_plot(idx):
        raise RuntimeError("synthetic plotting failure")

    bids_info.plot_3D_coreg = fail_plot

    gui.plot_3d_coreg()

    assert (
        "critical",
        "3D Coregistration Failed",
        "synthetic plotting failure",
    ) in FakeStatusMessage.events


def test_get_task_datproc_files_filters_by_task_prefix(tmp_path):
    (tmp_path / "rest_v2.py").write_text("")
    (tmp_path / "rest_v1.sh").write_text("")
    (tmp_path / "task_v1.py").write_text("")
    (tmp_path / "resting_extra.py").write_text("")

    assert get_task_datproc_files("rest", datproc_dir=str(tmp_path)) == ["rest_v2.py", "rest_v1.sh"]


def test_build_datproc_command_includes_quoted_mri():
    py_cmd = build_datproc_command(
        "/tmp/rest process.py",
        "/tmp/sub 01/sub-01_task-rest_meg.ds",
        "/tmp/sub 01/sub-01_T1w.nii.gz",
    )
    shell_cmd = build_datproc_command(
        "/tmp/rest process.sh",
        "/tmp/sub 01/sub-01_task-rest_meg.ds",
        "/tmp/sub 01/sub-01_T1w.nii.gz",
    )

    assert shlex.split(py_cmd)[-3:] == [
        "/tmp/sub 01/sub-01_task-rest_meg.ds",
        "-mri",
        "/tmp/sub 01/sub-01_T1w.nii.gz",
    ]
    assert shlex.split(py_cmd)[1] == "/tmp/rest process.py"
    assert shlex.split(shell_cmd) == [
        "/tmp/rest process.sh",
        "/tmp/sub 01/sub-01_task-rest_meg.ds",
        "-mri",
        "/tmp/sub 01/sub-01_T1w.nii.gz",
    ]


@pytest.mark.parametrize(
    "mri,mri_qa,error_text",
    [
        (None, "No MRIs", "no MRI is selected"),
        ("Multiple", "Undetermined", "multiple MRIs are unresolved"),
        ("missing_T1w.nii.gz", "GOOD", "selected MRI does not exist"),
        ("existing_T1w.nii.gz", "BAD", "MRI fiducial QA is 'BAD'"),
    ],
)
def test_get_datproc_mri_rejects_unready_inputs(tmp_path, mri, mri_qa, error_text):
    if mri == "existing_T1w.nii.gz":
        mri = str(tmp_path / mri)
        (tmp_path / "existing_T1w.nii.gz").touch()
    bids_info = FakeBidsInfo(mri=mri, mri_json_qa=mri_qa)

    with pytest.raises(ValueError, match=error_text):
        get_datproc_mri(bids_info)


def test_single_datproc_submits_selected_mri_and_reports_it(
        tmp_path, qapp, monkeypatch):
    processing_file = tmp_path / "rest_v1.py"
    processing_file.write_text("")
    mri = tmp_path / "sub-01_T1w.nii.gz"
    mri.touch()
    bids_info = FakeBidsInfo(mri=str(mri), mri_json_qa="GOOD")
    subject_gui = Subject_GUI(bids_info)
    submitted = []
    messages = []

    def fake_submit(processing_path, dataset_path, mri_path, **kwargs):
        submitted.append((processing_path, dataset_path, mri_path, kwargs))
        return "Submitted batch job 123"

    monkeypatch.setattr(qt_gui_module, "submit_datproc_job", fake_submit)
    monkeypatch.setattr(
        qt_gui_module.QMessageBox,
        "information",
        lambda parent, title, message: messages.append((title, message)),
    )
    dialog = qt_gui_module.DatprocSubmissionDialog(subject_gui)
    dialog.b_datproc_dir.setText(str(tmp_path))
    dialog.refresh_matches()

    assert shlex.split(dialog.l_cmd_preview.text())[-2:] == ["-mri", str(mri)]
    assert dialog.b_submit.isEnabled()

    dialog.submit_job()

    assert submitted[0][:3] == (
        str(processing_file),
        bids_info.meg_list[0].rel_path,
        str(mri),
    )
    assert "MRI: " + str(mri) in messages[0][1]
    assert "MRI QA: GOOD" in messages[0][1]


def test_single_datproc_blocks_failed_mri_qa(tmp_path, qapp, monkeypatch):
    (tmp_path / "rest_v1.py").write_text("")
    mri = tmp_path / "sub-01_T1w.nii.gz"
    mri.touch()
    bids_info = FakeBidsInfo(mri=str(mri), mri_json_qa="BAD")
    subject_gui = Subject_GUI(bids_info)
    warnings = []
    monkeypatch.setattr(
        qt_gui_module.QMessageBox,
        "warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    dialog = qt_gui_module.DatprocSubmissionDialog(subject_gui)
    dialog.b_datproc_dir.setText(str(tmp_path))
    dialog.refresh_matches()

    assert not dialog.b_submit.isEnabled()
    assert "MRI fiducial QA is 'BAD'" in dialog.l_cmd_preview.text()

    dialog.submit_job()

    assert warnings == [("MRI Not Ready", "sub-01: MRI fiducial QA is 'BAD'")]


def test_batch_datproc_skips_unready_subject_and_reports_mri(
        tmp_path, qapp, monkeypatch):
    processing_file = tmp_path / "rest_v1.py"
    processing_file.write_text("")
    valid_mri = tmp_path / "sub-01_T1w.nii.gz"
    valid_mri.touch()
    valid = FakeBidsInfo(
        subject="sub-01", mri=str(valid_mri), mri_json_qa="GOOD")
    skipped = FakeBidsInfo(
        subject="sub-02", mri="Multiple", mri_json_qa="Undetermined")
    skipped.meg_list.append(FakeMegDataset(
        "rest", "rest_run-02.ds", "/tmp/sub-02/rest_run-02.ds"))
    project = FakeProject({"sub-01": valid, "sub-02": skipped})
    window = BIDS_Project_Window(
        bids_project=project, gridsize_row=1, gridsize_col=2)
    submitted = []
    messages = []

    def fake_submit(processing_path, dataset_path, mri_path, **kwargs):
        submitted.append((processing_path, dataset_path, mri_path, kwargs))
        return "Submitted batch job 321"

    monkeypatch.setattr(qt_gui_module, "submit_datproc_job", fake_submit)
    monkeypatch.setattr(
        qt_gui_module.QMessageBox,
        "information",
        lambda parent, title, message: messages.append((title, message)),
    )
    dialog = qt_gui_module.ProjectDatprocSubmissionDialog(window)
    dialog.b_datproc_dir.setText(str(tmp_path))
    dialog.refresh_matches()

    assert shlex.split(dialog.l_cmd_preview.text().splitlines()[0])[-2:] == [
        "-mri", str(valid_mri)]
    assert "MRI skipped: 1" in dialog.l_dataset_count.text()

    dialog.submit_jobs()

    assert len(submitted) == 1
    assert submitted[0][:3] == (
        str(processing_file),
        valid.meg_list[0].rel_path,
        str(valid_mri),
    )
    summary = messages[0][1]
    assert "sub-01" in summary.splitlines()
    assert "Submitted batch job 321" not in summary
    assert str(valid_mri) not in summary
    assert "QA: GOOD" not in summary
    assert "Skipped for MRI readiness: 1 subject(s)" in summary
    assert summary.count("sub-02: multiple MRIs are unresolved") == 1


def test_batch_datproc_with_no_ready_mri_submits_nothing(
        tmp_path, qapp, monkeypatch):
    (tmp_path / "rest_v1.py").write_text("")
    bids_info = FakeBidsInfo(
        subject="sub-01", mri=None, mri_json_qa="No MRIs")
    window = BIDS_Project_Window(
        bids_project=FakeProject({"sub-01": bids_info}),
        gridsize_row=1,
        gridsize_col=1,
    )
    warnings = []
    monkeypatch.setattr(
        qt_gui_module,
        "submit_datproc_job",
        lambda *args, **kwargs: pytest.fail("an MRI-invalid job was submitted"),
    )
    monkeypatch.setattr(
        qt_gui_module.QMessageBox,
        "warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    dialog = qt_gui_module.ProjectDatprocSubmissionDialog(window)
    dialog.b_datproc_dir.setText(str(tmp_path))
    dialog.refresh_matches()

    dialog.submit_jobs()

    assert warnings[0][0] == "No Processing-Ready MRIs"
    assert "No slurm jobs were submitted" in warnings[0][1]
    assert "sub-01: no MRI is selected" in warnings[0][1]


def test_collect_task_datasets_returns_all_matching_project_datasets():
    subjects = {
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
    }

    matched = collect_task_datasets(FakeProject(subjects), "rest")

    assert [(subject, dset.task, dset.fname) for subject, _, dset in matched] == [
        ("sub-01", "rest", "rest.ds"),
        ("sub-02", "rest", "rest.ds"),
    ]


def test_extract_processing_log_info_and_subject_status(tmp_path, qapp):
    procfile = tmp_path / "rest_proc.py"
    procfile.write_text(
        "project = 'beamformer_test'\n"
        "log_dir = output_path.root / 'logging' / f'sub-{bids_path.subject}'\n"
        "log_fname = log_dir / 'beamformer_template.log'\n"
    )
    proc_hash = git_blob_hash(str(procfile))

    subject_log = tmp_path / "fake_bids" / "derivatives" / "beamformer_test" / "logging" / "sub-01"
    subject_log.mkdir(parents=True)
    logfile = subject_log / "beamformer_template.log"
    logfile.write_text(f"INFO START :: {proc_hash}\nINFO FINISHED :: {proc_hash}\n")

    assert extract_processing_log_info(str(procfile)) == {
        "project": "beamformer_test",
        "logfile_name": "beamformer_template.log",
    }
    assert get_subject_processing_logfile("/tmp/fake_bids", "sub-01", str(procfile)) == (
        "/tmp/fake_bids/derivatives/beamformer_test/logging/sub-01/beamformer_template.log"
    )
    assert get_subject_processing_status(str(logfile), proc_hash) == "(SUCCESS)"


def test_get_subject_processing_status_marks_started_without_finish_error(tmp_path):
    logfile = tmp_path / "beamformer_template.log"
    logfile.write_text("INFO START :: abc123\n")

    assert get_subject_processing_status(str(logfile), "abc123") == "(ERROR)"
    assert get_subject_processing_status(str(logfile), "different") == ""


def test_subject_selection_dialog_shows_status_labels(qapp):
    dialog = SubjectSelectionDialog(
        ["sub-01", "sub-02"],
        {"sub-01"},
        subject_statuses={"sub-01": "(SUCCESS)", "sub-02": "(ERROR)"},
    )

    labels = [label.text() for label in dialog.findChildren(qt_gui_module.QLabel)]

    assert "sub-01" in labels
    assert "sub-02" in labels
    assert "(SUCCESS)" in labels
    assert "(ERROR)" in labels


def test_subject_selection_buttons_show_and_toggle_state(qapp):
    dialog = SubjectSelectionDialog(
        ["sub-01", "sub-02"],
        {"sub-01"},
    )
    included_button = dialog.subject_buttons["sub-01"]
    excluded_button = dialog.subject_buttons["sub-02"]

    assert included_button.text() == "Included"
    assert excluded_button.text() == "Excluded"
    included_color = included_button.icon().pixmap(10, 10).toImage().pixelColor(5, 5)
    excluded_color = excluded_button.icon().pixmap(10, 10).toImage().pixelColor(5, 5)
    assert included_color.name() == qt_gui_module._INCLUDED_COLOR
    assert excluded_color.name() == qt_gui_module._EXCLUDED_COLOR

    included_button.click()

    assert dialog.get_selected_subjects() == []
    assert included_button.text() == "Excluded"
    toggled_color = included_button.icon().pixmap(10, 10).toImage().pixelColor(5, 5)
    assert toggled_color.name() == qt_gui_module._EXCLUDED_COLOR


def test_read_subject_ids_file_detects_csv_header_and_normalizes(tmp_path):
    subject_file = tmp_path / "subjects.csv"
    subject_file.write_text(
        "visit,BIDS ID,notes\n1,ON02811,first\n1,sub-ON02812,second\n"
        "1,ON02811,duplicate\n"
    )

    assert qt_gui_module.read_subject_ids_file(subject_file) == [
        "sub-ON02811",
        "sub-ON02812",
    ]


def test_read_subject_ids_file_detects_headerless_tsv(tmp_path):
    subject_file = tmp_path / "subjects.tsv"
    subject_file.write_text("ON02811\tinclude\nsub-ON02812\tinclude\n\t\n")

    assert qt_gui_module.read_subject_ids_file(subject_file) == [
        "sub-ON02811",
        "sub-ON02812",
    ]


def test_subject_selection_dialog_load_csv_replaces_selection_and_warns(
        tmp_path, qapp, monkeypatch):
    subject_file = tmp_path / "subjects.csv"
    subject_file.write_text("participant_id\n01\n03\n99\n")
    dialog = SubjectSelectionDialog(
        ["sub-01", "sub-02", "sub-03"],
        {"sub-02"},
    )
    warnings = []
    monkeypatch.setattr(
        qt_gui_module.QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(subject_file), "Subject lists (*.csv)"),
    )
    monkeypatch.setattr(
        qt_gui_module.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    dialog.load_subject_file()

    assert dialog.get_selected_subjects() == ["sub-01", "sub-03"]
    assert dialog.subject_buttons["sub-01"].text() == "Included"
    assert dialog.subject_buttons["sub-02"].text() == "Excluded"
    assert len(warnings) == 1
    assert warnings[0][1] == "Subjects Not Found"
    assert "sub-99" in warnings[0][2]


def test_subject_selection_dialog_cancel_load_preserves_selection(
        qapp, monkeypatch):
    dialog = SubjectSelectionDialog(["sub-01", "sub-02"], {"sub-02"})
    monkeypatch.setattr(
        qt_gui_module.QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("", ""),
    )

    dialog.load_subject_file()

    assert dialog.get_selected_subjects() == ["sub-02"]


def test_project_datproc_default_selection_excludes_success_and_error(qapp):
    project = FakeProject({
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
        "sub-03": FakeBidsInfo(subject="sub-03"),
    })
    window = BIDS_Project_Window(bids_project=project, gridsize_row=1, gridsize_col=2)
    dialog = qt_gui_module.ProjectDatprocSubmissionDialog(window)
    dialog.matched_subjects = ["sub-01", "sub-02", "sub-03"]
    dialog.selected_subjects = {"sub-01", "sub-02", "sub-03"}

    initial_selection = dialog.get_subject_selector_initial_selection({
        "sub-01": "(SUCCESS)",
        "sub-02": "(ERROR)",
        "sub-03": "",
    })

    assert initial_selection == {"sub-03"}

    dialog.selected_subjects = {"sub-02"}
    preserved_selection = dialog.get_subject_selector_initial_selection({
        "sub-01": "(SUCCESS)",
        "sub-02": "(ERROR)",
        "sub-03": "",
    })

    assert preserved_selection == {"sub-02"}


def test_return_message_box_response_saves_bads_and_segments(qapp, monkeypatch):
    gui = Subject_GUI(FakeBidsInfo())
    recorded = {}

    def fake_write_bad_chans_to_raw(fname=None, bad_chs=None):
        recorded["fname"] = fname
        recorded["bad_chs"] = bad_chs

    def fake_write_bad_segments(fname=None, annotations=None):
        recorded["segments_fname"] = fname
        recorded["annotations"] = annotations

    monkeypatch.setattr(gui, "write_bad_chans_to_raw", fake_write_bad_chans_to_raw)
    monkeypatch.setattr(gui, "write_bad_segments", fake_write_bad_segments)

    gui.return_message_box_response(SimpleNamespace(text=lambda: "&Save"))

    assert recorded["fname"] == "/tmp/sub-01/rest.ds"
    assert set(recorded["bad_chs"]) == {"MEG 001", "MEG 002"}
    assert recorded["segments_fname"] == "/tmp/sub-01/rest.ds"
    assert recorded["annotations"] == gui.bids_info.current_meg_dset.annotations


def test_write_bad_outputs(tmp_path, qapp):
    gui = Subject_GUI(FakeBidsInfo())
    ds_path = tmp_path / "subject.ds"
    ds_path.mkdir()

    gui.write_bad_chans_to_raw(fname=str(ds_path), bad_chs=["MEG 010", "MEG 011"])
    gui.write_bad_segments(
        fname=str(ds_path),
        annotations=[
            {"onset": 1.0, "duration": 0.25, "description": "bad_jump"},
            {"onset": 2.0, "duration": 0.5, "description": "GOOD"},
            {"onset": 4.0, "duration": 0.75, "description": "BAD_motion"},
        ],
    )

    assert (ds_path / "BadChannels").read_text() == "MEG 010\nMEG 011\n"
    assert (ds_path / "bad.segments").read_text() == (
        "0\t1.0\t1.25\tbad_jump\n"
        "0\t4.0\t4.75\tBAD_motion\n"
    )


def test_write_bad_segments_skips_empty_annotations(tmp_path, qapp):
    gui = Subject_GUI(FakeBidsInfo())
    ds_path = tmp_path / "subject.ds"
    ds_path.mkdir()

    gui.write_bad_segments(fname=str(ds_path), annotations=[])

    assert not (ds_path / "bad.segments").exists()


def test_project_window_task_filter_and_pagination(qapp):
    subjects = {
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
        "sub-03": FakeBidsInfo(subject="sub-03"),
    }
    window = BIDS_Project_Window(
        bids_project=FakeProject(subjects),
        gridsize_row=1,
        gridsize_col=2,
    )

    assert window.task_set == ["All : ", "rest : 3", "task : 3"]
    assert window.get_available_task_names() == ["rest", "task"]
    assert window.b_subject_number.text() == "Subject Totals: #3"
    assert window.b_current_page_idx.text() == "Page: 0 / 1"
    assert window.b_project_datproc.text() == "Batch Datproc"
    assert window.b_proc_review.text() == "ProcReview"
    assert window.b_add_dataset.text() == "Add Dataset"
    assert not hasattr(window, "b_run_megnet")

    top_layout = window.centralWidget().layout().itemAt(0).layout()
    top_button_text = [
        top_layout.itemAt(index).widget().text()
        for index in range(top_layout.count())
        if isinstance(top_layout.itemAt(index).widget(), qt_gui_module.QPushButton)
    ]
    assert top_button_text[:2] == ["BIDS Directory", "Add Dataset"]

    window.b_task_chooser.setCurrentIndex(1)
    window.filter_task_qa_vis()
    qapp.processEvents()
    assert window.selected_task == "rest"

    window.increment_page_idx()
    qapp.processEvents()
    assert window.page_idx == 1
    assert window.subject_start_idx == 2
    assert window.b_current_page_idx.text() == "Page: 1 / 1"

    window.decrement_page_idx()
    qapp.processEvents()
    assert window.page_idx == 0
    assert window.subject_start_idx == 0


def _write_review_script(path):
    path.write_text(
        "project = 'review'\n"
        "recipe = FilterSpec(l_freq=1, h_freq=40)\n"
        "template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe)\n"
        "artifacts = template.evaluate()\n"
        "save_if_requested(value, artifacts['epochs'], saver, False, logger)\n"
        "record = template.artifact('stc', description='face')\n"
        "save_if_requested(value, record, saver, False, logger)\n"
    )


def _finish_proc_review_scan(dialog, qapp):
    dialog._thread_pool.waitForDone(5000)
    for _ in range(20):
        qapp.processEvents()
        if not dialog._pending_evaluations:
            break


def test_proc_review_dialog_runs_pagination_and_subject_buttons(qapp, tmp_path, monkeypatch):
    script = tmp_path / "task_v1.py"
    _write_review_script(script)
    subjects = {
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
        "sub-03": FakeBidsInfo(subject="sub-03"),
    }
    subjects["sub-01"].meg_list = [
        FakeMegDataset(
            "task", "run01.ds",
            "/tmp/fake_bids/sub-01/meg/sub-01_task-task_run-01_meg.ds",
        ),
        FakeMegDataset(
            "task", "run02.ds",
            "/tmp/fake_bids/sub-01/meg/sub-01_task-task_run-02_meg.ds",
        ),
    ]
    subjects["sub-02"].meg_list = []
    subjects["sub-03"].meg_list = [
        FakeMegDataset(
            "task", "run01.ds",
            "/tmp/fake_bids/sub-03/meg/sub-03_task-task_run-01_meg.ds",
        )
    ]
    project = FakeProject(subjects)
    window = BIDS_Project_Window(bids_project=project, gridsize_row=1, gridsize_col=2)

    def fake_evaluate(spec, dataset):
        return tuple(
            SimpleNamespace(
                request=request,
                record=SimpleNamespace(
                    exists=request.key == "epochs",
                    status="present" if request.key == "epochs" else "missing",
                    path=f"/tmp/{request.key}",
                    component_paths=(),
                    validation_errors=(),
                ),
            )
            for request in spec.outputs
        )

    monkeypatch.setattr(qt_gui_module, "evaluate_dataset", fake_evaluate)
    dialog = ProcReviewDialog(window, script)
    dialog.b_subjects_per_page.setText("2")
    dialog._reset_and_refresh()
    _finish_proc_review_scan(dialog, qapp)

    assert dialog.task == "task"
    assert dialog.l_task.text() == "Task: task"
    assert not hasattr(dialog, "b_task")
    assert [dialog.b_run.itemText(index) for index in range(dialog.b_run.count())] == [
        "All", "run-01", "run-02"
    ]
    assert dialog.l_page.text() == "1 / 2"
    assert [button.text() for button in dialog.subject_buttons] == [
        "sub-01", "sub-02"
    ]
    assert len(dialog.indicators) == 6  # two runs plus one no-task placeholder, two outputs
    assert sum(indicator.present for indicator in dialog.indicators) == 2
    assert sum(isinstance(indicator, DerivativeFailureButton)
               for indicator in dialog.indicators) == 2

    dialog.next_page()
    assert dialog.l_page.text() == "2 / 2"
    assert [button.text() for button in dialog.subject_buttons] == ["sub-03"]
    dialog.b_hide_no_task.setChecked(True)
    assert dialog.l_page.text() == "1 / 1"


def test_proc_review_subject_focus_shows_run_paths_and_qa_placeholders(
    qapp, tmp_path, monkeypatch
):
    script = tmp_path / "task_v1.py"
    _write_review_script(script)
    subjects = {
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
    }
    subjects["sub-01"].meg_list = [FakeMegDataset(
        "task", "run01.ds",
        "/tmp/fake_bids/sub-01/meg/sub-01_task-task_run-01_meg.ds",
    )]
    subjects["sub-02"].meg_list = [
        FakeMegDataset(
            "task", "run01.ds",
            "/tmp/fake_bids/sub-02/meg/sub-02_task-task_run-01_meg.ds",
        ),
        FakeMegDataset(
            "task", "run02.ds",
            "/tmp/fake_bids/sub-02/meg/sub-02_task-task_run-02_meg.ds",
        ),
    ]
    window = BIDS_Project_Window(
        bids_project=FakeProject(subjects), gridsize_row=1, gridsize_col=2
    )

    def fake_evaluate(spec, dataset):
        run = "run-02" if "run-02" in str(dataset) else "run-01"
        derivative_dir = Path(
            f"/tmp/fake_bids/derivatives/review/sub-02/meg/{run}"
        )
        records = []
        for request in spec.outputs:
            path = derivative_dir / f"sub-02_{run}_{request.key}"
            components = ()
            if request.key == "stc":
                components = (
                    path.with_name(f"{path.name}-lh.stc"),
                    path.with_name(f"{path.name}-rh.stc"),
                )
            records.append(SimpleNamespace(
                request=request,
                record=SimpleNamespace(
                    exists=True,
                    status="present",
                    path=path,
                    component_paths=components,
                    validation_errors=(),
                ),
            ))
        return tuple(records)

    monkeypatch.setattr(qt_gui_module, "evaluate_dataset", fake_evaluate)
    dialog = ProcReviewDialog(window, script)
    dialog.b_subjects_per_page.setText("1")
    dialog._reset_and_refresh()
    dialog.b_run.setCurrentText("run-02")
    dialog.next_page()
    _finish_proc_review_scan(dialog, qapp)

    assert dialog.l_page.text() == "2 / 2"
    assert [button.text() for button in dialog.subject_buttons] == ["sub-02"]
    dialog.subject_buttons[0].click()
    _finish_proc_review_scan(dialog, qapp)

    assert dialog.focused_subject == "sub-02"
    assert dialog.l_page.text() == "Focused: sub-02"
    assert dialog.b_run.currentText() == "run-02"
    assert dialog.b_subject_run.count() == 2
    assert dialog.b_subject_run.currentText() == "run-02"
    assert not dialog.b_show_all.isHidden()
    assert not dialog.b_prev.isEnabled()
    assert not dialog.b_next.isEnabled()
    assert len(dialog.path_fields) == len(dialog.spec.outputs)
    assert all(not button.isEnabled() for button in dialog.qa_buttons)
    assert all(button.toolTip() == "QA not yet available"
               for button in dialog.qa_buttons)
    assert all(indicator.width() == 16 for indicator in dialog.indicators)
    assert dialog.path_fields[0].toPlainText().startswith("derivatives/")
    stc_text = dialog.path_fields[1].toPlainText()
    assert "\n" in stc_text
    assert all(line.startswith("derivatives/") for line in stc_text.splitlines())
    assert stc_text.endswith("-rh.stc")
    assert len(dialog.copy_path_buttons) == len(dialog.spec.outputs)
    assert len(dialog.copy_load_buttons) == len(dialog.spec.outputs)
    assert len(dialog.copy_load_all_buttons) == len(dialog.spec.outputs)
    assert all(button.toolTip() == "Copy Path"
               for button in dialog.copy_path_buttons)
    assert all(button.toolTip() == "Copy Load"
               for button in dialog.copy_load_buttons)
    assert all(button.toolTip() == "Copy Load All"
               for button in dialog.copy_load_all_buttons)
    assert all(button.width() == 34 for button in dialog.copy_path_buttons)
    assert all(button.width() == 34 for button in dialog.copy_load_buttons)
    assert all(button.width() == 42 for button in dialog.copy_load_all_buttons)

    stc_stem = (
        "/tmp/fake_bids/derivatives/review/sub-02/meg/run-02/"
        "sub-02_run-02_stc"
    )
    dialog.copy_path_buttons[1].click()
    assert QApplication.clipboard().text() == stc_stem
    assert dialog.l_scan_status.text() == "Copied path"

    dialog.copy_load_buttons[1].click()
    assert QApplication.clipboard().text() == (
        f"value = mne.read_source_estimate({stc_stem!r})"
    )
    assert dialog.l_scan_status.text() == "Copied load command"

    dialog.copy_load_all_buttons[0].click()
    copied_debug = QApplication.clipboard().text()
    assert "WARNING: an isolated recipe could not be generated" in copied_debug
    assert "%run " in copied_debug
    assert dialog.l_scan_status.text() == (
        "Copied full-script fallback (not isolated)"
    )

    dialog.b_subject_run.setCurrentText("run-01")
    _finish_proc_review_scan(dialog, qapp)
    assert "run-01" in dialog.path_fields[0].toPlainText()

    dialog.rescan()
    assert all(not button.isEnabled() for button in dialog.copy_path_buttons)
    assert all(not button.isEnabled() for button in dialog.copy_load_buttons)
    assert all(not button.isEnabled()
               for button in dialog.copy_load_all_buttons)
    _finish_proc_review_scan(dialog, qapp)
    assert all(button.isEnabled() for button in dialog.copy_path_buttons)
    assert all(button.isEnabled() for button in dialog.copy_load_buttons)
    assert all(button.isEnabled() for button in dialog.copy_load_all_buttons)

    dialog.b_show_all.click()
    assert dialog.focused_subject is None
    assert dialog.l_page.text() == "2 / 2"
    assert dialog.b_show_all.isHidden()
    assert [button.text() for button in dialog.subject_buttons] == ["sub-02"]

    dialog.subject_buttons[0].click()
    dialog.subject_buttons[0].click()
    assert dialog.focused_subject is None
    assert dialog.l_page.text() == "2 / 2"


def test_proc_review_prefetches_three_pages_and_reuses_cached_results(
    qapp, tmp_path, monkeypatch
):
    script = tmp_path / "task_v1.py"
    _write_review_script(script)
    subjects = {}
    for index in range(1, 7):
        subject = f"sub-{index:02d}"
        info = FakeBidsInfo(subject=subject)
        info.meg_list = [FakeMegDataset(
            "task",
            "run01.ds",
            f"/tmp/fake_bids/{subject}/meg/{subject}_task-task_run-01_meg.ds",
        )]
        subjects[subject] = info
    window = BIDS_Project_Window(
        bids_project=FakeProject(subjects), gridsize_row=1, gridsize_col=2
    )
    calls = []

    def fake_evaluate(spec, dataset):
        calls.append(str(dataset))
        return tuple(
            SimpleNamespace(
                request=request,
                record=SimpleNamespace(
                    exists=True,
                    status="present",
                    path=f"/tmp/{request.key}",
                    component_paths=(),
                    validation_errors=(),
                ),
            )
            for request in spec.outputs
        )

    monkeypatch.setattr(qt_gui_module, "evaluate_dataset", fake_evaluate)
    dialog = ProcReviewDialog(window, script)
    _finish_proc_review_scan(dialog, qapp)
    dialog.b_subjects_per_page.setText("1")
    calls.clear()

    dialog.rescan()
    assert all(
        isinstance(indicator, qt_gui_module.PendingDerivativeIndicator)
        for indicator in dialog.indicators
    )
    _finish_proc_review_scan(dialog, qapp)

    assert len(calls) == 4  # current page plus the next three pages
    assert len(set(calls)) == 4

    dialog.next_page()
    _finish_proc_review_scan(dialog, qapp)
    assert len(calls) == 5

    dialog.previous_page()
    _finish_proc_review_scan(dialog, qapp)
    assert len(calls) == 5

    dialog.b_hide_successful.setChecked(True)
    _finish_proc_review_scan(dialog, qapp)
    assert len(calls) == 6  # filtering completes the project-wide scan
    assert dialog.l_page.text() == "0 / 0"


def test_proc_review_filters_by_aggregate_first_failure(
    qapp, tmp_path, monkeypatch
):
    script = tmp_path / "task_v1.py"
    _write_review_script(script)
    subjects = {}
    for index in range(1, 6):
        subject = f"sub-{index:02d}"
        info = FakeBidsInfo(subject=subject)
        if index == 4:
            info.meg_list = []
        else:
            info.meg_list = [FakeMegDataset(
                "task",
                "run01.ds",
                f"/tmp/fake_bids/{subject}/meg/{subject}_task-task_run-01_meg.ds",
            )]
        if index == 5:
            info.meg_list.append(FakeMegDataset(
                "task",
                "run02.ds",
                f"/tmp/fake_bids/{subject}/meg/{subject}_task-task_run-02_meg.ds",
            ))
        subjects[subject] = info
    window = BIDS_Project_Window(
        bids_project=FakeProject(subjects), gridsize_row=1, gridsize_col=2
    )

    def fake_evaluate(spec, dataset):
        subject = Path(dataset).name.split("_", 1)[0]
        if subject == "sub-05":
            present = (True, "run-02" not in str(dataset))
        else:
            present = {
                "sub-01": (True, True),
                "sub-02": (False, False),
                "sub-03": (True, False),
            }[subject]
        return tuple(
            SimpleNamespace(
                request=request,
                record=SimpleNamespace(
                    exists=present[offset],
                    status="present" if present[offset] else "missing",
                    path=f"/tmp/{request.key}",
                    component_paths=(),
                    validation_errors=(),
                ),
            )
            for offset, request in enumerate(spec.outputs)
        )

    monkeypatch.setattr(qt_gui_module, "evaluate_dataset", fake_evaluate)
    dialog = ProcReviewDialog(window, script)
    _finish_proc_review_scan(dialog, qapp)

    dialog.b_hide_successful.setChecked(True)
    assert [button.text() for button in dialog.subject_buttons] == [
        "sub-02", "sub-03", "sub-04", "sub-05"
    ]

    dialog.b_failure_step.setCurrentText("stc: face")
    assert [button.text() for button in dialog.subject_buttons] == [
        "sub-03", "sub-04", "sub-05"
    ]

    dialog.b_failure_step.setCurrentText("epochs")
    assert [button.text() for button in dialog.subject_buttons] == [
        "sub-02", "sub-04"
    ]

    dialog.b_hide_no_task.setChecked(True)
    assert [button.text() for button in dialog.subject_buttons] == ["sub-02"]


def test_proc_review_global_run_filter_keeps_subject_scope(qapp, tmp_path, monkeypatch):
    script = tmp_path / "task_v1.py"
    _write_review_script(script)
    subjects = {
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
    }
    subjects["sub-01"].meg_list = [FakeMegDataset(
        "task", "run01.ds", "/tmp/sub-01_task-task_run-01_meg.ds"
    )]
    subjects["sub-02"].meg_list = [FakeMegDataset(
        "task", "run02.ds", "/tmp/sub-02_task-task_run-02_meg.ds"
    )]
    window = BIDS_Project_Window(bids_project=FakeProject(subjects), gridsize_row=1, gridsize_col=2)
    monkeypatch.setattr(qt_gui_module, "evaluate_dataset", lambda spec, dataset: ())
    dialog = ProcReviewDialog(window, script)

    dialog.b_run.setCurrentText("run-01")

    assert [button.text() for button in dialog.subject_buttons] == [
        "sub-01", "sub-02"
    ]
    assert len(dialog.indicators) == 4


def test_proc_review_first_failure_opens_logged_error(qapp, tmp_path, monkeypatch):
    script = tmp_path / "task_v1.py"
    _write_review_script(script)
    bids_info = FakeBidsInfo(subject="sub-01")
    bids_info.meg_list = [FakeMegDataset(
        "task", "run01.ds", "/tmp/fake_bids/sub-01/meg/sub-01_task-task_run-01_meg.ds"
    )]
    window = BIDS_Project_Window(
        bids_project=FakeProject({"sub-01": bids_info}),
        gridsize_row=1,
        gridsize_col=1,
    )

    def fake_evaluate(spec, dataset):
        return tuple(
            SimpleNamespace(
                request=request,
                record=SimpleNamespace(
                    exists=False,
                    status="missing",
                    path=f"/tmp/{request.key}",
                    component_paths=(),
                    validation_errors=(),
                ),
            )
            for request in spec.outputs
        )

    report = SimpleNamespace(display_text="Traceback\nValueError: no events")
    monkeypatch.setattr(qt_gui_module, "evaluate_dataset", fake_evaluate)
    monkeypatch.setattr(qt_gui_module, "find_processing_failure", lambda *args, **kwargs: report)
    monkeypatch.setattr(qt_gui_module.ProcessingFailureDialog, "exec", lambda self: 0)
    dialog = ProcReviewDialog(window, script)
    _finish_proc_review_scan(dialog, qapp)

    failure_buttons = [
        indicator for indicator in dialog.indicators
        if isinstance(indicator, DerivativeFailureButton)
    ]
    assert len(failure_buttons) == 1
    assert failure_buttons[0].emphasis_width == 4
    assert failure_buttons[0].minimumHeight() == 22

    failure_buttons[0].click()

    displayed = dialog.failure_dialog.error_text.toPlainText()
    assert "ValueError: no events" in displayed
    assert "First failed derivative: epochs" in [
        label.text() for label in dialog.failure_dialog.findChildren(qt_gui_module.QLabel)
    ][0]

    dialog.subject_buttons[0].click()
    focused_failure_buttons = [
        indicator for indicator in dialog.indicators
        if isinstance(indicator, DerivativeFailureButton)
    ]
    assert len(focused_failure_buttons) == 1
    assert focused_failure_buttons[0].width() == 16
    focused_failure_buttons[0].click()
    assert "ValueError: no events" in dialog.failure_dialog.error_text.toPlainText()


def test_processing_failure_dialog_loads_full_log_and_scrolls_to_bottom(qapp, tmp_path):
    application_log = tmp_path / "processing.log"
    application_log.write_text("application fallback", encoding="utf-8")
    execution_log = tmp_path / "job.err"
    execution_log.write_text(
        "\n".join([f"log line {index}" for index in range(200)] + ["FINAL ERROR"]),
        encoding="utf-8",
    )
    report = SimpleNamespace(
        display_text="Short failure summary",
        application_log=application_log,
        execution_log=execution_log,
    )
    dialog = qt_gui_module.ProcessingFailureDialog(
        "sub-01", "task", "run-01", "epochs", "missing", report
    )
    dialog.show()

    dialog.b_load_full_log.click()
    qapp.processEvents()

    displayed = dialog.error_text.toPlainText()
    assert f"Full log: {execution_log}" in displayed
    assert "FINAL ERROR" in displayed
    scrollbar = dialog.error_text.verticalScrollBar()
    assert scrollbar.value() == scrollbar.maximum()


def test_processing_failure_dialog_falls_back_to_application_log(qapp, tmp_path):
    application_log = tmp_path / "processing.log"
    application_log.write_text("application log contents", encoding="utf-8")
    report = SimpleNamespace(
        display_text="Short failure summary",
        application_log=application_log,
        execution_log=tmp_path / "missing-job.err",
    )
    dialog = qt_gui_module.ProcessingFailureDialog(
        "sub-01", "task", "run-01", "epochs", "missing", report
    )

    dialog.b_load_full_log.click()
    qapp.processEvents()

    displayed = dialog.error_text.toPlainText()
    assert f"Full log: {application_log}" in displayed
    assert "application log contents" in displayed


def test_proc_review_rejects_script_task_missing_from_project(qapp, tmp_path):
    script = tmp_path / "unknown_v1.py"
    _write_review_script(script)
    window = BIDS_Project_Window(
        bids_project=FakeProject({"sub-01": FakeBidsInfo(subject="sub-01")}),
        gridsize_row=1,
        gridsize_col=1,
    )

    with pytest.raises(qt_gui_module.ProcessingScriptError, match="not present"):
        ProcReviewDialog(window, script)


def test_proc_review_button_chooses_script_and_opens_dialog(qapp, tmp_path, monkeypatch):
    script = tmp_path / "task_v1.py"
    _write_review_script(script)
    window = BIDS_Project_Window(
        bids_project=FakeProject({"sub-01": FakeBidsInfo(subject="sub-01")}),
        gridsize_row=1,
        gridsize_col=1,
    )
    calls = []

    class FakeReviewDialog:
        def __init__(self, project_window, processing_script, parent=None):
            calls.append((project_window, processing_script, parent))

        def exec(self):
            calls.append("exec")

    monkeypatch.setattr(
        qt_gui_module.QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(script), "Python scripts (*.py)"),
    )
    monkeypatch.setattr(qt_gui_module, "ProcReviewDialog", FakeReviewDialog)

    window.open_proc_review_dialog()

    assert calls[0] == (window, str(script), window)
    assert calls[1] == "exec"


def test_add_dataset_opens_one_window_for_current_root(
        qapp, monkeypatch):
    class FakeSignal:
        def __init__(self):
            self.callback = None

        def connect(self, callback):
            self.callback = callback

    class FakeDatasetWindow:
        instances = []

        def __init__(self, bids_root=None):
            self.bids_root = bids_root
            self.bids_import_finished = FakeSignal()
            self.visible = False
            self.raised = 0
            self.activated = 0
            self.instances.append(self)

        def isVisible(self):
            return self.visible

        def show(self):
            self.visible = True

        def showNormal(self):
            return None

        def raise_(self):
            self.raised += 1

        def activateWindow(self):
            self.activated += 1

    fake_module = types.ModuleType("nih2mne.GUI.dataset_gui")
    fake_module.GUI_MainWindow = FakeDatasetWindow
    monkeypatch.setitem(
        sys.modules,
        "nih2mne.GUI.dataset_gui",
        fake_module,
    )

    project = FakeProject({"sub-01": FakeBidsInfo(subject="sub-01")})
    window = BIDS_Project_Window(
        bids_project=project,
        gridsize_row=1,
        gridsize_col=1,
    )
    reloads = []
    monkeypatch.setattr(
        window,
        "reload_bids_project",
        lambda: reloads.append(True),
    )

    window.open_dataset_gui()
    dataset_window = FakeDatasetWindow.instances[0]
    assert dataset_window.bids_root == project.bids_root
    assert dataset_window.visible is True

    dataset_window.bids_import_finished.callback()
    assert reloads == [True]

    window.open_dataset_gui()
    assert len(FakeDatasetWindow.instances) == 1
    assert dataset_window.raised == 1
    assert dataset_window.activated == 1


def test_reload_bids_project_rescans_and_refreshes_controls(
        qapp, monkeypatch):
    original = FakeProject({
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
    })
    window = BIDS_Project_Window(
        bids_project=original,
        gridsize_row=1,
        gridsize_col=2,
    )
    window.b_task_chooser.setCurrentIndex(1)
    window.filter_task_qa_vis()

    refreshed = FakeProject({
        "sub-01": FakeBidsInfo(subject="sub-01"),
        "sub-02": FakeBidsInfo(subject="sub-02"),
        "sub-03": FakeBidsInfo(subject="sub-03"),
    })
    refreshed.issues["Freesurfer_notStarted"] = ["sub-03"]
    calls = []

    def load_project(**kwargs):
        calls.append(kwargs)
        return refreshed

    monkeypatch.setattr(qt_gui_module, "bids_project", load_project)

    assert window.reload_bids_project() is True
    assert calls == [{"bids_root": original.bids_root, "force_update": True}]
    assert window.bids_project is refreshed
    assert window.selected_task == "rest"
    assert window.subject_keys == ["sub-01", "sub-02", "sub-03"]
    assert window.b_subject_number.text() == "Subject Totals: #3"
    assert window.b_current_page_idx.text() == "Page: 0 / 1"
    assert window.b_run_freesurfer.text() == "Run Freesurfer (N=1)"
    assert window.windowTitle() == f"BIDS Folder: {refreshed.bids_root}"


def test_project_window_proc_actions(qapp):
    class ProcInfo(FakeBidsInfo):
        def __init__(self, subject):
            super().__init__(subject=subject)
            self.proc_freesurfer_calls = 0
            self.mri_preproc_calls = []

        def proc_freesurfer(self):
            self.proc_freesurfer_calls += 1

        def mri_preproc(self, surf, fname):
            self.mri_preproc_calls.append({"surf": surf, "fname": fname})

    subjects = {
        "sub-01": ProcInfo("sub-01"),
        "sub-02": ProcInfo("sub-02"),
        "sub-03": ProcInfo("sub-03"),
    }
    project = FakeProject(subjects)
    project.issues = {
        "Freesurfer_notStarted": ["sub-01"],
        "Freesurfer_failed": ["sub-02"],
    }
    window = BIDS_Project_Window(
        bids_project=project,
        gridsize_row=1,
        gridsize_col=2,
    )

    window.proc_freesurfer()
    window.b_mri_volSurf_selection.setCurrentText("Vol")
    window.proc_mriprep()

    assert subjects["sub-01"].proc_freesurfer_calls == 1
    assert subjects["sub-02"].proc_freesurfer_calls == 0
    assert subjects["sub-03"].proc_freesurfer_calls == 0
    assert subjects["sub-01"].mri_preproc_calls == []
    assert subjects["sub-02"].mri_preproc_calls == []
    assert subjects["sub-03"].mri_preproc_calls == [{"surf": False, "fname": "all"}]


def test_window_reuses_running_qapplication(qapp, monkeypatch):
    class ExistingApplication:
        @classmethod
        def instance(cls):
            return qapp

    project = FakeProject({"sub-01": FakeBidsInfo(subject="sub-01")})
    monkeypatch.setattr(qt_gui_module, "QApplication", ExistingApplication)
    monkeypatch.setattr(qt_gui_module.os, "chdir", lambda _path: None)

    window = qt_gui_module.window(
        bids_project=project,
        num_rows=1,
        num_cols=1,
    )

    try:
        assert isinstance(window, BIDS_Project_Window)
        assert window.isVisible()
    finally:
        window.close()
