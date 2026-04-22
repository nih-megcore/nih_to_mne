#!/usr/bin/env python3

import os
import sys
from types import SimpleNamespace
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

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
SubjectSelectionDialog = qt_gui_module.SubjectSelectionDialog


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


def test_subject_gui_plot_save_and_override_actions(qapp):
    bids_info = FakeBidsInfo(mri="Multiple")
    gui = Subject_GUI(bids_info)

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
        }
    ]
    assert bids_info.plot_mri_fids_called is True
    assert bids_info.plot_3d_coreg_calls == [0]
    assert bids_info.saved is True
    assert bids_info.override_calls == ["/tmp/mri_b.nii.gz"]


def test_get_task_datproc_files_filters_by_task_prefix(tmp_path):
    (tmp_path / "rest_v2.py").write_text("")
    (tmp_path / "rest_v1.sh").write_text("")
    (tmp_path / "task_v1.py").write_text("")
    (tmp_path / "resting_extra.py").write_text("")

    assert get_task_datproc_files("rest", datproc_dir=str(tmp_path)) == ["rest_v2.py", "rest_v1.sh"]


def test_build_datproc_command_uses_python_for_py_files():
    py_cmd = build_datproc_command("/tmp/rest_v2.py", "/tmp/sub-01_task-rest_meg.ds")
    shell_cmd = build_datproc_command("/tmp/rest_v2.sh", "/tmp/sub-01_task-rest_meg.ds")

    assert py_cmd.endswith("/tmp/rest_v2.py /tmp/sub-01_task-rest_meg.ds")
    assert ".py" in py_cmd
    assert shell_cmd == "/tmp/rest_v2.sh /tmp/sub-01_task-rest_meg.ds"


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
