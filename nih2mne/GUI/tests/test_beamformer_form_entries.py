#!/usr/bin/env python3

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

from nih2mne.GUI.beamformer_form_entries import (
    BeamformerFormEntries,
    BeamformerFormWindow,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_beamformer_form_entries_capture_gui_values(qapp):
    window = BeamformerFormWindow()
    ui = window.ui

    ui.lineEdit_tmin.setText("-0.25")
    ui.lineEdit_tmax.setText("0.25")
    ui.lineEdit_fmin.setText("1")
    ui.lineEdit_fmax.setText("110")
    ui.lineEdit_BeamformerRegularization.setText("5")

    entries = BeamformerFormEntries.from_ui(ui)

    assert entries.tmin == "-0.25"
    assert entries.tmax == "0.25"
    assert entries.fmin == "1"
    assert entries.fmax == "110"
    assert entries.beamformer_regularization == "5"

    assert entries.to_dict()["tmin"] == "-0.25"
    assert entries.to_dict()["tmax"] == "0.25"
    assert entries.to_dict()["fmin"] == "1"
    assert entries.to_dict()["fmax"] == "110"
    assert entries.to_dict()["beamformer_regularization"] == "5"


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
