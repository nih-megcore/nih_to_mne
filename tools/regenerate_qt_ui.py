#!/usr/bin/env python3
"""Regenerate checked-in Qt Designer forms with PyQt6's ``pyuic``."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORM_TARGETS = {
    "nih2mne/GUI/beamformer_maker2.ui": "nih2mne/GUI/beamformer_maker.py",
    "nih2mne/GUI/templates/BIDS_creator_gui.ui": "nih2mne/GUI/templates/BIDS_creator_gui.py",
    "nih2mne/GUI/templates/file_staging_meg.ui": "nih2mne/GUI/templates/file_staging_meg.py",
    "nih2mne/GUI/templates/input_error_dset_tile_listWidgetBase.ui": "nih2mne/GUI/templates/input_error_dset_tile_listWidgetBase.py",
    "nih2mne/GUI/templates/input_meg_dset_tile_listWidgetBase.ui": "nih2mne/GUI/templates/input_meg_dset_tile_listWidgetBase.py",
    "nih2mne/GUI/templates/parse_marks_single_line.ui": "nih2mne/GUI/templates/parse_marks_single_line.py",
    "nih2mne/GUI/templates/trigger_processing_gui.ui": "nih2mne/GUI/templates/trigger_processing_gui.py",
    "nih2mne/GUI/templates/trigger_single_line.ui": "nih2mne/GUI/templates/trigger_single_line.py",
    "nih2mne/GUI/templates/OLDvers/BIDS_creator_gui.ui_manually_formatted": (
        "nih2mne/GUI/templates/OLDvers/BIDS_creator_gui.py"
    ),
}
PYQT_IMPORT = "from PyQt6 import QtCore, QtGui, QtWidgets"
COMPAT_IMPORT = "from nih2mne.GUI.qt_compat import QtCore, QtGui, QtWidgets"


def regenerate(source: Path, target: Path) -> None:
    """Compile one form and redirect its binding import through ``qt_compat``."""
    source_name = source.relative_to(ROOT)
    with tempfile.TemporaryDirectory(prefix="nih2mne-pyuic6-") as tmp_dir:
        generated_path = Path(tmp_dir) / target.name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyQt6.uic.pyuic",
                "-x",
                str(source_name),
                "-o",
                str(generated_path),
            ],
            check=True,
            cwd=ROOT,
        )
        generated = generated_path.read_text(encoding="utf-8")

    if generated.count(PYQT_IMPORT) != 1:
        raise RuntimeError(f"Unexpected PyQt6 import structure generated for {source}")
    target.write_text(generated.replace(PYQT_IMPORT, COMPAT_IMPORT), encoding="utf-8")


def main() -> None:
    for source_name, target_name in FORM_TARGETS.items():
        regenerate(ROOT / source_name, ROOT / target_name)


if __name__ == "__main__":
    main()
