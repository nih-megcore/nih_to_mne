#!/usr/bin/env python3

import os
import sys
import types
from collections import OrderedDict

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nih2mne.GUI.beamformer_maker import Ui_BeamformerScriptGenerator
import nih2mne.GUI.qt_compat as qt_compat
from nih2mne.GUI.qt_compat import QT_API, QT_BINDING, QtCore, QtWidgets
from nih2mne.GUI.templates.BIDS_creator_gui import Ui_MainWindow as Ui_BidsWindow
from nih2mne.GUI.templates.OLDvers.BIDS_creator_gui import (
    Ui_MainWindow as Ui_LegacyBidsWindow,
)
from nih2mne.GUI.templates.file_staging_meg import Ui_MainWindow as Ui_StagingWindow
from nih2mne.GUI.templates.input_error_dset_tile_listWidgetBase import (
    Ui_ErrorDatasetTile,
)
from nih2mne.GUI.templates.input_meg_dset_tile_listWidgetBase import Ui_InputDatasetTile
from nih2mne.GUI.templates.parse_marks_single_line import Ui_Form as Ui_ParseMarks
from nih2mne.GUI.templates.trigger_processing_gui import (
    Ui_MainWindow as Ui_TriggerWindow,
)
from nih2mne.GUI.templates.trigger_single_line import Ui_Form as Ui_TriggerLine


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.mark.parametrize(
    ("ui_class", "widget_class"),
    [
        (Ui_BeamformerScriptGenerator, QtWidgets.QWidget),
        (Ui_BidsWindow, QtWidgets.QMainWindow),
        (Ui_LegacyBidsWindow, QtWidgets.QMainWindow),
        (Ui_StagingWindow, QtWidgets.QMainWindow),
        (Ui_ErrorDatasetTile, QtWidgets.QWidget),
        (Ui_InputDatasetTile, QtWidgets.QWidget),
        (Ui_ParseMarks, QtWidgets.QWidget),
        (Ui_TriggerWindow, QtWidgets.QMainWindow),
        (Ui_TriggerLine, QtWidgets.QWidget),
    ],
)
def test_generated_form_setup(qapp, ui_class, widget_class):
    widget = widget_class()
    ui = ui_class()

    ui.setupUi(widget)

    assert widget.objectName()
    widget.deleteLater()


def test_selected_binding_is_shared_with_environment():
    assert QT_API in {"pyqt5", "pyqt6"}
    assert QT_BINDING == {"pyqt5": "PyQt5", "pyqt6": "PyQt6"}[QT_API]
    assert os.environ["QT_API"] == QT_API
    assert QtCore.__package__ == QT_BINDING


def test_mixed_loaded_bindings_are_rejected(monkeypatch):
    other_binding = "PyQt5" if QT_BINDING == "PyQt6" else "PyQt6"
    monkeypatch.setitem(sys.modules, other_binding, types.ModuleType(other_binding))

    with pytest.raises(RuntimeError, match="both already imported"):
        qt_compat._loaded_binding()


def test_pyqtgraph_uses_selected_binding():
    import pyqtgraph

    assert pyqtgraph.Qt.QT_LIB == QT_BINDING


def test_parse_marks_tile_uses_typed_check_states(qapp):
    from nih2mne.GUI.trigger_code_gui import parse_marks_tile

    events = OrderedDict(
        [
            ("one", {"name": "event_one"}),
            ("two", {"name": "event_two"}),
        ]
    )
    tile = parse_marks_tile(event_name_dict=events)

    assert tile.cb_OnLead.checkState() == QtCore.Qt.CheckState.Checked
    tile.set_onlag_selection()
    tile.cb_OnLag.setCheckState(QtCore.Qt.CheckState.Checked)
    assert tile.get_outputs()["mark_on"] == "lag"

    tile._disable()
    assert tile.combo_LeadSelection.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    tile.deleteLater()
