from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from PyQt5 import QtWidgets

try:
    from .beamformer_maker import Ui_Form
except ImportError:
    from beamformer_maker import Ui_Form


@dataclass
class BeamformerFormEntries:
    fname: str
    tmin: str
    tmax: str
    fmin: str
    fmax: str
    threshold_rejection: str
    beamformer_regularization: str
    megnet: bool
    conditions_of_interest: str
    contrast_type: str
    anat_overwrite: bool
    beamformer_overwrite: bool
    contrasts_overwrite: bool

    @classmethod
    def from_ui(cls, ui: Any) -> "BeamformerFormEntries":
        return cls(
            fname=ui.lineEdit_fname.text(),
            tmin=ui.lineEdit_tmin.text(),
            tmax=ui.lineEdit_tmax.text(),
            fmin=ui.lineEdit_fmin.text(),
            fmax=ui.lineEdit_fmax.text(),
            threshold_rejection=ui.lineEdit_ThresholdRejection.text(),
            beamformer_regularization=ui.lineEdit_BeamformerRegularization.text(),
            megnet=ui.cb_MEGNET.isChecked(),
            conditions_of_interest=ui.lineEdit_ConditionsOfInterest.text(),
            contrast_type=ui.comboBox_ContrastType.currentText(),
            anat_overwrite=ui.cb_AnatOverwrite.isChecked(),
            beamformer_overwrite=ui.cb_BeamformerOverwrite.isChecked(),
            contrasts_overwrite=ui.cb_ContrastsOverwrite.isChecked(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BeamformerFormWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pb_OpenFileDialog.clicked.connect(self.open_dataset_directory)

    def entries(self) -> BeamformerFormEntries:
        return BeamformerFormEntries.from_ui(self.ui)

    def _dialog_start_dir(self) -> str:
        current_path = self.ui.lineEdit_fname.text().strip()
        if current_path:
            if os.path.isdir(current_path):
                return current_path
            parent = os.path.dirname(current_path)
            if parent:
                return parent
        return os.getcwd()

    def open_dataset_directory(self) -> Optional[str]:
        dataset_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select .ds dataset directory",
            self._dialog_start_dir(),
            QtWidgets.QFileDialog.ShowDirsOnly,
        )
        if not dataset_dir:
            return None
        if not dataset_dir.endswith(".ds"):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid dataset",
                "Please select a directory with the .ds extension.",
            )
            return None

        self.ui.lineEdit_fname.setText(dataset_dir)
        return dataset_dir


def launch_gui() -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = BeamformerFormWindow()
    window.show()

    if owns_app:
        return app.exec_()
    return 0


if __name__ == "__main__":
    sys.exit(launch_gui())
