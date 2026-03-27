from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5 import QtWidgets

try:
    from .beamformer_maker import Ui_Form
except ImportError:
    from beamformer_maker import Ui_Form


TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "proc" / "beamformer_template.py"
TEMPLATE_INSERT_MARKER = "#%% << INSERT GUI COMPONENTS HERE >>"
SCRIPT_DIALOG_START_DIR = Path("~/megcore/datproc/").expanduser()
_VERSION_PATTERN = re.compile(r"_v(\d+)\.py$")


@dataclass
class BeamformerFormEntries:
    """Container for the current beamformer GUI values."""

    fname: str
    tmin: str
    tmax: str
    fmin: str
    fmax: str
    threshold_rejection: str
    beamformer_regularization: str
    megnet: bool
    muscle_detect: bool
    conditions_of_interest: str
    contrast_type: str
    anat_overwrite: bool
    beamformer_overwrite: bool
    contrasts_overwrite: bool

    @classmethod
    def from_ui(cls, ui: Any) -> "BeamformerFormEntries":
        """Read the current widget state from the generated Qt form."""
        return cls(
            fname=ui.lineEdit_fname.text(),
            tmin=ui.lineEdit_tmin.text(),
            tmax=ui.lineEdit_tmax.text(),
            fmin=ui.lineEdit_fmin.text(),
            fmax=ui.lineEdit_fmax.text(),
            threshold_rejection=ui.lineEdit_ThresholdRejection.text(),
            beamformer_regularization=ui.lineEdit_BeamformerRegularization.text(),
            megnet=ui.cb_MEGNET.isChecked(),
            muscle_detect=ui.cb_MuscleDetect.isChecked(),
            conditions_of_interest=ui.lineEdit_ConditionsOfInterest.text(),
            contrast_type=ui.comboBox_ContrastType.currentText(),
            anat_overwrite=ui.cb_AnatOverwrite.isChecked(),
            beamformer_overwrite=ui.cb_BeamformerOverwrite.isChecked(),
            contrasts_overwrite=ui.cb_ContrastsOverwrite.isChecked(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the dataclass fields as a plain dictionary."""
        return asdict(self)


def _split_csv_values(raw_text: str) -> List[str]:
    """Normalize comma or semicolon separated GUI text into a clean list."""
    separators_normalized = raw_text.replace(";", ",")
    return [item.strip() for item in separators_normalized.split(",") if item.strip()]


def _normalize_contrast_type(raw_value: str) -> str:
    """Translate GUI display labels into the script values expected downstream."""
    normalized = raw_value.strip()
    if not normalized:
        return "percent"

    label_map = {
        "log10 ratio": "logratio",
        "percent change": "percent",
    }
    return label_map.get(normalized.lower(), normalized)


def _muscle_detection_block(enabled: bool) -> List[str]:
    """Render the raw-preprocessing lines for optional muscle artifact rejection."""
    if not enabled:
        return ["# Muscle detection disabled from GUI"]

    return [
        "_musc_annot = annotate_muscle_zscore(raw, threshold=4, ch_type='mag', min_length_good=0.1, " ,
        "                       filter_freq=(110, 140), n_jobs=n_jobs, verbose=None)",
        "raw.set_annotations(raw.annotations + _musc_annot[0])",
    ]


def _task_id_from_dataset_path(dataset_path: str) -> str:
    """Derive a task id from the dataset name using BIDS rules when available."""
    dataset_name = Path(dataset_path).name
    if dataset_name.endswith(".ds"):
        dataset_name = dataset_name[:-3]
    if not dataset_name:
        return "beamformer"

    if "task-" in dataset_name:
        task_id = dataset_name.split("task-", 1)[-1].split("_", 1)[0].strip()
        if task_id:
            return task_id

    parts = [part for part in dataset_name.split("_") if part]
    if len(parts) > 1 and parts[1]:
        return parts[1]
    if parts:
        return parts[0]
    return "beamformer"


def _next_script_version(task_id: str, output_dir: Path = SCRIPT_DIALOG_START_DIR) -> int:
    """Return the next available version number for a task id in the output directory."""
    if not output_dir.exists():
        return 1

    versions = []
    for candidate in output_dir.glob(f"{task_id}_v*.py"):
        match = _VERSION_PATTERN.search(candidate.name)
        if match:
            versions.append(int(match.group(1)))
    if not versions:
        return 1
    return max(versions) + 1


def _default_script_name(dataset_path: str, output_dir: Path = SCRIPT_DIALOG_START_DIR) -> str:
    """Build a default versioned script name from the selected dataset path."""
    task_id = _task_id_from_dataset_path(dataset_path)
    version = _next_script_version(task_id, output_dir=output_dir)
    return f"{task_id}_v{version}.py"


def build_gui_component_block(entries: BeamformerFormEntries) -> str:
    """Render the Python block inserted into the beamformer template marker."""
    conds_oi = _split_csv_values(entries.conditions_of_interest)
    contrasts_type = _normalize_contrast_type(entries.contrast_type)

    try:
        beam_reg_fraction: Any = float(entries.beamformer_regularization) / 100.0
    except ValueError:
        beam_reg_fraction = entries.beamformer_regularization

    block_lines = [
        "# GUI entries >>",
        "#%% GUI Components",
        f"dataset_path = pathlib.Path({repr(entries.fname)})",
        "entity_map = {}",
        "for part in dataset_path.stem.split('_'):",
        "    if '-' in part:",
        "        key, value = part.split('-', 1)",
        "        entity_map[key] = value",
        "",
        "subject_dir = next((parent for parent in dataset_path.parents if parent.name.startswith('sub-')), None)",
        "if subject_dir is None:",
        "    raise ValueError(f'Unable to determine BIDS subject from {dataset_path}')",
        "",
        "bids_root = subject_dir.parent",
        "subject = entity_map.get('sub', subject_dir.name.replace('sub-', '', 1))",
        "run = entity_map.get('run')",
        "ses = entity_map.get('ses')",
        "task_type = entity_map.get('task')",
        "project = 'beamformer'",
        "",
        f"epo_tmin = {entries.tmin}",
        f"epo_tmax = {entries.tmax}",
        "epo_baseline = None",
        f"f_min = {entries.fmin}",
        f"f_max = {entries.fmax}",
        "er_run = '01'",
        "",
        f"reject_dict = dict(mag={entries.threshold_rejection})",
        "cov_cv = 5",
        "cov_method = 'shrunk'",
        f"beam_reg = {repr(beam_reg_fraction)}",
        "beam_ori = 'max-power'",
        "",
        f"conds_OI = {repr(conds_oi)}",
        "contrasts_OI = []",
        f"contrasts_type = {repr(contrasts_type)}",
        "",
        f"use_megnet_ica = {repr(entries.megnet)}",
        f"use_muscle_detection = {repr(entries.muscle_detect)}",
        f"overwrite_anats = {repr(entries.anat_overwrite)}",
        f"overwrite_preproc = {repr(entries.anat_overwrite)}",
        f"overwrite_beam = {repr(entries.beamformer_overwrite)}",
        f"overwrite_contrasts = {repr(entries.contrasts_overwrite)}",
        "# GUI entries <<",
    ]
    return "\n".join(block_lines)


def render_beamformer_script(
    entries: BeamformerFormEntries,
    template_path: Path = TEMPLATE_PATH,
) -> str:
    """Load the template, inject the GUI block at the marker, and return the script text."""
    template_text = template_path.read_text(encoding="utf-8")
    if TEMPLATE_INSERT_MARKER not in template_text:
        raise ValueError(
            f"Template marker {repr(TEMPLATE_INSERT_MARKER)} not found in {template_path}"
        )

    before, after = template_text.split(TEMPLATE_INSERT_MARKER, 1)
    gui_block = build_gui_component_block(entries)
    script_text = f"{before.rstrip()}\n\n{gui_block}\n\n{after.lstrip()}"

    muscle_block = "\n".join(
        [
            "_musc_annot = annotate_muscle_zscore(raw, threshold=4, ch_type='mag', min_length_good=0.1, " ,
            "                       filter_freq=(110, 140), n_jobs=n_jobs, verbose=None)",
            "raw.set_annotations(raw.annotations + _musc_annot[0])",
        ]
    )
    return script_text.replace(muscle_block, "\n".join(_muscle_detection_block(entries.muscle_detect)))


class BeamformerFormWindow(QtWidgets.QWidget):
    """Runtime wrapper around the generated form with file-dialog actions."""

    def __init__(self) -> None:
        """Create the Qt form and connect the browse and write buttons."""
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pb_OpenFileDialog.clicked.connect(self.open_dataset_directory)
        self.ui.pb_WriteScript.clicked.connect(self.write_script_via_dialog)

    def entries(self) -> BeamformerFormEntries:
        """Return the current form values as a typed dataclass."""
        return BeamformerFormEntries.from_ui(self.ui)

    def _dialog_start_dir(self) -> str:
        """Choose a reasonable starting directory for the dataset browser."""
        current_path = self.ui.lineEdit_fname.text().strip()
        if current_path:
            if os.path.isdir(current_path):
                return current_path
            parent = os.path.dirname(current_path)
            if parent:
                return parent
        return os.getcwd()

    def _script_save_path(self) -> str:
        """Build the default save path shown in the script file dialog."""
        current_path = self.ui.lineEdit_fname.text().strip()
        default_name = _default_script_name(current_path)
        return str(SCRIPT_DIALOG_START_DIR / default_name)

    def open_dataset_directory(self) -> Optional[str]:
        """Open a directory picker and keep only selections ending in .ds."""
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

    def write_script_via_dialog(self) -> Optional[str]:
        """Render the template from the current GUI state and save it through a file dialog."""
        script_text = render_beamformer_script(self.entries())
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Write beamformer script",
            self._script_save_path(),
            "Python files (*.py)",
        )
        if not save_path:
            return None

        Path(save_path).write_text(script_text, encoding="utf-8")
        return save_path


def launch_gui() -> int:
    """Launch the standalone beamformer form window."""
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
