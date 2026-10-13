from pathlib import Path

import mne
import numpy as np
import pytest

from nih2mne.utilities.door_ssp_annotation_plotter import (
    annotation_path_for_dataset,
    episode_plot_filename,
    episode_window,
    plot_dataset_annotations,
    plot_episode,
)
from nih2mne.utilities.door_ssp_classifier import ProcessingConfig


def _raw(sfreq=100.0, duration=5.0):
    info = mne.create_info(["Fz", "C3", "C4"], sfreq, ch_types="eeg")
    info.set_montage("standard_1020")
    data = np.vstack(
        [
            np.sin(np.linspace(0, 20, int(sfreq * duration))),
            np.cos(np.linspace(0, 20, int(sfreq * duration))),
            -np.sin(np.linspace(0, 20, int(sfreq * duration))),
        ]
    ) * 1e-12
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_annotation_path_prefers_standard_and_supports_legacy(tmp_path):
    dataset = tmp_path / "MEG_EmptyRoom_test.ds"
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    legacy = annotation_dir / "MEG_EmptyRoom_test.ds_door_artifacts-annot.fif"
    legacy.touch()
    assert annotation_path_for_dataset(dataset, annotation_dir) == legacy
    standard = annotation_dir / "MEG_EmptyRoom_test_door_artifacts-annot.fif"
    standard.touch()
    assert annotation_path_for_dataset(dataset, annotation_dir) == standard


def test_episode_window_uses_duration_around_midpoint():
    raw = _raw()
    window = episode_window(
        raw,
        onset=1.0,
        duration=2.0,
        description="door_artifact_positive",
    )
    assert window.midpoint == 2.0
    assert window.start_index == 100
    assert window.midpoint_index == 200
    assert window.stop_index == 301
    with pytest.raises(ValueError, match="duration must be positive"):
        episode_window(raw, onset=1.0, duration=0.0, description="bad")


def test_episode_plot_writes_decodable_png(tmp_path):
    import matplotlib.image as mpimg

    raw = _raw()
    window = episode_window(
        raw,
        onset=1.0,
        duration=2.0,
        description="door_artifact_positive",
    )
    output = tmp_path / episode_plot_filename(Path("sample.ds"), 1, 2.0)
    topo_scale, butterfly_scale = plot_episode(
        raw.get_data(),
        raw,
        window,
        output,
        dataset=Path("sample.ds"),
        episode_number=1,
        dpi=80,
    )
    assert topo_scale > 0
    assert butterfly_scale > 0
    image = mpimg.imread(output)
    assert image.ndim == 3
    assert image.size > 0


def test_dataset_plotter_skips_annotations_shorter_than_three_seconds(tmp_path):
    annotation_path = tmp_path / "door-annot.fif"
    mne.Annotations(
        onset=[1.0, 5.0],
        duration=[0.5, 2.99],
        description=["short", "also_short"],
    ).save(annotation_path)
    count = plot_dataset_annotations(
        Path("unused.ds"),
        annotation_path,
        tmp_path / "plots",
        ProcessingConfig(),
        dpi=80,
        overwrite=False,
    )
    assert count == 0
    assert not (tmp_path / "plots").exists()
