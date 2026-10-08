from pathlib import Path

import mne
import numpy as np
import pytest

from nih2mne.utilities.door_artifact_classifier import (
    EVENT_ID,
    MatchResult,
    align_data_to_template,
    classify_spatial_template,
    create_spatial_template,
    find_template_matches,
    load_spatial_template,
    matches_to_annotations,
    output_paths,
    spatial_corr,
    write_match_outputs,
    write_spatial_template,
)
from nih2mne.utilities.interactive_ctf_topomap import prepare_display_data


def _raw(data, sfreq=10.0):
    info = mne.create_info(
        [f"MEG{index:03d}" for index in range(len(data))],
        sfreq,
        ch_types="mag",
    )
    return mne.io.RawArray(np.asarray(data, dtype=float), info)


def test_spatial_corr_is_scale_independent_and_handles_constant_samples():
    template = np.array([1.0, 0.0, -1.0])
    data = np.column_stack(
        [
            template * 1e-12,
            template * -1e12,
            np.ones(3),
        ]
    )
    np.testing.assert_allclose(spatial_corr(data, template), [1.0, -1.0, 0.0])


def test_spatial_corr_rejects_zero_variance_template():
    with pytest.raises(ValueError, match="zero spatial variance"):
        spatial_corr(np.ones((3, 2)), np.ones(3))


def test_prepare_display_data_matches_viewer_centering():
    raw = _raw([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
    display, channel_dc = prepare_display_data(raw)
    np.testing.assert_allclose(channel_dc[:, 0], [2.0, 6.0])
    np.testing.assert_allclose(display.mean(axis=1), 0.0, atol=1e-15)


def test_template_round_trip_preserves_data_and_metadata(tmp_path):
    raw = _raw(
        [
            np.arange(30.0),
            np.arange(30.0) * -2,
            np.sin(np.arange(30.0)),
        ]
    )
    display, _ = prepare_display_data(raw)
    template = create_spatial_template(
        raw,
        display,
        tmin=1.0,
        tmax=2.0,
        source_dataset=Path("sample.ds"),
    )
    path = tmp_path / "door-ave.fif"
    write_spatial_template(template, path)
    loaded = load_spatial_template(path)
    np.testing.assert_allclose(loaded.data, template.data)
    assert loaded.ch_names == template.ch_names
    assert loaded.nave == 11
    assert "sample.ds" in loaded.info["description"]


def test_align_data_to_template_reorders_and_rejects_missing_channels():
    data = np.array([[1.0], [2.0], [3.0]])
    aligned = align_data_to_template(
        data,
        ["a", "b", "c"],
        ["c", "a"],
    )
    np.testing.assert_array_equal(aligned[:, 0], [3.0, 1.0])
    with pytest.raises(ValueError, match="missing 1 template channels"):
        align_data_to_template(data, ["a", "b", "c"], ["missing"])


def test_classifier_rejects_preprocessing_mismatch():
    raw = _raw(
        [
            np.arange(30.0),
            np.arange(30.0) * -2,
            np.sin(np.arange(30.0)),
        ]
    )
    display, _ = prepare_display_data(raw)
    template = create_spatial_template(raw, display, tmin=1.0, tmax=2.0)
    with raw.info._unlock():
        raw.info["highpass"] = 0.1
    with pytest.raises(ValueError, match="highpass values differ"):
        classify_spatial_template(raw, display, template)


def test_find_template_matches_uses_absolute_score_and_polarity_codes():
    correlation = np.zeros(30)
    correlation[[5, 7, 20]] = [-0.9, 0.85, 0.95]
    times = np.arange(30) / 10.0
    result = find_template_matches(
        correlation,
        times,
        10.0,
        threshold=0.8,
        min_distance=0.5,
        prominence=0.05,
    )
    # Samples 5 and 7 are too close; the stronger reversed match is retained.
    np.testing.assert_array_equal(result.peak_indices, [5, 20])
    np.testing.assert_allclose(result.onsets, [0.5, 2.0])
    np.testing.assert_array_equal(result.event_codes, [2, 1])


def test_annotations_and_outputs_preserve_codes(tmp_path):
    result = MatchResult(
        correlation=np.array([-0.9, 0.95]),
        peak_indices=np.array([0, 1]),
        onsets=np.array([1.25, 2.5]),
        event_codes=np.array([2, 1]),
    )
    annotations = matches_to_annotations(result)
    assert annotations.description.tolist() == [
        "door_artifact_negative",
        "door_artifact_positive",
    ]

    raw = _raw(np.zeros((3, 400)), sfreq=100.0)
    raw.set_annotations(annotations)
    expected_events, event_id = mne.events_from_annotations(raw, event_id=EVENT_ID)
    events_path = tmp_path / "door_evts.npy"
    annotations_path = tmp_path / "door_artifacts-annot.fif"
    write_match_outputs(
        expected_events,
        annotations,
        events_path,
        annotations_path,
    )
    np.testing.assert_array_equal(np.load(events_path), expected_events)
    loaded = mne.read_annotations(annotations_path)
    np.testing.assert_allclose(loaded.onset, annotations.onset)
    assert loaded.description.tolist() == annotations.description.tolist()
    assert event_id == EVENT_ID

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_match_outputs(
            expected_events,
            annotations,
            events_path,
            annotations_path,
        )


def test_output_paths_are_local_or_dataset_prefixed(tmp_path):
    dataset = tmp_path / "MEG_EmptyRoom_20260923_001.ds"
    assert output_paths(dataset, None) == (
        dataset / "door_evts.npy",
        dataset / "door_artifacts-annot.fif",
    )
    shared = tmp_path / "shared"
    assert output_paths(dataset, shared) == (
        shared / "MEG_EmptyRoom_20260923_001_door_evts.npy",
        shared / "MEG_EmptyRoom_20260923_001_door_artifacts-annot.fif",
    )
