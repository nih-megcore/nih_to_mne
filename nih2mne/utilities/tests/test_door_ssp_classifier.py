from pathlib import Path

import mne
import numpy as np
import pytest

from nih2mne.utilities.door_ssp_classifier import (
    DEFAULT_3ORD_PROJECTOR,
    DEFAULT_PROJECTOR,
    EventEpochRecord,
    ProcessingConfig,
    apply_projector,
    assign_burst_ids,
    build_projector_from_segments,
    discover_event_files,
    discover_emptyroom_datasets,
    epoch_plot_filename,
    filter_edge_guard_seconds,
    find_projected_matches,
    orient_projector,
    orient_projector_to_event_codes,
    parse_args,
    projected_matches_to_annotations,
    projected_pt_score,
    plot_grade3_epoch,
    read_selected_epoch_records,
    resolve_classification_projector,
    select_training_windows,
    strongest_epoch_indices,
    summarize_score,
    validate_event_array,
    write_event_epoch_csv,
)
from nih2mne.utilities.interactive_ctf_topomap import (
    ARTIFACT_CHANNEL_NAME,
    append_artifact_channel,
    artifact_projection_timeseries,
)


def _info(names=("a", "b", "c"), sfreq=100.0):
    return mne.create_info(list(names), sfreq, ch_types="mag")


def _projector(vector, names=("a", "b", "c")):
    vector = np.asarray(vector, dtype=float)[np.newaxis]
    return mne.Projection(
        data={
            "data": vector,
            "nrow": 1,
            "ncol": len(names),
            "row_names": None,
            "col_names": list(names),
        },
        desc="test-PCA-01",
        explained_var=0.9,
    )


def test_projected_pt_score_aligns_channels_and_preserves_sign():
    projector = _projector([1.0, 0.0, 0.0])
    data = np.array([[2e-12], [3e-12], [1e-12]])
    score = projected_pt_score(data, ["b", "c", "a"], projector)
    np.testing.assert_allclose(score, [1.0])


def test_projected_pt_score_omits_missing_channels_and_renormalizes():
    projector = _projector([3.0, 4.0, 12.0])
    data = np.array([[8e-12], [6e-12]])
    with pytest.warns(RuntimeWarning, match="missing 1 projector channels.*c"):
        score = projected_pt_score(data, ["b", "a"], projector)
    np.testing.assert_allclose(score, [10.0])


def test_projected_pt_score_rejects_no_usable_projector_weight():
    projector = _projector([1.0, 0.0], names=("a", "b"))
    with pytest.raises(ValueError, match="none of the projector channels"):
        projected_pt_score(np.ones((1, 2)), ["missing"], projector)
    with pytest.warns(RuntimeWarning, match="missing 1 projector channels"):
        with pytest.raises(ValueError, match="zero on the available"):
            projected_pt_score(np.ones((1, 2)), ["b"], projector)


def test_artifact_projection_can_be_appended_as_misc_channel():
    projector = _projector([1.0, 0.0, 0.0])
    data = np.array(
        [
            [2e-12, 4e-12],
            [3e-12, 6e-12],
            [1e-12, -2e-12],
        ]
    )
    score = artifact_projection_timeseries(
        data,
        ["b", "c", "a"],
        projector,
    )
    np.testing.assert_allclose(score, [1.0, -2.0])

    raw = mne.io.RawArray(
        data,
        _info(names=("b", "c", "a")),
        verbose="ERROR",
    )
    browser_raw, appended_score = append_artifact_channel(
        raw,
        data,
        projector,
    )
    assert browser_raw.ch_names[-1] == ARTIFACT_CHANNEL_NAME
    assert browser_raw.get_channel_types()[-1] == "misc"
    np.testing.assert_allclose(appended_score, score)
    np.testing.assert_allclose(browser_raw.get_data(picks="misc")[0], score)
    assert raw.ch_names == ["b", "c", "a"]


def test_orient_projector_uses_template_positive_direction():
    info = _info()
    template = mne.EvokedArray(
        np.array([[1.0], [0.0], [-1.0]]),
        info,
        tmin=0.0,
    )
    projector, correlation = orient_projector(
        _projector([-1.0, 0.0, 1.0]),
        template,
    )
    assert correlation == pytest.approx(1.0)
    np.testing.assert_allclose(projector["data"]["data"][0], [1.0, 0.0, -1.0])


def test_burst_assignment_and_strongest_epoch_selection():
    burst_ids = assign_burst_ids(
        np.array([0, 50, 200, 351, 400]),
        100.0,
        gap_s=1.5,
        first_id=7,
    )
    np.testing.assert_array_equal(burst_ids, [7, 7, 7, 8, 8])
    selected = strongest_epoch_indices(
        burst_ids,
        np.array([1.0, 5.0, 3.0, 2.0, 4.0]),
    )
    np.testing.assert_array_equal(selected, [1, 4])


def test_event_validation_and_discovery(tmp_path):
    dataset = tmp_path / "date" / "MEG_EmptyRoom_test.ds"
    dataset.mkdir(parents=True)
    path = dataset / "door_evts.npy"
    events = np.array([[100, 0, 1], [200, 0, 2]], dtype=np.int64)
    np.save(path, events)
    assert discover_event_files(tmp_path) == [path]
    np.testing.assert_array_equal(validate_event_array(events, path), events)
    with pytest.raises(ValueError, match="unknown event codes"):
        validate_event_array(np.array([[100, 0, 9]]), path)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_event_array(np.array([[100, 0, 1], [100, 0, 2]]), path)


def test_epoch_audit_selection_and_absolute_plot_filename(tmp_path):
    records = [
        EventEpochRecord(
            dataset="/data/MEG_EmptyRoom_test.ds",
            event_file="/data/MEG_EmptyRoom_test.ds/door_evts.npy",
            source_sample=100,
            resampled_sample=10,
            onset_s=78.7133333333,
            event_code=2,
            burst_id=4,
            epoch_rms_pt=0.2,
            selected=True,
        ),
        EventEpochRecord(
            dataset="/data/MEG_EmptyRoom_test.ds",
            event_file="/data/MEG_EmptyRoom_test.ds/door_evts.npy",
            source_sample=200,
            resampled_sample=20,
            onset_s=80.0,
            event_code=1,
            burst_id=4,
            epoch_rms_pt=0.1,
            selected=False,
        ),
    ]
    audit = tmp_path / "audit.csv"
    write_event_epoch_csv(records, audit, overwrite=False)
    selected = read_selected_epoch_records(audit)
    assert selected == records[:1]
    assert epoch_plot_filename(selected[0], tmin=-0.25, tmax=0.5) == (
        "MEG_EmptyRoom_test_78.463_79.213.png"
    )


def test_plot_grade3_epoch_writes_decodable_png(tmp_path):
    info = mne.create_info(["Fz", "C3", "C4"], 100.0, ch_types="eeg")
    info.set_montage("standard_1020")
    times = np.linspace(-0.25, 0.5, 76)
    epoch = np.vstack(
        [
            np.sin(2 * np.pi * 4 * times),
            np.cos(2 * np.pi * 4 * times),
            -np.sin(2 * np.pi * 4 * times),
        ]
    ) * 1e-12
    record = EventEpochRecord(
        dataset="/data/MEG_EmptyRoom_test.ds",
        event_file="/data/MEG_EmptyRoom_test.ds/door_evts.npy",
        source_sample=100,
        resampled_sample=10,
        onset_s=10.0,
        event_code=1,
        burst_id=2,
        epoch_rms_pt=0.7,
        selected=True,
    )
    output = tmp_path / "epoch.png"
    topo_scale, butterfly_scale = plot_grade3_epoch(
        epoch,
        info,
        times,
        record,
        output,
        tmin=-0.25,
        tmax=0.5,
        dpi=80,
    )
    assert topo_scale > 0
    assert butterfly_scale > 0
    assert output.stat().st_size > 0


def test_orient_projector_to_event_polarity():
    data = np.zeros((2, 3, 3))
    data[0, :, 1] = [1.0, 0.0, -1.0]
    data[1, :, 1] = [-1.0, 0.0, 1.0]
    projector, correlation = orient_projector_to_event_codes(
        _projector([-1.0, 0.0, 1.0]),
        data,
        np.array([1, 2]),
        zero_sample=1,
    )
    assert correlation == pytest.approx(1.0)
    np.testing.assert_allclose(projector["data"]["data"][0], [1.0, 0.0, -1.0])


def test_filter_edge_guard_matches_current_preprocessing():
    guard = filter_edge_guard_seconds(ProcessingConfig())
    assert guard == pytest.approx(16.5)


def test_summarize_score_excludes_filter_edges():
    times = np.arange(100, dtype=float)
    score = np.zeros(100)
    score[2] = 50.0
    score[40] = -20.0
    row = summarize_score(Path("sample.ds"), times, score, edge_guard_s=10.0)
    assert row["overall_max_time_s"] == 2.0
    assert row["overall_abs_max_pt"] == 50.0
    assert row["interior_max_time_s"] == 40.0
    assert row["interior_signed_max_pt"] == -20.0


def test_select_training_windows_excludes_seed_and_errors():
    rows = [
        {
            "dataset": "seed.ds",
            "status": "ok",
            "interior_abs_max_pt": 100.0,
            "interior_signed_max_pt": -100.0,
            "interior_max_time_s": 20.0,
        },
        {
            "dataset": "bad.ds",
            "status": "error",
            "interior_abs_max_pt": "",
            "interior_signed_max_pt": "",
            "interior_max_time_s": "",
        },
        {
            "dataset": "second.ds",
            "status": "ok",
            "interior_abs_max_pt": 20.0,
            "interior_signed_max_pt": -20.0,
            "interior_max_time_s": 30.0,
        },
        {
            "dataset": "first.ds",
            "status": "ok",
            "interior_abs_max_pt": 30.0,
            "interior_signed_max_pt": 30.0,
            "interior_max_time_s": 40.0,
        },
    ]
    windows = select_training_windows(
        rows,
        exclude_dataset=Path("seed.ds"),
        top_n=2,
        duration=6.0,
    )
    assert [window.dataset for window in windows] == ["first.ds", "second.ds"]
    assert [window.onset for window in windows] == [40.0, 30.0]


def test_find_projected_matches_applies_threshold_edges_and_codes():
    times = np.arange(50) / 10.0
    score = np.zeros(50)
    score[[2, 20, 35]] = [20.0, -30.0, 40.0]
    result = find_projected_matches(
        score,
        times,
        10.0,
        threshold_pt=15.0,
        min_distance=0.5,
        prominence_pt=1.0,
        edge_guard_s=0.5,
        min_duration=0.1,
    )
    np.testing.assert_array_equal(result.center_indices, [20, 35])
    np.testing.assert_array_equal(result.peak_indices, result.center_indices)
    np.testing.assert_array_equal(result.event_codes, [2, 1])
    annotations = projected_matches_to_annotations(result)
    np.testing.assert_allclose(annotations.onset, [2.0, 3.5])
    np.testing.assert_allclose(annotations.duration, [0.1, 0.1])
    assert annotations.description.tolist() == [
        "door_artifact_negative",
        "door_artifact_positive",
    ]


def test_projected_matches_merge_flicker_and_split_baseline_returns():
    sfreq = 100.0
    score = np.zeros(1200)
    score[100:180] = -20.0
    score[190:300] = -20.0
    score[350:550] = 30.0
    score[695:900] = -25.0
    result = find_projected_matches(
        score,
        np.arange(len(score)) / sfreq,
        sfreq,
        threshold_pt=10.0,
        min_distance=0.1,
        prominence_pt=1.0,
        edge_guard_s=0.0,
        baseline_z=3.0,
        baseline_dwell=0.25,
        min_duration=1.0,
    )
    np.testing.assert_array_equal(result.center_indices, [199, 449, 797])
    np.testing.assert_allclose(result.episode_onsets, [1.0, 3.5, 6.95])
    np.testing.assert_allclose(result.durations, [2.0, 2.0, 2.05])
    np.testing.assert_array_equal(result.event_codes, [2, 1, 2])


def test_projected_matches_keep_episodes_clipped_to_safe_edges():
    sfreq = 10.0
    score = np.zeros(100)
    score[10:20] = 20.0
    score[80:90] = -20.0
    result = find_projected_matches(
        score,
        np.arange(len(score)) / sfreq,
        sfreq,
        threshold_pt=10.0,
        min_distance=0.1,
        prominence_pt=1.0,
        edge_guard_s=1.0,
        min_duration=0.5,
    )
    np.testing.assert_allclose(result.episode_onsets, [1.0, 8.0])
    np.testing.assert_allclose(result.durations, [1.0, 1.0])


def test_projected_matches_allow_a_zero_width_quiet_baseline():
    score = np.zeros(100)
    result = find_projected_matches(
        score,
        np.arange(len(score)) / 10.0,
        10.0,
        threshold_pt=10.0,
        min_distance=0.5,
        prominence_pt=0.0,
        edge_guard_s=1.0,
    )
    assert len(result.onsets) == 0
    assert result.baseline_center_pt == 0.0
    assert result.baseline_scale_pt == 0.0


def test_projected_matches_require_three_second_episodes_by_default():
    sfreq = 100.0
    score = np.zeros(1500)
    score[100:300] = -20.0
    score[500:900] = 30.0
    result = find_projected_matches(
        score,
        np.arange(len(score)) / sfreq,
        sfreq,
        threshold_pt=10.0,
        min_distance=0.5,
        prominence_pt=0.0,
        edge_guard_s=0.0,
    )
    np.testing.assert_array_equal(result.center_indices, [699])
    np.testing.assert_allclose(result.durations, [4.0])


@pytest.mark.parametrize(
    ("baseline_z", "baseline_dwell"),
    [(0.0, 0.25), (3.0, 0.0)],
)
def test_projected_matches_reject_invalid_baseline_settings(
    baseline_z,
    baseline_dwell,
):
    with pytest.raises(ValueError, match="baseline"):
        find_projected_matches(
            np.zeros(100),
            np.arange(100) / 10.0,
            10.0,
            threshold_pt=10.0,
            min_distance=0.5,
            prominence_pt=0.0,
            edge_guard_s=1.0,
            baseline_z=baseline_z,
            baseline_dwell=baseline_dwell,
        )


def test_classify_cli_accepts_baseline_and_compensation_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "door_ssp_classifier.py",
            "classify",
            "--dataset",
            "sample.ds",
            "--projector",
            "door-proj.fif",
            "--threshold-pt",
            "10",
            "--baseline-z",
            "4",
            "--baseline-dwell",
            "0.5",
            "--compensation-grade",
            "3",
        ],
    )
    args = parse_args()
    assert args.baseline_z == 4.0
    assert args.baseline_dwell == 0.5
    assert args.min_duration == 3.0
    assert args.compensation_grade == 3


@pytest.mark.parametrize(
    ("grade", "expected"),
    [(0, DEFAULT_PROJECTOR), (3, DEFAULT_3ORD_PROJECTOR)],
)
def test_classification_projector_defaults_follow_compensation_grade(
    grade,
    expected,
):
    assert resolve_classification_projector(None, grade) == expected


def test_classification_projector_override_and_unsupported_grade():
    override = Path("custom-proj.fif")
    assert resolve_classification_projector(override, 2) == override
    with pytest.raises(ValueError, match="no default projector.*grade 2"):
        resolve_classification_projector(None, 2)


def test_build_and_apply_projector_removes_component():
    rng = np.random.default_rng(4)
    spatial = np.array([1.0, -2.0, 0.5])
    temporal = np.sin(np.linspace(0, 20, 1000))
    data = spatial[:, np.newaxis] * temporal + rng.normal(scale=0.01, size=(3, 1000))
    info = _info(sfreq=100.0)
    projector, shape = build_projector_from_segments(
        [data[:, :500], data[:, 500:]],
        info,
        description="door-test-PCA-01",
    )
    assert shape == (3, 1000)
    assert projector["explained_var"] > 0.99
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    cleaned = apply_projector(raw, projector)
    vector = projector["data"]["data"][0]
    assert np.linalg.norm(vector @ cleaned.get_data()) < 1e-10
    assert not raw.proj


def test_discover_emptyroom_datasets_uses_dataset_name(tmp_path):
    expected = tmp_path / "2026" / "MEG_EmptyRoom_20260101_001.ds"
    expected.mkdir(parents=True)
    (tmp_path / "2026" / "MEG_Noise_20260101_001.ds").mkdir()
    nested = expected / "hz.ds"
    nested.mkdir()
    assert discover_emptyroom_datasets(tmp_path) == [expected]
