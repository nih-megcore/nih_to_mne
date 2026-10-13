from pathlib import Path

import matplotlib.image as mpimg
import mne
import numpy as np
import pytest

from nih2mne.utilities.door_compensation_norm_plotter import (
    SelectedEvent,
    compensation_norms,
    parse_source_png,
    plot_norm_comparison,
    resolve_selected_events,
)


def test_parse_source_png():
    parsed = parse_source_png(
        Path("MEG_EmptyRoom_test_event-012_mid-45.802s.png")
    )
    assert parsed == ("MEG_EmptyRoom_test", 12, 45.802)
    with pytest.raises(ValueError, match="unrecognized"):
        parse_source_png(Path("unrelated.png"))


def test_resolve_selected_event_uses_midpoint_not_stale_number(tmp_path):
    dataset = tmp_path / "date" / "MEG_EmptyRoom_test.ds"
    dataset.mkdir(parents=True)
    png_dir = tmp_path / "pngs"
    png_dir.mkdir()
    source = png_dir / "MEG_EmptyRoom_test_event-001_mid-20.000s.png"
    source.touch()
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    annotation_path = annotation_dir / "MEG_EmptyRoom_test_door_artifacts-annot.fif"
    mne.Annotations(
        onset=[1.0, 18.0],
        duration=[4.0, 4.0],
        description=["first", "midpoint_match"],
    ).save(annotation_path)
    selected = resolve_selected_events(png_dir, tmp_path, annotation_dir)
    assert len(selected) == 1
    assert selected[0].event_number == 1
    assert selected[0].onset == 18.0
    assert selected[0].midpoint == 20.0
    assert selected[0].description == "midpoint_match"


def test_resolve_selected_event_rejects_ambiguous_midpoint(tmp_path):
    dataset = tmp_path / "date" / "MEG_EmptyRoom_test.ds"
    dataset.mkdir(parents=True)
    png_dir = tmp_path / "pngs"
    png_dir.mkdir()
    (png_dir / "MEG_EmptyRoom_test_event-001_mid-3.000s.png").touch()
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    annotation_path = annotation_dir / "MEG_EmptyRoom_test_door_artifacts-annot.fif"
    mne.Annotations(
        onset=[1.0, 2.0],
        duration=[4.0, 2.0],
        description=["a", "b"],
    ).save(annotation_path)
    with pytest.raises(ValueError, match="expected one annotation"):
        resolve_selected_events(png_dir, tmp_path, annotation_dir)


def test_compensation_norms_use_channel_l2_and_ratio():
    grade0 = np.array([[3e-12, 0.0], [4e-12, 10e-12]])
    grade3 = np.array([[0.0, 3e-12], [2e-12, 4e-12]])
    norm0, norm3, ratio = compensation_norms(grade0, grade3)
    np.testing.assert_allclose(norm0, [5.0, 10.0])
    np.testing.assert_allclose(norm3, [2.0, 5.0])
    np.testing.assert_allclose(ratio, [2.5, 2.0])
    with pytest.raises(ValueError, match="zero denominator"):
        compensation_norms(grade0, np.zeros_like(grade0))


def test_plot_norm_comparison_writes_decodable_three_panel_png(tmp_path):
    times = np.linspace(-2.0, 2.0, 401)
    norm0 = 20.0 + 10.0 * np.cos(times)
    norm3 = 2.0 + np.cos(times) ** 2
    ratio = norm0 / norm3
    event = SelectedEvent(
        source_png=Path("sample_event-001_mid-10.000s.png"),
        dataset=Path("sample.ds"),
        annotation_path=Path("sample-annot.fif"),
        event_number=1,
        onset=8.0,
        midpoint=10.0,
        duration=4.0,
        description="door_artifact_positive",
    )
    output = tmp_path / "comparison.png"
    plot_norm_comparison(times, norm0, norm3, ratio, event, output, dpi=80)
    image = mpimg.imread(output)
    assert image.ndim == 3
    assert image.size > 0
