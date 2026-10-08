#!/usr/bin/env python3
"""Calibrate and apply an SSP classifier for CTF door artifacts."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
from scipy.signal import find_peaks

if __package__:
    from .door_artifact_classifier import (
        EVENT_ID,
        NEGATIVE_DESCRIPTION,
        POSITIVE_DESCRIPTION,
        align_data_to_template,
        annotations_to_source_events,
        classify_spatial_template,
        load_spatial_template,
        output_paths,
        spatial_corr,
        write_match_outputs,
    )
    from .interactive_ctf_topomap import prepare_display_data, prepare_raw
else:
    from door_artifact_classifier import (
        EVENT_ID,
        NEGATIVE_DESCRIPTION,
        POSITIVE_DESCRIPTION,
        align_data_to_template,
        annotations_to_source_events,
        classify_spatial_template,
        load_spatial_template,
        output_paths,
        spatial_corr,
        write_match_outputs,
    )
    from interactive_ctf_topomap import prepare_display_data, prepare_raw


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMPTYROOM_ROOT = Path("/fast2/EMPTY")
DEFAULT_SEED_DATASET = Path(
    "/fast2/EMPTY/20260923/MEG_EmptyRoom_20260923_001.ds"
)
DEFAULT_TEMPLATE = PACKAGE_ROOT / "templates" / "door_artifact_template.fif"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "templates"
DEFAULT_WINDOW_DURATION = 6.0
DEFAULT_TOP_N = 5
DEFAULT_3ORD_PROJECTOR = (
    DEFAULT_OUTPUT_DIR / "door_artifact_final-proj_3ord.fif"
)
DEFAULT_3ORD_TMIN = -0.25
DEFAULT_3ORD_TMAX = 0.50
DEFAULT_BURST_GAP = 1.5
DEFAULT_3ORD_METADATA = (
    DEFAULT_OUTPUT_DIR / "door_artifact_final-proj_3ord.json"
)
DEFAULT_3ORD_EPOCH_AUDIT = (
    DEFAULT_OUTPUT_DIR / "door_artifact_final-proj_3ord_epochs.csv"
)
DEFAULT_3ORD_PLOT_DIR = Path("/tmp/EMP_TOPOS")


@dataclass(frozen=True)
class ProcessingConfig:
    """Preprocessing shared by SSP creation, calibration, and classification."""

    resample_sfreq: float = 300.0
    highpass: float = 0.1
    lowpass: float = 40.0
    compensation_grade: int = 0
    remove_dc: bool = True
    n_jobs: int = -1


@dataclass(frozen=True)
class TrainingWindow:
    """One dataset interval contributing to an SSP covariance."""

    dataset: str
    onset: float
    duration: float
    seed_score_pt: float


@dataclass(frozen=True)
class ProjectedMatchResult:
    """Continuous projected-pT values and thresholded local maxima."""

    score_pt: np.ndarray
    peak_indices: np.ndarray
    onsets: np.ndarray
    event_codes: np.ndarray


@dataclass
class EventEpochRecord:
    """Audit information for one event-centered third-order epoch."""

    dataset: str
    event_file: str
    source_sample: int
    resampled_sample: int
    onset_s: float
    event_code: int
    burst_id: int
    epoch_rms_pt: float
    selected: bool = False


SCAN_FIELDS = (
    "dataset",
    "status",
    "error_type",
    "error",
    "duration_s",
    "overall_max_time_s",
    "overall_signed_max_pt",
    "overall_abs_max_pt",
    "interior_max_time_s",
    "interior_signed_max_pt",
    "interior_abs_max_pt",
    "abs_p50_pt",
    "abs_p90_pt",
    "abs_p95_pt",
    "abs_p99_pt",
    "abs_p99_9_pt",
)

EVENT_EPOCH_FIELDS = tuple(EventEpochRecord.__dataclass_fields__)


def preprocess_dataset(
    dataset: Path,
    config: ProcessingConfig,
) -> tuple[mne.io.BaseRaw, np.ndarray]:
    """Apply the calibration preprocessing and whole-run channel DC removal."""
    raw = prepare_raw(
        Path(dataset),
        resample_sfreq=config.resample_sfreq,
        highpass=config.highpass,
        lowpass=config.lowpass,
        compensation_grade=config.compensation_grade,
        n_jobs=config.n_jobs,
    )
    data_t, _ = prepare_display_data(raw, remove_dc=config.remove_dc)
    return raw, data_t


def filter_edge_guard_seconds(config: ProcessingConfig) -> float:
    """Return half the zero-phase FIR length in seconds."""
    taps = mne.filter.create_filter(
        None,
        config.resample_sfreq,
        config.highpass,
        config.lowpass,
        verbose="ERROR",
    )
    return (len(taps) - 1) / (2.0 * config.resample_sfreq)


def discover_emptyroom_datasets(root: Path) -> list[Path]:
    """Find CTF directories whose own names identify EmptyRoom recordings."""
    return sorted(
        path
        for path in Path(root).rglob("MEG_EmptyRoom*.ds")
        if path.is_dir()
    )


def discover_event_files(
    root: Path,
    event_filename: str = "door_evts.npy",
) -> list[Path]:
    """Find dataset-local door event arrays without matching nested files."""
    return sorted(
        path
        for path in Path(root).rglob(event_filename)
        if path.is_file() and path.parent.name.endswith(".ds")
    )


def validate_event_array(events: np.ndarray, path: Path) -> np.ndarray:
    """Validate and return an integer MNE event array."""
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(f"event array must have shape (n_events, 3): {path}")
    if not np.issubdtype(events.dtype, np.integer):
        raise ValueError(f"event array must use an integer dtype: {path}")
    events = events.astype(np.int64, copy=False)
    if len(events) == 0:
        return events
    if np.any(np.diff(events[:, 0]) <= 0):
        raise ValueError(f"event samples must be strictly increasing: {path}")
    unknown = sorted(set(events[:, 2]) - set(EVENT_ID.values()))
    if unknown:
        raise ValueError(f"unknown event codes {unknown} in {path}")
    return events


def read_selected_epoch_records(path: Path) -> list[EventEpochRecord]:
    """Read and validate selected rows from a third-order epoch audit."""
    records = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(EVENT_EPOCH_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"epoch audit is missing columns: {sorted(missing)}")
        for row in reader:
            selected_text = row["selected"].strip().lower()
            if selected_text not in {"true", "false"}:
                raise ValueError(
                    f"invalid selected value {row['selected']!r} in {path}"
                )
            if selected_text == "false":
                continue
            records.append(
                EventEpochRecord(
                    dataset=row["dataset"],
                    event_file=row["event_file"],
                    source_sample=int(row["source_sample"]),
                    resampled_sample=int(row["resampled_sample"]),
                    onset_s=float(row["onset_s"]),
                    event_code=int(row["event_code"]),
                    burst_id=int(row["burst_id"]),
                    epoch_rms_pt=float(row["epoch_rms_pt"]),
                    selected=True,
                )
            )
    if not records:
        raise ValueError(f"epoch audit contains no selected rows: {path}")
    return records


def epoch_plot_filename(
    record: EventEpochRecord,
    *,
    tmin: float,
    tmax: float,
) -> str:
    """Build a collision-readable absolute-time PNG filename."""
    dataset_name = Path(record.dataset).stem
    absolute_tmin = record.onset_s + tmin
    absolute_tmax = record.onset_s + tmax
    return f"{dataset_name}_{absolute_tmin:.3f}_{absolute_tmax:.3f}.png"


def assign_burst_ids(
    event_samples: np.ndarray,
    sfreq: float,
    *,
    gap_s: float,
    first_id: int = 0,
) -> np.ndarray:
    """Group ordered samples when consecutive detections are near in time."""
    event_samples = np.asarray(event_samples, dtype=np.int64)
    if event_samples.ndim != 1:
        raise ValueError("event_samples must be one-dimensional")
    if sfreq <= 0 or gap_s <= 0:
        raise ValueError("sfreq and burst gap must be positive")
    if len(event_samples) == 0:
        return np.empty(0, dtype=int)
    if np.any(np.diff(event_samples) <= 0):
        raise ValueError("event samples must be strictly increasing")
    starts = np.r_[True, np.diff(event_samples) / sfreq > gap_s]
    return np.cumsum(starts, dtype=int) - 1 + int(first_id)


def strongest_epoch_indices(
    burst_ids: np.ndarray,
    epoch_rms: np.ndarray,
) -> np.ndarray:
    """Select the highest-RMS epoch in every burst, breaking ties by order."""
    burst_ids = np.asarray(burst_ids, dtype=int)
    epoch_rms = np.asarray(epoch_rms, dtype=float)
    if burst_ids.ndim != 1 or epoch_rms.shape != burst_ids.shape:
        raise ValueError("burst IDs and epoch RMS must be equal-length vectors")
    if not np.isfinite(epoch_rms).all():
        raise ValueError("epoch RMS values must be finite")
    selected = []
    for burst_id in np.unique(burst_ids):
        candidates = np.flatnonzero(burst_ids == burst_id)
        selected.append(int(candidates[np.argmax(epoch_rms[candidates])]))
    return np.asarray(selected, dtype=int)


def load_grade3_event_epochs(
    dataset: Path,
    events: np.ndarray,
    config: ProcessingConfig,
    *,
    tmin: float,
    tmax: float,
) -> tuple[mne.Epochs, np.ndarray, float, int]:
    """Load every requested event epoch with CTF compensation grade 3."""
    if not tmin < 0 < tmax:
        raise ValueError("third-order epochs must span time zero")
    raw = mne.io.read_raw_ctf(
        dataset,
        system_clock="ignore",
        preload=False,
    )
    source_sfreq = float(raw.info["sfreq"])
    source_first_samp = int(raw.first_samp)
    if np.any(events[:, 0] < raw.first_samp) or np.any(
        events[:, 0] > raw.last_samp
    ):
        raise ValueError(f"event sample lies outside recording: {dataset}")

    raw.load_data()
    raw.apply_gradient_compensation(3)
    if raw.compensation_grade != 3:
        raise RuntimeError(
            f"expected compensation grade 3, got {raw.compensation_grade}"
        )
    raw, resampled_events = raw.resample(
        config.resample_sfreq,
        events=events,
        n_jobs=config.n_jobs,
    )
    raw.filter(config.highpass, config.lowpass, n_jobs=config.n_jobs)
    picks = mne.pick_types(
        raw.info,
        meg=True,
        ref_meg=False,
        exclude="bads",
    )
    raw.pick(picks)
    if raw.compensation_grade != 3:
        raise RuntimeError("compensation grade changed while selecting MEG channels")
    if set(raw.get_channel_types()) != {"mag"}:
        raise RuntimeError("third-order epoch input contains non-magnetometers")
    if config.remove_dc:
        raw._data -= raw._data.mean(axis=1, keepdims=True)

    epochs = mne.Epochs(
        raw,
        resampled_events,
        event_id=EVENT_ID,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        proj=False,
        reject_by_annotation=False,
        event_repeated="error",
        on_missing="ignore",
        verbose="ERROR",
    )
    if len(epochs) != len(events):
        dropped = [index for index, reason in enumerate(epochs.drop_log) if reason]
        raise RuntimeError(
            f"loaded {len(epochs)}/{len(events)} epochs from {dataset}; "
            f"dropped indices: {dropped[:10]}"
        )
    if raw.compensation_grade != 3:
        raise RuntimeError("epochs were not loaded from compensation-grade-3 data")
    return epochs, resampled_events, source_sfreq, source_first_samp


def projector_vector(projector: mne.Projection) -> tuple[np.ndarray, list[str]]:
    """Extract and validate one SSP vector and its ordered channel names."""
    data = np.asarray(projector["data"]["data"], dtype=float)
    names = list(projector["data"]["col_names"])
    if data.shape != (1, len(names)):
        raise ValueError(
            "expected one SSP vector with one value per channel, got "
            f"{data.shape} for {len(names)} channels"
        )
    norm = np.linalg.norm(data[0])
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("SSP vector has invalid or zero norm")
    return data[0] / norm, names


def orient_projector(
    projector: mne.Projection,
    template: mne.Evoked,
) -> tuple[mne.Projection, float]:
    """Orient an SSP vector so positive scores match the door template."""
    projector = deepcopy(projector)
    vector, names = projector_vector(projector)
    template_lookup = {name: index for index, name in enumerate(template.ch_names)}
    missing = [name for name in names if name not in template_lookup]
    if missing:
        raise ValueError(
            f"template is missing {len(missing)} projector channels: {missing[:5]}"
        )
    template_data = template.data[
        [template_lookup[name] for name in names],
        0,
    ]
    correlation = float(spatial_corr(vector[:, np.newaxis], template_data)[0])
    if correlation < 0:
        projector["data"]["data"][0] *= -1.0
        correlation *= -1.0
    return projector, correlation


def projected_pt_score(
    data_t: np.ndarray,
    data_ch_names: list[str],
    projector: mne.Projection,
) -> np.ndarray:
    """Return the signed amplitude along one SSP vector in pT."""
    vector, projector_names = projector_vector(projector)
    aligned = align_data_to_template(data_t, data_ch_names, projector_names)
    return (vector @ aligned) * 1e12


def build_projector_from_segments(
    segments_t: Iterable[np.ndarray],
    info: mne.Info,
    *,
    description: str,
) -> tuple[mne.Projection, tuple[int, int]]:
    """Compute one axial-magnetometer SSP from concatenated data segments."""
    segments_t = [np.asarray(segment, dtype=float) for segment in segments_t]
    if not segments_t:
        raise ValueError("at least one segment is required")
    expected_channels = len(info["ch_names"])
    if any(
        segment.ndim != 2 or segment.shape[0] != expected_channels
        for segment in segments_t
    ):
        raise ValueError("all segments must have one row per info channel")
    pooled_t = np.concatenate(segments_t, axis=1)
    pooled_raw = mne.io.RawArray(pooled_t, info.copy(), verbose="ERROR")
    projectors = mne.compute_proj_raw(
        pooled_raw,
        start=0.0,
        stop=None,
        duration=None,
        n_grad=0,
        n_mag=1,
        n_eeg=0,
        meg="separate",
        verbose="ERROR",
    )
    if len(projectors) != 1:
        raise RuntimeError(f"expected one magnetometer SSP, got {len(projectors)}")
    projectors[0]["desc"] = description
    return projectors[0], pooled_t.shape


def extract_window(
    raw: mne.io.BaseRaw,
    data_t: np.ndarray,
    onset: float,
    duration: float,
) -> np.ndarray:
    """Extract an exact-duration processed interval using rounded samples."""
    recording_end = raw.times[-1] + 1 / raw.info["sfreq"]
    if onset < 0 or duration <= 0 or onset + duration > recording_end:
        raise ValueError(
            f"window {onset:g}-{onset + duration:g} s is outside the recording"
        )
    start = int(raw.time_as_index(onset, use_rounding=True)[0])
    length = int(round(duration * raw.info["sfreq"]))
    stop = start + length
    if stop > raw.n_times:
        raise ValueError("rounded SSP window extends beyond the recording")
    return data_t[:, start:stop].copy()


def summarize_score(
    dataset: Path,
    times: np.ndarray,
    score_pt: np.ndarray,
    edge_guard_s: float,
) -> dict[str, object]:
    """Summarize overall and filter-edge-excluded projected amplitudes."""
    score_pt = np.asarray(score_pt, dtype=float)
    times = np.asarray(times, dtype=float)
    if score_pt.shape != times.shape or score_pt.ndim != 1:
        raise ValueError("score and times must be equal-length one-dimensional arrays")
    absolute = np.abs(score_pt)
    overall_index = int(np.argmax(absolute))
    interior = (times >= edge_guard_s) & (times <= times[-1] - edge_guard_s)
    if not interior.any():
        raise ValueError("recording is too short for the configured FIR edge guard")
    interior_indices = np.flatnonzero(interior)
    interior_index = int(
        interior_indices[np.argmax(absolute[interior_indices])]
    )
    percentiles = np.percentile(absolute, [50, 90, 95, 99, 99.9])
    return {
        "dataset": str(dataset),
        "status": "ok",
        "error_type": "",
        "error": "",
        "duration_s": float(times[-1] + (times[1] - times[0])),
        "overall_max_time_s": float(times[overall_index]),
        "overall_signed_max_pt": float(score_pt[overall_index]),
        "overall_abs_max_pt": float(absolute[overall_index]),
        "interior_max_time_s": float(times[interior_index]),
        "interior_signed_max_pt": float(score_pt[interior_index]),
        "interior_abs_max_pt": float(absolute[interior_index]),
        "abs_p50_pt": float(percentiles[0]),
        "abs_p90_pt": float(percentiles[1]),
        "abs_p95_pt": float(percentiles[2]),
        "abs_p99_pt": float(percentiles[3]),
        "abs_p99_9_pt": float(percentiles[4]),
    }


def error_scan_row(dataset: Path, error: Exception) -> dict[str, object]:
    """Represent a failed dataset without dropping it from calibration."""
    row = {field: "" for field in SCAN_FIELDS}
    row.update(
        dataset=str(dataset),
        status="error",
        error_type=type(error).__name__,
        error=str(error),
    )
    return row


def scan_projector(
    datasets: Iterable[Path],
    projector: mne.Projection,
    config: ProcessingConfig,
    edge_guard_s: float,
) -> list[dict[str, object]]:
    """Score each dataset independently and continue past readable failures."""
    datasets = list(datasets)
    rows: list[dict[str, object]] = []
    for number, dataset in enumerate(datasets, start=1):
        try:
            raw, data_t = preprocess_dataset(dataset, config)
            score_pt = projected_pt_score(data_t, raw.ch_names, projector)
            row = summarize_score(dataset, raw.times, score_pt, edge_guard_s)
            print(
                f"[{number:02d}/{len(datasets)}] "
                f"interior={row['interior_abs_max_pt']:9.3f} pT "
                f"at {row['interior_max_time_s']:8.3f} s  {dataset}",
                flush=True,
            )
            del raw, data_t, score_pt
            gc.collect()
        except Exception as error:
            row = error_scan_row(dataset, error)
            print(
                f"[{number:02d}/{len(datasets)}] ERROR "
                f"{type(error).__name__}: {error}  {dataset}",
                flush=True,
            )
        rows.append(row)
    return rows


def select_training_windows(
    rows: Iterable[dict[str, object]],
    *,
    exclude_dataset: Path,
    top_n: int,
    duration: float,
) -> list[TrainingWindow]:
    """Select the strongest successful independent interior maxima."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    exclude = str(Path(exclude_dataset))
    candidates = [
        row
        for row in rows
        if row["status"] == "ok" and row["dataset"] != exclude
    ]
    candidates.sort(
        key=lambda row: (-float(row["interior_abs_max_pt"]), str(row["dataset"]))
    )
    if len(candidates) < top_n:
        raise RuntimeError(
            f"only {len(candidates)} independent successful datasets for top {top_n}"
        )
    return [
        TrainingWindow(
            dataset=str(row["dataset"]),
            onset=float(row["interior_max_time_s"]),
            duration=float(duration),
            seed_score_pt=float(row["interior_signed_max_pt"]),
        )
        for row in candidates[:top_n]
    ]


def build_pooled_projector(
    windows: Iterable[TrainingWindow],
    config: ProcessingConfig,
    template: mne.Evoked,
) -> tuple[mne.Projection, dict[str, object]]:
    """Process, concatenate, and project the selected training intervals."""
    windows = list(windows)
    segments = []
    reference_names: list[str] | None = None
    reference_info: mne.Info | None = None
    for number, window in enumerate(windows, start=1):
        raw, data_t = preprocess_dataset(Path(window.dataset), config)
        if reference_names is None:
            reference_names = list(raw.ch_names)
            reference_info = raw.info.copy()
        aligned = align_data_to_template(data_t, raw.ch_names, reference_names)
        segment = extract_window(raw, aligned, window.onset, window.duration)
        segments.append(segment)
        print(
            f"[pool {number}/{len(windows)}] {window.dataset} "
            f"{window.onset:.3f}-{window.onset + window.duration:.3f} s",
            flush=True,
        )
        del raw, data_t, aligned
        gc.collect()
    assert reference_info is not None
    projector, pooled_shape = build_projector_from_segments(
        segments,
        reference_info,
        description="door-artifact-final-PCA-01",
    )
    projector, template_correlation = orient_projector(projector, template)
    metadata = {
        "pooled_shape": list(pooled_shape),
        "explained_variance": float(projector["explained_var"]),
        "template_spatial_correlation": float(template_correlation),
    }
    return projector, metadata


def scan_statistics(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Summarize per-dataset interior maxima and reported failures."""
    rows = list(rows)
    successful = [row for row in rows if row["status"] == "ok"]
    failures = [row for row in rows if row["status"] != "ok"]
    values = np.array(
        [float(row["interior_abs_max_pt"]) for row in successful],
        dtype=float,
    )
    return {
        "successful_datasets": len(successful),
        "failed_datasets": len(failures),
        "interior_max_pt": {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        },
        "failures": [
            {
                "dataset": row["dataset"],
                "error_type": row["error_type"],
                "error": row["error"],
            }
            for row in failures
        ],
    }


def write_scan_csv(
    rows: Iterable[dict[str, object]],
    path: Path,
    *,
    overwrite: bool,
) -> None:
    """Write deterministic dataset-level calibration results."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"scan report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, object], path: Path, *, overwrite: bool) -> None:
    """Write stable, human-readable calibration provenance."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"calibration metadata already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_projector(
    projector: mne.Projection,
    path: Path,
    *,
    overwrite: bool,
) -> None:
    """Write one SSP projector with explicit overwrite behavior."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"projector already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions",
        )
        mne.write_proj(path, [projector], overwrite=overwrite)


def write_event_epoch_csv(
    records: Iterable[EventEpochRecord],
    path: Path,
    *,
    overwrite: bool,
) -> None:
    """Write the complete third-order event-selection audit trail."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"epoch audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_EPOCH_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _align_epoch_channels(
    data: np.ndarray,
    ch_names: list[str],
    reference_names: list[str],
) -> np.ndarray:
    """Reorder three-dimensional epoch data to a reference channel list."""
    if data.ndim != 3 or data.shape[1] != len(ch_names):
        raise ValueError("epoch data shape does not match channel names")
    lookup = {name: index for index, name in enumerate(ch_names)}
    missing = [name for name in reference_names if name not in lookup]
    extra = [name for name in ch_names if name not in set(reference_names)]
    if missing or extra:
        raise ValueError(
            f"incompatible epoch channels; missing={missing[:5]}, extra={extra[:5]}"
        )
    return data[:, [lookup[name] for name in reference_names], :]


def orient_projector_to_event_codes(
    projector: mne.Projection,
    epoch_data: np.ndarray,
    event_codes: np.ndarray,
    zero_sample: int,
) -> tuple[mne.Projection, float]:
    """Orient a grade-3 SSP so code 1 has the positive reference direction."""
    projector = deepcopy(projector)
    vector, _ = projector_vector(projector)
    event_codes = np.asarray(event_codes, dtype=int)
    if epoch_data.ndim != 3 or epoch_data.shape[0] != len(event_codes):
        raise ValueError("epoch data and event codes do not match")
    if not 0 <= zero_sample < epoch_data.shape[2]:
        raise ValueError("zero sample is outside the epochs")
    unknown = sorted(set(event_codes) - set(EVENT_ID.values()))
    if unknown:
        raise ValueError(f"unknown event codes: {unknown}")
    signs = np.where(
        event_codes == EVENT_ID[POSITIVE_DESCRIPTION],
        1.0,
        -1.0,
    )
    reference = np.mean(
        epoch_data[:, :, zero_sample] * signs[:, np.newaxis],
        axis=0,
    )
    correlation = float(spatial_corr(vector[:, np.newaxis], reference)[0])
    if correlation < 0:
        projector["data"]["data"][0] *= -1.0
        correlation *= -1.0
    return projector, correlation


def run_build_3ord(args: argparse.Namespace) -> None:
    """Build one door-artifact SSP from event epochs in grade-3 data."""
    config = ProcessingConfig(
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=3,
        remove_dc=True,
        n_jobs=args.n_jobs,
    )
    output = args.output
    output_stem = output.name[:-4] if output.name.endswith(".fif") else output.name
    metadata_path = output.with_name(f"{output_stem}.json")
    audit_path = output.with_name(f"{output_stem}_epochs.csv")
    _preflight_outputs(
        (output, metadata_path, audit_path),
        args.overwrite,
    )

    event_files = discover_event_files(args.root, args.event_filename)
    if not event_files:
        raise FileNotFoundError(
            f"no {args.event_filename} files found below {args.root}"
        )

    records: list[EventEpochRecord] = []
    selected_epochs: list[np.ndarray] = []
    selected_codes: list[int] = []
    reference_names: list[str] | None = None
    reference_info: mne.Info | None = None
    next_burst_id = 0
    empty_files = 0
    nonempty_files = 0

    for number, event_file in enumerate(event_files, start=1):
        events = validate_event_array(
            np.load(event_file, allow_pickle=False),
            event_file,
        )
        if len(events) == 0:
            empty_files += 1
            print(
                f"[{number:02d}/{len(event_files)}] 0 events  {event_file.parent}",
                flush=True,
            )
            continue
        nonempty_files += 1
        dataset = event_file.parent
        epochs, resampled_events, source_sfreq, source_first_samp = (
            load_grade3_event_epochs(
                dataset,
                events,
                config,
                tmin=args.tmin,
                tmax=args.tmax,
            )
        )
        epoch_data = epochs.get_data(copy=True)
        if reference_names is None:
            reference_names = list(epochs.ch_names)
            reference_info = epochs.info.copy()
        else:
            epoch_data = _align_epoch_channels(
                epoch_data,
                epochs.ch_names,
                reference_names,
            )

        epoch_rms_pt = np.sqrt(np.mean(epoch_data**2, axis=(1, 2))) * 1e12
        burst_ids = assign_burst_ids(
            events[:, 0],
            source_sfreq,
            gap_s=args.burst_gap,
            first_id=next_burst_id,
        )
        selected_local = strongest_epoch_indices(burst_ids, epoch_rms_pt)
        selected_set = set(selected_local.tolist())
        for index, event in enumerate(events):
            selected = index in selected_set
            records.append(
                EventEpochRecord(
                    dataset=str(dataset),
                    event_file=str(event_file),
                    source_sample=int(event[0]),
                    resampled_sample=int(resampled_events[index, 0]),
                    onset_s=float(
                        (event[0] - source_first_samp) / source_sfreq
                    ),
                    event_code=int(event[2]),
                    burst_id=int(burst_ids[index]),
                    epoch_rms_pt=float(epoch_rms_pt[index]),
                    selected=selected,
                )
            )
            if selected:
                selected_epochs.append(epoch_data[index].copy())
                selected_codes.append(int(event[2]))
        next_burst_id = int(burst_ids[-1]) + 1
        print(
            f"[{number:02d}/{len(event_files)}] {len(events):3d} epochs -> "
            f"{len(selected_local):2d} bursts  {dataset}",
            flush=True,
        )
        del epochs, epoch_data
        gc.collect()

    if reference_info is None or reference_names is None:
        raise RuntimeError("all discovered event files were empty")
    pooled_data = np.stack(selected_epochs)
    pooled_codes = np.asarray(selected_codes, dtype=int)
    synthetic_events = np.column_stack(
        [
            np.arange(len(pooled_data), dtype=int),
            np.zeros(len(pooled_data), dtype=int),
            pooled_codes,
        ]
    )
    pooled_epochs = mne.EpochsArray(
        pooled_data,
        reference_info,
        events=synthetic_events,
        event_id=EVENT_ID,
        tmin=args.tmin,
        baseline=None,
        proj=False,
        verbose="ERROR",
    )
    projectors = mne.compute_proj_epochs(
        pooled_epochs,
        n_grad=0,
        n_mag=1,
        n_eeg=0,
        n_jobs=config.n_jobs,
        desc_prefix="door-artifact-final-3ord",
        meg="separate",
        verbose="ERROR",
    )
    if len(projectors) != 1:
        raise RuntimeError(f"expected one third-order SSP, got {len(projectors)}")
    projector = projectors[0]
    projector["desc"] = "door-artifact-final-3ord-PCA-01"
    zero_sample = int(np.argmin(np.abs(pooled_epochs.times)))
    projector, orientation_correlation = orient_projector_to_event_codes(
        projector,
        pooled_data,
        pooled_codes,
        zero_sample,
    )

    vector, _ = projector_vector(projector)
    component_before = np.einsum("c,ect->et", vector, pooled_data)
    cleaned_epochs = pooled_epochs.copy()
    cleaned_epochs.add_proj([projector])
    cleaned_epochs.apply_proj()
    cleaned_data = cleaned_epochs.get_data(copy=False)
    component_after = np.einsum("c,ect->et", vector, cleaned_data)
    total_variance_reduction = 1.0 - float(
        np.sum(cleaned_data**2) / np.sum(pooled_data**2)
    )
    post_projection_max_pt = float(np.max(np.abs(component_after)) * 1e12)
    if post_projection_max_pt > 1e-6:
        raise RuntimeError(
            "SSP validation failed; residual component is "
            f"{post_projection_max_pt:g} pT"
        )

    metadata = {
        "kind": "door_artifact_third_order_ssp",
        "root": str(args.root),
        "event_filename": args.event_filename,
        "processing": asdict(config),
        "epoch": {
            "tmin_s": args.tmin,
            "tmax_s": args.tmax,
            "baseline": None,
            "reject_by_annotation": False,
        },
        "selection": {
            "burst_gap_s": args.burst_gap,
            "criterion": "maximum full-epoch grade-3 RMS",
            "candidate_epochs": len(records),
            "selected_epochs": len(selected_epochs),
        },
        "inputs": {
            "event_files": len(event_files),
            "nonempty_event_files": nonempty_files,
            "empty_event_files": empty_files,
        },
        "projector": {
            "description": projector["desc"],
            "components": 1,
            "channels": len(reference_names),
            "explained_variance": float(projector["explained_var"]),
            "polarity_reference_correlation": orientation_correlation,
            "selected_component_rms_pt": float(
                np.sqrt(np.mean(component_before**2)) * 1e12
            ),
            "total_variance_reduction": total_variance_reduction,
            "post_projection_max_component_pt": post_projection_max_pt,
        },
        "outputs": {
            "projector": str(output),
            "metadata": str(metadata_path),
            "epoch_audit": str(audit_path),
        },
    }
    write_projector(projector, output, overwrite=args.overwrite)
    write_event_epoch_csv(records, audit_path, overwrite=args.overwrite)
    write_json(metadata, metadata_path, overwrite=args.overwrite)
    print(
        f"Wrote grade-3 SSP from {len(records)} epochs / "
        f"{len(selected_epochs)} bursts: {output}",
        flush=True,
    )


def plot_grade3_epoch(
    epoch_t: np.ndarray,
    info: mne.Info,
    times: np.ndarray,
    record: EventEpochRecord,
    output: Path,
    *,
    tmin: float,
    tmax: float,
    dpi: int,
) -> tuple[float, float]:
    """Save one epoch-average topomap and epoch butterfly figure."""
    import matplotlib.pyplot as plt

    epoch_pt = np.asarray(epoch_t, dtype=float) * 1e12
    times = np.asarray(times, dtype=float)
    if epoch_pt.shape != (len(info["ch_names"]), len(times)):
        raise ValueError("epoch data shape does not match info and times")
    average_topography_pt = np.mean(epoch_pt, axis=1)
    topo_vlim_pt = float(
        np.nanpercentile(np.abs(average_topography_pt), 99.5)
    )
    butterfly_vlim_pt = float(np.nanpercentile(np.abs(epoch_pt), 99.5))
    if (
        not np.isfinite(topo_vlim_pt)
        or topo_vlim_pt <= 0
        or not np.isfinite(butterfly_vlim_pt)
        or butterfly_vlim_pt <= 0
    ):
        raise ValueError(f"invalid plot scale for burst {record.burst_id}")
    absolute_tmin = record.onset_s + tmin
    absolute_tmax = record.onset_s + tmax
    dataset_name = Path(record.dataset).stem

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        layout="constrained",
    )
    image, _ = mne.viz.plot_topomap(
        average_topography_pt,
        info,
        axes=axes[0],
        show=False,
        sensors=True,
        contours=0,
        extrapolate="local",
        cmap="RdBu_r",
        vlim=(-topo_vlim_pt, topo_vlim_pt),
    )
    colorbar = fig.colorbar(image, ax=axes[0], shrink=0.82)
    colorbar.set_label("pT")
    axes[0].set_title(
        f"Epoch-average topomap ({times[0]:.3f} to {times[-1]:.3f} s)"
    )

    axes[1].plot(times, epoch_pt.T, color="black", alpha=0.16, linewidth=0.45)
    axes[1].axvline(0.0, color="crimson", linewidth=1.2, label="event")
    axes[1].set_xlim(float(times[0]), float(times[-1]))
    axes[1].set_ylim(-butterfly_vlim_pt, butterfly_vlim_pt)
    axes[1].set_xlabel("Time relative to event (s)")
    axes[1].set_ylabel("Amplitude (pT)")
    axes[1].set_title("Grade-3 epoch butterfly")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"{dataset_name} | absolute {absolute_tmin:.3f}–{absolute_tmax:.3f} s\n"
        f"code={record.event_code}, burst={record.burst_id}, "
        f"epoch RMS={record.epoch_rms_pt:.4g} pT, "
        f"topo scale=±{topo_vlim_pt:.4g} pT, "
        f"butterfly scale=±{butterfly_vlim_pt:.4g} pT"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return topo_vlim_pt, butterfly_vlim_pt


def run_plot_3ord(args: argparse.Namespace) -> None:
    """Recreate and plot every selected third-order SSP training epoch."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    with args.metadata.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    processing = metadata.get("processing", {})
    epoch_metadata = metadata.get("epoch", {})
    if processing.get("compensation_grade") != 3:
        raise ValueError("plot metadata does not describe compensation grade 3")
    if not processing.get("remove_dc"):
        raise ValueError("plot metadata must specify whole-run DC removal")
    tmin = float(epoch_metadata["tmin_s"])
    tmax = float(epoch_metadata["tmax_s"])
    config = ProcessingConfig(
        resample_sfreq=float(processing["resample_sfreq"]),
        highpass=float(processing["highpass"]),
        lowpass=float(processing["lowpass"]),
        compensation_grade=3,
        remove_dc=True,
        n_jobs=args.n_jobs,
    )
    records = read_selected_epoch_records(args.audit)
    paths = [
        args.output_dir / epoch_plot_filename(record, tmin=tmin, tmax=tmax)
        for record in records
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("absolute-time plot filenames are not unique")
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} epoch plots; "
            f"first collision: {existing[0]}"
        )

    grouped: dict[str, list[tuple[int, EventEpochRecord]]] = defaultdict(list)
    for output_index, record in enumerate(records):
        grouped[record.dataset].append((output_index, record))

    created: list[Path] = []
    for dataset_number, (dataset, indexed_records) in enumerate(
        grouped.items(),
        start=1,
    ):
        event_array = np.array(
            [
                [record.source_sample, 0, record.event_code]
                for _, record in indexed_records
            ],
            dtype=np.int64,
        )
        event_array = validate_event_array(event_array, Path(dataset))
        epochs, _, _, _ = load_grade3_event_epochs(
            Path(dataset),
            event_array,
            config,
            tmin=tmin,
            tmax=tmax,
        )
        epoch_data = epochs.get_data(copy=False)
        for local_index, (output_index, record) in enumerate(indexed_records):
            output = paths[output_index]
            topo_vlim_pt, butterfly_vlim_pt = plot_grade3_epoch(
                epoch_data[local_index],
                epochs.info,
                epochs.times,
                record,
                output,
                tmin=tmin,
                tmax=tmax,
                dpi=args.dpi,
            )
            created.append(output)
            print(
                f"[{len(created):02d}/{len(records)}] "
                f"topo=±{topo_vlim_pt:.4g} pT, "
                f"butterfly=±{butterfly_vlim_pt:.4g} pT  {output.name}",
                flush=True,
            )
        print(
            f"Loaded grade-3 epochs from dataset "
            f"{dataset_number}/{len(grouped)}: {dataset}",
            flush=True,
        )
        del epochs, epoch_data
        gc.collect()

    import matplotlib.image as mpimg

    for path in created:
        if path.stat().st_size == 0:
            raise RuntimeError(f"empty plot file: {path}")
        image = mpimg.imread(path)
        if image.ndim not in (2, 3) or image.size == 0:
            raise RuntimeError(f"could not decode plot file: {path}")
    print(f"Wrote and verified {len(created)} plots in {args.output_dir}")


def find_projected_matches(
    score_pt: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    *,
    threshold_pt: float,
    min_distance: float,
    prominence_pt: float,
    edge_guard_s: float,
) -> ProjectedMatchResult:
    """Find local projected-pT peaks above an explicit absolute threshold."""
    if threshold_pt <= 0 or min_distance <= 0 or prominence_pt < 0:
        raise ValueError(
            "threshold and distance must be positive; prominence nonnegative"
        )
    score_pt = np.asarray(score_pt, dtype=float)
    times = np.asarray(times, dtype=float)
    values = np.abs(score_pt).copy()
    values[(times < edge_guard_s) | (times > times[-1] - edge_guard_s)] = 0.0
    peaks, _ = find_peaks(
        values,
        height=threshold_pt,
        distance=max(1, int(round(min_distance * sfreq))),
        prominence=prominence_pt,
    )
    peaks = peaks.astype(int, copy=False)
    return ProjectedMatchResult(
        score_pt=score_pt,
        peak_indices=peaks,
        onsets=times[peaks],
        event_codes=np.where(score_pt[peaks] >= 0, 1, 2).astype(int),
    )


def projected_matches_to_annotations(
    result: ProjectedMatchResult,
) -> mne.Annotations:
    """Convert projected-pT matches to polarity-aware point annotations."""
    descriptions = np.where(
        result.event_codes == EVENT_ID[POSITIVE_DESCRIPTION],
        POSITIVE_DESCRIPTION,
        NEGATIVE_DESCRIPTION,
    )
    return mne.Annotations(
        onset=result.onsets,
        duration=np.zeros(len(result.onsets)),
        description=descriptions,
        orig_time=None,
    )


def apply_projector(
    raw: mne.io.BaseRaw,
    projector: mne.Projection,
) -> mne.io.BaseRaw:
    """Return a projected copy, leaving the caller's Raw object untouched."""
    cleaned = raw.copy()
    cleaned.add_proj([projector], remove_existing=False)
    cleaned.apply_proj()
    return cleaned


def _preflight_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite calibration outputs: {joined}")


def run_calibration(args: argparse.Namespace) -> None:
    """Build seed/final SSPs and scan the complete EmptyRoom collection."""
    config = ProcessingConfig(
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=0,
        remove_dc=True,
        n_jobs=args.n_jobs,
    )
    output_dir = args.output_dir
    outputs = {
        "seed_projector": output_dir / "door_artifact_seed-proj.fif",
        "final_projector": output_dir / "door_artifact_final-proj.fif",
        "seed_scan": output_dir / "door_artifact_seed_scan.csv",
        "final_scan": output_dir / "door_artifact_final_scan.csv",
        "metadata": output_dir / "door_artifact_ssp_calibration.json",
    }
    _preflight_outputs(outputs.values(), args.overwrite)

    template = load_spatial_template(args.template)
    seed_raw, seed_data = preprocess_dataset(args.seed_dataset, config)
    template_matches = classify_spatial_template(
        seed_raw,
        seed_data,
        template,
        threshold=args.seed_correlation_threshold,
        min_distance=args.seed_min_distance,
        prominence=args.seed_prominence,
        remove_dc=True,
    )
    negative = np.flatnonzero(template_matches.event_codes == 2)
    if len(negative) == 0:
        raise RuntimeError("seed classifier found no code-2 matches")
    strongest = int(
        negative[
            np.argmax(
                np.abs(
                    template_matches.correlation[
                        template_matches.peak_indices[negative]
                    ]
                )
            )
        ]
    )
    seed_peak_index = int(template_matches.peak_indices[strongest])
    seed_onset = float(seed_raw.times[seed_peak_index])
    seed_segment = extract_window(
        seed_raw,
        seed_data,
        seed_onset,
        args.window_duration,
    )
    seed_projector, seed_shape = build_projector_from_segments(
        [seed_segment],
        seed_raw.info,
        description="door-artifact-seed-PCA-01",
    )
    seed_projector, seed_template_correlation = orient_projector(
        seed_projector,
        template,
    )
    write_projector(seed_projector, outputs["seed_projector"], overwrite=args.overwrite)
    print(
        f"Seed SSP: {seed_onset:.3f}-{seed_onset + args.window_duration:.3f} s, "
        f"explained variance={seed_projector['explained_var']:.6f}",
        flush=True,
    )
    del seed_raw, seed_data, seed_segment
    gc.collect()

    datasets = discover_emptyroom_datasets(args.root)
    edge_guard_s = filter_edge_guard_seconds(config)
    seed_rows = scan_projector(datasets, seed_projector, config, edge_guard_s)
    write_scan_csv(seed_rows, outputs["seed_scan"], overwrite=args.overwrite)
    windows = select_training_windows(
        seed_rows,
        exclude_dataset=args.seed_dataset,
        top_n=args.top_n,
        duration=args.window_duration,
    )
    final_projector, final_projector_metadata = build_pooled_projector(
        windows,
        config,
        template,
    )
    write_projector(
        final_projector,
        outputs["final_projector"],
        overwrite=args.overwrite,
    )
    final_rows = scan_projector(datasets, final_projector, config, edge_guard_s)
    write_scan_csv(final_rows, outputs["final_scan"], overwrite=args.overwrite)

    metadata = {
        "processing": asdict(config),
        "root": str(args.root),
        "candidate_datasets": len(datasets),
        "filter_edge_guard_s": edge_guard_s,
        "window_duration_s": args.window_duration,
        "top_n": args.top_n,
        "accepted_threshold_pt": None,
        "seed": {
            "dataset": str(args.seed_dataset),
            "onset_s": seed_onset,
            "window_shape": list(seed_shape),
            "explained_variance": float(seed_projector["explained_var"]),
            "template_spatial_correlation": seed_template_correlation,
            "scan": scan_statistics(seed_rows),
        },
        "final": {
            "training_windows": [asdict(window) for window in windows],
            "projector": final_projector_metadata,
            "scan": scan_statistics(final_rows),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(metadata, outputs["metadata"], overwrite=args.overwrite)
    print("Calibration complete. No threshold was selected.", flush=True)


def run_classification(args: argparse.Namespace) -> None:
    """Apply an explicit projected-pT threshold to one dataset."""
    config = ProcessingConfig(
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=0,
        remove_dc=True,
        n_jobs=args.n_jobs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions",
        )
        projectors = mne.read_proj(args.projector)
    if len(projectors) != 1:
        raise ValueError(f"expected one projector, got {len(projectors)}")
    raw, data_t = preprocess_dataset(args.dataset, config)
    score_pt = projected_pt_score(data_t, raw.ch_names, projectors[0])
    result = find_projected_matches(
        score_pt,
        raw.times,
        raw.info["sfreq"],
        threshold_pt=args.threshold_pt,
        min_distance=args.min_distance,
        prominence_pt=args.prominence_pt,
        edge_guard_s=filter_edge_guard_seconds(config),
    )
    annotations = projected_matches_to_annotations(result)
    events = annotations_to_source_events(args.dataset, annotations)
    events_path, annotations_path = output_paths(args.dataset, args.output_dir)
    write_match_outputs(
        events,
        annotations,
        events_path,
        annotations_path,
        overwrite=args.overwrite,
    )
    print("onset_s   projected_pT   code")
    for peak, onset, code in zip(
        result.peak_indices,
        result.onsets,
        result.event_codes,
    ):
        print(f"{onset:8.3f}  {result.score_pt[peak]:+12.6f}  {code}")
    print(f"Wrote {events_path}")
    print(f"Wrote {annotations_path}")


def _add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resample", type=float, default=300.0)
    parser.add_argument("--highpass", type=float, default=0.1)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--n-jobs", type=int, default=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Use '<command> -h' for command-specific options, for example "
            "'build-3ord -h' or 'plot-3ord -h'."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser(
        "calibrate",
        help="Build grade-0 seed/final SSPs and scan EmptyRoom datasets",
    )
    calibrate.add_argument("--root", type=Path, default=DEFAULT_EMPTYROOM_ROOT)
    calibrate.add_argument(
        "--seed-dataset",
        type=Path,
        default=DEFAULT_SEED_DATASET,
    )
    calibrate.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    calibrate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    calibrate.add_argument("--window-duration", type=float, default=6.0)
    calibrate.add_argument("--top-n", type=int, default=5)
    calibrate.add_argument("--seed-correlation-threshold", type=float, default=0.8)
    calibrate.add_argument("--seed-min-distance", type=float, default=0.5)
    calibrate.add_argument("--seed-prominence", type=float, default=0.05)
    calibrate.add_argument("--overwrite", action="store_true")
    _add_processing_arguments(calibrate)

    classify = commands.add_parser(
        "classify",
        help="Classify one dataset with an explicit projected-pT threshold",
    )
    classify.add_argument("--dataset", type=Path, required=True)
    classify.add_argument("--projector", type=Path, required=True)
    classify.add_argument("--threshold-pt", type=float, required=True)
    classify.add_argument("--min-distance", type=float, default=0.5)
    classify.add_argument("--prominence-pt", type=float, default=0.0)
    classify.add_argument("--output-dir", type=Path, default=None)
    classify.add_argument("--overwrite", action="store_true")
    _add_processing_arguments(classify)

    build_3ord = commands.add_parser(
        "build-3ord",
        help="Build a one-component grade-3 SSP from door event epochs",
        description=(
            "Load all door event epochs with CTF compensation grade 3, "
            "collapse nearby detections into bursts, retain the strongest "
            "epoch per burst, and fit one artifact SSP component."
        ),
    )
    build_3ord.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_EMPTYROOM_ROOT,
        help=f"EmptyRoom root (default: {DEFAULT_EMPTYROOM_ROOT})",
    )
    build_3ord.add_argument(
        "--event-filename",
        default="door_evts.npy",
        help="Dataset-local event filename (default: door_evts.npy)",
    )
    build_3ord.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_3ORD_PROJECTOR,
        help=f"Output projector (default: {DEFAULT_3ORD_PROJECTOR})",
    )
    build_3ord.add_argument(
        "--tmin",
        type=float,
        default=DEFAULT_3ORD_TMIN,
        help=f"Epoch start in seconds (default: {DEFAULT_3ORD_TMIN})",
    )
    build_3ord.add_argument(
        "--tmax",
        type=float,
        default=DEFAULT_3ORD_TMAX,
        help=f"Epoch end in seconds (default: {DEFAULT_3ORD_TMAX})",
    )
    build_3ord.add_argument(
        "--burst-gap",
        type=float,
        default=DEFAULT_BURST_GAP,
        help=(
            "A larger inter-event gap starts a new burst in seconds "
            f"(default: {DEFAULT_BURST_GAP})"
        ),
    )
    build_3ord.add_argument("--overwrite", action="store_true")
    _add_processing_arguments(build_3ord)

    plot_3ord = commands.add_parser(
        "plot-3ord",
        help="Plot the selected grade-3 SSP training epochs",
        description=(
            "Recreate each selected grade-3 epoch and save a 1x2 epoch-average "
            "topomap and butterfly figure using absolute-time filenames."
        ),
    )
    plot_3ord.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_3ORD_EPOCH_AUDIT,
        help=f"Epoch audit CSV (default: {DEFAULT_3ORD_EPOCH_AUDIT})",
    )
    plot_3ord.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_3ORD_METADATA,
        help=f"Build metadata JSON (default: {DEFAULT_3ORD_METADATA})",
    )
    plot_3ord.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_3ORD_PLOT_DIR,
        help=f"PNG output directory (default: {DEFAULT_3ORD_PLOT_DIR})",
    )
    plot_3ord.add_argument("--dpi", type=int, default=150)
    plot_3ord.add_argument("--n-jobs", type=int, default=-1)
    plot_3ord.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mne.set_log_level("WARNING")
    if args.command == "calibrate":
        run_calibration(args)
    elif args.command == "classify":
        run_classification(args)
    elif args.command == "build-3ord":
        run_build_3ord(args)
    else:
        run_plot_3ord(args)


if __name__ == "__main__":
    main()
