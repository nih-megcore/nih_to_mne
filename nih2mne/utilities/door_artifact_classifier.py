#!/usr/bin/env python3
"""Build and apply a spatial-correlation template for CTF door artifacts."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.signal import find_peaks

if __package__:
    from .interactive_ctf_topomap import prepare_display_data, prepare_raw
else:
    from interactive_ctf_topomap import prepare_display_data, prepare_raw


DEFAULT_TEMPLATE_TMIN = 93.0
DEFAULT_TEMPLATE_TMAX = 95.0
DEFAULT_THRESHOLD = 0.8
DEFAULT_MIN_DISTANCE = 0.5
DEFAULT_PROMINENCE = 0.05
POSITIVE_DESCRIPTION = "door_artifact_positive"
NEGATIVE_DESCRIPTION = "door_artifact_negative"
EVENT_ID = {
    POSITIVE_DESCRIPTION: 1,
    NEGATIVE_DESCRIPTION: 2,
}


@dataclass(frozen=True)
class MatchResult:
    """Continuous correlation and discrete artifact matches."""

    correlation: np.ndarray
    peak_indices: np.ndarray
    onsets: np.ndarray
    event_codes: np.ndarray


def spatial_corr(data: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Calculate scale-independent spatial correlation at every time point."""
    data = np.asarray(data, dtype=float)
    template = np.asarray(template, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must have shape (n_channels, n_times), got {data.shape}"
        )
    if template.ndim != 1:
        raise ValueError(
            f"template must have shape (n_channels,), got {template.shape}"
        )
    if data.shape[0] != template.size:
        raise ValueError(
            "data and template channel counts differ: "
            f"{data.shape[0]} != {template.size}"
        )
    if not np.isfinite(data).all() or not np.isfinite(template).all():
        raise ValueError("data and template must contain only finite values")

    template = template - template.mean()
    template_norm = np.linalg.norm(template)
    if template_norm == 0:
        raise ValueError("template has zero spatial variance")
    template = template / template_norm

    centered = data - data.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    valid = norms > 0
    correlation = np.zeros(data.shape[1], dtype=float)
    correlation[valid] = template @ (centered[:, valid] / norms[valid])
    return np.clip(correlation, -1.0, 1.0)


def create_spatial_template(
    raw: mne.io.BaseRaw,
    display_data_t: np.ndarray,
    *,
    tmin: float,
    tmax: float,
    source_dataset: Path | None = None,
    remove_dc: bool = True,
) -> mne.Evoked:
    """Average an inclusive time window into a one-sample Evoked template."""
    if not 0 <= tmin < tmax <= raw.times[-1]:
        raise ValueError(
            f"template window [{tmin}, {tmax}] is outside "
            f"the recording [0, {raw.times[-1]}]"
        )
    if display_data_t.shape != (len(raw.ch_names), raw.n_times):
        raise ValueError(
            "display data shape does not match raw: "
            f"{display_data_t.shape} != {(len(raw.ch_names), raw.n_times)}"
        )

    selected = (raw.times >= tmin) & (raw.times <= tmax)
    nave = int(selected.sum())
    if nave == 0:
        raise ValueError("template window contains no samples")
    template_t = display_data_t[:, selected].mean(axis=1, keepdims=True)

    provenance = {
        "kind": "door_artifact_spatial_template",
        "source_dataset": str(source_dataset) if source_dataset else None,
        "tmin": float(tmin),
        "tmax": float(tmax),
        "remove_dc": bool(remove_dc),
    }
    info = raw.info.copy()
    with info._unlock():
        info["description"] = json.dumps(provenance, sort_keys=True)
    return mne.EvokedArray(
        template_t,
        info,
        tmin=0.0,
        comment=f"door artifact average {tmin:g}-{tmax:g} s",
        nave=nave,
    )


def write_spatial_template(
    template: mne.Evoked,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a template after checking overwrite policy."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"template already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions",
        )
        mne.write_evokeds(output, template, overwrite=overwrite)


def load_spatial_template(path: Path) -> mne.Evoked:
    """Load and validate a one-sample spatial template."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This filename .* does not conform to MNE naming conventions",
        )
        template = mne.read_evokeds(path, condition=0, proj=False)
    if template.data.shape[1] != 1:
        raise ValueError(
            f"spatial template must contain one sample, got {template.data.shape[1]}"
        )
    return template


def align_data_to_template(
    data: np.ndarray,
    data_ch_names: list[str],
    template_ch_names: list[str],
) -> np.ndarray:
    """Reorder data rows to the template and reject missing channels."""
    if data.shape[0] != len(data_ch_names):
        raise ValueError("data row count does not match data channel names")
    if len(set(data_ch_names)) != len(data_ch_names):
        raise ValueError("data channel names are not unique")
    if len(set(template_ch_names)) != len(template_ch_names):
        raise ValueError("template channel names are not unique")
    lookup = {name: index for index, name in enumerate(data_ch_names)}
    missing = [name for name in template_ch_names if name not in lookup]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"data are missing {len(missing)} template channels: {preview}"
        )
    return data[[lookup[name] for name in template_ch_names]]


def find_template_matches(
    correlation: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    prominence: float = DEFAULT_PROMINENCE,
) -> MatchResult:
    """Find separated local maxima of absolute spatial correlation."""
    correlation = np.asarray(correlation, dtype=float)
    times = np.asarray(times, dtype=float)
    if correlation.ndim != 1 or times.shape != correlation.shape:
        raise ValueError(
            "correlation and times must be one-dimensional and equal length"
        )
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if sfreq <= 0 or min_distance <= 0 or prominence < 0:
        raise ValueError(
            "sfreq and min_distance must be positive; prominence nonnegative"
        )

    distance_samples = max(1, int(round(min_distance * sfreq)))
    peaks, _ = find_peaks(
        np.abs(correlation),
        height=threshold,
        distance=distance_samples,
        prominence=prominence,
    )
    peaks = peaks.astype(int, copy=False)
    event_codes = np.where(correlation[peaks] >= 0, 1, 2).astype(int)
    return MatchResult(
        correlation=correlation,
        peak_indices=peaks,
        onsets=times[peaks],
        event_codes=event_codes,
    )


def classify_spatial_template(
    raw: mne.io.BaseRaw,
    display_data_t: np.ndarray,
    template: mne.Evoked,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    prominence: float = DEFAULT_PROMINENCE,
    remove_dc: bool = True,
) -> MatchResult:
    """Score one processed recording and identify template matches."""
    for key in ("sfreq", "highpass", "lowpass"):
        if not np.isclose(raw.info[key], template.info[key]):
            raise ValueError(
                f"recording and template {key} values differ: "
                f"{raw.info[key]} != {template.info[key]}"
            )
    try:
        provenance = json.loads(template.info["description"] or "{}")
    except json.JSONDecodeError:
        provenance = {}
    template_remove_dc = provenance.get("remove_dc")
    if template_remove_dc is not None and template_remove_dc != remove_dc:
        raise ValueError(
            "recording and template channel-centering settings differ: "
            f"{remove_dc} != {template_remove_dc}"
        )
    aligned = align_data_to_template(
        display_data_t,
        raw.ch_names,
        template.ch_names,
    )
    correlation = spatial_corr(aligned, template.data[:, 0])
    return find_template_matches(
        correlation,
        raw.times,
        raw.info["sfreq"],
        threshold=threshold,
        min_distance=min_distance,
        prominence=prominence,
    )


def matches_to_annotations(result: MatchResult) -> mne.Annotations:
    """Represent point matches in seconds with polarity-aware descriptions."""
    descriptions = np.where(
        result.event_codes == 1,
        POSITIVE_DESCRIPTION,
        NEGATIVE_DESCRIPTION,
    )
    return mne.Annotations(
        onset=result.onsets,
        duration=np.zeros(len(result.onsets)),
        description=descriptions,
        orig_time=None,
    )


def annotations_to_source_events(
    dataset: Path,
    annotations: mne.Annotations,
) -> np.ndarray:
    """Convert second-based annotations to the original CTF sample grid."""
    source_raw = mne.io.read_raw_ctf(
        dataset,
        system_clock="ignore",
        preload=False,
    )
    source_raw.set_annotations(annotations)
    events, _ = mne.events_from_annotations(source_raw, event_id=EVENT_ID)
    return events


def output_paths(dataset: Path, output_dir: Path | None) -> tuple[Path, Path]:
    """Resolve dataset-local or collision-safe shared output paths."""
    dataset = Path(dataset)
    if output_dir is None:
        return dataset / "door_evts.npy", dataset / "door_artifacts-annot.fif"
    prefix = f"{dataset.stem}_"
    output_dir = Path(output_dir)
    return (
        output_dir / f"{prefix}door_evts.npy",
        output_dir / f"{prefix}door_artifacts-annot.fif",
    )


def write_match_outputs(
    events: np.ndarray,
    annotations: mne.Annotations,
    events_path: Path,
    annotations_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write both match formats after preflighting all destinations."""
    existing = [path for path in (events_path, annotations_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing outputs: {joined}")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(events_path, events, allow_pickle=False)
    annotations.save(annotations_path, overwrite=overwrite)


def _add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resample", type=float, default=300.0)
    parser.add_argument("--highpass", type=float, default=0.1)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--compensation-grade", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Use literal channel values instead of channel-centered data",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-template")
    _add_preprocessing_arguments(build)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--tmin", type=float, default=DEFAULT_TEMPLATE_TMIN)
    build.add_argument("--tmax", type=float, default=DEFAULT_TEMPLATE_TMAX)
    build.add_argument("--overwrite", action="store_true")

    classify = subparsers.add_parser("classify")
    _add_preprocessing_arguments(classify)
    classify.add_argument("--template", type=Path, required=True)
    classify.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    classify.add_argument(
        "--min-distance",
        type=float,
        default=DEFAULT_MIN_DISTANCE,
        help="Minimum separation between peaks in seconds",
    )
    classify.add_argument("--prominence", type=float, default=DEFAULT_PROMINENCE)
    classify.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional shared directory; filenames receive the dataset prefix",
    )
    classify.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _process_dataset(args: argparse.Namespace) -> tuple[mne.io.BaseRaw, np.ndarray]:
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"CTF dataset not found: {args.dataset}")
    raw = prepare_raw(
        args.dataset,
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=args.compensation_grade,
        n_jobs=args.n_jobs,
    )
    display_data_t, _ = prepare_display_data(raw, remove_dc=not args.keep_dc)
    return raw, display_data_t


def main() -> None:
    args = parse_args()
    raw, display_data_t = _process_dataset(args)

    if args.command == "build-template":
        template = create_spatial_template(
            raw,
            display_data_t,
            tmin=args.tmin,
            tmax=args.tmax,
            source_dataset=args.dataset,
            remove_dc=not args.keep_dc,
        )
        write_spatial_template(template, args.output, overwrite=args.overwrite)
        print(
            f"Wrote {args.output}: {len(template.ch_names)} channels, "
            f"nave={template.nave}, window={args.tmin:g}-{args.tmax:g} s"
        )
        return

    template = load_spatial_template(args.template)
    result = classify_spatial_template(
        raw,
        display_data_t,
        template,
        threshold=args.threshold,
        min_distance=args.min_distance,
        prominence=args.prominence,
        remove_dc=not args.keep_dc,
    )
    annotations = matches_to_annotations(result)
    events = annotations_to_source_events(args.dataset, annotations)
    events_path, annotations_path = output_paths(args.dataset, args.output_dir)
    write_match_outputs(
        events,
        annotations,
        events_path,
        annotations_path,
        overwrite=args.overwrite,
    )

    print("onset_s   corr       code")
    for peak, onset, code in zip(
        result.peak_indices,
        result.onsets,
        result.event_codes,
    ):
        print(f"{onset:8.3f}  {result.correlation[peak]:+0.6f}  {code}")
    print(f"Wrote {events_path}")
    print(f"Wrote {annotations_path}")


if __name__ == "__main__":
    main()
