#!/usr/bin/env python3
"""Compare grade-0 and grade-3 channel norms for curated door events."""

from __future__ import annotations

import argparse
import gc
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

if __package__:
    from .door_ssp_annotation_plotter import (
        DEFAULT_ANNOTATION_DIR,
        DEFAULT_ROOT,
        annotation_path_for_dataset,
        episode_window,
    )
    from .door_ssp_classifier import (
        ProcessingConfig,
        discover_emptyroom_datasets,
        preprocess_dataset,
    )
else:
    from door_ssp_annotation_plotter import (
        DEFAULT_ANNOTATION_DIR,
        DEFAULT_ROOT,
        annotation_path_for_dataset,
        episode_window,
    )
    from door_ssp_classifier import (
        ProcessingConfig,
        discover_emptyroom_datasets,
        preprocess_dataset,
    )


DEFAULT_PNG_DIR = DEFAULT_ROOT / "OUT" / "PNGS"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "OUT" / "NORM_COMPARISONS"
DEFAULT_MATCH_TOLERANCE = 0.010
PNG_PATTERN = re.compile(
    r"^(?P<dataset>.+)_event-(?P<number>\d+)_"
    r"mid-(?P<midpoint>\d+(?:\.\d+)?)s\.png$"
)


@dataclass(frozen=True)
class SelectedEvent:
    """One curated PNG resolved to its dataset and duration annotation."""

    source_png: Path
    dataset: Path
    annotation_path: Path
    event_number: int
    onset: float
    midpoint: float
    duration: float
    description: str


def parse_source_png(path: Path) -> tuple[str, int, float]:
    """Parse the dataset stem, display number, and midpoint from a PNG."""
    match = PNG_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"unrecognized door plot filename: {Path(path).name}")
    return (
        match.group("dataset"),
        int(match.group("number")),
        float(match.group("midpoint")),
    )


def _dataset_lookup(root: Path) -> dict[str, Path]:
    """Build an unambiguous lookup from dataset stem to CTF directory."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for dataset in discover_emptyroom_datasets(root):
        grouped[dataset.stem].append(dataset)
    duplicates = {stem: paths for stem, paths in grouped.items() if len(paths) > 1}
    if duplicates:
        stem = sorted(duplicates)[0]
        raise ValueError(f"multiple datasets have stem {stem!r}: {duplicates[stem]}")
    return {stem: paths[0] for stem, paths in grouped.items()}


def resolve_selected_events(
    png_dir: Path,
    root: Path,
    annotation_dir: Path,
    *,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> list[SelectedEvent]:
    """Resolve every curated PNG to one dataset and one annotation."""
    if match_tolerance <= 0:
        raise ValueError("match tolerance must be positive")
    pngs = sorted(Path(png_dir).glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no PNG files found in {png_dir}")
    datasets = _dataset_lookup(root)
    annotation_cache: dict[Path, mne.Annotations] = {}
    selected = []
    for source_png in pngs:
        dataset_stem, event_number, png_midpoint = parse_source_png(source_png)
        if dataset_stem not in datasets:
            raise FileNotFoundError(
                f"no dataset matches {dataset_stem!r} from {source_png.name}"
            )
        dataset = datasets[dataset_stem]
        annotation_path = annotation_path_for_dataset(dataset, annotation_dir)
        if annotation_path is None:
            raise FileNotFoundError(f"no annotations found for {dataset}")
        if annotation_path not in annotation_cache:
            annotation_cache[annotation_path] = mne.read_annotations(annotation_path)
        annotations = annotation_cache[annotation_path]
        midpoints = annotations.onset + annotations.duration / 2.0
        differences = np.abs(midpoints - png_midpoint)
        matches = np.flatnonzero(differences <= match_tolerance)
        if len(matches) != 1:
            raise ValueError(
                f"expected one annotation within {match_tolerance:g} s of "
                f"{png_midpoint:g} s for {source_png.name}, got {len(matches)}"
            )
        index = int(matches[0])
        duration = float(annotations.duration[index])
        if duration <= 0:
            raise ValueError(f"matched annotation has nonpositive duration: {source_png}")
        selected.append(
            SelectedEvent(
                source_png=source_png,
                dataset=dataset,
                annotation_path=annotation_path,
                event_number=event_number,
                onset=float(annotations.onset[index]),
                midpoint=float(midpoints[index]),
                duration=duration,
                description=str(annotations.description[index]),
            )
        )
    return selected


def validate_preprocessed_pair(
    raw_grade0: mne.io.BaseRaw,
    data_grade0: np.ndarray,
    raw_grade3: mne.io.BaseRaw,
    data_grade3: np.ndarray,
) -> None:
    """Require grade-0 and grade-3 data to share one sample/channel grid."""
    if raw_grade0.ch_names != raw_grade3.ch_names:
        raise ValueError("grade-0 and grade-3 channel names differ")
    if data_grade0.shape != data_grade3.shape:
        raise ValueError(
            "grade-0 and grade-3 data shapes differ: "
            f"{data_grade0.shape} != {data_grade3.shape}"
        )
    if data_grade0.shape != (len(raw_grade0.ch_names), raw_grade0.n_times):
        raise ValueError("grade-0 data shape does not match raw")
    if data_grade3.shape != (len(raw_grade3.ch_names), raw_grade3.n_times):
        raise ValueError("grade-3 data shape does not match raw")
    if not np.array_equal(raw_grade0.times, raw_grade3.times):
        raise ValueError("grade-0 and grade-3 time grids differ")


def compensation_norms(
    data_grade0_t: np.ndarray,
    data_grade3_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return channel L2 norms in pT and their pointwise grade-0/grade-3 ratio."""
    data_grade0_t = np.asarray(data_grade0_t, dtype=float)
    data_grade3_t = np.asarray(data_grade3_t, dtype=float)
    if data_grade0_t.ndim != 2 or data_grade0_t.shape != data_grade3_t.shape:
        raise ValueError("grade-0 and grade-3 snippets must have equal 2-D shapes")
    if not np.isfinite(data_grade0_t).all() or not np.isfinite(data_grade3_t).all():
        raise ValueError("compensation snippets must contain only finite values")
    norm_grade0_pt = np.linalg.norm(data_grade0_t, axis=0) * 1e12
    norm_grade3_pt = np.linalg.norm(data_grade3_t, axis=0) * 1e12
    if np.any(norm_grade3_pt <= 0):
        raise ValueError("grade-3 channel norm contains a zero denominator")
    ratio = norm_grade0_pt / norm_grade3_pt
    if not np.isfinite(ratio).all():
        raise ValueError("compensation norm ratio contains non-finite values")
    return norm_grade0_pt, norm_grade3_pt, ratio


def comparison_output_path(event: SelectedEvent, output_dir: Path) -> Path:
    """Return a collision-safe filename derived from the curated PNG."""
    return Path(output_dir) / f"{event.source_png.stem}_norm0-norm3-ratio.png"


def plot_norm_comparison(
    times: np.ndarray,
    norm_grade0_pt: np.ndarray,
    norm_grade3_pt: np.ndarray,
    ratio: np.ndarray,
    event: SelectedEvent,
    output: Path,
    *,
    dpi: int,
) -> None:
    """Write vertically stacked grade-0, grade-3, and ratio time series."""
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=float)
    norm_grade0_pt = np.asarray(norm_grade0_pt, dtype=float)
    norm_grade3_pt = np.asarray(norm_grade3_pt, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    if not (
        times.ndim == 1
        and norm_grade0_pt.shape == times.shape
        and norm_grade3_pt.shape == times.shape
        and ratio.shape == times.shape
    ):
        raise ValueError("times, norms, and ratio must be equal-length vectors")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    norm_limit = 1.05 * float(max(np.max(norm_grade0_pt), np.max(norm_grade3_pt)))
    ratio_limit = 1.05 * float(np.max(ratio))
    if norm_limit <= 0 or ratio_limit <= 0:
        raise ValueError("plot values must have positive finite scales")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(times, norm_grade0_pt, color="tab:blue", linewidth=1.0)
    axes[0].set_title("Compensation grade 0: channel Euclidean norm")
    axes[1].plot(times, norm_grade3_pt, color="tab:orange", linewidth=1.0)
    axes[1].set_title("Compensation grade 3: channel Euclidean norm")
    for axis in axes[:2]:
        axis.set_ylim(0.0, norm_limit)
        axis.set_ylabel("L2 norm (pT)")

    axes[2].plot(times, ratio, color="tab:green", linewidth=1.0)
    axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=0.9, label="ratio=1")
    axes[2].set_ylim(0.0, ratio_limit)
    axes[2].set_ylabel("Grade 0 / grade 3")
    axes[2].set_xlabel("Time relative to annotation midpoint (s)")
    axes[2].set_title("Pointwise compensation norm ratio")
    axes[2].legend(loc="upper right")
    for axis in axes:
        axis.axvline(0.0, color="crimson", linewidth=1.1)
        axis.grid(alpha=0.2)
        axis.set_xlim(float(times[0]), float(times[-1]))

    end = event.onset + event.duration
    fig.suptitle(
        f"{event.dataset.stem} | source event {event.event_number:03d} | "
        f"{event.description}\n"
        f"absolute {event.onset:.3f}-{end:.3f} s | "
        f"midpoint={event.midpoint:.3f} s | duration={event.duration:.3f} s"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_dataset_events(
    dataset: Path,
    events: list[SelectedEvent],
    output_dir: Path,
    base_config: ProcessingConfig,
    *,
    dpi: int,
) -> list[Path]:
    """Preprocess both grades once and plot every selected dataset event."""
    config_grade0 = ProcessingConfig(
        resample_sfreq=base_config.resample_sfreq,
        highpass=base_config.highpass,
        lowpass=base_config.lowpass,
        compensation_grade=0,
        remove_dc=base_config.remove_dc,
        n_jobs=base_config.n_jobs,
    )
    config_grade3 = ProcessingConfig(
        resample_sfreq=base_config.resample_sfreq,
        highpass=base_config.highpass,
        lowpass=base_config.lowpass,
        compensation_grade=3,
        remove_dc=base_config.remove_dc,
        n_jobs=base_config.n_jobs,
    )
    raw_grade0, data_grade0 = preprocess_dataset(dataset, config_grade0)
    raw_grade3, data_grade3 = preprocess_dataset(dataset, config_grade3)
    validate_preprocessed_pair(raw_grade0, data_grade0, raw_grade3, data_grade3)

    created = []
    for event in events:
        window = episode_window(
            raw_grade0,
            onset=event.onset,
            duration=event.duration,
            description=event.description,
        )
        selection = slice(window.start_index, window.stop_index)
        norm_grade0, norm_grade3, ratio = compensation_norms(
            data_grade0[:, selection],
            data_grade3[:, selection],
        )
        times = raw_grade0.times[selection] - event.midpoint
        output = comparison_output_path(event, output_dir)
        plot_norm_comparison(
            times,
            norm_grade0,
            norm_grade3,
            ratio,
            event,
            output,
            dpi=dpi,
        )
        created.append(output)
        print(
            f"  midpoint={event.midpoint:8.3f} s  "
            f"ratio median={np.median(ratio):8.3f}  {output.name}",
            flush=True,
        )
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--match-tolerance", type=float, default=DEFAULT_MATCH_TOLERANCE)
    parser.add_argument("--resample", type=float, default=300.0)
    parser.add_argument("--highpass", type=float, default=0.1)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg", force=True)
    mne.set_log_level("WARNING")
    selected = resolve_selected_events(
        args.png_dir,
        args.root,
        args.annotation_dir,
        match_tolerance=args.match_tolerance,
    )
    outputs = [comparison_output_path(event, args.output_dir) for event in selected]
    if len(set(outputs)) != len(outputs):
        raise ValueError("comparison output filenames are not unique")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} comparison plots; "
            f"first collision: {existing[0]}"
        )
    base_config = ProcessingConfig(
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=0,
        remove_dc=True,
        n_jobs=args.n_jobs,
    )
    grouped: dict[Path, list[SelectedEvent]] = defaultdict(list)
    for event in selected:
        grouped[event.dataset].append(event)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    failures = []
    for number, (dataset, events) in enumerate(grouped.items(), start=1):
        try:
            paths = plot_dataset_events(
                dataset,
                events,
                args.output_dir,
                base_config,
                dpi=args.dpi,
            )
            created.extend(paths)
            print(
                f"[{number:02d}/{len(grouped):02d}] OK {len(paths)} plots  {dataset}",
                flush=True,
            )
        except Exception as error:
            failures.append((dataset, error))
            print(
                f"[{number:02d}/{len(grouped):02d}] ERROR "
                f"{type(error).__name__}: {error}  {dataset}",
                flush=True,
            )
        finally:
            gc.collect()

    import matplotlib.image as mpimg

    for path in created:
        if path.stat().st_size == 0:
            raise RuntimeError(f"empty comparison plot: {path}")
        image = mpimg.imread(path)
        if image.ndim not in (2, 3) or image.size == 0:
            raise RuntimeError(f"could not decode comparison plot: {path}")
    print(
        f"Finished: selected={len(selected)}, created={len(created)}, "
        f"failed_datasets={len(failures)}, output={args.output_dir}",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"comparison plotting failed for {len(failures)} datasets")
    if len(created) != len(selected):
        raise RuntimeError(
            f"created {len(created)} plots for {len(selected)} selected events"
        )


if __name__ == "__main__":
    main()
