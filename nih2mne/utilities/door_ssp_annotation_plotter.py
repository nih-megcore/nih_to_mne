#!/usr/bin/env python3
"""Plot grade-0 MEG data for duration annotations from the door SSP classifier."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

if __package__:
    from .door_ssp_classifier import (
        DEFAULT_MIN_DURATION,
        ProcessingConfig,
        discover_emptyroom_datasets,
        preprocess_dataset,
    )
else:
    from door_ssp_classifier import (
        DEFAULT_MIN_DURATION,
        ProcessingConfig,
        discover_emptyroom_datasets,
        preprocess_dataset,
    )


DEFAULT_ROOT = Path("/fast2/EMPTY")
DEFAULT_ANNOTATION_DIR = DEFAULT_ROOT / "OUTS"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "OUT" / "PNGS"
ANNOTATION_SUFFIX = "_door_artifacts-annot.fif"


@dataclass(frozen=True)
class EpisodeWindow:
    """One annotation interval represented on the resampled data grid."""

    onset: float
    midpoint: float
    duration: float
    description: str
    start_index: int
    midpoint_index: int
    stop_index: int


def annotation_path_for_dataset(
    dataset: Path,
    annotation_dir: Path,
) -> Path | None:
    """Return the standard or legacy classifier annotation for a dataset."""
    dataset = Path(dataset)
    annotation_dir = Path(annotation_dir)
    candidates = (
        annotation_dir / f"{dataset.stem}{ANNOTATION_SUFFIX}",
        annotation_dir / f"{dataset.name}{ANNOTATION_SUFFIX}",
    )
    return next((path for path in candidates if path.is_file()), None)


def episode_window(
    raw: mne.io.BaseRaw,
    *,
    onset: float,
    duration: float,
    description: str,
) -> EpisodeWindow:
    """Map one duration annotation to an inclusive resampled-data window."""
    onset = float(onset)
    duration = float(duration)
    if not np.isfinite(onset) or not np.isfinite(duration):
        raise ValueError("annotation onset and duration must be finite")
    if duration <= 0:
        raise ValueError("annotation duration must be positive")
    end = onset + duration
    recording_end = raw.times[-1] + 1.0 / raw.info["sfreq"]
    tolerance = 0.5 / raw.info["sfreq"]
    if onset < -tolerance or end > recording_end + tolerance:
        raise ValueError(
            f"annotation {onset:g}-{end:g} s is outside recording "
            f"0-{recording_end:g} s"
        )
    onset = max(0.0, onset)
    end = min(recording_end, end)
    midpoint = onset + (end - onset) / 2.0
    start_index = int(raw.time_as_index(onset, use_rounding=True)[0])
    midpoint_index = int(raw.time_as_index(midpoint, use_rounding=True)[0])
    stop_index = int(raw.time_as_index(end, use_rounding=True)[0]) + 1
    start_index = min(max(start_index, 0), raw.n_times - 1)
    midpoint_index = min(max(midpoint_index, start_index), raw.n_times - 1)
    stop_index = min(max(stop_index, start_index + 1), raw.n_times)
    return EpisodeWindow(
        onset=onset,
        midpoint=midpoint,
        duration=end - onset,
        description=str(description),
        start_index=start_index,
        midpoint_index=midpoint_index,
        stop_index=stop_index,
    )


def episode_plot_filename(
    dataset: Path,
    episode_number: int,
    midpoint: float,
) -> str:
    """Create a unique, time-readable PNG filename."""
    return (
        f"{Path(dataset).stem}_event-{episode_number:03d}_"
        f"mid-{midpoint:.3f}s.png"
    )


def plot_episode(
    data_t: np.ndarray,
    raw: mne.io.BaseRaw,
    window: EpisodeWindow,
    output: Path,
    *,
    dataset: Path,
    episode_number: int,
    dpi: int,
) -> tuple[float, float]:
    """Write a midpoint topomap and duration-derived butterfly plot."""
    import matplotlib.pyplot as plt

    data_t = np.asarray(data_t, dtype=float)
    if data_t.shape != (len(raw.ch_names), raw.n_times):
        raise ValueError("processed data shape does not match raw")
    epoch_pt = (
        data_t[:, window.start_index : window.stop_index] * 1e12
    )
    midpoint_pt = data_t[:, window.midpoint_index] * 1e12
    times = (
        raw.times[window.start_index : window.stop_index] - window.midpoint
    )
    topo_vlim_pt = float(np.nanpercentile(np.abs(midpoint_pt), 99.5))
    butterfly_vlim_pt = float(np.nanpercentile(np.abs(epoch_pt), 99.5))
    if (
        not np.isfinite(topo_vlim_pt)
        or topo_vlim_pt <= 0
        or not np.isfinite(butterfly_vlim_pt)
        or butterfly_vlim_pt <= 0
    ):
        raise ValueError("episode has an invalid plot scale")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        layout="constrained",
    )
    image, _ = mne.viz.plot_topomap(
        midpoint_pt,
        raw.info,
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
    axes[0].set_title(f"Grade-0 topomap at midpoint ({window.midpoint:.3f} s)")

    axes[1].plot(times, epoch_pt.T, color="black", alpha=0.16, linewidth=0.45)
    axes[1].axvline(0.0, color="crimson", linewidth=1.2, label="midpoint")
    axes[1].set_xlim(float(times[0]), float(times[-1]))
    axes[1].set_ylim(-butterfly_vlim_pt, butterfly_vlim_pt)
    axes[1].set_xlabel("Time relative to annotation midpoint (s)")
    axes[1].set_ylabel("Amplitude (pT)")
    axes[1].set_title("Grade-0 duration butterfly")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    end = window.onset + window.duration
    fig.suptitle(
        f"{Path(dataset).stem} | event {episode_number:03d} | "
        f"{window.description}\n"
        f"absolute {window.onset:.3f}-{end:.3f} s | "
        f"midpoint={window.midpoint:.3f} s | duration={window.duration:.3f} s | "
        f"topo scale=+/-{topo_vlim_pt:.4g} pT | "
        f"butterfly scale=+/-{butterfly_vlim_pt:.4g} pT"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return topo_vlim_pt, butterfly_vlim_pt


def plot_dataset_annotations(
    dataset: Path,
    annotation_path: Path,
    output_dir: Path,
    config: ProcessingConfig,
    *,
    dpi: int,
    overwrite: bool,
    min_duration: float = DEFAULT_MIN_DURATION,
) -> int:
    """Preprocess one dataset once and plot qualifying duration annotations."""
    if min_duration <= 0:
        raise ValueError("min-duration must be positive")
    annotations = mne.read_annotations(annotation_path)
    if len(annotations) == 0:
        return 0
    if np.any(annotations.duration < 0):
        raise ValueError(
            f"annotations must not have negative duration: {annotation_path}"
        )
    selected = np.flatnonzero(annotations.duration >= min_duration)
    if len(selected) == 0:
        return 0

    raw, data_t = preprocess_dataset(dataset, config)
    windows = [
        episode_window(
            raw,
            onset=onset,
            duration=duration,
            description=description,
        )
        for onset, duration, description in zip(
            annotations.onset[selected],
            annotations.duration[selected],
            annotations.description[selected],
        )
    ]
    outputs = [
        Path(output_dir)
        / episode_plot_filename(dataset, number, window.midpoint)
        for number, window in enumerate(windows, start=1)
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} episode plots; "
            f"first collision: {existing[0]}"
        )

    for number, (window, output) in enumerate(
        zip(windows, outputs),
        start=1,
    ):
        topo_scale, butterfly_scale = plot_episode(
            data_t,
            raw,
            window,
            output,
            dataset=dataset,
            episode_number=number,
            dpi=dpi,
        )
        print(
            f"  [{number:03d}/{len(windows):03d}] "
            f"mid={window.midpoint:8.3f} s "
            f"topo=+/-{topo_scale:.4g} pT "
            f"butterfly=+/-{butterfly_scale:.4g} pT  {output.name}",
            flush=True,
        )
    return len(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resample", type=float, default=300.0)
    parser.add_argument("--highpass", type=float, default=0.1)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help="Minimum annotation duration to plot in seconds",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    import matplotlib

    matplotlib.use("Agg", force=True)
    mne.set_log_level("WARNING")
    config = ProcessingConfig(
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=0,
        remove_dc=True,
        n_jobs=args.n_jobs,
    )
    datasets = discover_emptyroom_datasets(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    missing = 0
    empty = 0
    failed = 0
    for number, dataset in enumerate(datasets, start=1):
        annotation_path = annotation_path_for_dataset(
            dataset,
            args.annotation_dir,
        )
        if annotation_path is None:
            missing += 1
            print(
                f"[{number:03d}/{len(datasets):03d}] SKIP missing annotations  "
                f"{dataset}",
                flush=True,
            )
            continue
        try:
            count = plot_dataset_annotations(
                dataset,
                annotation_path,
                args.output_dir,
                config,
                dpi=args.dpi,
                overwrite=args.overwrite,
                min_duration=args.min_duration,
            )
            if count == 0:
                empty += 1
                status = "SKIP empty/no qualifying annotations"
            else:
                plotted += count
                status = f"OK {count} plots"
            print(
                f"[{number:03d}/{len(datasets):03d}] {status}  {dataset}",
                flush=True,
            )
        except Exception as error:
            failed += 1
            print(
                f"[{number:03d}/{len(datasets):03d}] ERROR "
                f"{type(error).__name__}: {error}  {dataset}",
                flush=True,
            )
        finally:
            gc.collect()
    print(
        f"Finished: datasets={len(datasets)}, plots={plotted}, "
        f"missing={missing}, empty={empty}, failed={failed}, "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
