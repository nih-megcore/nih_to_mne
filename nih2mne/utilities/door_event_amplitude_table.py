#!/usr/bin/env python3
"""Summarize grade-0 and grade-3 amplitudes for annotated door events."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

if __package__:
    from .door_compensation_norm_plotter import (
        compensation_norms,
        validate_preprocessed_pair,
    )
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
    from door_compensation_norm_plotter import (
        compensation_norms,
        validate_preprocessed_pair,
    )
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


DEFAULT_OUTPUT = DEFAULT_ROOT / "door_artifact_amplitudes_min5s.csv"
DEFAULT_MIN_DURATION = 5.0
DATE_PATTERN = re.compile(r"_(?P<date>\d{8})(?:_|$)")
RUN_PATTERN = re.compile(r"_\d{8}_(?P<run>\d+)(?:\D|$)")
COLUMNS = (
    "date",
    "dataset name",
    "run number",
    "epoch number",
    "max amplitude in the event window in 0 order correction (pT)",
    "amplitude of the 3rd order compensated data at the same time (pT)",
    "decibel difference between the two amplitudes (dB)",
)


def dataset_date(dataset: Path) -> str:
    """Return the recording date as an ISO-formatted string."""
    match = DATE_PATTERN.search(Path(dataset).stem)
    if match is None:
        raise ValueError(f"could not find YYYYMMDD date in {Path(dataset).name}")
    value = match.group("date")
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def dataset_run(dataset: Path) -> int:
    """Return the numeric run following the recording date."""
    match = RUN_PATTERN.search(Path(dataset).stem)
    if match is None:
        raise ValueError(f"could not find run number in {Path(dataset).name}")
    return int(match.group("run"))


def amplitude_values(
    data_grade0_t: np.ndarray,
    data_grade3_t: np.ndarray,
) -> tuple[float, float, float]:
    """Return peak grade-0 norm, simultaneous grade-3 norm, and dB ratio."""
    norm_grade0_pt, norm_grade3_pt, _ = compensation_norms(
        data_grade0_t,
        data_grade3_t,
    )
    peak_index = int(np.argmax(norm_grade0_pt))
    amplitude_grade0 = float(norm_grade0_pt[peak_index])
    amplitude_grade3 = float(norm_grade3_pt[peak_index])
    difference_db = float(20.0 * np.log10(amplitude_grade0 / amplitude_grade3))
    if not np.isfinite(difference_db):
        raise ValueError("amplitude difference in dB is not finite")
    return amplitude_grade0, amplitude_grade3, difference_db


def processing_config(grade: int, args: argparse.Namespace) -> ProcessingConfig:
    """Build the shared preprocessing configuration for one compensation grade."""
    return ProcessingConfig(
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=grade,
        remove_dc=True,
        n_jobs=args.n_jobs,
    )


def dataset_rows(
    dataset: Path,
    annotation_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Calculate table rows for qualifying annotations in one dataset."""
    annotations = mne.read_annotations(annotation_path)
    qualifying = [
        index
        for index, duration in enumerate(annotations.duration)
        if float(duration) >= args.min_duration
    ]
    if not qualifying:
        return []

    raw_grade0, data_grade0 = preprocess_dataset(
        dataset,
        processing_config(0, args),
    )
    raw_grade3, data_grade3 = preprocess_dataset(
        dataset,
        processing_config(3, args),
    )
    validate_preprocessed_pair(raw_grade0, data_grade0, raw_grade3, data_grade3)

    rows = []
    for index in qualifying:
        window = episode_window(
            raw_grade0,
            onset=float(annotations.onset[index]),
            duration=float(annotations.duration[index]),
            description=str(annotations.description[index]),
        )
        selection = slice(window.start_index, window.stop_index)
        amplitude_grade0, amplitude_grade3, difference_db = amplitude_values(
            data_grade0[:, selection],
            data_grade3[:, selection],
        )
        rows.append(
            {
                COLUMNS[0]: dataset_date(dataset),
                COLUMNS[1]: dataset.name,
                COLUMNS[2]: dataset_run(dataset),
                COLUMNS[3]: index + 1,
                COLUMNS[4]: amplitude_grade0,
                COLUMNS[5]: amplitude_grade3,
                COLUMNS[6]: difference_db,
            }
        )
    return rows


def _dataset_rows_job(
    job: tuple[Path, Path, argparse.Namespace],
) -> tuple[Path, list[dict[str, object]]]:
    """Process one dataset in a worker process."""
    dataset, annotation_path, args = job
    return dataset, dataset_rows(dataset, annotation_path, args)


def write_table(rows: list[dict[str, object]], output: Path) -> None:
    """Write sorted rows using the stable public column order."""
    table = pd.DataFrame(rows, columns=COLUMNS)
    table.sort_values(
        [COLUMNS[0], COLUMNS[2], COLUMNS[3]],
        kind="stable",
        inplace=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--resample", type=float, default=300.0)
    parser.add_argument("--highpass", type=float, default=0.1)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--dataset-jobs",
        type=int,
        default=1,
        help="Number of datasets to process concurrently (default: 1)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_duration <= 0:
        raise ValueError("min-duration must be positive")
    if args.dataset_jobs <= 0:
        raise ValueError("dataset-jobs must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    mne.set_log_level("WARNING")

    datasets = discover_emptyroom_datasets(args.root)
    jobs = []
    for dataset in datasets:
        annotation_path = annotation_path_for_dataset(dataset, args.annotation_dir)
        if annotation_path is None:
            continue
        annotations = mne.read_annotations(annotation_path)
        if any(float(value) >= args.min_duration for value in annotations.duration):
            jobs.append((dataset, annotation_path, args))

    rows = []
    checkpoint = args.output.with_suffix(".partial.csv")
    if args.dataset_jobs == 1:
        results = map(_dataset_rows_job, jobs)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.dataset_jobs,
        )
        results = executor.map(_dataset_rows_job, jobs)
    try:
        for number, (dataset, current) in enumerate(results, start=1):
            rows.extend(current)
            write_table(rows, checkpoint)
            print(
                f"[{number:03d}/{len(jobs):03d}] {len(current)} events  {dataset}",
                flush=True,
            )
            gc.collect()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    write_table(rows, args.output)
    checkpoint.unlink(missing_ok=True)
    print(
        f"Wrote {len(rows)} events from {len(jobs)} datasets to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
