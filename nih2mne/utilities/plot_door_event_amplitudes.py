#!/usr/bin/env python3
"""Plot grade-0 and grade-3 door-event amplitudes over recording date."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from .door_event_amplitude_table import COLUMNS, DEFAULT_OUTPUT as DEFAULT_CSV
else:
    from door_event_amplitude_table import COLUMNS, DEFAULT_OUTPUT as DEFAULT_CSV


DEFAULT_OUTPUT = DEFAULT_CSV.with_name("door_artifact_amplitudes_over_time.png")


def exclude_events(table: pd.DataFrame, specifications: list[str]) -> pd.DataFrame:
    """Remove explicitly named ``DATASET:EPOCH`` rows from a table."""
    filtered = table.copy()
    for specification in specifications:
        try:
            dataset, epoch_text = specification.rsplit(":", 1)
            epoch = int(epoch_text)
        except (ValueError, TypeError):
            raise ValueError(
                f"invalid exclusion {specification!r}; expected DATASET:EPOCH"
            ) from None
        selected = (filtered[COLUMNS[1]] == dataset) & (
            filtered[COLUMNS[3]] == epoch
        )
        count = int(selected.sum())
        if count != 1:
            raise ValueError(
                f"exclusion {specification!r} matched {count} rows; expected one"
            )
        filtered = filtered.loc[~selected].copy()
    return filtered


def plot_amplitudes(
    table: pd.DataFrame,
    output: Path,
    *,
    dpi: int = 150,
    size_inches: float = 6.0,
) -> None:
    """Write a two-row, date-indexed amplitude plot."""
    missing = [column for column in COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if table.empty:
        raise ValueError("CSV contains no events")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if size_inches <= 0:
        raise ValueError("size-inches must be positive")

    table = table.copy()
    table[COLUMNS[0]] = pd.to_datetime(table[COLUMNS[0]], errors="raise")
    table.sort_values([COLUMNS[0], COLUMNS[2], COLUMNS[3]], inplace=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(size_inches, size_inches),
        sharex=True,
        layout="constrained",
    )
    series = (
        (COLUMNS[4], "0-order amplitude", "tab:blue"),
        (COLUMNS[5], "3rd-order compensated amplitude", "tab:orange"),
    )
    for axis, (column, title, color) in zip(axes, series, strict=True):
        axis.plot(
            table[COLUMNS[0]],
            table[column],
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            color=color,
        )
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("L2 norm (pT)", fontsize=9)
        axis.tick_params(axis="both", labelsize=8)
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Recording date", fontsize=9)
    fig.suptitle("Door-artifact amplitudes (duration ≥ 5 s)", fontsize=12)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--size-inches", type=float, default=6.0)
    parser.add_argument(
        "--exclude-event",
        action="append",
        default=[],
        metavar="DATASET:EPOCH",
        help="Exclude one dataset/epoch row; may be supplied more than once",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    table = pd.read_csv(args.csv)
    table = exclude_events(table, args.exclude_event)
    plot_amplitudes(
        table,
        args.output,
        dpi=args.dpi,
        size_inches=args.size_inches,
    )
    print(
        f"Wrote {len(table)} events to {args.output} "
        f"(excluded {len(args.exclude_event)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
