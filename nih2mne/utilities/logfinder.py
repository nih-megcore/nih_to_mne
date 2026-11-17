"""Find task logfiles and align their event times with MEG triggers."""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from glob import glob
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import yaml


_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")


def _identifier(value, label):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a nonempty identifier without path or glob characters")
    return value


def _string_list(value, label):
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a nonempty list of strings")
    return value


def logfind(task: str, subject: str) -> str:
    """Return the sole logfile matching a task's configured sources.

    ``subject`` is the MEG subject hash substituted for ``{subject_hash}`` in
    each pattern. Each source has ``directories`` and ``patterns`` lists and an
    optional ``recursive`` boolean. Missing, invalid, and ambiguous searches
    raise an exception instead of selecting an arbitrary file.
    """
    task = _identifier(task, "task")
    subject = _identifier(subject, "subject")
    config_path = Path.home() / "megcore" / "trigproc" / f"{task}_logfind.yml"
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid logfile configuration: {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Invalid logfile configuration: {config_path}")
    sources = config.get("log_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{config_path}: log_sources must be a nonempty list")

    matches = set()
    for index, source in enumerate(sources, start=1):
        label = f"{config_path}: log_sources[{index}]"
        if not isinstance(source, dict):
            raise ValueError(f"{label} must be a mapping")
        directories = _string_list(source.get("directories"), f"{label}.directories")
        patterns = _string_list(source.get("patterns"), f"{label}.patterns")
        recursive = source.get("recursive", False)
        if not isinstance(recursive, bool):
            raise ValueError(f"{label}.recursive must be a boolean")
        for directory in directories:
            root = Path(directory).expanduser()
            if not root.is_absolute():
                root = config_path.parent / root
            for pattern in patterns:
                try:
                    relative_pattern = pattern.format(subject_hash=subject)
                except (KeyError, ValueError, IndexError) as exc:
                    raise ValueError(f"{label}: invalid pattern {pattern!r}") from exc
                if Path(relative_pattern).is_absolute() or ".." in Path(relative_pattern).parts:
                    raise ValueError(f"{label}: pattern must stay within its directory")
                search = f"**/{relative_pattern}" if recursive else relative_pattern
                for filename in glob(str(root / search), recursive=recursive):
                    path = Path(filename)
                    if path.is_file():
                        matches.add(str(path.resolve()))

    if not matches:
        raise FileNotFoundError(
            f"No logfile found for task {task!r}, subject {subject!r} using {config_path}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple logfiles found for task {task!r}, subject {subject!r}: "
            + ", ".join(sorted(matches))
        )
    return matches.pop()


@dataclass
class TimeOffsetResult:
    """Log-to-MEG timing estimate and the events used to calculate it.

    ``offset_seconds`` is the applied constant offset for first/middle, or the
    median summary for local correction. ``corrected_log`` contains MEG times.
    ``drift_ms_per_minute`` is the change in that difference per minute of
    logfile time. A positive drift means the required offset grows over time.
    """

    offset_seconds: float
    drift_ms_per_minute: float
    matches: pd.DataFrame
    unmatched_log: pd.DataFrame
    unmatched_trigger: pd.DataFrame
    corrected_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    correction_method: str = "middle"


def _event_code(value):
    """Use the same representation for integral numeric codes on both sides."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
    code = str(value).strip()
    return code or None


def find_time_offset(
    logfile, meg_filename, log_code, onset_time, *, method="middle",
    time_columns=None, output_file=None, offset_error=0.020,
) -> TimeOffsetResult:
    """Compare CSV events with UPPT001 and estimate offset and linear drift.

    ``log_code`` and ``onset_time`` name CSV columns. Logfile times must be
    numeric seconds on one timeline. UPPT001 is read with ``detect_digital``.
    Equal codes are aligned in event order, allowing gaps on either side.
    The median of ``trigger_onset - log_onset`` is the single offset; ordinary
    least squares of that difference against elapsed log time estimates drift.

    ``matches`` contains source row indices, code, both onsets, the difference,
    and its residual after subtracting the single offset. Unmatched tables
    contain event rows that did not participate in the estimate. At least three
    matched events at distinct times are required.

    ``method`` selects the earliest matched offset (first), median offset
    (middle, default), or time interpolation between matched offsets (local).
    Local correction holds the nearest offset beyond the matched time range.
    Gaps are allowed, but repeated codes can make sequence alignment ambiguous.
    ``offset_error`` limits each trial's applied offset relative to the median,
    in seconds; violations raise ValueError listing all offending source rows.

    The returned ``corrected_log`` preserves all original rows and columns,
    correcting ``onset_time`` and .started/.stopped columns by default. Supply
    ``time_columns`` to select other columns (onset_time is always included).
    Durations and reaction times are not automatically selected. Missing times
    remain missing. ``output_file`` optionally writes this table as a new CSV;
    existing files, including the source, are never overwritten.
    """
    from nih2mne.utilities.trigger_utilities import detect_digital

    if method not in ("first", "middle", "local"):
        raise ValueError("method must be 'first', 'middle', or 'local'")
    try:
        offset_error = float(offset_error)
    except (TypeError, ValueError) as exc:
        raise ValueError("offset_error must be finite and nonnegative seconds") from exc
    if not np.isfinite(offset_error) or offset_error < 0:
        raise ValueError("offset_error must be finite and nonnegative seconds")
    if output_file is not None and Path(output_file).resolve() == Path(logfile).resolve():
        raise ValueError("output_file must differ from the source logfile")

    original = pd.read_csv(logfile)
    log = original
    missing = [column for column in (log_code, onset_time) if column not in log]
    if missing:
        raise ValueError(f"Missing logfile column(s): {', '.join(missing)}")

    if time_columns is None:
        time_columns = [c for c in original if c.endswith((".started", ".stopped"))]
    elif isinstance(time_columns, str):
        time_columns = [time_columns]
    columns = list(dict.fromkeys([onset_time, *time_columns]))
    numeric_times = {}
    for column in columns:
        if column not in original:
            raise ValueError(f"Missing logfile column: {column}")
        values = pd.to_numeric(original[column], errors="coerce")
        invalid = original[column].notna() & ~np.isfinite(values)
        if invalid.any():
            raise ValueError(f"{column!r} must contain finite numeric seconds or missing values")
        numeric_times[column] = values.to_numpy(dtype=float)

    log = log[[log_code, onset_time]].copy()
    log["code"] = log[log_code].map(_event_code)
    log = log.loc[log["code"].notna()].copy()
    log["log_onset"] = pd.to_numeric(log[onset_time], errors="coerce")
    if not np.isfinite(log["log_onset"].to_numpy(dtype=float)).all():
        raise ValueError(f"{onset_time!r} must contain finite numeric seconds for coded events")
    log["log_row"] = log.index
    log = log.sort_values("log_onset", kind="stable").reset_index(drop=True)

    triggers = detect_digital(meg_filename, channel="UPPT001")
    if not {"onset", "condition"}.issubset(triggers.columns):
        raise ValueError("detect_digital must return onset and condition columns")
    triggers = triggers.copy()
    triggers["code"] = triggers["condition"].map(_event_code)
    triggers["trigger_onset"] = pd.to_numeric(triggers["onset"], errors="coerce")
    if triggers["code"].isna().any() or not np.isfinite(
        triggers["trigger_onset"].to_numpy(dtype=float)
    ).all():
        raise ValueError("UPPT001 events must have codes and finite numeric onsets")
    triggers["trigger_row"] = triggers.index
    triggers = triggers.sort_values("trigger_onset", kind="stable").reset_index(drop=True)

    matcher = SequenceMatcher(None, log["code"].tolist(), triggers["code"].tolist(), autojunk=False)
    pairs = [
        (log_index + step, trigger_index + step)
        for log_index, trigger_index, size in matcher.get_matching_blocks()
        for step in range(size)
    ]
    if len(pairs) < 3:
        raise ValueError("At least three matching UPPT001 and logfile events are required")

    log_indices, trigger_indices = zip(*pairs)
    matched_log = log.iloc[list(log_indices)]
    matched_trigger = triggers.iloc[list(trigger_indices)]
    log_times = matched_log["log_onset"].to_numpy(dtype=float)
    trigger_times = matched_trigger["trigger_onset"].to_numpy(dtype=float)
    if np.unique(log_times).size < 2:
        raise ValueError("Matched logfile events must span distinct times")
    differences = trigger_times - log_times
    median_offset = float(np.median(differences))
    offset = float(differences[0]) if method == "first" else median_offset
    elapsed_minutes = (log_times - log_times.min()) / 60
    drift = float(np.polyfit(elapsed_minutes, differences, 1)[0] * 1000)
    matches = pd.DataFrame({
        "log_row": matched_log["log_row"].to_numpy(),
        "trigger_row": matched_trigger["trigger_row"].to_numpy(),
        "code": matched_log["code"].to_numpy(),
        "log_onset": log_times,
        "trigger_onset": trigger_times,
        "difference_seconds": differences,
        "residual_seconds": differences - median_offset,
    })
    unmatched_log = log.drop(index=list(log_indices)).reset_index(drop=True)
    unmatched_trigger = triggers.drop(index=list(trigger_indices)).reset_index(drop=True)
    if method == "local":
        if not unmatched_log.empty or unmatched_trigger["code"].isin(log["code"]).any():
            warnings.warn(
                "Local correction has unmatched logfile or relevant UPPT001 events; "
                "inspect matches and unmatched tables for ambiguous alignment.",
                UserWarning, stacklevel=2,
            )
        anchors, indices = np.unique(log_times, return_index=True)
        anchor_offsets = differences[indices]
        if not np.allclose(differences, np.interp(log_times, anchors, anchor_offsets),
                           rtol=0, atol=1e-12):
            raise ValueError("Conflicting offsets at identical logfile anchor times")

    def offsets(times):
        if method == "local":
            return np.interp(times, anchors, anchor_offsets)
        return np.full_like(times, offset, dtype=float)

    trial_times = numeric_times[onset_time]
    trial_offsets = offsets(trial_times)
    deviations = np.abs(trial_offsets - median_offset)
    # Allow equality despite floating point roundoff in time subtraction.
    exceeded = np.isfinite(trial_times) & (deviations > offset_error) & ~np.isclose(
        deviations, offset_error, rtol=0, atol=1e-12,
    )
    if exceeded.any():
        details = [
            f"row {row}, code {original.iloc[row][log_code]!r}: "
            f"offset={trial_offsets[row]:.9f} s, deviation={deviations[row] * 1000:.6f} ms"
            for row in np.flatnonzero(exceeded)
        ]
        raise ValueError(
            f"Trial offsets exceed offset_error={offset_error:g} s relative to the "
            f"middle offset ({median_offset:.9f} s):\n" + "\n".join(details)
        )

    corrected = original.copy()
    for column, times in numeric_times.items():
        corrected[column] = times + offsets(times)
    matches["applied_offset_seconds"] = offsets(log_times)
    matches["corrected_residual_seconds"] = differences - offsets(log_times)
    if output_file is not None:
        corrected.to_csv(output_file, index=False, mode="x")
    return TimeOffsetResult(
        offset, drift, matches, unmatched_log, unmatched_trigger, corrected, method,
    )
