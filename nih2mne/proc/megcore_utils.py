"""Shared runtime helpers for NIH MEG processing pipelines."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess

import mne_bids
import numpy as np


__all__ = [
    "check_covariance",
    "ensure_dataset_description",
    "git_blob_hash",
    "load_or_make",
    "make_logger",
    "normalize_beamfilt_ori",
    "save_filtered",
    "save_if_requested",
    "slurm_context",
]


def git_blob_hash(path: str | Path) -> str:
    """Return Git's blob hash for a processing script."""

    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def slurm_context() -> tuple[int, str | None]:
    """Return the worker count and SLURM output path, when applicable."""

    job_id = os.getenv("SLURM_JOB_ID")
    if not job_id:
        return -1, None
    result = subprocess.run(
        ["scontrol", "show", "job", job_id], capture_output=True, text=True
    )
    stdout = next(
        (
            token.removeprefix("StdOut=")
            for token in result.stdout.split()
            if token.startswith("StdOut=")
        ),
        None,
    )
    return int(os.getenv("SLURM_CPUS_PER_TASK", "1")), stdout


def make_logger(path: Path, *, logger_name: str) -> logging.Logger:
    """Create a file logger with an identity supplied by the calling routine."""

    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_or_make(record, make, save, logger):
    """Load an existing template artifact or create and save it."""

    if record.exists:
        logger.info("Using existing %s: %s", record.key, record.path)
        return record.load_file()
    if record.path.exists():
        raise RuntimeError(
            f"Invalid {record.key} artifact at {record.path}: "
            f"{'; '.join(record.validation_errors)}"
        )
    logger.info("Creating %s: %s", record.key, record.path)
    try:
        value = make()
        record.path.parent.mkdir(parents=True, exist_ok=True)
        save(value, record.path)
        return value
    except Exception as error:
        logger.exception("Failed to create %s: %s", record.key, record.path)
        raise RuntimeError(f"Failed to create {record.key}: {record.path}") from error


def save_filtered(raw, data_record, json_record, overwrite, logger):
    """Save filtered data and the filter recipe sidecar."""

    data_record.path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not data_record.path.exists():
        raw.save(data_record.path, overwrite=overwrite)
        logger.info("Saved filtered raw data: %s", data_record.path)
    else:
        logger.info("Keeping existing filtered raw data: %s", data_record.path)

    if overwrite or not json_record.path.exists():
        json_record.path.write_text(
            json.dumps(json_record.expected_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved filter metadata: %s", json_record.path)
    elif json_record.status == "invalid":
        logger.warning("Keeping invalid filter metadata: %s", json_record.path)


def save_if_requested(value, record, save, overwrite, logger):
    """Save a result unless an existing output should be preserved."""

    record.path.parent.mkdir(parents=True, exist_ok=True)
    if record.exists and not overwrite:
        logger.info("Keeping existing %s: %s", record.key, record.path)
        return
    if record.status == "invalid" and not overwrite:
        raise RuntimeError(
            f"Invalid {record.key} artifact at {record.path}: "
            f"{'; '.join(record.validation_errors)}"
        )
    save(value, record.path)
    logger.info("Saved %s: %s", record.key, record.path)


def check_covariance(name: str, covariance, logger) -> None:
    """Log and validate a covariance matrix's condition number."""

    condition = np.linalg.cond(covariance.data)
    logger.info("%s covariance condition number: %s", name, condition)
    if condition >= 1e12:
        raise ValueError(f"{name} covariance is ill-conditioned")
    if condition >= 1e8:
        logger.warning("%s covariance is poorly conditioned", name)
    elif condition >= 1e6:
        logger.warning("%s covariance has moderate conditioning", name)


def normalize_beamfilt_ori(forward, filters) -> None:
    """Align max-power beamformer orientations to anatomical source normals."""

    src_anat_ori = np.vstack(
        [space["nn"][space["vertno"]] for space in forward["src"]]
    )
    ori_flip = np.sign(
        np.real(np.sum(filters["max_power_ori"] * src_anat_ori, axis=1))
    )
    filters["weights"] = np.real_if_close(
        filters["weights"] * ori_flip[:, np.newaxis]
    )
    filters["max_power_ori"] = np.real_if_close(
        filters["max_power_ori"] * ori_flip[:, np.newaxis]
    )


def ensure_dataset_description(
    record, name: str, logger, *, generated_by: str
) -> None:
    """Create a derivative dataset description when one is missing."""

    if record.exists:
        return
    if record.path.exists():
        logger.warning("Keeping invalid dataset description: %s", record.path)
        return
    record.path.parent.mkdir(parents=True, exist_ok=True)
    mne_bids.make_dataset_description(
        path=record.path.parent,
        name=name,
        dataset_type="derivative",
        generated_by=[{"Name": generated_by}],
    )
