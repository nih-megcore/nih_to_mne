"""Shared runtime helpers for NIH MEG processing pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

import mne_bids
import numpy as np


__all__ = [
    "DerivativeProvenance",
    "ProvenanceContext",
    "check_covariance",
    "ensure_dataset_description",
    "git_blob_hash",
    "load_or_make",
    "make_logger",
    "normalize_beamfilt_ori",
    "provenance_sidecar_path",
    "save_filtered",
    "save_if_requested",
    "slurm_context",
]


_PROVENANCE_KEY = "NIH2MNEProvenance"


def _package_versions() -> Mapping[str, str]:
    packages = {}
    for distribution in ("mne", "mne-bids", "nih2mne"):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "unknown"
    return MappingProxyType(packages)


@dataclass(frozen=True)
class ProvenanceContext:
    """Run-level information shared by derivative provenance records."""

    bids_root: Path
    script_filename: str
    script_hash: str
    packages: Mapping[str, str] = field(default_factory=_package_versions)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bids_root",
            Path(self.bids_root).expanduser().resolve(strict=False),
        )
        object.__setattr__(self, "script_filename", Path(self.script_filename).name)
        object.__setattr__(self, "packages", MappingProxyType(dict(self.packages)))

    @classmethod
    def from_script(
        cls,
        bids_root: str | Path,
        script: str | Path,
        *,
        script_hash: str | None = None,
    ) -> "ProvenanceContext":
        """Create a portable context without retaining the script's location."""

        script_path = Path(script)
        return cls(
            bids_root=Path(bids_root).expanduser().resolve(strict=False),
            script_filename=script_path.name,
            script_hash=script_hash or git_blob_hash(script_path),
        )

    def derivative(
        self,
        *,
        calls: Sequence[str],
        inputs: Mapping[str, str | Path],
    ) -> "DerivativeProvenance":
        """Describe the resolved calls and input paths for one derivative."""

        return DerivativeProvenance(self, tuple(calls), dict(inputs))

    def portable_path(self, path: str | Path) -> str:
        """Represent a path using the literal ``{bids_root}`` placeholder."""

        return _portable_bids_path(path, self.bids_root)


@dataclass(frozen=True)
class DerivativeProvenance:
    """Artifact-specific processing calls and their file inputs."""

    context: ProvenanceContext
    calls: tuple[str, ...]
    inputs: Mapping[str, str | Path]


def _portable_bids_path(path: str | Path, bids_root: Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = bids_root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(bids_root)
    except ValueError as error:
        raise ValueError(
            f"provenance input is outside the BIDS root {bids_root}: {resolved}"
        ) from error
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"provenance input is not portable: {path}")
    if relative == Path("."):
        return "{bids_root}"
    return f"{{bids_root}}/{relative.as_posix()}"


def _portable_inputs(provenance: DerivativeProvenance) -> dict[str, str]:
    return {
        name: _portable_bids_path(path, provenance.context.bids_root)
        for name, path in provenance.inputs.items()
    }


def _provenance_payload(
    provenance: DerivativeProvenance,
    *,
    portable_inputs: Mapping[str, str] | None = None,
) -> dict:
    context = provenance.context
    return {
        "ProcessingDateTime": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "Script": {
            "Filename": context.script_filename,
            "Hash": context.script_hash,
            "HashType": "git-blob-sha1",
        },
        "Calls": list(provenance.calls),
        "Inputs": dict(portable_inputs or _portable_inputs(provenance)),
        "Packages": dict(context.packages),
    }


def provenance_sidecar_path(record) -> Path:
    """Return the JSON sidecar path for a non-JSON derivative record."""

    if record.key in {"raw", "noise_raw"}:
        raise ValueError("provenance sidecars cannot be written for raw BIDS inputs")
    path = Path(record.path)
    if path.suffix.lower() == ".json":
        raise ValueError(f"artifact {record.key!r} is already JSON metadata")
    if path.name.lower().endswith(".fif.gz"):
        return path.with_name(f"{path.name[:-7]}.json")
    if path.suffix:
        return path.with_suffix(".json")
    return path.with_name(f"{path.name}.json")


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read derivative JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"derivative JSON must contain an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_provenance(
    record,
    provenance: DerivativeProvenance,
    logger: logging.Logger,
    *,
    json_path: Path | None = None,
    base_metadata: Mapping | None = None,
    portable_inputs: Mapping[str, str] | None = None,
    repair_invalid: bool = False,
) -> Path:
    path = json_path or provenance_sidecar_path(record)
    try:
        metadata = _read_json_object(path)
    except RuntimeError:
        if not repair_invalid:
            raise
        metadata = {}
    if base_metadata is not None:
        metadata.update(base_metadata)
    metadata[_PROVENANCE_KEY] = _provenance_payload(
        provenance, portable_inputs=portable_inputs
    )
    _atomic_write_json(path, metadata)
    logger.info("Saved %s provenance: %s", record.key, path)
    return path


def git_blob_hash(path: str | Path) -> str:
    """Return Git's blob hash for a processing script."""

    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def slurm_context() -> tuple[int, str | None]:
    """Return the worker count and SLURM output path, when applicable."""

    job_id = os.getenv("SLURM_JOB_ID")
    if not job_id:
        return -1, None
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id], capture_output=True, text=True
        )
    except OSError:
        return int(os.getenv("SLURM_CPUS_PER_TASK", "1")), None
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


def load_or_make(
    record,
    make,
    save,
    logger,
    *,
    overwrite=False,
    provenance: DerivativeProvenance | None = None,
):
    """Load a valid derivative artifact, or create and save a new one.

    Parameters
    ----------
    record : ArtifactRecord
        Description of the expected derivative, normally obtained from
        ``MEGDerivativeTemplate.evaluate()`` or
        ``MEGDerivativeTemplate.artifact()``. The record supplies the output
        ``path``, artifact ``key``, validation ``status`` and ``load_file()``
        method used here. A valid existing record is loaded unless
        ``overwrite`` is true; an invalid record raises rather than being
        silently replaced.
    make : callable
        Zero-argument function that constructs the artifact in memory. It is
        called as ``value = make()`` only when a new result is required. For
        example, this can be ``lambda: mne.make_forward_solution(...)``.
    save : callable
        Function responsible for writing the newly constructed value. It is
        called as ``save(value, record.path)`` after the destination directory
        has been created. This wrapper accommodates the different save APIs
        used by MNE objects, for example
        ``lambda value, path: value.save(path, overwrite=True)``.
    logger : logging.Logger
        Logger used to record whether the artifact was loaded or created and
        to capture a traceback if creation or saving fails.
    overwrite : bool, keyword-only
        If false (default), load a valid existing artifact and reject an
        invalid one. If true, always call ``make`` and ``save`` and replace any
        existing output according to the behavior of the supplied ``save``
        callable.
    provenance : DerivativeProvenance | None, keyword-only
        Portable processing metadata to write beside a newly created or
        overwritten derivative. Reused artifacts are left unchanged.

    Returns
    -------
    object
        The object returned by ``record.load_file()`` for a reused artifact,
        or the object returned by ``make()`` for a newly created artifact.

    Raises
    ------
    RuntimeError
        If a non-overwritten artifact is invalid.
    Exception
        Any exception raised while loading, constructing, saving, or recording
        provenance for the artifact is written to ``logger`` with its full
        traceback and then re-raised unchanged.
    """

    if record.exists and not overwrite:
        logger.info("Using existing %s: %s", record.key, record.path)
        try:
            return record.load_file()
        except Exception:
            logger.exception("Failed to load %s: %s", record.key, record.path)
            raise
    if record.status == "invalid" and not overwrite:
        raise RuntimeError(
            f"Invalid {record.key} artifact at {record.path}: "
            f"{'; '.join(record.validation_errors)}"
        )
    logger.info("Creating %s: %s", record.key, record.path)
    try:
        portable_inputs = (
            _portable_inputs(provenance) if provenance is not None else None
        )
        value = make()
        record.path.parent.mkdir(parents=True, exist_ok=True)
        save(value, record.path)
        if provenance is not None:
            _write_provenance(
                record,
                provenance,
                logger,
                portable_inputs=portable_inputs,
                repair_invalid=True,
            )
        return value
    except Exception:
        logger.exception("Failed to create %s: %s", record.key, record.path)
        raise


def save_filtered(
    raw,
    data_record,
    json_record,
    overwrite,
    logger,
    *,
    provenance: DerivativeProvenance | None = None,
):
    """Save filtered data and its derivative JSON metadata.

    When ``provenance`` is supplied, it is merged into the filter JSON only if
    the filtered data are created or overwritten. Existing cached derivatives
    are not backfilled or modified.
    """

    invalid_records = [
        record for record in (data_record, json_record) if record.status == "invalid"
    ]
    if invalid_records and not overwrite:
        details = "; ".join(
            f"{record.key}: {', '.join(record.validation_errors)}"
            for record in invalid_records
        )
        raise RuntimeError(f"Invalid filtered artifact pair: {details}")

    data_record.path.parent.mkdir(parents=True, exist_ok=True)
    save_data = overwrite or not data_record.path.exists()
    portable_inputs = (
        _portable_inputs(provenance)
        if save_data and provenance is not None
        else None
    )
    if save_data:
        raw.save(data_record.path, overwrite=overwrite)
        logger.info("Saved filtered raw data: %s", data_record.path)
    else:
        logger.info("Keeping existing filtered raw data: %s", data_record.path)

    if overwrite or not json_record.path.exists() or portable_inputs is not None:
        if provenance is not None and portable_inputs is not None:
            _write_provenance(
                data_record,
                provenance,
                logger,
                json_path=Path(json_record.path),
                base_metadata=json_record.expected_metadata,
                portable_inputs=portable_inputs,
                repair_invalid=overwrite,
            )
        else:
            _atomic_write_json(
                Path(json_record.path), json_record.expected_metadata
            )
            logger.info("Saved filter metadata: %s", json_record.path)


def save_if_requested(
    value,
    record,
    save,
    overwrite,
    logger,
    *,
    provenance: DerivativeProvenance | None = None,
):
    """Save a result and optional provenance unless preserving existing output.

    The provenance sidecar is written only after ``save(value, record.path)``
    succeeds. A valid cached output remains untouched when overwrite is false.
    Save and provenance exceptions are logged with their full tracebacks and
    re-raised unchanged.
    """

    if record.exists and not overwrite:
        logger.info("Keeping existing %s: %s", record.key, record.path)
        return
    if record.status == "invalid" and not overwrite:
        raise RuntimeError(
            f"Invalid {record.key} artifact at {record.path}: "
            f"{'; '.join(record.validation_errors)}"
        )
    try:
        record.path.parent.mkdir(parents=True, exist_ok=True)
        portable_inputs = (
            _portable_inputs(provenance) if provenance is not None else None
        )
        save(value, record.path)
        if provenance is not None:
            _write_provenance(
                record,
                provenance,
                logger,
                portable_inputs=portable_inputs,
                repair_invalid=True,
            )
        logger.info("Saved %s: %s", record.key, record.path)
    except Exception:
        logger.exception("Failed to save %s: %s", record.key, record.path)
        raise


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
    record,
    name: str,
    logger,
    *,
    generated_by: str,
    generated_by_version: str | None = None,
    overwrite: bool = False,
) -> None:
    """Create a derivative dataset description when one is missing."""

    if record.exists and not overwrite:
        return
    if record.status == "invalid" and not overwrite:
        raise RuntimeError(
            f"Invalid dataset description at {record.path}: "
            f"{'; '.join(record.validation_errors)}"
        )
    record.path.parent.mkdir(parents=True, exist_ok=True)
    generator = {"Name": generated_by}
    if generated_by_version is not None:
        generator["Version"] = generated_by_version
    mne_bids.make_dataset_description(
        path=record.path.parent,
        name=name,
        dataset_type="derivative",
        generated_by=[generator],
        overwrite=overwrite,
    )
    logger.info("Created dataset description: %s", record.path)
