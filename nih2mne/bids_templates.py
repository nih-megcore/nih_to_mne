"""Schema-driven paths for MEG raw data and MNE derivatives.

This module only describes and inspects artifacts.  It does not create
directories, filter data, or write derivative files.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from mne_bids import BIDSPath


ArtifactStatus = Literal["present", "missing", "invalid"]
ArtifactKind = Literal["file", "directory", "raw", "json"]
Loader = str | Callable[..., Any] | None

_PROJECT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_LABEL_PATTERN = re.compile(r"[A-Za-z0-9]+")
_NOISE_ENTITY_KEYS = frozenset({"task", "acquisition", "run", "recording"})


def _frequency(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _frequency_token(value: float) -> str:
    """Render a positive frequency as a BIDS-label-safe token."""

    text = format(value, ".12g").lower()
    return text.replace(".", "p").replace("+", "").replace("-", "m")


@dataclass(frozen=True)
class FilterSpec:
    """Parameters that identify one filtered derivative stream.

    The frequency-changing parameters are encoded in :attr:`processing_label`.
    Other MNE parameters are retained in the expected JSON metadata.  Use
    ``variant`` when two algorithms with the same frequencies must coexist.
    """

    l_freq: float | None = None
    h_freq: float | None = None
    notch_freqs: Sequence[float] = ()
    resample_sfreq: float | None = None
    method: str | None = None
    phase: str | None = None
    filter_length: str | int | None = None
    l_trans_bandwidth: str | float | None = None
    h_trans_bandwidth: str | float | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        l_freq = _frequency(self.l_freq, "l_freq")
        h_freq = _frequency(self.h_freq, "h_freq")
        if l_freq is not None and h_freq is not None and l_freq >= h_freq:
            raise ValueError("l_freq must be less than h_freq")

        try:
            normalized_notch = []
            for value in self.notch_freqs:
                normalized = _frequency(value, "notch_freqs")
                if normalized is None:
                    raise TypeError("notch_freqs cannot contain None")
                normalized_notch.append(normalized)
            notch = tuple(sorted(set(normalized_notch)))
        except TypeError as error:
            if "not iterable" in str(error):
                raise TypeError("notch_freqs must be a sequence of frequencies") from error
            raise
        resample = _frequency(self.resample_sfreq, "resample_sfreq")

        if all(value is None for value in (l_freq, h_freq, resample)) and not notch:
            raise ValueError("FilterSpec must describe filtering or resampling")
        if self.variant is not None and not _LABEL_PATTERN.fullmatch(self.variant):
            raise ValueError("variant must contain only letters and numbers")
        for name in ("method", "phase"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")

        object.__setattr__(self, "l_freq", l_freq)
        object.__setattr__(self, "h_freq", h_freq)
        object.__setattr__(self, "notch_freqs", notch)
        object.__setattr__(self, "resample_sfreq", resample)

    @property
    def processing_label(self) -> str:
        """Return the alphanumeric value used by the ``proc`` entity."""

        label = "filt"
        if self.l_freq is not None or self.h_freq is not None:
            low = _frequency_token(self.l_freq) if self.l_freq is not None else "0"
            high = _frequency_token(self.h_freq) if self.h_freq is not None else "inf"
            label += f"{low}to{high}"
        if self.notch_freqs:
            label += "n" + "x".join(_frequency_token(v) for v in self.notch_freqs)
        if self.resample_sfreq is not None:
            label += "r" + _frequency_token(self.resample_sfreq)
        if self.variant:
            label += self.variant
        return label

    def software_filter_metadata(self) -> dict[str, dict[str, Any]]:
        """Return the BIDS ``SoftwareFilters`` value for this recipe."""

        parameters: dict[str, Any] = {}
        optional = {
            "HighpassCutoffHz": self.l_freq,
            "LowpassCutoffHz": self.h_freq,
            "NotchFrequenciesHz": list(self.notch_freqs) or None,
            "ResampleFrequencyHz": self.resample_sfreq,
            "Method": self.method,
            "Phase": self.phase,
            "FilterLength": self.filter_length,
            "HighpassTransitionBandwidth": self.l_trans_bandwidth,
            "LowpassTransitionBandwidth": self.h_trans_bandwidth,
        }
        parameters.update({key: value for key, value in optional.items() if value is not None})
        return {"MNE-Python": parameters}


@dataclass(frozen=True)
class ArtifactSpec:
    """Declarative definition of one artifact in a derivative template."""

    key: str
    pipeline: Literal["raw", "preprocessing", "project"]
    source: Literal["primary", "noise", "shared"]
    kind: ArtifactKind
    suffix: str | None = None
    extension: str | None = None
    entities: tuple[str, ...] = ()
    uses_filter: bool = False
    loader: Loader = None
    location: Literal["input", "dataset", "meg", "nested"] = "meg"
    parent_key: str | None = None
    metadata_for: str | None = None

    def __post_init__(self) -> None:
        if not self.key or not re.fullmatch(r"[a-z][a-z0-9_]*", self.key):
            raise ValueError("artifact keys must be lowercase snake_case")
        if self.location in {"meg", "nested"} and not self.suffix:
            raise ValueError(f"{self.key} requires a suffix")
        if self.location == "nested" and not self.parent_key:
            raise ValueError(f"{self.key} requires parent_key")


@dataclass(frozen=True)
class ArtifactRecord:
    """Evaluated path, status, metadata, and loader for one artifact."""

    key: str
    path: Path
    kind: ArtifactKind
    status: ArtifactStatus
    validation_errors: tuple[str, ...] = ()
    expected_metadata: Mapping[str, Any] | None = None
    _loader: Loader = field(default=None, repr=False, compare=False)
    _load_target: Any = field(default=None, repr=False, compare=False)

    @property
    def exists(self) -> bool:
        """Whether an object of the expected filesystem type exists."""

        return self.status != "missing"

    def load_file(self, **kwargs: Any) -> Any:
        """Load this artifact using its configured MNE or JSON reader."""

        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self._loader is None:
            raise TypeError(f"artifact '{self.key}' does not have a file loader")
        target = self._load_target if self._load_target is not None else self.path
        if callable(self._loader):
            return self._loader(target, **kwargs)
        return _run_loader(self._loader, target, kwargs)


def _run_loader(loader: str, target: Any, kwargs: dict[str, Any]) -> Any:
    """Import heavy readers only when an artifact is loaded."""

    if loader == "json":
        encoding = kwargs.pop("encoding", "utf-8")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected JSON loader arguments: {unexpected}")
        return json.loads(Path(target).read_text(encoding=encoding))

    if loader == "raw_bids":
        from mne_bids import read_raw_bids

        return read_raw_bids(target, **kwargs)

    import mne

    if loader == "raw":
        generic_reader = getattr(mne.io, "read_raw", None)
        if generic_reader is not None:
            return generic_reader(target, **kwargs)
        name = Path(target).name.lower()
        if name.endswith((".fif", ".fif.gz")):
            return mne.io.read_raw_fif(target, **kwargs)
        if name.endswith(".ds"):
            return mne.io.read_raw_ctf(target, **kwargs)
        if name.endswith((".con", ".sqd")):
            return mne.io.read_raw_kit(target, **kwargs)
        if name.endswith(".pdf"):
            return mne.io.read_raw_bti(target, **kwargs)
        raise ValueError(f"cannot select an MNE raw reader for: {target}")

    readers: dict[str, Callable[..., Any]] = {
        "raw_fif": mne.io.read_raw_fif,
        "epochs": mne.read_epochs,
        "covariance": mne.read_cov,
        "bem": mne.read_bem_solution,
        "source_space": mne.read_source_spaces,
        "transform": mne.read_trans,
        "forward": mne.read_forward_solution,
        "ica": mne.preprocessing.read_ica,
        "beamformer": mne.beamformer.read_beamformer,
    }
    try:
        reader = readers[loader]
    except KeyError as error:
        raise RuntimeError(f"unknown artifact loader: {loader}") from error
    return reader(target, **kwargs)


def _project_name(project: str) -> str:
    if not isinstance(project, str) or not _PROJECT_PATTERN.fullmatch(project):
        raise ValueError(
            "project must be one directory name containing only letters, numbers, "
            "'.', '_', or '-'"
        )
    if project in {".", ".."}:
        raise ValueError("project cannot be '.' or '..'")
    return project


def _bids_source(path: Path, root: Path) -> str:
    try:
        return f"bids::{path.relative_to(root).as_posix()}"
    except ValueError:
        return path.as_posix()


def _metadata_errors(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    """Compare expected metadata as a recursive subset of actual metadata."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{prefix or 'metadata'} must be an object"]
        errors: list[str] = []
        for key, expected_value in expected.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if key not in actual:
                errors.append(f"missing {location}")
            else:
                errors.extend(_metadata_errors(actual[key], expected_value, location))
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{prefix} must be a list"]
        if prefix == "Sources":
            return [
                f"{prefix} does not contain {value!r}"
                for value in expected
                if value not in actual
            ]
        if actual != expected:
            return [f"{prefix} does not match the filter recipe"]
        return []
    if isinstance(expected, Real) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, Real):
            return [f"{prefix} must be numeric"]
        if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
            return [f"{prefix} does not match the filter recipe"]
        return []
    if actual != expected:
        return [f"{prefix} does not match the filter recipe"]
    return []


class MEGDerivativeTemplate:
    """Generate and evaluate paths for one primary MEG derivative stream."""

    ARTIFACT_SPECS = (
        ArtifactSpec("raw", "raw", "primary", "raw", loader="raw_bids", location="input"),
        ArtifactSpec("noise_raw", "raw", "noise", "raw", loader="raw", location="input"),
        ArtifactSpec(
            "preprocessing_dataset_description",
            "preprocessing",
            "shared",
            "json",
            loader="json",
            location="dataset",
        ),
        ArtifactSpec(
            "project_dataset_description",
            "project",
            "shared",
            "json",
            loader="json",
            location="dataset",
        ),
        ArtifactSpec(
            "bem", "preprocessing", "shared", "file", "bem", ".fif",
            ("subject", "session"), loader="bem",
        ),
        ArtifactSpec(
            "src", "preprocessing", "shared", "file", "src", ".fif",
            ("subject", "session", "space"), loader="source_space",
        ),
        ArtifactSpec(
            "trans", "preprocessing", "primary", "file", "trans", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            loader="transform",
        ),
        ArtifactSpec(
            "fwd", "preprocessing", "primary", "file", "fwd", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording", "space"),
            loader="forward",
        ),
        ArtifactSpec(
            "filtered_raw", "project", "primary", "file", "meg", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="raw_fif",
        ),
        ArtifactSpec(
            "filtered_raw_json", "project", "primary", "json", "meg", ".json",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="json", metadata_for="filtered_raw",
        ),
        ArtifactSpec(
            "epochs", "project", "primary", "file", "epo", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="epochs",
        ),
        ArtifactSpec(
            "covariance", "project", "primary", "file", "cov", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="covariance",
        ),
        ArtifactSpec(
            "ica_dir", "project", "primary", "directory", "ica", None,
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True,
        ),
        ArtifactSpec(
            "ica", "project", "primary", "file", "ica", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="ica", location="nested", parent_key="ica_dir",
        ),
        ArtifactSpec(
            "lcmv", "project", "primary", "file", "lcmv", ".h5",
            ("subject", "session", "task", "acquisition", "run", "recording", "space"),
            uses_filter=True, loader="beamformer",
        ),
        ArtifactSpec(
            "noise_filtered_raw", "project", "noise", "file", "meg", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="raw_fif",
        ),
        ArtifactSpec(
            "noise_filtered_raw_json", "project", "noise", "json", "meg", ".json",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="json", metadata_for="noise_filtered_raw",
        ),
        ArtifactSpec(
            "noise_epochs", "project", "noise", "file", "epo", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="epochs",
        ),
        ArtifactSpec(
            "noise_covariance", "project", "noise", "file", "cov", ".fif",
            ("subject", "session", "task", "acquisition", "run", "recording"),
            uses_filter=True, loader="covariance",
        ),
    )

    def __init__(
        self,
        primary: BIDSPath,
        *,
        project: str,
        filter_spec: FilterSpec,
        noise: BIDSPath | str | Path | None = None,
        noise_entities: Mapping[str, str | int | None] | None = None,
        space: str | None = None,
    ) -> None:
        if not isinstance(primary, BIDSPath):
            raise TypeError("primary must be an mne_bids.BIDSPath")
        self._validate_raw_bids_path(primary, name="primary")
        if not isinstance(filter_spec, FilterSpec):
            raise TypeError("filter_spec must be a FilterSpec")
        self.primary = primary.copy()
        self.project = _project_name(project)
        self.filter_spec = filter_spec
        self.space = space
        self.noise = noise.copy() if isinstance(noise, BIDSPath) else noise

        if isinstance(noise, BIDSPath):
            self._validate_raw_bids_path(noise, name="noise", require_task=False)
            if noise_entities is not None:
                raise ValueError("noise_entities cannot be used with a noise BIDSPath")
            self.noise_entities = {
                key: getattr(noise, key) for key in _NOISE_ENTITY_KEYS
            }
            if self.noise_entities["task"] is None:
                self.noise_entities["task"] = "noise"
        elif noise is not None:
            provided = dict(noise_entities or {})
            unknown = set(provided) - _NOISE_ENTITY_KEYS
            if unknown:
                raise ValueError(f"unsupported noise entities: {sorted(unknown)}")
            self.noise_entities = {key: provided.get(key) for key in _NOISE_ENTITY_KEYS}
            if self.noise_entities["task"] is None:
                self.noise_entities["task"] = "noise"
            self.noise = Path(noise).expanduser().resolve(strict=False)
        else:
            if noise_entities is not None:
                raise ValueError("noise_entities requires a noise input")
            self.noise_entities = {}

        keys = [spec.key for spec in self.ARTIFACT_SPECS]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate artifact keys: {duplicates}")

    @staticmethod
    def _validate_raw_bids_path(
        bids_path: BIDSPath, *, name: str, require_task: bool = True
    ) -> None:
        missing = [field for field in ("root", "subject") if getattr(bids_path, field) is None]
        if require_task and bids_path.task is None:
            missing.append("task")
        if missing:
            raise ValueError(f"{name} BIDSPath is missing: {', '.join(missing)}")
        if bids_path.datatype not in {None, "meg"}:
            raise ValueError(f"{name} BIDSPath datatype must be 'meg'")
        if bids_path.suffix not in {None, "meg"}:
            raise ValueError(f"{name} BIDSPath suffix must be 'meg'")
        if bids_path.datatype is None and bids_path.suffix is None:
            raise ValueError(f"{name} BIDSPath must identify MEG data")
        if bids_path.extension is None:
            raise ValueError(f"{name} BIDSPath must include a raw-data extension")

    @property
    def bids_root(self) -> Path:
        return Path(self.primary.root)

    def _pipeline_root(self, pipeline: str) -> Path:
        if pipeline == "preprocessing":
            return self.bids_root / "derivatives" / "preprocessing"
        if pipeline == "project":
            return self.bids_root / "derivatives" / self.project
        raise ValueError(f"pipeline {pipeline!r} does not have a derivative root")

    def _input_path(self, source: str) -> Path:
        target = self.primary if source == "primary" else self.noise
        if isinstance(target, BIDSPath):
            return Path(target.fpath)
        if target is None:
            raise ValueError(f"no {source} input is configured")
        return Path(target)

    def _entity_context(self, source: str) -> dict[str, Any]:
        primary = self.primary
        context = {
            "subject": primary.subject,
            "session": primary.session,
            "task": primary.task,
            "acquisition": primary.acquisition,
            "run": primary.run,
            "recording": primary.recording,
            "space": self.space,
        }
        if source == "noise":
            context.update(self.noise_entities)
        return context

    def _artifact_path(self, spec: ArtifactSpec, paths: Mapping[str, Path]) -> Path:
        if spec.location == "input":
            return self._input_path(spec.source)
        root = self._pipeline_root(spec.pipeline)
        if spec.location == "dataset":
            return root / "dataset_description.json"

        context = self._entity_context(spec.source)
        entities = {key: context.get(key) for key in spec.entities}
        entities = {key: value for key, value in entities.items() if value is not None}
        if spec.uses_filter:
            entities["processing"] = self.filter_spec.processing_label
        bids_path = BIDSPath(
            root=root,
            datatype="meg",
            suffix=spec.suffix,
            extension=spec.extension,
            check=False,
            **entities,
        )
        path = Path(bids_path.fpath)
        if spec.location == "nested":
            path = paths[spec.parent_key] / path.name
        return path

    def _filter_metadata(self, source: str) -> dict[str, Any]:
        source_path = self._input_path(source)
        metadata: dict[str, Any] = {
            "Description": "MEG data filtered with MNE-Python.",
            "Sources": [_bids_source(source_path, self.bids_root)],
            "SoftwareFilters": self.filter_spec.software_filter_metadata(),
        }
        if self.filter_spec.resample_sfreq is not None:
            metadata["SamplingFrequency"] = self.filter_spec.resample_sfreq
        return metadata

    @staticmethod
    def _dataset_description_errors(path: Path) -> tuple[str, ...]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return (f"invalid JSON: {error}",)
        if not isinstance(data, dict):
            return ("dataset description must be a JSON object",)
        errors = []
        for key in ("Name", "BIDSVersion"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                errors.append(f"{key} must be a non-empty string")
        if data.get("DatasetType") != "derivative":
            errors.append("DatasetType must be 'derivative'")
        generated_by = data.get("GeneratedBy")
        if not isinstance(generated_by, list) or not generated_by:
            errors.append("GeneratedBy must be a non-empty list")
        elif any(
            not isinstance(item, dict)
            or not isinstance(item.get("Name"), str)
            or not item["Name"].strip()
            for item in generated_by
        ):
            errors.append("each GeneratedBy entry must have a non-empty Name")
        return tuple(errors)

    @staticmethod
    def _json_metadata_errors(
        path: Path, expected: Mapping[str, Any]
    ) -> tuple[str, ...]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return (f"invalid JSON: {error}",)
        return tuple(_metadata_errors(data, expected))

    def _record(
        self, spec: ArtifactSpec, path: Path, expected_metadata: Mapping[str, Any] | None
    ) -> ArtifactRecord:
        errors: tuple[str, ...] = ()
        if not path.exists():
            status: ArtifactStatus = "missing"
        elif spec.kind == "directory" and not path.is_dir():
            status = "invalid"
            errors = ("expected a directory",)
        elif spec.kind in {"file", "json"} and not path.is_file():
            status = "invalid"
            errors = ("expected a regular file",)
        else:
            if spec.location == "dataset":
                errors = self._dataset_description_errors(path)
            elif spec.kind == "json" and expected_metadata is not None:
                errors = self._json_metadata_errors(path, expected_metadata)
            status = "invalid" if errors else "present"

        load_target: Any = path
        loader = spec.loader
        if spec.key == "raw":
            load_target = self.primary.copy()
        elif spec.key == "noise_raw" and isinstance(self.noise, BIDSPath):
            load_target = self.noise.copy()
            loader = "raw_bids"

        return ArtifactRecord(
            key=spec.key,
            path=path,
            kind=spec.kind,
            status=status,
            validation_errors=errors,
            expected_metadata=expected_metadata,
            _loader=loader,
            _load_target=load_target,
        )

    def evaluate(self) -> dict[str, ArtifactRecord]:
        """Return exact expected paths and their current filesystem status."""

        records: dict[str, ArtifactRecord] = {}
        paths: dict[str, Path] = {}
        for spec in self.ARTIFACT_SPECS:
            if spec.source == "noise" and self.noise is None:
                continue
            path = self._artifact_path(spec, paths)
            paths[spec.key] = path
            expected_metadata: Mapping[str, Any] | None = None
            if spec.metadata_for is not None:
                expected_metadata = self._filter_metadata(spec.source)
            elif spec.location == "dataset":
                expected_metadata = {"DatasetType": "derivative"}
            records[spec.key] = self._record(spec, path, expected_metadata)
        return records


__all__ = [
    "ArtifactRecord",
    "ArtifactSpec",
    "FilterSpec",
    "MEGDerivativeTemplate",
]
