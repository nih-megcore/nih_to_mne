"""Run the orthohull and SAM processing stages for a BIDS CTF dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


class PipelineError(RuntimeError):
    """Raised when the SAM pipeline inputs or outputs are invalid."""


@dataclass(frozen=True)
class BIDSContext:
    """BIDS entities and paths derived from an input CTF dataset."""

    dataset: Path
    bids_root: Path
    subject: str
    session: str | None
    run: str | None


@dataclass(frozen=True)
class PipelinePaths:
    """Generated paths returned after a successful pipeline run."""

    mri: Path
    ortho_mri: Path
    hull: Path
    sam_output: Path


def _single_entity(tokens: list[str], key: str, *, required: bool) -> str | None:
    prefix = f"{key}-"
    values = [token for token in tokens if token.startswith(prefix)]
    if not values:
        if required:
            raise PipelineError(f"dataset filename is missing the {prefix} entity")
        return None
    if len(values) != 1 or values[0] == prefix:
        raise PipelineError(f"dataset filename has an invalid {prefix} entity")
    return values[0]


def parse_bids_dataset(dataset: str | Path) -> BIDSContext:
    """Validate an absolute BIDS CTF path and derive its root and entities."""

    original = Path(dataset).expanduser()
    if not original.is_absolute():
        raise PipelineError(f"dataset path must be fully qualified: {dataset}")
    if not original.is_dir():
        raise PipelineError(f"CTF dataset directory does not exist: {original}")

    resolved = original.resolve()
    if not resolved.name.endswith("_meg.ds"):
        raise PipelineError(
            f"CTF dataset must have a BIDS name ending in '_meg.ds': {resolved.name}"
        )

    tokens = resolved.name.removesuffix(".ds").split("_")
    subject = _single_entity(tokens, "sub", required=True)
    session = _single_entity(tokens, "ses", required=False)
    run = _single_entity(tokens, "run", required=False)

    if resolved.parent.name != "meg":
        raise PipelineError(f"BIDS CTF dataset must be inside a 'meg' directory: {resolved}")

    entity_parent = resolved.parent.parent
    if session is not None:
        if entity_parent.name != session:
            raise PipelineError(
                f"filename session {session!r} does not match directory "
                f"{entity_parent.name!r}"
            )
        subject_dir = entity_parent.parent
    else:
        if entity_parent.name.startswith("ses-"):
            raise PipelineError(
                f"dataset is in {entity_parent.name!r} but its filename has no session entity"
            )
        subject_dir = entity_parent

    if subject_dir.name != subject:
        raise PipelineError(
            f"filename subject {subject!r} does not match directory {subject_dir.name!r}"
        )

    return BIDSContext(
        dataset=resolved,
        bids_root=subject_dir.parent,
        subject=subject,
        session=session,
        run=run,
    )


def _relocate_project_path(path: Path, old_root: object, new_root: Path) -> Path:
    if old_root is None:
        return path
    try:
        relative = path.resolve(strict=False).relative_to(
            Path(str(old_root)).expanduser().resolve(strict=False)
        )
    except (OSError, ValueError):
        return path
    return new_root / relative


def load_selected_mri(context: BIDSContext) -> Path:
    """Load the MRI selected in the subject's megQA pickle."""

    pickle_path = (
        context.bids_root / "derivatives" / "megQA" / f"{context.subject}.pkl"
    )
    if not pickle_path.is_file():
        raise PipelineError(f"megQA pickle does not exist: {pickle_path}")

    try:
        import dill

        with pickle_path.open("rb") as stream:
            qa_record = dill.load(stream)
    except Exception as error:
        raise PipelineError(f"could not load megQA pickle {pickle_path}: {error}") from error

    selected = getattr(qa_record, "mri", None)
    if selected in (None, "Multiple"):
        detail = "no MRI is selected" if selected is None else "multiple MRIs are unresolved"
        raise PipelineError(f"{detail} in {pickle_path}")

    mri = Path(str(selected)).expanduser()
    if not mri.is_absolute():
        mri = Path(str(getattr(qa_record, "bids_root", context.bids_root))) / mri
    mri = _relocate_project_path(
        mri, getattr(qa_record, "bids_root", None), context.bids_root
    ).resolve(strict=False)
    if not mri.is_file():
        raise PipelineError(f"selected MRI does not exist: {mri}")
    if not (mri.name.endswith(".nii") or mri.name.endswith(".nii.gz")):
        raise PipelineError(f"selected MRI must be a NIfTI file: {mri}")

    validate_fiducial_sidecar(mri)
    return mri


def validate_fiducial_sidecar(mri: Path) -> Path:
    """Require the three finite BIDS voxel-space fiducials used by orthohull."""

    suffix = ".nii.gz" if mri.name.endswith(".nii.gz") else ".nii"
    sidecar = mri.with_name(f"{mri.name.removesuffix(suffix)}.json")
    if not sidecar.is_file():
        raise PipelineError(f"MRI JSON sidecar does not exist: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        landmarks = metadata["AnatomicalLandmarkCoordinates"]
        coordinates = [landmarks[name] for name in ("NAS", "LPA", "RPA")]
        valid = all(
            len(coordinate) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in coordinate
            )
            for coordinate in coordinates
        )
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        raise PipelineError(
            f"MRI JSON must contain numeric NAS, LPA, and RPA coordinates: {sidecar}"
        ) from error
    if not valid:
        raise PipelineError(
            f"NAS, LPA, and RPA must each contain three finite coordinates: {sidecar}"
        )
    return sidecar


def _stage_mri(mri: Path, ortho_mri: Path) -> Path:
    """Copy the selected BIDS MRI and sidecar into the derivative workspace."""

    sidecar = validate_fiducial_sidecar(mri)
    staged_mri = ortho_mri / mri.name
    staged_sidecar = ortho_mri / sidecar.name
    try:
        if mri.resolve() != staged_mri.resolve():
            shutil.copy2(mri, staged_mri)
        if sidecar.resolve() != staged_sidecar.resolve():
            shutil.copy2(sidecar, staged_sidecar)
    except OSError as error:
        raise PipelineError(f"could not stage MRI in {ortho_mri}: {error}") from error
    return staged_mri


def _validate_project(project: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project) or project in {
        ".",
        "..",
    }:
        raise PipelineError(
            "project must be a single derivative directory name containing only "
            "letters, numbers, '.', '_', or '-'"
        )
    return project


def sam_output_path(context: BIDSContext, project: str) -> Path:
    """Return the requested subject/session/run SAM derivative root."""

    output = context.bids_root / "derivatives" / _validate_project(project)
    output /= context.subject
    if context.session is not None:
        output /= context.session
    if context.run is not None:
        output /= context.run
    return output


def _command_path(command: str, which: Callable[[str], str | None]) -> None:
    if which(command) is None:
        raise PipelineError(f"required command was not found on PATH: {command}")


def _run(
    command: list[str],
    *,
    runner: Callable[..., object],
    cwd: Path | None = None,
) -> None:
    location = f" (in {cwd})" if cwd is not None else ""
    print(f"$ {shlex.join(command)}{location}", flush=True)
    runner(command, cwd=cwd, check=True)


def run_pipeline(
    dataset: str | Path,
    parameter_file: str | Path,
    *,
    project: str = "SAMsrc",
    force_orthohull: bool = False,
    runner: Callable[..., object] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> PipelinePaths:
    """Prepare a selected T1 hull and run covariance, weights, and 3-D SAM."""

    context = parse_bids_dataset(dataset)
    parameter = Path(parameter_file).expanduser().resolve(strict=False)
    if not parameter.is_file():
        raise PipelineError(f"SAM parameter file does not exist: {parameter}")

    output = sam_output_path(context, project)
    mri = load_selected_mri(context)
    ortho_mri = (
        context.bids_root
        / "derivatives"
        / "preprocessing"
        / context.subject
        / "orthoMRI"
    )
    hull = ortho_mri / "hull.shape"
    reuse_hull = hull.is_file() and hull.stat().st_size > 0 and not force_orthohull

    for command in ("sam_cov", "sam_wts", "sam_3d"):
        _command_path(command, which)
    if not reuse_hull:
        _command_path("orthohull", which)

    ortho_mri.mkdir(parents=True, exist_ok=True)
    if reuse_hull:
        print(f"Reusing existing MRI hull: {hull}", flush=True)
    else:
        staged_mri = _stage_mri(mri, ortho_mri)
        _run(["orthohull", str(staged_mri)], runner=runner, cwd=ortho_mri)
        if not hull.is_file() or hull.stat().st_size == 0:
            raise PipelineError(f"orthohull did not create a nonempty hull: {hull}")

    common = ["-r", str(context.dataset), "-m", str(parameter)]
    output_arg = str(output)
    _run(
        ["sam_cov", *common, "-o_SAMdir", output_arg],
        runner=runner,
    )
    _run(
        [
            "sam_wts",
            *common,
            "--MRIDirectory",
            str(ortho_mri),
            "--MRIPattern",
            "%M/%s",
            "-i_SAMdir",
            output_arg,
            "-o_SAMdir",
            output_arg,
        ],
        runner=runner,
    )
    _run(
        [
            "sam_3d",
            *common,
            "--ImageDirectory",
            output_arg,
            "-i_SAMdir",
            output_arg,
            "-o_SAMdir",
            output_arg,
        ],
        runner=runner,
    )
    return PipelinePaths(mri=mri, ortho_mri=ortho_mri, hull=hull, sam_output=output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an orthohull from the T1 selected in megQA, then run "
            "sam_cov, sam_wts, and sam_3d for a BIDS CTF dataset."
        )
    )
    parser.add_argument("dataset", help="Fully qualified BIDS CTF .ds directory")
    parser.add_argument("parameter_file", help="SAM analysis parameter file")
    parser.add_argument(
        "--project",
        default="SAMsrc",
        help="Derivative project directory name (default: SAMsrc)",
    )
    parser.add_argument(
        "--force-orthohull",
        action="store_true",
        help="Regenerate the MRI hull even when hull.shape already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    args = _parser().parse_args(argv)
    try:
        paths = run_pipeline(
            args.dataset,
            args.parameter_file,
            project=args.project,
            force_orthohull=args.force_orthohull,
        )
    except PipelineError as error:
        print(f"sam_bids_pipeline: error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(
            f"sam_bids_pipeline: command failed with exit status {error.returncode}: "
            f"{shlex.join(str(value) for value in error.cmd)}",
            file=sys.stderr,
        )
        return error.returncode or 1

    print(f"SAM pipeline complete: {paths.sam_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
