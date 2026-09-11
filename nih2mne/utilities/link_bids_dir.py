"""Create a lightweight BIDS directory backed by links to another dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence


DERIVATIVE_MODES = ("symlink", "copy")


def _validate_extra_derivatives(names: Optional[Iterable[str]]) -> list[str]:
    """Validate and de-duplicate derivative directory basenames."""
    validated = []
    for name in names or ():
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(
                "Extra derivative entries must be directory basenames, "
                f"not paths: {name!r}"
            )
        if name not in validated:
            validated.append(name)
    return validated


def _prepare_paths(
    bids_root: os.PathLike | str,
    new_root_dir: os.PathLike | str | None,
) -> tuple[Path, Path]:
    """Return validated, absolute source and destination paths."""
    source = Path(bids_root).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"BIDS root is not a directory: {source}")
    source = source.resolve()

    destination = (
        Path.cwd() / "BIDS" if new_root_dir is None else Path(new_root_dir).expanduser()
    ).resolve(strict=False)

    if os.path.lexists(destination):
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination == source:
        raise ValueError("The source and destination BIDS roots must be different")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("The destination cannot be located inside the source BIDS root")

    return source, destination


def _raw_entries(source: Path) -> list[Path]:
    """Select top-level BIDS files and subject directories."""
    return sorted(
        (
            entry
            for entry in source.iterdir()
            if entry.name != "derivatives"
            and (entry.is_file() or (entry.name.startswith("sub-") and entry.is_dir()))
        ),
        key=lambda entry: entry.name,
    )


def _derivative_entries(
    source: Path,
    *,
    freesurfer: bool,
    preproc: bool,
    extra_derivatives: Sequence[str],
) -> list[Path]:
    """Select immediate child directories from the source derivatives folder."""
    derivatives = source / "derivatives"
    available = {}
    if derivatives.is_dir():
        available = {
            entry.name: entry
            for entry in derivatives.iterdir()
            if entry.is_dir()
        }

    requested = []
    if freesurfer:
        requested.append("freesurfer")

    preproc_matches = sorted(name for name in available if name.startswith("preproc"))
    if preproc:
        requested.extend(preproc_matches)
        if not preproc_matches:
            print(f"No preproc* derivative directories found in {derivatives}; skipping")

    requested.extend(extra_derivatives)
    requested = list(dict.fromkeys(requested))

    selected = []
    for name in requested:
        entry = available.get(name)
        if entry is None:
            print(f"Derivative directory not found in {derivatives}; skipping: {name}")
            continue
        selected.append(entry)
    return selected


def link_bids_dir(
    bids_root: os.PathLike | str,
    new_root_dir: os.PathLike | str | None = None,
    *,
    freesurfer: bool = True,
    preproc: bool = True,
    extra_derivatives: Optional[Iterable[str]] = None,
    derivative_mode: str = "symlink",
) -> Path:
    """Create a BIDS root containing links to selected source content.

    Raw ``sub-*`` directories and top-level files are always symlinked. The
    selected derivative directories are either symlinked or recursively copied
    according to ``derivative_mode``.

    Parameters
    ----------
    bids_root
        Existing source BIDS directory.
    new_root_dir
        Exact destination directory to create. Defaults to ``$PWD/BIDS``.
    freesurfer
        Include ``derivatives/freesurfer`` when it exists.
    preproc
        Include all immediate derivative directories beginning with ``preproc``.
    extra_derivatives
        Exact basenames of additional immediate derivative directories.
    derivative_mode
        Either ``"symlink"`` or ``"copy"`` for all selected derivatives.
    """
    if derivative_mode not in DERIVATIVE_MODES:
        choices = ", ".join(DERIVATIVE_MODES)
        raise ValueError(f"derivative_mode must be one of: {choices}")

    extras = _validate_extra_derivatives(extra_derivatives)
    source, destination = _prepare_paths(bids_root, new_root_dir)
    raw_entries = _raw_entries(source)
    derivative_entries = _derivative_entries(
        source,
        freesurfer=freesurfer,
        preproc=preproc,
        extra_derivatives=extras,
    )

    destination.mkdir(parents=True)
    destination_derivatives = destination / "derivatives"
    destination_derivatives.mkdir()

    for entry in raw_entries:
        (destination / entry.name).symlink_to(
            entry.resolve(), target_is_directory=entry.is_dir()
        )

    for entry in derivative_entries:
        output = destination_derivatives / entry.name
        if derivative_mode == "symlink":
            output.symlink_to(entry.resolve(), target_is_directory=True)
        else:
            shutil.copytree(entry, output)

    print(
        f"Created {destination} with {len(raw_entries)} raw BIDS links and "
        f"{len(derivative_entries)} {derivative_mode} derivative directories"
    )
    return destination


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight BIDS directory linked to raw data in an "
            "existing BIDS root."
        )
    )
    parser.add_argument("-bids_root", required=True, help="Source BIDS directory")
    parser.add_argument(
        "-new_root_dir",
        help="Exact destination to create (default: $PWD/BIDS)",
    )
    parser.add_argument(
        "-freesurfer",
        "--freesurfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the freesurfer derivative directory (default: enabled)",
    )
    parser.add_argument(
        "-preproc",
        "--preproc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include derivative directories beginning with preproc (default: enabled)",
    )
    parser.add_argument(
        "-extra_derivatives",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Exact names of additional derivative directories to include",
    )
    parser.add_argument(
        "-derivative_mode",
        choices=DERIVATIVE_MODES,
        default="symlink",
        help="Transfer selected derivatives by symlink or recursive copy",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Command-line entry point."""
    args = _get_parser().parse_args(argv)
    return link_bids_dir(
        bids_root=args.bids_root,
        new_root_dir=args.new_root_dir,
        freesurfer=args.freesurfer,
        preproc=args.preproc,
        extra_derivatives=args.extra_derivatives,
        derivative_mode=args.derivative_mode,
    )


if __name__ == "__main__":
    main()
