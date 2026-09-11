"""Generate a fast scalp surface for coregistration visualization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def _subject_directory(subject: str, subjects_dir: os.PathLike | str) -> Path:
    """Return a validated FreeSurfer subject directory."""
    subject = os.fspath(subject)
    if not subject or subject in {".", ".."} or Path(subject).name != subject:
        raise ValueError(f"subject must be a directory name, got {subject!r}")
    return Path(subjects_dir).expanduser().resolve() / subject


def _head_surface_candidates(subject: str, subjects_dir: os.PathLike | str):
    """Yield head surfaces in MNE ``plot_alignment`` search order."""
    subject_dir = _subject_directory(subject, subjects_dir)
    yield subject_dir / "bem" / "outer_skin.surf"
    yield subject_dir / "bem" / "flash" / "outer_skin.surf"
    yield subject_dir / "bem" / f"{subject}-head-sparse.fif"
    yield subject_dir / "bem" / f"{subject}-head.fif"


def find_head_surface(
    subject: str,
    subjects_dir: os.PathLike | str,
) -> Path | None:
    """Return the first existing head surface recognized by MNE."""
    return next(
        (
            path
            for path in _head_surface_candidates(subject, subjects_dir)
            if path.is_file()
        ),
        None,
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground component and fill its holes."""
    from scipy.ndimage import binary_fill_holes, label

    labels, component_count = label(mask)
    if component_count == 0:
        raise ValueError("The MRI threshold did not produce any head voxels")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    mask = labels == component_sizes.argmax()
    return binary_fill_holes(mask)


def _crop_and_pad(
    mask: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the foreground extent and add a background border."""
    occupied = [
        np.flatnonzero(mask.any(axis=axes))
        for axes in ((1, 2), (0, 2), (0, 1))
    ]
    starts = np.array([indices[0] for indices in occupied], dtype=float)
    stops = [indices[-1] + 1 for indices in occupied]
    cropped = mask[
        int(starts[0]):stops[0],
        int(starts[1]):stops[1],
        int(starts[2]):stops[2],
    ]
    return np.pad(cropped, padding, mode="constant"), starts


def _orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Ensure that face winding gives a positive enclosed volume."""
    signed_volume = np.einsum(
        "ij,ij->i",
        vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
    ).sum()
    if signed_volume < 0:
        faces = faces[:, [0, 2, 1]]
    return faces


def make_fast_head_surface(
    subject: str,
    subjects_dir: os.PathLike | str,
    *,
    threshold: float = 20,
    step_size: int = 2,
    overwrite: bool = False,
) -> Path:
    """Create a lightweight ``outer_skin.surf`` from FreeSurfer ``T1.mgz``.

    This surface is intended for visual inspection of MEG/MRI coregistration,
    not for high-accuracy BEM modeling.

    Parameters
    ----------
    subject
        FreeSurfer subject directory name.
    subjects_dir
        FreeSurfer ``SUBJECTS_DIR``.
    threshold
        MRI intensity separating the head from the background.
    step_size
        Marching-cubes voxel stride. Larger values are faster and coarser.
    overwrite
        Replace ``bem/outer_skin.surf`` when it already exists.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if isinstance(step_size, bool) or not isinstance(step_size, (int, np.integer)):
        raise TypeError("step_size must be an integer")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")

    subject_dir = _subject_directory(subject, subjects_dir)
    existing = find_head_surface(subject, subjects_dir)
    if existing is not None and not overwrite:
        return existing

    t1_path = subject_dir / "mri" / "T1.mgz"
    if not t1_path.is_file():
        raise FileNotFoundError(f"FreeSurfer T1.mgz not found: {t1_path}")

    import nibabel as nib
    from mne import write_surface
    from skimage.measure import marching_cubes

    image = nib.load(t1_path)
    if not hasattr(image.header, "get_vox2ras_tkr"):
        raise TypeError(f"MRI does not provide a FreeSurfer vox2ras_tkr: {t1_path}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"T1.mgz must contain 3D data, got shape {data.shape}")

    mask = _largest_component(np.isfinite(data) & (data > threshold))
    mask, crop_start = _crop_and_pad(mask, int(step_size))
    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        gradient_direction="descent",
        step_size=int(step_size),
        allow_degenerate=False,
    )
    vertices += crop_start - float(step_size)
    vertices = nib.affines.apply_affine(
        image.header.get_vox2ras_tkr(), vertices
    ).astype(np.float32, copy=False)
    faces = _orient_faces_outward(vertices, faces).astype(np.int32, copy=False)

    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise RuntimeError("Marching cubes did not produce a valid head surface")

    bem_dir = subject_dir / "bem"
    output = bem_dir / "outer_skin.surf"
    bem_dir.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=".outer_skin-", suffix=".surf", dir=bem_dir
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    # Let MNE create the file with the process umask instead of retaining the
    # owner-only mode assigned by mkstemp.
    temp_path.unlink()
    try:
        write_surface(
            temp_path,
            vertices,
            faces,
            create_stamp="Fast visualization surface generated by nih2mne",
            overwrite=True,
        )
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return output


__all__ = ["find_head_surface", "make_fast_head_surface"]
