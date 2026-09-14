from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pytest

from nih2mne.utilities.fast_head_surface import (
    _smooth_surface,
    find_head_surface,
    make_fast_head_surface,
)


def _write_synthetic_t1(tmp_path: Path, subject: str = "sub-01"):
    subject_dir = tmp_path / "subjects" / subject
    mri_dir = subject_dir / "mri"
    mri_dir.mkdir(parents=True)

    shape = (40, 42, 44)
    center = np.array((20, 21, 22))
    coordinates = np.indices(shape).transpose(1, 2, 3, 0)
    data = np.zeros(shape, dtype=np.float32)
    data[np.linalg.norm(coordinates - center, axis=-1) <= 10] = 100
    data[2:4, 2:4, 2:4] = 100  # disconnected noise to be discarded

    image = nib.MGHImage(data, np.eye(4))
    t1_path = mri_dir / "T1.mgz"
    image.to_filename(t1_path)
    return subject_dir.parent, subject, center, image.header.get_vox2ras_tkr()


def _signed_volume(vertices, faces):
    return np.einsum(
        "ij,ij->i",
        vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
    ).sum() / 6


def _mesh_roughness(vertices, faces):
    neighbors = [set() for _ in vertices]
    for face in faces:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            neighbors[face[first]].add(face[second])
            neighbors[face[second]].add(face[first])
    residuals = [
        np.linalg.norm(vertex - vertices[list(adjacent)].mean(axis=0))
        for vertex, adjacent in zip(vertices, neighbors)
    ]
    return np.mean(residuals)


def test_surface_smoothing_reduces_jaggedness_without_shrinkage():
    from skimage.measure import marching_cubes

    shape = (41, 41, 41)
    center = np.array((20, 20, 20))
    coordinates = np.indices(shape).transpose(1, 2, 3, 0)
    mask = np.linalg.norm(coordinates - center, axis=-1) <= 14
    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.uint8), level=0.5, allow_degenerate=False
    )

    smoothed = _smooth_surface(vertices, faces)

    assert smoothed.shape == vertices.shape
    assert np.isfinite(smoothed).all()
    assert _mesh_roughness(smoothed, faces) < _mesh_roughness(vertices, faces)
    diagonal = np.linalg.norm(np.ptp(vertices, axis=0))
    assert np.linalg.norm(smoothed.mean(axis=0) - vertices.mean(axis=0)) < (
        diagonal * 1e-3
    )
    volume_change = abs(
        _signed_volume(smoothed, faces) / _signed_volume(vertices, faces) - 1
    )
    assert volume_change < 0.05


def test_make_fast_head_surface_writes_mne_surface(tmp_path):
    subjects_dir, subject, center, vox2ras_tkr = _write_synthetic_t1(tmp_path)

    output = make_fast_head_surface(subject, subjects_dir)

    assert output == subjects_dir / subject / "bem" / "outer_skin.surf"
    assert find_head_surface(subject, subjects_dir) == output
    vertices, faces = mne.read_surface(output)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) > 100
    assert len(faces) > 100

    surface_center = nib.affines.apply_affine(vox2ras_tkr, center)
    distances = np.linalg.norm(vertices - surface_center, axis=1)
    assert distances.max() < 13
    assert distances.min() > 7

    assert _signed_volume(vertices, faces) > 0


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("bem/outer_skin.surf"),
        Path("bem/flash/outer_skin.surf"),
        Path("bem/sub-01-head-sparse.fif"),
        Path("bem/sub-01-head.fif"),
    ],
)
def test_find_head_surface_uses_mne_search_order(tmp_path, relative_path):
    subjects_dir = tmp_path / "subjects"
    candidate = subjects_dir / "sub-01" / relative_path
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"existing")

    assert find_head_surface("sub-01", subjects_dir) == candidate
    assert make_fast_head_surface("sub-01", subjects_dir) == candidate
    assert candidate.read_bytes() == b"existing"


def test_missing_t1_does_not_leave_partial_surface(tmp_path):
    subjects_dir = tmp_path / "subjects"
    (subjects_dir / "sub-01").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="T1.mgz not found"):
        make_fast_head_surface("sub-01", subjects_dir)

    assert not (subjects_dir / "sub-01" / "bem").exists()


def test_empty_threshold_does_not_leave_partial_surface(tmp_path):
    subjects_dir, subject, _, _ = _write_synthetic_t1(tmp_path)

    with pytest.raises(ValueError, match="did not produce any head voxels"):
        make_fast_head_surface(subject, subjects_dir, threshold=200)

    assert not (subjects_dir / subject / "bem").exists()


@pytest.mark.parametrize("subject", ["../sub-01", "nested/sub-01", ".", ""])
def test_subject_must_be_a_directory_name(tmp_path, subject):
    with pytest.raises(ValueError, match="must be a directory name"):
        find_head_surface(subject, tmp_path)
