from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pytest

from nih2mne.utilities.fast_head_surface import (
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

    signed_volume = np.einsum(
        "ij,ij->i",
        vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
    ).sum()
    assert signed_volume > 0


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
