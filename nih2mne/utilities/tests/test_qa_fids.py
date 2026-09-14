import json

import nibabel as nib
import numpy as np
import pytest

import nih2mne.utilities.qa_fids as qa_fids


def _synthetic_head_image(constant=None):
    data = np.zeros((12, 12, 12), dtype=np.float32)
    head = np.zeros(data.shape, dtype=bool)
    head[3:9, 3:9, 3:9] = True
    if constant is None:
        data[head] = np.linspace(10, 110, head.sum(), dtype=np.float32)
    else:
        data[head] = constant
    return nib.Nifti1Image(data, np.eye(4)), data, head


def test_normalize_mri_uses_connected_head_percentiles():
    image, data, head = _synthetic_head_image()
    data[1, 1, 1] = 10_000  # disconnected bright artifact
    image = nib.Nifti1Image(data, image.affine)

    normalized_image = qa_fids._normalize_mri_for_display(image)
    normalized = normalized_image.get_fdata(dtype=np.float32)

    lower, upper = np.percentile(data[head], (5, 95))
    expected = np.zeros(data.shape, dtype=np.float32)
    expected[head] = np.clip(
        (data[head] - lower) * (128.0 / (upper - lower)),
        0,
        128,
    )
    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_array_equal(normalized_image.affine, image.affine)
    assert normalized_image.get_data_dtype() == np.dtype(np.float32)
    assert normalized[1, 1, 1] == 0


def test_normalize_constant_head_displays_foreground():
    image, _, head = _synthetic_head_image(constant=7)

    normalized = qa_fids._normalize_mri_for_display(image).get_fdata()

    assert np.all(normalized[head] == 128)
    assert np.all(normalized[~head] == 0)


def test_normalize_empty_image_fails_clearly():
    image = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4))

    with pytest.warns(UserWarning, match="empty mask"):
        with pytest.raises(ValueError, match="finite head intensities"):
            qa_fids._normalize_mri_for_display(image)


@pytest.mark.parametrize("block", [False, True])
def test_plot_fids_uses_normalized_volume_for_all_panels(
    tmp_path,
    monkeypatch,
    block,
):
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    t1_path = anat_dir / "sub-01_T1w.nii.gz"
    image, _, _ = _synthetic_head_image()
    nib.save(image, t1_path)
    t1_path.with_name("sub-01_T1w.json").write_text(
        json.dumps(
            {
                "AnatomicalLandmarkCoordinates": {
                    "LPA": [4, 5, 6],
                    "NAS": [5, 6, 7],
                    "RPA": [6, 7, 8],
                }
            }
        )
    )

    axes = [object(), object(), object()]
    plot_calls = []
    show_calls = []
    monkeypatch.setattr(
        qa_fids.plt,
        "subplots",
        lambda *args, **kwargs: (object(), axes),
    )
    monkeypatch.setattr(
        qa_fids,
        "plot_anat",
        lambda image, **kwargs: plot_calls.append((image, kwargs)),
    )
    monkeypatch.setattr(
        qa_fids.plt,
        "show",
        lambda *args, **kwargs: show_calls.append((args, kwargs)),
    )

    outfile = tmp_path / "sub-01_fids_qa.png"
    qa_fids.plot_fids_qa(
        subjid="01",
        bids_root=tmp_path,
        outfile=outfile,
        block=block,
    )

    assert len(plot_calls) == 3
    assert all(call[0] is plot_calls[0][0] for call in plot_calls)
    assert [call[1]["axes"] for call in plot_calls] == axes
    assert [call[1]["title"] for call in plot_calls] == ["LPA", "NAS", "RPA"]
    for _, kwargs in plot_calls:
        assert kwargs["vmin"] == 0
        assert kwargs["vmax"] == 128
        assert kwargs["dim"] == 0
        if block:
            assert "output_file" not in kwargs
        else:
            assert kwargs["output_file"] == outfile
    expected_show_calls = [((), {"block": True})] if block else [((), {})]
    assert show_calls == expected_show_calls
