from pathlib import Path

import pytest

from nih2mne.utilities.link_bids_dir import link_bids_dir, main


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sub-01").mkdir()
    (source / "sub-01" / "data.txt").write_text("raw data")
    (source / "participants.tsv").write_text("participant_id\nsub-01\n")
    (source / ".bidsignore").write_text("tmp\n")
    (source / "sourcedata").mkdir()
    (source / "sourcedata" / "ignored.txt").write_text("ignored")

    derivatives = source / "derivatives"
    derivatives.mkdir()
    for name in ("freesurfer", "preprocessing", "preproc-clean", "megQA"):
        directory = derivatives / name
        directory.mkdir()
        (directory / "result.txt").write_text(name)
    return source


def test_default_links_and_extra_derivatives(tmp_path, capsys):
    source = _make_source(tmp_path)
    destination = tmp_path / "linked"

    result = link_bids_dir(
        source,
        destination,
        extra_derivatives=["megQA", "missing", "freesurfer"],
    )

    assert result == destination
    assert (destination / "sub-01").is_symlink()
    assert (destination / "sub-01").resolve() == (source / "sub-01").resolve()
    assert (destination / "participants.tsv").is_symlink()
    assert (destination / ".bidsignore").is_symlink()
    assert not (destination / "sourcedata").exists()
    assert (destination / "derivatives").is_dir()
    assert not (destination / "derivatives").is_symlink()

    expected = {"freesurfer", "preprocessing", "preproc-clean", "megQA"}
    assert {entry.name for entry in (destination / "derivatives").iterdir()} == expected
    assert all(
        (destination / "derivatives" / name).is_symlink() for name in expected
    )
    assert "skipping: missing" in capsys.readouterr().out


def test_copy_mode_and_disabled_automatic_derivatives(tmp_path):
    source = _make_source(tmp_path)
    destination = tmp_path / "copied"

    link_bids_dir(
        source,
        destination,
        freesurfer=False,
        preproc=False,
        extra_derivatives=["megQA"],
        derivative_mode="copy",
    )

    copied = destination / "derivatives" / "megQA"
    assert copied.is_dir()
    assert not copied.is_symlink()
    assert (copied / "result.txt").read_text() == "megQA"
    assert {entry.name for entry in (destination / "derivatives").iterdir()} == {
        "megQA"
    }

    (source / "derivatives" / "megQA" / "result.txt").write_text("changed")
    assert (copied / "result.txt").read_text() == "megQA"


def test_missing_derivatives_are_reported_and_skipped(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset_description.json").write_text("{}")

    destination = link_bids_dir(source, tmp_path / "linked")

    assert list((destination / "derivatives").iterdir()) == []
    output = capsys.readouterr().out
    assert "No preproc* derivative directories" in output
    assert "skipping: freesurfer" in output


def test_existing_destination_fails_without_changes(tmp_path):
    source = _make_source(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="Destination already exists"):
        link_bids_dir(source, destination)

    assert marker.read_text() == "keep"
    assert list(destination.iterdir()) == [marker]


@pytest.mark.parametrize("name", ["../megQA", "nested/megQA", ".", ""])
def test_extra_derivatives_must_be_basenames(tmp_path, name):
    source = _make_source(tmp_path)
    destination = tmp_path / "linked"

    with pytest.raises(ValueError, match="must be directory basenames"):
        link_bids_dir(source, destination, extra_derivatives=[name])

    assert not destination.exists()


def test_invalid_mode_and_nested_destination_fail_before_creation(tmp_path):
    source = _make_source(tmp_path)

    with pytest.raises(ValueError, match="derivative_mode"):
        link_bids_dir(source, tmp_path / "bad-mode", derivative_mode="move")
    assert not (tmp_path / "bad-mode").exists()

    nested = source / "BIDS"
    with pytest.raises(ValueError, match="inside the source"):
        link_bids_dir(source, nested)
    assert not nested.exists()


def test_cli_defaults_to_pwd_bids_and_accepts_no_flags(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    result = main(
        ["-bids_root", str(source), "--no-freesurfer", "--no-preproc"]
    )

    assert result == working_directory / "BIDS"
    assert (result / "sub-01").is_symlink()
    assert list((result / "derivatives").iterdir()) == []
