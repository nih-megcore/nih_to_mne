# Changelog

All notable changes for the upcoming merge to `main` are documented here.

## [0.6.2] - Unreleased

### Added

- Added BIDS project utilities to link BIDS directories, recover `RUNDICT`
  metadata, read conversion logs, and update or reinitialize MEGQA files.
- Added completion indicators, log-derived input/output reporting, and surfaced
  processing errors in the BIDS conversion GUI and command-line workflow.
- Added the `sam_bids_pipeline` command-line pipeline for SAM source coding.
- Added fast, marching-cubes-based head-surface generation, including smoothed
  scalp output and a BEM head-surface file.
- Added command-line entry points for `make_bids_log_reader.py`,
  `update_megQA_files`, `sam_bids_pipeline`, and `link_bids_dir.py`.
- Added automated coverage for the new BIDS, head-surface, GUI compatibility,
  and SAM pipeline functionality.

### Changed

- Made PyQt6 the default GUI binding and added a PyQt5 compatibility path via
  `QT_API=pyqt5` for supported legacy environments.
- Added tooling to regenerate checked-in Qt Designer form code with PyQt6.
- Copy MEGQA files by default while linking datasets.
- Improved BIDS range selection and MRI fiducial-view autoscaling.
- Use the optional C-based YAML loader when available for faster configuration
  loading.
- Added `scikit-image`, `pyvista>=0.45`, and PyQt6 dependencies; removed the
  Python version upper bound from package metadata.

### Fixed

- Corrected an extra output dimension in tests and improved GitHub Actions
  installation behavior.
- Fixed BIDS MRI JSON lookup, git-annex dereferencing, and completion-popup
  behavior.

