import pathlib
from . import GUI
from . import utilities
from . import dataQA

_test_data_dir=pathlib.Path(__file__).parent.resolve() / 'test_data'
_test_meg_data_dir = _test_data_dir / '20010101'
_test_mri_data_dir = _test_data_dir / 'MRI'