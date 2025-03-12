import pathlib
from . import GUI
from . import utilities
from . import dataQA

_test_data_dir=pathlib.Path(__file__).parent.resolve() / 'test_data'
_test_meg_data_dir = _test_data_dir / '20010101'
_test_mri_data_dir = _test_data_dir / 'MRI'


class test_data():
    def __init__(self, test_data_dir=_test_data_dir):
        self.test_data_dir = _test_data_dir
        self.meg_data_dir = _test_meg_data_dir
        self.mri_data_dir = _test_mri_data_dir
        self.meg_airpuff_fname = self.meg_data_dir / 'ABABABAB_airpuff_20010101_001.ds'
        self.meg_hariri_fname = self.meg_data_dir / 'ABABABAB_haririhammer_20010101_002.ds'
        self.mri_nii = self.mri_data_dir / 'ABABABAB_refaced_T1w.nii.gz'
        self.mri_trans = self.mri_data_dir / 'ABABABAB-trans.fif'
        
    
    def is_present(self):
        self.has_meg =  (self.meg_data_dir / 'ABABABAB_airpuff_20010101_001.ds/ABABABAB_airpuff_20010101_001.res4').exists()
        self.has_mri =  (self.mri_data_dir / 'ABABABAB_refaced_T1w.nii.gz').exists()
        if self.has_meg and self.has_mri:
            return True
        else:
            return False
        