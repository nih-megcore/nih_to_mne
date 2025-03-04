
#>>>> https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate 
# <<<<

install_test:
	#conda install --channel=conda-forge --name=base mamba -y
	#conda env remove -n nih2mne_test
	mamba create --override-channels --channel=conda-forge --name=nih2mne_test "mne=1.5" "python<3.12" "numba<0.60" pip -y  
	($(CONDA_ACTIVATE) nih2mne_test ; pip install -e .[testing]; pip install pytest pytest-reportlog )
	git submodule init
	git pull --recurse-submodules

install_headless_env:
	#conda install --channel=conda-forge --name=base mamba -y
	conda env remove -n nih2mne_test
	mamba create --override-channels --channel=conda-forge --name=nih2mne_test "mne=1.5" "python<3.12" "numba<0.60"  pip "vtk>=9.2=*osmesa*" "mesalib=21.2.5" -y
	($(CONDA_ACTIVATE) nih2mne_test ; pip install -e .[testing]; pip install pytest pytest-reportlog )
	git submodule init
	git pull --recurse-submodules

install_system_requirements:
	dnf install Xvfb -y
	dnf install git git-annex -y

create_bids:
	cd nih2mne/test_data ; make_meg_bids.py -bids_dir ./BIDS -subjid_input ABABABAB -meg_input_dir 20010101 -bids_id S01 -mri_bsight MRI/ABABABAB_refaced_T1w.nii.gz -mri_bsight MRI/ABABABAB_elec.txt

test:
	($(CONDA_ACTIVATE) nih2mne_test ; pytest -vv  )  

test_headless:
	($(CONDA_ACTIVATE) nih2mne_test ; cd nih_to_mne; pytest -vv ) #cd enigma_MEG; xvfb-run -a pytest -vv --report-log=/tmp/enigma_MEG_test_logfile.txt )

get_data:
	git submodule update --init --recursive

