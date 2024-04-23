import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nih2mne", 
    version="0.2",
    author="Jeff Stout",
    author_email="stoutjd@nih.gov",
    description="Adapt coregistration from NIH MEG data for use with MNE python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nih-megcore/nih_to_mne",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: UNLICENSE",
        "Operating System :: Linux/Unix",
    ],
    python_requires='>=3.6',
    install_requires=['mne', 'numpy', 'pytest', 'joblib', 'nibabel','mne_bids','pandas', 'pyctf-lite @ git+https://github.com/nih-megcore/pyctf-lite@v1.0#egg=pyctf-lite', 'wget'],
    scripts=['cmdline/make_bids_fs_swarm.sh',
        'nih2mne/bstags.py',
        'nih2mne/calc_mnetrans.py',
        'nih2mne/make_meg_bids.py',
        'nih2mne/utilities/make_meg_bids_fromcsv.py',
        'nih2mne/utilities/print_bids_table.py',
        'nih2mne/utilities/fix_dsname.py',
        'nih2mne/eyetracking_preprocessing.py',
        'nih2mne/megcore_prep_mri_bids.py',
        'nih2mne/utilities/qa_fids.py'],
)
