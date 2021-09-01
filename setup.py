import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nih2mne", 
    version="0.1",
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
    python_requires='<3.9',
    install_requires=['mne', 'numpy', 'pytest', 'joblib'],
    scripts=['nih2mne/bstags.py',
        'nih2mne/bids_to_mnetrans.py'],
)
