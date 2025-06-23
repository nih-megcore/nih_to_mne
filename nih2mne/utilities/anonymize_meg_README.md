# `anonymize_meg.py`

This script is used to anonymize all MEG data that's been converted by these nih2mne BIDS scripts before, but was not anonymized the first time around.

It takes as input:

1. The previously created (ant not yet anonymous) BIDS-converted data root directory ; and
2. The new-to-you temporary output directory to create the whole anonymized BIDS tree within.

```shell
usage: anonymize_meg.py [-h] bids_indir bids_outdir

Anonymize all MEG data in a BIDS directory using CTF tools.

positional arguments:
  bids_indir   Input BIDS directory containing MEG data participant directories formatted as "sub-*"
  bids_outdir  Output BIDS directory to deposit anonymized data

options:
  -h, --help   show this help message and exit
```
