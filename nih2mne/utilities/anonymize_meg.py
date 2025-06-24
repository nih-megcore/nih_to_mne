import argparse
import numpy
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from make_meg_bids import anonymize_meg
from make_meg_bids import anonymize_finalize


def cli():
    parser = argparse.ArgumentParser(description='Anonymize all MEG data in a BIDS directory using CTF tools.')

    parser.add_argument('bids_indir', type=Path, help='Input BIDS directory containing MEG data participant directories formatted as "sub-*"')
    parser.add_argument('bids_outdir', type=Path, help='Output BIDS directory to deposit anonymized data')

    args = parser.parse_args()

    if not args.bids_indir.is_dir():
        raise ValueError(f"Input BIDS directory does not exist or is not a directory: {args.bids_indir}")

    if str(args.bids_indir.resolve()) == str(args.bids_outdir.resolve()):
        raise ValueError(f"Input BIDS directory and output BIDS directory are identical (which is not allowed): {args.bids_indir}")

    if not args.bids_outdir.parent.is_dir():
        raise ValueError(f"Output BIDS directory's parent does not exist or is not a directory: {args.bids_outdir.parent}")
    else:
        args.bids_outdir.mkdir(parents=True, exist_ok=True)

    return args.bids_indir, args.bids_outdir


if __name__ == '__main__':
    # run the CLI
    indir, outdir = cli()

    # crawl all subdirectories in indir looking for MEG data and call the anonymization function on each *.ds directory
    inmegs = [g for g in indir.glob('sub-*/meg/*.ds')] + [g for g in indir.glob('sub-*/ses-*/meg/*.ds')]

    # collect all unique parents of meg/*.ds directories
    parents = numpy.unique([inmeg.parent for inmeg in inmegs])

    for inmeg in inmegs:
        outmeg = outdir / inmeg.relative_to(indir).parent
        outmeg.parent.mkdir(parents=True, exist_ok=True)
        print(f"Anonymizing {inmeg} to {outmeg}")

        # anonymize the MEG data from inmeg to outmeg
        anonymize_finalize(anonymize_meg(inmeg.resolve(), outmeg.resolve()))

    # copy over parent-level JSON and TSV files from the parents of *.ds folders
    print(f"Copying in JSONs and TSVs from parent directories")
    for parent in parents:
        jsons_tsvs = [g for g in parent.glob('*.json')] + [g for g in parent.glob('*.tsv')]
        for in_json_tsv in jsons_tsvs:
            out_json_tsv = outdir / in_json_tsv.relative_to(indir)
            shutil.copyfile(in_json_tsv.resolve(), out_json_tsv.resolve())

    print(f"Anonymization complete. Anonymized data saved to: {outdir}")
