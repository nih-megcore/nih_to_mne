"""

"""

import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat
import mne_bids
import glob
import json
import nibabel as nib

def plot_fids_qa(subjid=None, bids_root=None, outfile=None, block=False, 
                 mri_override=None):
    '''
    Plot triaxial images of the fiducial locations and save out image

    Parameters
    ----------
    subjid : str, 
        Subject ID . 
    bids_root : str | path
        Top directory of BIDS dataset
    outfile : str
        Defaults to current directory {subject}_fids_qa.png

    Returns
    -------
    None.

    '''
    if subjid[0:4]!='sub-':
        subjid='sub-'+subjid
    if outfile == None:
        outfile = op.join(os.getcwd(), f'{subjid}_fids_qa.png')
    if mri_override != None:
        tmp = mri_override
    else:
        tmp = glob.glob(op.join(bids_root, subjid, '**/*T1w.nii.gz'), recursive=True)
    if len(tmp) == 0:
        tmp = glob.glob(op.join(bids_root, subjid, '**/*T1w.nii'), recursive=True)
        if len(tmp) == 0:
            raise ValueError(f"{subjid} :: Could not find T1w as a *T1w.nii or *T1w.nii.gz")
    if len(tmp) > 1:
        raise ValueError(f"{subjid} :: More than one T1w files")
            
    t1w_bids_path = mne_bids.get_bids_path_from_fname(tmp[0])
    jsonfile = str(t1w_bids_path.copy().update(extension='.json'))
    
    mr = nib.load(t1w_bids_path)
    
    with open(jsonfile, 'r') as f:
        json_out = json.load(f)
    nas_vox = json_out['AnatomicalLandmarkCoordinates']['NAS']
    lpa_vox = json_out['AnatomicalLandmarkCoordinates']['LPA']
    rpa_vox = json_out['AnatomicalLandmarkCoordinates']['RPA']
    
    nas = np.matmul(mr.affine[0:3,0:3] , nas_vox) + mr.affine[0:3,3]
    lpa = np.matmul(mr.affine[0:3,0:3] , lpa_vox) + mr.affine[0:3,3]
    rpa = np.matmul(mr.affine[0:3,0:3] , rpa_vox) + mr.affine[0:3,3]
    
    mri_pos = {'LPA':lpa, 'NAS': nas , 'RPA': rpa}
    
    # Plot it
    if block==False:
        fig, axs = plt.subplots(3, 1, figsize=(7, 7), facecolor="k")
        for point_idx, label in enumerate(("LPA", "NAS", "RPA")):
            plot_anat(
                str(t1w_bids_path),
                axes=axs[point_idx],
                cut_coords=mri_pos[label],#, :],
                title=label,
                vmax=160,
                output_file = outfile
            )
        plt.show()
    else:
        fig, axs = plt.subplots(3, 1, figsize=(7, 7), facecolor="k")
        for point_idx, label in enumerate(("LPA", "NAS", "RPA")):
            plot_anat(
                str(t1w_bids_path),
                axes=axs[point_idx],
                cut_coords=mri_pos[label],#, :],
                title=label,
                vmax=160,
            )
        plt.show(block=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject',
                        help='Subject ID - with or without sub- prefix', 
                        default=False)
    parser.add_argument('-loop_all', 
                        help='Alternative to -subject flag.  Loops over all subjects',
                        action='store_true',
                        default=False
                        )
    parser.add_argument('-bids_root',
                        help='BIDS root of the project', 
                        required=True)
    parser.add_argument('-out_image', 
                        help='''Image output path.  Defaults to current directory
                        subjid_fids_qa.png'''
                        )
    args = parser.parse_args()
    bids_root = op.abspath(args.bids_root)
    if args.subject == False:
        if args.loop_all == False:
            raise ValueError('Need to input either a subject or loop_all flag')
    if args.subject != False:
        plot_fids_qa(subjid=args.subject, bids_root=bids_root, outfile=None)
    elif args.loop_all == True:
        qa_dir = op.join(op.dirname(bids_root), 'QA_fids')
        if not op.exists(qa_dir):
            os.mkdir(qa_dir)
        os.chdir(qa_dir)
        subjects = glob.glob(op.join(bids_root, 'sub-*'))
        subjects = [op.basename(i) for i in subjects]
        for subject in subjects:
            print(f'Running subject {subject}')
            try:
                plot_fids_qa(subjid=subject, 
                             bids_root=bids_root)
            except:
                print(f'Failed to create QA image for {subject}')
                         
if __name__ == '__main__':
    main()    
    