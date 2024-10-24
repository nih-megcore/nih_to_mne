#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:37:34 2024

@author: jstout
"""

import os, os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
import json




import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mri', 
                        help='Nifti format MRI used with brainsight', 
                        required=True
                        )
    parser.add_argument('-elec', 
                        help = 'Electrodes.txt file name', 
                        required=True
                        )
    parser.add_argument('-plot',
                        default=False,
                        help='Plot the results in a triaxial image', 
                        action='store_true')
    args = parser.parse_args()


elec_fname = args.elec
mri_fname = args.mri
do_plot = args.plot

#%%
mri = nib.load(mri_fname)
dframe = pd.read_csv(elec_fname, skiprows=6, sep='\t')

locs_ras = {}
for val in ['Nasion', 'Left Ear', 'Right Ear']:
    row = dframe[dframe['# Electrode Name']==val]
    tmp = row['Loc. X'], row['Loc. Y'], row['Loc. Z']
    output = [i.values[0] for i in tmp]
    locs_ras[val] = np.array(output)


# set the fids as voxel coords
inv_rot = np.linalg.inv(mri.affine[0:3,0:3])
translation =  mri.affine[0:3,3]
nas_vox = np.matmul(inv_rot, locs_ras['Nasion']) - translation
lpa_vox = np.matmul(inv_rot, locs_ras['Left Ear']) - translation
rpa_vox = np.matmul(inv_rot, locs_ras['Right Ear']) - translation


# if json exists read it

# else create it

fids_json_out = {"AnatomicalLandmarkCoordinates": {
    "NAS":list(nas_vox),
    "LPA":list(lpa_vox), 
    "RPA":list(rpa_vox)
    }}

if mri_fname.endswith('.gz'):
    json_fname = mri_fname.replace('.nii.gz', '.json')
else:
    json_fname = mri_fname.replace('.nii','.json')

# Write the json
with open(json_fname, 'w') as f:
    json.dump(fids_json_out, f)


if do_plot:
    from matplotlib import pyplot as plt
    from nilearn.plotting import plot_anat
    
    # Do a full read of the json
    with open(json_fname, 'r') as f:
        json_out = json.load(f)
    nas_vox = json_out['AnatomicalLandmarkCoordinates']['NAS']
    lpa_vox = json_out['AnatomicalLandmarkCoordinates']['LPA']
    rpa_vox = json_out['AnatomicalLandmarkCoordinates']['RPA']
    
    nas = np.matmul(mri.affine[0:3,0:3] , nas_vox) + mri.affine[0:3,3]
    lpa = np.matmul(mri.affine[0:3,0:3] , lpa_vox) + mri.affine[0:3,3]
    rpa = np.matmul(mri.affine[0:3,0:3] , rpa_vox) + mri.affine[0:3,3]
    
    mri_pos = {'LPA':lpa, 'NAS': nas , 'RPA': rpa}
    
    # Plot it
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), facecolor="k")
    for point_idx, label in enumerate(("LPA", "NAS", "RPA")):
        plot_anat(
            mri_fname,
            axes=axs[point_idx],
            cut_coords=mri_pos[label],#, :],
            title=label,
            vmax=160,
        )
    plt.show(block=True)


    