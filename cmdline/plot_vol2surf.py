#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:29:11 2024

@author: jstout
"""

import nilearn
from nilearn import datasets
import numpy as np
from nilearn import surface
from nilearn import plotting
from matplotlib.pyplot import subplots


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-threshold', default=0.1, type=float)
    parser.add_argument('-group_nii',
                          help='This is the group data that hase been volumetrically warped to fsaverage'
                          )
    args = parser.parse_args()
    threshold = args.threshold
    ave_fname = args.group_nii

fsaverage = datasets.fetch_surf_fsaverage()
curv_right = surface.load_surf_data(fsaverage.curv_right)
curv_right_sign = np.sign(curv_right)

curv_left = surface.load_surf_data(fsaverage.curv_left)
curv_left_sign = np.sign(curv_left)

texture_rh = surface.vol_to_surf(ave_fname, fsaverage.pial_right)
texture_lh = surface.vol_to_surf(ave_fname, fsaverage.pial_left)



fig, ax = subplots(2,2, subplot_kw={'projection': '3d'})

## Lateral
plotting.plot_surf_stat_map(
    fsaverage.infl_left, texture_lh, hemi='left',
    title='Surface left hemisphere', colorbar=True,
    threshold=threshold, bg_map=curv_right_sign, axes=ax[0,0]
)

plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture_rh, hemi='right',
    title='Surface right hemisphere', colorbar=True,
    threshold=threshold, bg_map=curv_right_sign, axes=ax[0,1]
)

## Medial
plotting.plot_surf_stat_map(
    fsaverage.infl_left, texture_lh, hemi='left',
    title='Surface left hemisphere', view='medial',colorbar=True,
    threshold=threshold, bg_map=curv_right_sign, axes=ax[1,0]
)

plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture_rh, hemi='right',
    title='Surface right hemisphere', view='medial', colorbar=True,
    threshold=threshold, bg_map=curv_right_sign, axes=ax[1,1]
)

fig.show()
_ = input('Enter anything to close the plot')
