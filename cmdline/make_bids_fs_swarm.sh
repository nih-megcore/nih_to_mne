#!/bin/bash

shopt -s extglob 
shopt -s globstar

SUBJECTS_DIR=$(pwd)/derivatives/freesurfer/subjects 

if [ ! -d ${SUBJECTS_DIR} ]
then 
	mkdir -p ${SUBJECTS_DIR}
fi

if [ ! -d logdir ]
then 
	mkdir logdir
fi

subj_paths=$(ls -d $(pwd)/sub-*)

#Create the swarm file - may want to put tests for multiple T1w.nii.gz
for i in ${subj_paths}; do subjid=$(basename $i); echo recon-all -all -i ${subjid}/ses-1/anat/*T1w.nii.gz -s ${subjid} >> swarm_fs.sh; done
echo Here are the last 5 entries:
tail -5 swarm_fs.sh

read -p "Check the swarm_fs.sh file.  Press y to run and n to exit" quit_val
if [ ${quit_val} == y ] 
then
	module load freesurfer
	export SUBJECTS_DIR=${SUBJECTS_DIR}
	swarm -f ./swarm_fs.sh -g 6 -t 2 --time=24:00:00 --logdir=./logdir
else
	echo Exiting without running
fi
