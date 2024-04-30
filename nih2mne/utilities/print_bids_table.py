import pandas as pd
import glob 

import os, os.path as op
import mne_bids


def get_sessions(subjid):
    return [i for i in os.listdir(subjid) if i[0:3]=='ses']

def get_megs(subjid, ses='1', extension='.ds'):
    '''Return all MEG scans'''
    candidate_ses = glob.glob(f'{subjid}/ses-*')
    candidate_ses = [i.split('-')[-1] for i in candidate_ses]
    if '-' in ses:
        ses==ses.split('-')[-1]
    if ses not in candidate_ses:
        ses='0'+ses
        if ses not in candidate_ses:
            return None
    ses='ses-'+ses
    megs = glob.glob(f'{bids_dir}/{subjid}/{ses}/meg/*{extension}')
    return megs
    
def get_tag_output(fname, tag='task'):
    tmp = op.basename(fname)
    return tmp.split(f'{tag}-')[-1].split('_')[0]

def get_bids_table(bids_dir, ses='1'):
    init_dir = os.getcwd()
    os.chdir(bids_dir)
    subjids = glob.glob('sub-*')
    dsets=[]
    for sub in subjids:
        dsets+=get_megs(sub, ses=ses)
    dframe=pd.DataFrame(dsets, columns=['fname'])
    dframe['task']=dframe.fname.apply(get_tag_output, tag='task')
    dframe['run']=dframe.fname.apply(get_tag_output, tag='run')
    dframe['subjid']=dframe.fname.apply(get_tag_output, tag='sub')
    os.chdir(init_dir)
    return dframe
                                     

def main():    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_dir', 
                        help='Location of the bids directory')
    parser.add_argument('-session',
                        help='Session of data acq',
                        default='1')
    parser.add_argument('-output_fname', 
                        help='''If set the value counts of all the tasks will
                        be written to a csv table.  This has more information
                        than the print_task_counts''')
    parser.add_argument('-print_task_counts',
                        help='''Print out the number of task runs 
                        and number of subjects in the bids dataset''',
                        action='store_true')
    args = parser.parse_args()
    bids_dir=args.bids_dir 
    dframe = get_bids_table(bids_dir, ses=args.session)
    
    if args.print_task_counts:
        num_subjs = len(dframe.subjid.unique())
        num_tasks = len(dframe.task.unique())
        print(f'There are {num_subjs} subjects in the dataset\n')
        print(dframe.subjid.unique())
        print(f'\n\n')
        print(f'There are {num_tasks} tasks in the dataset:{dframe.task.unique()}\n')
        print('Task Count')
        print(dframe.task.value_counts())
    
    if args.output_fname:
        out_fname = op.abspath(args.output_fname)
        pivot = dframe.pivot_table(columns='task', index='subjid', aggfunc='count')
        pivot.to_csv(out_fname)
        print(f'Table saved to {out_fname}')
        
if __name__ == '__main__':
    main()    
    
    
