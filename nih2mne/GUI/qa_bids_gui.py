#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:50:57 2024

@author: jstout

TODO: 
    1)Functionalize the window loop operations, so they can be called outside 
    of the while loop - basically this is preventing the read_cfg operation from
    changing the layout.
    2)Button to launch full logfile read
    3) Button to view MRI coreg
    4) Button to view 3D coreg in helmet (requires runtime to eval headsurf)

Code layout:
    1) Define call options (opts)
        a) Override default opts with config file
    2) Define layout based on options    
    3) Read / Write Config options from opts object
    4) RUN: Format opts object into commandline string and execute
        After running - open Summary text
    
                       
"""
import PySimpleGUI as sg
import os,os.path as op
import glob
import subprocess

CFG_VERSION = 1.0

font = ("Arial", 25)
sg.set_options(font=font)

global size_mult, font_size, x_size, y_size
size_mult=2
font_size=12  
x_size = 500*size_mult
y_size = 600*size_mult

def collapse(layout, key, visible):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this section visible / invisible
    :param visible: visible determines if section is rendered visible or invisible on initialization
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key, visible=visible, pad=(0,0)))

class window_opts:
    def __init__(self, config=False):
        # Button Entries
        self.anonymize = False
        self.ignore_mri_checks = False

        # Standard Entries
        self.bids_dir = op.join(os.getcwd(), 'bids_dir')
        self.subjects = self.get_subjs()
        
        
        self.meg_input_dir = None
        self.bids_session = 1
        self.subjid_input = None
        self.bids_id = None
        self.coreg = 'Brainsight'

        ## Afni Coreg:
        self.mri_brik = None

        ## Brainsight Coreg:
        self.mri_bsight = None
        self.mri_bsight_elec = None

        ## Optional Overrides:
        self.ignore_eroom = False
        self.autocrop_zeros = False
        self.freesurfer = None
        self.eventID_csv = None
        # Run standardize_eventID_list.py
        
        self.config = config
        if config != False:
            write_opts = read_cfg(config)
            self.update_opts(opts=write_opts)
    
    def update_opts(self, opts=None):
        for key, val in opts.items():
            if hasattr(self, key):
                setattr(self, key, val)
    
    def get_subjs(self):
        '''List the subjects in the bids directory'''
        tmp_ = glob.glob(op.join(self.bids_dir, 'sub-*'))
        return [op.basename(i) for i in tmp_]
    
    
    

def read_cfg(cfg_fname):
    with open(cfg_fname, 'r') as f:
        lines = f.readlines()
    lines = [i.strip('\n') for i in lines]
    write_opts={}
    for i in lines:
        _key, _val = i.split(':')
        if _val == 'True':
            _val=True
        elif _val == 'None':
            _val=None
        elif _val == 'False':
            _val=False
        write_opts[_key]=_val
    return write_opts
    
        

def make_layout(opts=None):
    '''Generate 3 sections - Project / Subject_RAW / Subject_Preproc'''  
    required_opts = [
        [sg.Text('BIDS DIR:'), 
         sg.InputText(key='-BIDS_DIR-', enable_events=True),
         sg.FolderBrowse(target='-BIDS_DIR-')
         ],
        ]
           
    project_opts = [
        [sg.Text(' -- PROJECT_QA -- ')],
        [sg.Button('Compute Table'), sg.Button('View Table', disabled=True)], 
        [sg.Button('Missing Data', disabled=True)], 
        [sg.Button('Assess Noise Levels', disabled=True)]
        ]
    
    subject_required_opts = [
        [sg.Text(' -- Subject Section -- ')],
        [sg.Button('Select Subject')]
         ]
    subject_raw_opts = [
        [sg.Text(' -- SUBJECT INPUT QA -- ')],
        [sg.Button('MRI ANON', disabled=True),
         sg.Button('Fiducials', disabled=True)],
        [sg.Button('3D Coreg', disabled=True),
         sg.Button('Data QA', disabled=True)]
        ]
    
    subject_preproc_opts = [
        [sg.Text('  -- SUBJECT Preproc QA --')],
        [sg.Button('Run MR Prep', disabled=False), sg.Button('Run MR Prep (VOL)', disabled=False)]
        ]    
         
    # -- Assemble Layout --
    layout = required_opts
    layout.append([sg.Text(' ')])
    layout.append(project_opts)
    layout.append([sg.Text(' ')])
    layout.append(subject_raw_opts)
    layout.append([sg.Text(' ')])
    layout.append(subject_preproc_opts)
    layout.append([sg.Text(' ')])
    layout.append([sg.Button('Print CMD', key='-PRINT_CMD-'), sg.Button('RUN', key='-RUN-'), 
                   sg.Button('Write Cfg', key='-WRITE_CFG-'), sg.Button('EXIT')])
    return layout
        
def selector_POPUP(data, text_item='Select'):
    layout = [
        [sg.Text(text_item)],
        [sg.Listbox(data, size=(20,5), key='SELECTED')],
        [sg.Button('OK')],
    ]
    window = sg.Window('POPUP', layout).Finalize()
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'OK':
            break
        else:
            print('OVER')
    window.close()
    if values and values['SELECTED']:
        return values['SELECTED']
    
def write_cfg(opts, fname=None):
    '''
    From a list of writeable options - save these in key:value pair format sep=:
    '''
    save_opt_list = ['anonymize', 'autocrop_zeros', 'freesurfer', 'ignore_eroom',
                    'ignore_mri_checks', 'coreg', 'bids_dir', 'bids_session']
    write_opts = []
    for opt_tag in save_opt_list:
        if hasattr(opts, opt_tag):
            write_opts.append(f'{opt_tag}:{getattr(opts,opt_tag)}\n')
    with open(fname, 'w') as f:
        f.writelines(write_opts)

def get_window(options=None):
    layout = make_layout(opts=options)
    window = sg.Window('MEG BIDS conversion', layout, resizable=True, auto_size_buttons=True, 
                   scaling=True, size=(x_size, y_size))
    del layout
    return window

single_flag_list = ['anonymize', 'autocrop_zeros', 'freesurfer', 'ignore_eroom',
                    'ignore_mri_checks']
drop_flag_list = ['coreg', 'read_from_config', 'config', 'update_opts', 'error_log', 'full_log', 'fids_qa']
def format_cmd(opts):
    '''
    Write out the commandline options from the opts object.  Special cases 
    for the single flag vs flag w/option entry.

    Parameters
    ----------
    opts : opt object
        DESCRIPTION.

    Returns
    -------
    cmd : str

    '''
    arglist = ['make_meg_bids.py']
    for i in value_writedict.values():
        if i in drop_flag_list:
            if (i == 'coreg') and (opts.coreg == 'None'):
                arglist.append('-ignore_mri_checks')
                continue
            else:
                continue
        flag_val =  getattr(opts, i)
        if i in single_flag_list:
            if flag_val == True:
                arglist += [f'-{i}']
            else:
                continue
        else:
            if flag_val != None:
                arglist += [f'-{i} {getattr(opts, i)}']
    cmd = ' '.join(arglist)
    return cmd 

config_fname = False
if __name__=='__main__':
    ''' Get config file from the commandline - preset to False above'''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='Config file: typically ending in cfg', 
                        default=False)
    args = parser.parse_args()
    if hasattr(args, 'config'):
        config_fname = args.config

def qa_gui(config_fname=False):    
    ## Setup and run gui
    opts = window_opts(config=config_fname) #This defaults to False if not set
    window = get_window(opts)
    
    # Conversion dictionary between GUI variables and cmdline vars
    _tmp = [i for i in dir(opts) if not i.startswith('_')]
    global value_writedict
    value_writedict = {f'-{i.upper()}-':i for i in _tmp}
    
    while True:
        event, values = window.read()
        
        if event == '-READ_CFG-':
            cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', 
                                          default_extension='.cfg')
            write_opts = read_cfg(cfg_fname)
            opts.update_opts(write_opts)
        
        # Update object options if event triggered
        if event in value_writedict.keys():
            setattr(opts, value_writedict[event], values[event])
        
        ## BIDS PROJECT LEVEL
        if event == 'Compute Table':
            from nih2mne.utilities.print_bids_table import gui_interface as pbids_table
            opts.bids_table_fname = op.join(op.dirname(opts.bids_dir), 'bids_prep_logs','BIDS_table.csv')
            pbids_table(opts.bids_dir, out_fname =opts.bids_table_fname)
            window['View Table'].update(disabled=False)
        
        if event == 'View Table':
            cmd = f'xdg-open {opts.bids_table_fname}'
            subprocess.run(cmd.split())
            
        ## BIDS SUBJECT LEVEL QA 
        
        
        ## MRI Preprocess
        if (event == 'Run MR Prep') or (event == 'Run MR Prep (VOL)'):
            subjects = sorted(glob.glob('sub-*', root_dir=opts.bids_dir))
            subject = selector_POPUP(subjects, text_item='Select a subject')[0]
            dsets = glob.glob(f'**/{subject}_*.ds', recursive=True, root_dir=op.join(opts.bids_dir, subject))
            dsets = ['ALL'] + dsets
            dsets_sel = selector_POPUP(dsets, text_item = 'Select a dataset to run MR preprocessing')
            if dsets_sel[0] == 'ALL':
                dsets.remove('ALL')
                dsets_sel = dsets
            #Add full path for processing
            dsets_sel = [op.join(opts.bids_dir, subject, i) for i in dsets_sel]
            for dset in dsets_sel:
                cmd = f'megcore_prep_mri_bids.py -bids_root {opts.bids_dir} -filename {dset}'
                if event == 'Run MR Prep (VOL)':
                    cmd+=' -volume'
                subprocess.run(cmd.split())
            
        
        # Logic for displaying coreg options
        if event == '-COREG-':
            if values['-COREG-'] == 'BrainSight':
                window['-COREG_BSIGHT-'].update(visible=True)
                window['-COREG_AFNI-'].update(visible=False)
            elif values['-COREG-'] == 'Afni':
                window['-COREG_AFNI-'].update(visible=True)
                window['-COREG_BSIGHT-'].update(visible=False)
            elif values['-COREG-'] == 'None':
                window['-COREG_BSIGHT-'].update(visible=False)
                window['-COREG_AFNI-'].update(visible=False)
        
        # If input directory is chosen - automatically select subject id and set
        if event == '-MEG_INPUT_DIR-':
            search_dir = values['-MEG_INPUT_DIR-']
            tmp = glob.glob(op.join(search_dir, '*.ds'))
            tmp = [op.basename(i) for i in tmp]
            tmp = list(set([i.split('_')[0] for i in tmp]))
            if len(tmp) == 1:
                values['-SUBJID_INPUT-']=tmp[0]
                opts.subjid_input =tmp[0]
                window['-SUBJID_INPUT-'].update(tmp[0])
            if len(tmp) > 1:
                tmp = subject_selector_POPUP(tmp)
                window['-SUBJID_INPUT-'].update(tmp[0])
                opts.subjid_input=tmp[0]
                
        if event == '-PRINT_CMD-':
            cmd = format_cmd(opts)
            print(cmd)
        
        if event == '-WRITE_CFG-':
            cfg_fname = sg.popup_get_file('ConfigFile ending in .cfg', default_path='nih_bids_gui.cfg', 
                                          default_extension='.cfg')
            write_cfg(opts, fname=cfg_fname)
        
        
        if event == '-RUN-':
            print(f'Running the command: {cmd}')
            cmd = format_cmd(opts)
            out_txt = subprocess.run(cmd.split(), check=True, capture_output=True)
            summary = []
            _start = False
            for i in str(out_txt.stdout).split('\\n'):
                if '########### SUMMARY #################' in i:
                    _start = True
                if _start:
                    summary.append(i)
            sg.popup_get_text('\n'.join(summary), title='SUMMARY')
            _tmp = op.dirname(opts.bids_dir)
            setattr(opts, 'error_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_err_log.txt'))
            setattr(opts, 'full_log',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_log.txt'))
            setattr(opts, 'fids_qa',  op.join(_tmp, 'bids_prep_logs' , opts.subjid_input + '_fids_qa.png'))  
            window['-CHECK_TRIAX_COREG-'].update(disabled=False)
            print('FINISHED')
            
        if event == '-CHECK_TRIAX_COREG-':
            subprocess.run(f'xdg-open {opts.fids_qa}'.split())
        if event == sg.WIN_CLOSED or event == 'EXIT': # if user closes window or clicks cancel
            break
        
    window.close()

#%%
# Main window
# Check bids data

bidsroot_template = 'bidsroot'
projectroot_template = 'projectroot'


# Data checks 
data_checks = {
    'MEGraw':{'key':
              '*.ds'
              },
    'MRIraw':{'key':
              ['*.nii.gz', '*.nii']
              },
    'MRIfree':{'key':
                   ['surf/lh.pial', 'surf/rh.pial'],
               'logkey':[]
                   },
    'Coreg':[   ], 
    'MRIprep':[  ],
    }
    
def check_fs_recon(subjid, subjects_dir):
    '''
    Returns status of freesurfer reconstruction

    Parameters
    ----------
    subjid : str
        Subject ID
    subjects_dir : str
        Freesurfer subjects dir

    Returns
    -------
    out_dict: dict
        

    '''
    logfile = op.join(subjects_dir, subjid, 'scripts', 'recon-all.log')
    with open(logfile) as f:
        fs_success_line = f.readlines()[-1]
    if 'finished without error' in fs_success_line:
        finished = True
    else:
        finished = False
    has_lhpial = os.path.exists(op.join(subjects_dir, subjid, 'surf', 'lh.pial'))
    has_rhpial = os.path.exists(op.join(subjects_dir, subjid, 'surf', 'rh.pial'))
    out_dict = dict(fs_success = finished,
                lhpial = has_lhpial, 
                rhpial = has_rhpial)
                
    return out_dict


class qa_megraw_object:
    '''Current minimal template - add more later'''
    def __init__(self, fname):
        self.rel_path = fname
        self.fname = op.basename(fname)
        self._get_task()
        self._is_emptyroom()
    
    def _get_task(self):
        tmp = self.fname.split('_')
        task_stub = [i for i in tmp if i[0:4]=='task'][0]
        self.task = task_stub.split('-')[-1]
        
    def _is_emptyroom(self):
        eroom_tags = ['empty', 'er','emptyroom', 'noise']
        if self.task.lower() in eroom_tags:
            self.is_emptyroom = True
        else:
            self.is_emptyroom = False
    
    def load(self, load_val=False):
        self.raw = mne.io.read_raw_ctf(self.rel_path, 
                                       preload=load_val,
                                       system_clock='ignore',
                                       clean_names=True)
    def _calc_bad_segments(self):
        'Template for calculation'
    
    def _calc_jumps(self):
        'Template for calc'
    
    def _is_valid(self, set_value=None):
        '''Fill in more of this -- maybe '''
        if set_value != None:
            self.is_valid = set_value
        else:
            self.is_valid = True
        
        

    def __repr__(self):
        return f'megraw: {self.task} : {self.fname}'
            

class meglist_class:
    def __init__(self, subject=None, bids_root=None):
        if subject[0:4]!='sub-':
            subject='sub-'+subject
        dsets = glob.glob(f'{op.join(subject, "**", "*.ds")}',
                          root_dir=bids_root, recursive=True)
        tmp = [qa_megraw_object(i) for i in dsets]
        self.meg_list = tmp
    
    @property
    def meg_count(self):
        return len(self.meg_list)
    
class qa_mri_class:    
    def __init__(self, subject=None, bids_root=None):
        tmp = glob.glob(f'{op.join(subject,"**", "anat/*")}')
        all_mri_list = []
        for i in tmp:
            if (tmp[-4:]=='.nii') or (tmp[-6]=='.nii.gz'):
                all_mri_list.append(i)
        self.all_mris = all_mri_list
        
    def _sort_T1(self):
        
        
    

class subject_tile(qa_mri_class, meglist_class):
    def __init__(self, subject, bids_root=None, subjects_dir=None):
        self.subject = subject
        if bids_root==None:
            self.bids_root=os.getcwd()
        else:
            self.bids_root = bids_root
        
        if subjects_dir==None:
            self.subjects_dir = op.join('derivatives','freesurfer','subjects')
        else:
            self.subjects_dir = subjects_dir
        
        # MEG component
        self.meg_list = meglist_class(subject, self.bids_root)
        self.fs_recon = check_fs_recon(subject, self.subjects_dir)
        
        
        # # MRI component
        # mri_stuff = mri_class(self, 
        # self.mri = 
        
        
        
        
        

# Test Section
topdir = '/fast2/BIDS'
tmp = meglist_class(subject='ON08710')
tmp.meg_list
test= tmp.meg_list[0]
test.load()

















