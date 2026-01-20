#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:53:16 2025

@author: jstout

Operational inputs from the BIDS_creator_gui.py 

# pb (pushbutton), te (text edit), list (list), 
pb_Anonymize   #flipflop toggle Y/N
te_meghash  #MEGHASH manual entry -- fill from inputs
te_BIDS_id  #BIDS id - manual entry only
te_bids_dir #Text edit -- filled by pb_BIDSdir selection
pb_BIDS_dir  #Push button to get file/directory browser selector
cb_Bids_Session #Combobox with bids session options
te_BRIKfname #Manual entry set by pb if used
pb_BRIKfname #File selector
te_brainsight_elec #Filled by pb_brainsight --- Do a datacheck upon selection for formatting
pb_BrainsightElec #File selection popup - fill te_brainsight_elec
te_brainsight_mri #Brainsight text
pb_BrainsightMRI #File selector, fill te_brainsight_mri
list_fname_conversion   #Order datasets -- but also check current bidsDIR and runs...  
cb_crop_zeros #Checkbox do crop zero operation
cb_emptyroom  #Checkbox do pull emptyroom from biowulf
pb_print_cmd  #Print command
pb_run  #Run operation
**pb_check_inpout #pushbutton to read current bids config, update run outputs, in -->> out naming in list

"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from nih2mne.GUI.templates.BIDS_creator_gui import Ui_MainWindow
import sys
import os, os.path as op
from nih2mne.make_meg_bids import make_bids
from nih2mne.make_meg_bids import _read_electrodes_file
from nih2mne.calc_mnetrans import coords_from_oblique_afni
from nih2mne.config import DEFAULTS
import shutil
from mne_bids import BIDSPath
from nih2mne.make_meg_bids import _gen_taskrundict, _proc_meg_bids
from collections import OrderedDict

#%% Setup Defaults for GUI browse functions
BIDS_DEFAULTS = DEFAULTS['BIDS_gen']
NULL_VALS = ['', None, 'None', []]
if BIDS_DEFAULTS['bids_root'] not in NULL_VALS:
    DEFAULT_BIDS_ROOT = BIDS_DEFAULTS['bids_root']
else:
    DEFAULT_BIDS_ROOT = op.join(os.getcwd(), 'BIDS')

if BIDS_DEFAULTS['mri_dir'] not in NULL_VALS:
    DEFAULT_MRI_ROOT = BIDS_DEFAULTS['mri_dir']
else:
    DEFAULT_MRI_ROOT = os.getcwd()

if BIDS_DEFAULTS['coreg_type'] not in NULL_VALS:
    DEFAULT_MRI_TAB = BIDS_DEFAULTS['coreg_type']
else:
    DEFAULT_MRI_TAB = 'Brainsight'

if BIDS_DEFAULTS['anonymize'].upper() in ['Y','N']:
    if BIDS_DEFAULTS['anonymize'].upper() == 'Y':
        DEFAULT_ANONYMIZE = True
    else:
        DEFAULT_ANONYMIZE = False
else:
    DEFAULT_ANONYMIZE = False

if BIDS_DEFAULTS['crop_zeros'].upper() in ['Y','N']:
    if BIDS_DEFAULTS['crop_zeros'] == 'Y':
        DEFAULT_CROPZ = True
    else:
        DEFAULT_CROPZ = False
else:
    DEFAULT_CROPZ = False
    
if BIDS_DEFAULTS['emptyroom'].upper() in ['Y','N']:
    if BIDS_DEFAULTS['emptyroom'] == 'Y':
        DEFAULT_EROOM = True
    else:
        DEFAULT_EROOM = False
else:
    DEFAULT_EROOM = False

if BIDS_DEFAULTS['bids_session_list'][0] not in [None, False]:
    DEFAULT_BIDS_SESSION_LIST = [str(i).strip() for i in BIDS_DEFAULTS['bids_session_list']]
    if BIDS_DEFAULTS['bids_session'] not in [None, False]:
        DEFAULT_BIDS_SESSION = str(BIDS_DEFAULTS['bids_session'])
    else:
        DEFAULT_BIDS_SESSION = str(BIDS_DEFAULTS['bids_session_list'][0])
else:
    DEFAULT_BIDS_SESSION_LIST = ['1']
    DEFAULT_BIDS_SESSION = '1'
    
    

#%% 

class BIDS_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, meghash='None', bids_id='None', meg_dsets=None):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Collect all bids options in self.opts
        self.opts = dict(anonymize=DEFAULT_ANONYMIZE, 
                         subjid_input=meghash, 
                         bids_id=bids_id,
                         bids_dir=DEFAULT_BIDS_ROOT, 
                         bids_session=DEFAULT_BIDS_SESSION,
                         meg_dataset_list = meg_dsets,
                         
                         #MRI_none
                         mri_none = True,
                         #MRI_bsight
                         mri_bsight = False,
                         mri_elec = False,
                         #MRI_afni
                         mri_brik = False,
                         
                         #Options
                         crop_zeros=DEFAULT_CROPZ,
                         include_empty_room=DEFAULT_EROOM,
                         
                         )
        
        #### Fill out default text in text edit lines 
        self.ui.te_meghash.setPlainText(str(self.opts['subjid_input']))
        self.ui.te_BIDS_id.setPlainText(str(self.opts['bids_id']))
        self.ui.te_bids_dir.setPlainText(str(self.opts['bids_dir']))
        
        #### Fill combobox
        self.ui.cb_Bids_Session.addItems(DEFAULT_BIDS_SESSION_LIST)
        
        ### Connect TextEdit lines
        self.ui.te_meghash.textChanged.connect(self._update_meghash)
        self.ui.te_BIDS_id.textChanged.connect(self._update_bids_id)
        self.ui.te_bids_dir.textChanged.connect(self._update_bids_dir)
        self.ui.cb_Bids_Session.currentIndexChanged.connect(self._update_bids_ses)
        
        ### Connect buttons    
        self.ui.pb_Anonymize.clicked.connect(self._action_pb_anonymize)   #flipflop toggle
        self.ui.pb_BIDS_dir.clicked.connect(self._action_pb_BIDS_dir)
        self.ui.pb_print_cmd.clicked.connect(self._action_print_cmd)
        self.ui.pb_BRIKfname.clicked.connect(self._action_pb_BRIKfname)
        self.ui.pb_BrainsightElec.clicked.connect(self._action_pb_BrainsightElec)
        self.ui.pb_BrainsightMRI.clicked.connect(self._action_pb_BrainsightMRI)
        self.ui.pb_run.clicked.connect(self._action_pb_run)
        self.ui.pb_CheckOutputs.clicked.connect(self._action_pb_CheckOutputs)
        
        ### Connect checkboxes
        self.ui.cb_crop_zeros.stateChanged.connect(self._action_cb_crop_zeros)
        self.ui.cb_emptyroom.stateChanged.connect(self._action_cb_emptyroom)
        
        ### Initialize w/ DEFAULTS
        if self.opts['include_empty_room']: self.ui.cb_emptyroom.setCheckState(2)
        if self.opts['crop_zeros']: self.ui.cb_crop_zeros.setCheckState(2)
        if self.opts['anonymize']: 
            if shutil.which('newDs'):
                self.ui.pb_Anonymize.setText('Anonymize: Y')
            else:
                self.ui.pb_Anonymize.setText('Anonymize: N  (no CTF code)')
                self.opts['anonymize']=False
        else:
            self.ui.pb_Anonymize.setText('Anonymize: N')
        
    ############ >> Action Section  ##########
    def _action_pb_run(self):
        'Run the BIDS conversion - Loop over all items in list'
        self._action_pb_CheckOutputs()  #Initialize io_mapping
        for idx, key in enumerate(self.io_mapping.keys()):
            self._set_single_filelist_text(idx=idx, prefix='Processing')
            QApplication.processEvents() #Force text update live
            _meg_fname = key
            _bids_path = self.io_mapping[key]['bidspath']
            try:
                _proc_meg_bids(meg_fname=_meg_fname, bids_path=_bids_path,
                                    anonymize=False, tmpdir=None, ignore_eroom=True, 
                                    crop_trailing_zeros=False, 
                                   )
                self._set_single_filelist_text(idx=idx, prefix='Done')
            except BaseException as e:
                self._set_single_filelist_text(idx=idx, prefix='Error')
            QApplication.processEvents()  #Force text update live
    
    def _action_pb_CheckOutputs(self):
        'Map the input files to output and display in filelist'
        self._make_task_dict()  #Generates the in_out_mapping
        self.io_mapping
        self._set_filelist_text()
        
        
    def _action_pb_BrainsightElec(self):
        'Browse for electrodes file'
        fname = self.open_file_dialog(file_filters='*.txt', 
                                      default_dir=DEFAULT_MRI_ROOT)
        if fname:
            self.ui.te_brainsight_elec.setPlainText(fname)
            self.opts['mri_elec'] = fname
            self.set_mri_type('bsight')
            
            #Check the validity of the electrodes file
            try:
                _fids = _read_electrodes_file(fname)
            except:
                _fids = False
            print(f'{_fids}')
            if _fids == False:
                print('Changing text')
                self.ui.label_10.setText('Brainsight Elec: !FORMAT_ERROR!')
                self.ui.pb_BrainsightElec.setText('Retry')
            else:
                self.ui.label_10.setText("Brainsight Elec (txt): ")
                self.ui.pb_BrainsightElec.setText('Browse')
            
    def _action_pb_BrainsightMRI(self):
        'Browse for brainsight MRI'
        fname = self.open_file_dialog(file_filters='NIFTI files (*.nii *.nii.gz)', 
                                      default_dir=DEFAULT_MRI_ROOT)
        if fname:
            self.ui.te_brainsight_mri.setPlainText(fname)
            self.opts['mri_bsight'] = fname        
            self.set_mri_type('bsight')

    def _action_pb_BRIKfname(self):
        'Browse for Afni file'
        fname = self.open_file_dialog(file_filters='AFNI files (*.BRIK *.BRIK.gz)', 
                                      default_dir=DEFAULT_MRI_ROOT)
        if fname:
            self.ui.te_BRIKfname.setPlainText(fname)
            self.opts['mri_brik'] = fname
            self.set_mri_type('afni')
            
            try:
                _ = coords_from_oblique_afni(fname)
                self.ui.pb_BRIKfname.setText('Browse')
                self.ui.label_BRIKFILE.setText('BRIK File:')
            except:
                self.ui.pb_BRIKfname.setText('Retry')
                self.ui.label_BRIKFILE.setText('BRIK File: !FORMAT_ERROR!')
            
    def _action_pb_BIDS_dir(self):
        directory = self.open_folder_dialog()
        self.ui.te_bids_dir.setPlainText(directory)
        self.opts['bids_dir'] = directory
        
    def _action_cb_emptyroom(self):
        if self.ui.cb_emptyroom.checkState() == 2:
            self.opts['include_empty_room'] = True
        else:
            self.opts['include_empty_room'] = False

    def _action_cb_crop_zeros(self):
        if self.ui.cb_crop_zeros.checkState() == 2:
            self.opts['crop_zeros'] = True
        else:
            self.opts['crop_zeros'] = False
    
    def _action_print_cmd(self):
        print(self.opts)
        
    def _update_bids_dir(self):
        self.opts['bids_dir']=self.ui.te_bids_dir.toPlainText().strip()
    
    def _update_bids_id(self):
        self.opts['bids_id'] = self.ui.te_BIDS_id.toPlainText().strip()
    
    def _update_bids_ses(self):
        _bids_session = self.ui.cb_Bids_Session.currentText()
        self.opts['bids_session']=_bids_session
            
    def _update_meghash(self):
        self.opts['subjid_input']=self.ui.te_meghash.toPlainText().strip()
        
    def _action_pb_anonymize(self):
        if self.opts['anonymize']==False:
            if shutil.which('newDs'):
                self.ui.pb_Anonymize.setText('Anonymize: Y')
                self.opts['anonymize']=True
            else:
                self.ui.pb_Anonymize.setText('Anonymize: N  (no CTF code)')
                self.opts['anonymize']=False
        elif self.opts['anonymize']==True: 
            self.ui.pb_Anonymize.setText('Anonymize: N')
            self.opts['anonymize']=False
        else:
            print(f'current text: {self.ui.pb_Anonymize.text()}')
    
    ##### <<< 
    
    ## Helper functions
    def set_mri_type(self, mr_type):
        ''' Clear out other MRI information from opts'''
        if mr_type == 'afni':
            self.opts['mri_elec'] = False
            self.opts['mri_bsight'] = False
            self.opts['mri_none'] = False
        elif mr_type == 'bsight':
            self.opts['mri_brik'] = False
            self.opts['mri_none'] = False
        elif mr_type == 'none':
            self.opts['mri_elec'] = False
            self.opts['mri_bsight'] = False
            self.opts['mri_brik'] = False
    
    def open_file_dialog(self, file_filters='*', default_dir=os.getcwd()):
        # Open file dialog
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",  # Dialog title
            default_dir,
            file_filters,
            options=options
        )
        return fileName
    
    def open_folder_dialog(self, default_dir=os.getcwd()):
        options = QtWidgets.QFileDialog.Options()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Directory",  # Dialog title
            default_dir, 
            options=options
        )
        return directory
    
    def _get_bids_path(self, task=None, run=None):
        try:
            bids_path = BIDSPath(subject=self.opts['bids_id'],
                                 session=self.opts['bids_session'], 
                                 task=task,
                                 run=run, 
                                 root=self.opts['bids_dir'], 
                                 suffix='meg', 
                                 extension='.ds')
        except BaseException as e:
            bids_path = False
            print(f'{e}')
        return bids_path
    
    def _make_task_dict(self):
        task_dict = _gen_taskrundict(self.opts['meg_dataset_list'])
        f_out_attributes = OrderedDict() #{}
        
        #Extract out the ordered runs
        for task in task_dict.keys():
            for idx, filename in enumerate(task_dict[task]):
                run = str(idx+1)
                if len(run)==1: run = '0'+run
                bpath = self._get_bids_path(task=task, run=run)
                f_out_attributes[filename] = {'run':run,
                                                   'task':task,
                                                   'bidspath':bpath
                                                   }
        
        for key in f_out_attributes.keys():
            f_out_attributes[key]['out_fname'] = str(f_out_attributes[key]['bidspath'].fpath)
            
        self.io_mapping = f_out_attributes
        
    def _set_filelist_text(self, prefix=None):
        for idx,input_key in enumerate(self.io_mapping):
            self._set_single_filelist_text(idx=idx, prefix=prefix)
    
    def _set_single_filelist_text(self, idx=None, prefix=None):
        input_key = list(self.io_mapping.keys())[idx]
        _txt = f'{idx+1}) '
        if prefix != None: _txt+=f'[{prefix}] '
        _txt += f'{op.basename(str(input_key))} --> '
        try:
            _out_fname = self.io_mapping[input_key]['out_fname']
            _strip_outfname = _out_fname.replace(self.opts['bids_dir'],'')
            _txt += f'(bidsDir){_strip_outfname}'
        except:
            _txt += 'Failed to assess conversion'
        self.ui.list_fname_conversion.item(idx).setText(_txt)
        
        
        
        
    
    # def _extract_task(self, fname):
    #     base_fname = op.basename(fname)
    #     _splits = base_fname.split('_')
    #     if len(_splits)==4:
    #         taskname = _splits[1]
    #     elif _splits[0].startswith('sub-'):
    #         taskname = fname.split('task-')[-1].split('_')[0]
    #     else:
    #         taskname = 'NA'
    #     return taskname
        
        
        
        
        
    
#  Need to pipe in the rest of the arguments into the class
# Then initialize in the above
# Then add function to ignore folder and just use the meg_list (in make-meg_bids)
class Args:
    def __init__(self, opts):
        self.anonymize = opts['anonymize']
        self.meg_dataset_list = opts['meg_dataset_list']
        if opts['bids_dir'] != '':
            self.bids_dir = opts['bids_dir']
        
        if opts['mri_bsight'] != False:
            self.mri_bsight = opts['mri_bsight']
            self.ignore_mri_checks = False
            # self.mri_brik = False
        
        if opts['mri_elec'] != False:
            self.mri_bsight_elec = opts['mri_elec']
            self.ignore_mri_checks = False
            # self.mri_brik = False
        
        if opts['mri_brik'] != False:
            self.mri_brik = opts['mri_brik']
            self.ignore_mri_checks = False
            self.mri_bsight = False
            self.mri_bsight_elec = None  #This should be false, but it checks against None
        else:
            self.mri_brik = False
        
        if opts['mri_none'] == True:
            self.ignore_mri_checks = True
        
        self.bids_session = opts['bids_session']
        
        if opts['bids_id'] != None:
            self.bids_id = opts['bids_id']
        
        if opts['crop_zeros'] == True:
            self.autocrop_zeros = True
        else:
            self.autocrop_zeros = False
        
        if opts['include_empty_room'] == False:
            self.ignore_eroom = True
        else:
            self.ignore_eroom = False
        
        # Add required tags to force make_meg_bids to run
        if 'subjid_input' in opts:
            if opts['subjid_input'] not in [False, None, 'None', 'False', '']:
                self.subjid_input = opts['subjid_input'].strip()
        elif 'subjid' in opts:
            if opts['subjid'] not in [False, None, 'None', 'False', '']:
                self.subjid_input = opts['subjid'].strip()
        else:
            self.subjid_input = False
        
        self.eventID_csv = None
        self.freesurfer = False
        self.mri_prep_s = False
        self.mri_prep_v = False
        
            
            
        

# def test_Args():
#     opts = {'anonymize': True, 'meghash': 'None', 'bids_id': 'S01', 
#             'bids_dir': '/tmp/BIDS', 'bids_session': '1', 
#             'meg_dataset_list': [], 'mri_none': False, 
#             'mri_bsight': '/tmp/Uploaded_cohort3/DUJGWRKZ/DUJGWRKZ.nii', 
#             'mri_elec': '/tmp/Uploaded_cohort3/DUJGWRKZ/Exported Electrodes.txt', 
#             'mri_brik': False, 'crop_zeros': True, 'include_empty_room': False}
     
#             #'subjid_input' : False}
#     args = Args(opts)
#     # make_bids(args)
    
# test_Args()

# app = QtWidgets.QApplication(sys.argv)
# MainWindow = QtWidgets.QMainWindow()
# ui = BIDS_MainWindow()
# ui.show()       

# class test_BIDS_MainWindow():
#     def __init__(self):
#         app = QtWidgets.QApplication(sys.argv)
#         MainWindow = QtWidgets.QMainWindow()
#         ui = BIDS_MainWindow()
#         ui.show()
    
    
