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
te_bids_session #Manual entry default to 1
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
from nih2mne.GUI.templates.BIDS_creator_gui import Ui_MainWindow
import sys
import os, os.path as op
from nih2mne.make_meg_bids import make_bids

class BIDS_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, meghash='None', bids_id='None', meg_dsets=None):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Collect all bids options in self.opts
        self.opts = dict(anonymize=False, 
                         subjid_input=meghash, 
                         bids_id=bids_id,
                         bids_dir=op.join(os.getcwd(), 'BIDS'),
                         bids_session='1',
                         meg_dataset_list = meg_dsets,
                         
                         #MRI_none
                         mri_none = True,
                         #MRI_bsight
                         mri_bsight = False,
                         mri_elec = False,
                         #MRI_afni
                         mri_brik = False,
                         
                         #Options
                         crop_zeros=False,
                         include_empty_room=False,
                         
                         )
        
        #### Fill out default text in text edit lines 
        self.ui.te_meghash.setPlainText(str(self.opts['subjid_input']))
        self.ui.te_BIDS_id.setPlainText(str(self.opts['bids_id']))
        self.ui.te_bids_dir.setPlainText(str(self.opts['bids_dir']))
        self.ui.te_bids_session.setPlainText(str(self.opts['bids_session']))
        
        ### Connect TextEdit lines
        self.ui.te_meghash.textChanged.connect(self._update_meghash)
        self.ui.te_BIDS_id.textChanged.connect(self._update_bids_id)
        self.ui.te_bids_dir.textChanged.connect(self._update_bids_dir)
        self.ui.te_bids_session.textChanged.connect(self._update_bids_ses)
        
        ### Connect buttons    
        self.ui.pb_Anonymize.clicked.connect(self._action_pb_anonymize)   #flipflop toggle
        self.ui.pb_BIDS_dir.clicked.connect(self._action_pb_BIDS_dir)
        self.ui.pb_print_cmd.clicked.connect(self._action_print_cmd)
        self.ui.pb_BRIKfname.clicked.connect(self._action_pb_BRIKfname)
        self.ui.pb_BrainsightElec.clicked.connect(self._action_pb_BrainsightElec)
        self.ui.pb_BrainsightMRI.clicked.connect(self._action_pb_BrainsightMRI)
        self.ui.pb_run.clicked.connect(self._action_pb_run)
        
        ### Connect checkboxes
        self.ui.cb_crop_zeros.stateChanged.connect(self._action_cb_crop_zeros)
        self.ui.cb_emptyroom.stateChanged.connect(self._action_cb_emptyroom)
        
    ############ >> Action Section  ##########
    def _action_pb_run(self):
        args = Args(self.opts)
        make_bids(args)
        
        
    def _action_pb_BrainsightElec(self):
        fname = self.open_file_dialog(file_filters='*.txt')
        if fname:
            self.ui.te_brainsight_elec.setPlainText(fname)
            self.opts['mri_elec'] = fname
            self.set_mri_type('bsight')
        
    def _action_pb_BrainsightMRI(self):
        fname = self.open_file_dialog(file_filters='NIFTI files (*.nii *.nii.gz)')
        if fname:
            self.ui.te_brainsight_mri.setPlainText(fname)
            self.opts['mri_bsight'] = fname        
            self.set_mri_type('bsight')

    def _action_pb_BRIKfname(self):
        fname = self.open_file_dialog(file_filters='AFNI files (*.BRIK *.BRIK.gz)')
        if fname:
            self.ui.te_BRIKfname.setPlainText(fname)
            self.opts['mri_brik'] = fname
            self.set_mri_type('afni')

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
        self.opts['bids_session']=self.ui.te_bids_session.toPlainText().strip()
            
    def _update_meghash(self):
        self.opts['subjid_input']=self.ui.te_meghash.toPlainText().strip()
        
    def _action_pb_anonymize(self):
        if self.opts['anonymize']==False:
            self.ui.pb_Anonymize.setText('Anonymize: Y')
            self.opts['anonymize']=True
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
    
    
