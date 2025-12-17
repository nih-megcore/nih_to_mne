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

class BIDS_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, meghash='None', bids_id='None', ):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Collect all bids options in self.opts
        self.opts = dict(anonymize=False, 
                         meghash=meghash, 
                         bids_id=bids_id,
                         bids_dir=op.join(os.getcwd(), 'BIDS'),
                         bids_session='1',
                         crop_zeros=False,
                         include_empty_room=False
                         )

        #### Fill out default text in text edit lines 
        self.ui.te_meghash.setPlainText(str(self.opts['meghash']))
        self.ui.te_BIDS_id.setPlainText(str(self.opts['bids_id']))
        self.ui.te_bids_dir.setPlainText(str(self.opts['bids_dir']))
        self.ui.te_bids_session.setPlainText(str(self.opts['bids_session']))
        
        ### Connect TextEdit lines
        self.ui.te_meghash.textChanged.connect(self._update_meghash)
        self.ui.te_BIDS_id.textChanged.connect(self._update_bids_id)
        self.ui.te_bids_dir.textChanged.connect(self._update_bids_dir)
        self.ui.te_bids_session.textChanged.connect(self._update_bids_ses)
        
        ### Connect buttons    ----- FIX commented items -----
        self.ui.pb_Anonymize.clicked.connect(self._action_pb_anonymize)   #flipflop toggle
        # self.ui.pb_BIDS_dir.clicked.connect(self._action_pb_BIDS_dir)
        self.ui.pb_print_cmd.clicked.connect(self._action_print_cmd)
        # self.ui.pb_BRIKfname.clicked.connect(self._action_pb_BRIKfname)
        # self.ui.pb_BrainsightElec.clicked.connect(self._action_pb_BrainsightElec)
        # self.ui.pb_BrainsightMRI.clicked.connect(self._action_pb_BrainsightMRI)
        # self.ui.pb_run.clicked.connect(self._action_pb_run)
        
        ### Connect checkboxes
        self.ui.cb_crop_zeros.stateChanged.connect(self._action_cb_crop_zeros)
        self.ui.cb_emptyroom.stateChanged.connect(self._action_cb_emptyroom)
        
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
        self.opts['bids_dir']=self.ui.te_bids_dir.toPlainText()
    
    def _update_bids_id(self):
        self.opts['bids_id'] = self.ui.te_BIDS_id.toPlainText()
    
    def _update_bids_ses(self):
        self.opts['bids_session']=self.ui.te_bids_session.toPlainText()
            
    def _update_meghash(self):
        self.opts['meghash']=self.ui.te_meghash.toPlainText()
        

    def _action_pb_anonymize(self):
        if self.opts['anonymize']==False:
            self.ui.pb_Anonymize.setText('Anonymize: Y')
            self.opts['anonymize']=True
        elif self.opts['anonymize']==True: 
            self.ui.pb_Anonymize.setText('Anonymize: N')
            self.opts['anonymize']=False
        else:
            print(f'current text: {self.ui.pb_Anonymize.text()}')
        
        

# def _test():        
#     #### <<<<<<<<< start of copy/paste
#     self.anonymize = False
#     self.ignore_mri_checks = False

#     # Standard Entries
#     self.bids_dir = op.join(os.getcwd(), 'bids_dir')
#     self.meg_input_dir = None
#     self.bids_session = 1
#     self.subjid_input = None
#     self.bids_id = None
#     self.coreg = 'Brainsight'

#     ## Afni Coreg:
#     self.mri_brik = None

#     ## Brainsight Coreg:
#     self.mri_bsight = None
#     self.mri_bsight_elec = None

#     ## Optional Overrides:
#     self.ignore_eroom = False
#     self.autocrop_zeros = False
#     self.freesurfer = None
#     self.eventID_csv = None
#     # Run standardize_eventID_list.py
    
#     self.config = config
#     if config != False:
#         write_opts = read_cfg(config)
#         self.update_opts(opts=write_opts)
# ##############3 << 
        
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = BIDS_MainWindow()
ui.show()       
        
        
        
        

class test_BIDS_MainWindow():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = BIDS_MainWindow()
        ui.show()
    
    
