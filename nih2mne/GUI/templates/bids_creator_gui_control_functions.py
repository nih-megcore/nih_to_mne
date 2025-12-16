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


app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()


class BIDS_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        ####  Setup addon features >>>
        self.ui.FileDrop.dragEnterEvent = self.dragEnterEvent
        self.ui.FileDrop.dropEvent = self.dropEvent
        self.ui.pb_DeleteAllEntries.clicked.connect(self.clear_all_entries)
        self.ui.pb_LaunchBidsCreator.clicked.connect(self.open_bids_creator)


class control_funcs():
    def __init__(self):
        