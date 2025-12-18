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
    def __init__(self, meghash='None', bids_id='None', ):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Collect all bids options in self.opts
        meg_dsets = [self.ui.list_fname_conversion.item(i) for i in range(self.ui.list_fname_conversion.count())]
        # meg_dsets = [self.ui.list_fname_conversion.item(i).text() for i in range(self.ui.list_fname_conversion.count())]
        self.opts = dict(anonymize=False, 
                         meghash=meghash, 
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
    
    ##### <<< 
    
    ## Helper functions
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
    
    # def format_cmd(self):
    #     cmd = ['make_meg_bids.py']
    #     if self.opts['anonymize']:
    #         cmd += ' -anonymize'
    #     if self.opts['bids_id'] != 'None':
    #         cmd += 
        

#  Need to pipe in the rest of the arguments into the class
# Then initialize in the above
# Then add function to ignore folder and just use the meg_list (in make-meg_bids)
class Args:
    def __init__(self, opts):
        self.anonymize = opts['anonymize']
        self.meg_dataset_list = opts['meg_dataset_list']
        if opts['bids_dir'] != '':
            self.bids_dir = opts['bids_dir']
            # !!!!!!!!!!!   Do a check that bids_dir is accessible and real path
        
        if opts['mri_bsight'] != False:
            self.mri_bsight = opts['mri_bsight']
        
        if opts['mri_elec'] != False:
            self.mri_bsight_elec = opts['mri_elec']
        
        if opts['mri_brik'] != False:
            self.mri_brik = opts['mri_brik']
        
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
        
# ########### FORMAT OPTIONS >>>>>>
# cmd = format_cmd(opts)
# print(f'Running the command: {cmd}')
# out_txt = subprocess.run(cmd.split(), check=True, capture_output=True)
# summary = []
# _start = False
# for i in str(out_txt.stdout).split('\\n'):
#     if '########### SUMMARY #################' in i:
#         _start = True
#     if _start:
#         summary.append(i)

# single_flag_list = ['anonymize', 'autocrop_zeros', 'freesurfer', 'ignore_eroom',
#                     'ignore_mri_checks']
# drop_flag_list = ['coreg', 'read_from_config', 'config', 'update_opts', 'error_log', 'full_log', 'fids_qa']
# def format_cmd(opts):
#     '''
#     Write out the commandline options from the opts object.  Special cases 
#     for the single flag vs flag w/option entry.

#     Parameters
#     ----------
#     opts : opt object
#         DESCRIPTION.

#     Returns
#     -------
#     cmd : str

#     '''
#     arglist = ['make_meg_bids.py']
#     for i in value_writedict.values():
#         if i in drop_flag_list:
#             if (i == 'coreg') and (opts.coreg == 'None'):
#                 arglist.append('-ignore_mri_checks')
#                 continue
#             else:
#                 continue
#         flag_val =  getattr(opts, i)
#         if i in single_flag_list:
#             if flag_val == True:
#                 arglist += [f'-{i}']
#             else:
#                 continue
#         else:
#             if flag_val != None:
#                 arglist += [f'-{i} {getattr(opts, i)}']
#     cmd = ' '.join(arglist)
#     return cmd 

##############  <<<<<<<<<<<        
 
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
    
    
