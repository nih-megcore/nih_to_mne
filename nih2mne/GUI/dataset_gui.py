#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:39:56 2025

@author: jstout

Ui_MainWindow - From QT designer
Ui_InputDatasetTile - From Qt designer (converted without -x)


#make glob case insensitive -- for the task match on trigproc files

"""

from nih2mne.GUI.templates.file_staging_meg import Ui_MainWindow
from nih2mne.GUI.templates.input_meg_dset_tile_listWidgetBase import \
    Ui_InputDatasetTile
from PyQt5 import QtWidgets, QtCore, QtGui
import os, os.path as op
import mne
import glob
import pandas as pd
from nih2mne.utilities.calc_hm import get_localizer_dframe, compute_movement
import shutil
import copy
import numpy as np

TRIG_FILE_LOC = op.expanduser(f'~/megcore/trigproc')
LOG_FILE_LOC = op.expanduser(f'~/meglogs/')


class GUI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        ####  Setup addon features >>>
        self.ui.FileDrop.dragEnterEvent = self.dragEnterEvent
        self.ui.FileDrop.dropEvent = self.dropEvent
        self.ui.pb_DeleteAllEntries.clicked.connect(self.clear_all_entries)

        #### <<< 
        
        # This should have been part of UI design, but posthoc added
        self.ui.scrollAreaWidgetContents = QtWidgets.QListWidget()
        self.ui.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 853, 303))
        self.ui.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.ui.scrollArea.setWidget(self.ui.scrollAreaWidgetContents)
        
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()        
        
    def dropEvent(self, event):
        self.meg_files = [url.toLocalFile() for url in event.mimeData().urls()]
        print(f"Dropped: {', '.join(self.meg_files)}")
        self.populate_file_tiles()
        
    def populate_file_tiles(self):
        for i in self.meg_files: 
            _tmp_tile = InputDatasetTile(fname=i)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(_tmp_tile.sizeHint())
            self.ui.scrollAreaWidgetContents.addItem(item)
            self.ui.scrollAreaWidgetContents.setItemWidget(item, _tmp_tile)
            # self.scrollAreaWidgetContents.addItem(Ui_InputDatasetTile())
            
    def clear_all_entries(self):
        '''Delete all of the entries in the dataset list'''
        contents = self.ui.scrollAreaWidgetContents
        while contents.count():
            item = contents.takeItem(0)
        
    def delete_tile(self):
        '''Takes an input from the child tile in list and removes the tile
        from the tile_list'''



class InputDatasetTile(QtWidgets.QWidget):
    def __init__(self, parent=None, fname=None):
        super().__init__(parent)
        
        # Create instance of the UI class and set it up
        self.ui = Ui_InputDatasetTile()
        self.ui.setupUi(self)
        
        # Determine filename info
        self.fname = fname
        base_fname = op.basename(fname)
        _splits = base_fname.split('_')
        if len(_splits)==4:
            subjid = _splits[0]
            taskname = _splits[1]
            date = _splits[2]
            run = _splits[3].replace('.ds','')
        else:
            subjid = 'NA'
            taskname = 'NA'
            date = 'NA'
            run = 'NA'
            
        self.load_meg()
        self.taskname = taskname
        
        self.ui.ReadoutFilename.setText(f'File: {base_fname}')
        self.ui.ReadoutSubjid.setText(f'  Subjid: {subjid}')
        self.ui.ReadoutTaskname.setText(f'Task: {taskname}')
        self.ui.label_Duration.setText(f'Duration: {round(self.raw.times[-1])}s')
        # self.ui.pushButton = [Delete entry] THIS is Delete
        
        ## Process Triggers
        self.fill_procfile_list()
        self.ui.pb_TrigProcess.clicked.connect(self.trigprocess)
        
        ## Plotting
        self.ui.pb_PlotTrig.clicked.connect(self.plot_trig)
        self.ui.pb_PlotData.clicked.connect(self.plot_data)
        self.ui.pb_FFT.clicked.connect(self.plot_fft)
        
        ## Compute status info
        self._compute_movement()
        
        ## Info 
        self.set_events_label()
        self.set_status_label() 
        
    def load_meg(self):
        ''' Added as a method, so reloading data (updating annotation) can be done easily'''
        self.raw = mne.io.read_raw_ctf(self.fname, preload=False, 
                                  system_clock='ignore', clean_names=True)
    
    def _compute_movement(self):
        if shutil.which('calcHeadPos') == None:
            self.head_movement = 'NoCTFCode'
            return
        _has_hz = op.exists(op.join(self.fname, 'hz.ds'))
        _has_hz2 = op.exists(op.join(self.fname, 'hz2.ds'))
        if _has_hz and _has_hz2:
            dframe = get_localizer_dframe(self.fname)
            move_dict = compute_movement(dframe)
            self.head_movement = move_dict['Max']
        elif not _has_hz:
            self.head_movement = 'No hz.ds'
        elif not _has_hz2:
            self.head_movement = 'No hz2.ds'
    
    def _check_trailing_zeros(self):
        tmp_ = self.raw.copy().pick('meg')
        max_time_ = tmp_.times[-1]
        if max_time_ < 10:
            return None
        tmp_.crop(max_time_ - 10, None) #Pull just the last 10 seconds
        tmp_.load_data()
        test_vals = np.ones(tmp_._data.shape) * tmp_._data
        if (test_vals).sum() == 0.0:
            self.early_termination = True
        else:
            self.early_termination = False
        
        
    def set_status_label(self):
        '''
        Default head trans
        '''
        from numbers import Number
        status_text = ''
        if glob.glob(op.join(self.fname, 'MarkerFile.mrk')).__len__()==0:
            status_text+='No MrkFile! : '
        
        # Add movement info    
        if isinstance(self.head_movement, Number):
            _mvt_text = f'MVT={self.head_movement:.2f}cm'
        else:
            _mvt_text = self.head_movement
        status_text+=_mvt_text + ': '
        
        # Check early termination
        self._check_trailing_zeros()
        if self.early_termination:
            _term_text = f'Early Termination Detected: '
        else:
            _term_text = ''
        status_text+=_term_text
        self.ui.lbl_Status.setText(f'STATUS: {status_text}')
        
    def set_events_label(self):
        'Extract the annotations and list the counts in EventInfo'
        evt_dframe = pd.DataFrame(self.raw.annotations)
        if len(evt_dframe) != 0:
            val_counts = evt_dframe.description.value_counts()
            evt_text = ' '.join([f"{k}:({v})" for k, v in val_counts.items()])
            self.ui.lbl_EventInfo.setText(f'EVENTS: {evt_text}')
        else:
            self.ui.lbl_EventInfo.setText(f'EVENTS: NONE')
    
    def plot_trig(self):
        tmp_ = self.raw.copy()
        tmp_.pick_types(misc=True, meg=False, eeg=False)
        if 'SCLK01' in tmp_.ch_names:
            _chs = copy.copy(tmp_.ch_names)
            _chs.remove('SCLK01')
            tmp_.pick(_chs)
        tmp_.load_data()
        tmp_.plot(scalings=10) #10 was empirically determined
        
    def plot_data(self):
        tmp_ = self.raw.copy()
        tmp_.pick_types(meg=True)
        tmp_.load_data()
        tmp_.plot()
        
    def plot_fft(self):
        ## FUTURE -- REMOVE ZEROS from data before FFT 
        tmp_ = self.raw.copy()
        tmp_.pick_types(meg=True)
        tmp_.load_data()
        psd_ = tmp_.compute_psd(fmin=0, fmax=100)
        psd_.plot()
    
    def fill_procfile_list(self):
        if op.exists(TRIG_FILE_LOC):
            self.trigfile_dir = TRIG_FILE_LOC
        else:
            self.trigfile_dir = None
        
        task_trig_files = glob.glob(op.join(self.trigfile_dir, self.taskname.lower() + '*'))
        if len(task_trig_files) > 0:
            _tt_files = [op.basename(i) for i in task_trig_files]
            self.ui.ProcFileComboBox.addItems(_tt_files)
    
    def trigprocess(self):
        import subprocess
        current_trigfile = self.ui.ProcFileComboBox.currentText()
        print(f'No associated trigger processing file for task: {self.taskname.lower()}')
        if (current_trigfile is None) or (current_trigfile is ""):
            return
        cmd = f'{self.trigfile_dir}/{current_trigfile} {self.fname}'
        subprocess.run(cmd.split())  #Add errror processing
        self.load_meg() #Reload to get the newly created annotations
        self.set_events_label()
        self.set_status_label() 
  


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = GUI_MainWindow() 
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

