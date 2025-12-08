#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:39:56 2025

@author: jstout

Ui_MainWindow - From QT designer
Ui_InputDatasetTile - From Qt designer (converted without -x)

"""

from nih2mne.GUI.templates.file_staging_meg import Ui_MainWindow
from nih2mne.GUI.templates.input_meg_dset_tile_listWidgetBase import \
    Ui_InputDatasetTile
from PyQt5 import QtWidgets, QtCore, QtGui
import os, os.path as op
import mne


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
            
        self.raw = mne.io.read_raw_ctf(fname, preload=False, 
                                  system_clock='ignore')
        
        self.ui.ReadoutFilename.setText(f'File: {base_fname}')
        self.ui.ReadoutSubjid.setText(f'Subjid: {subjid}')
        self.ui.ReadoutTaskname.setText(f'Task: {taskname}')
        self.ui.label_Duration.setText(f'Duration: {round(self.raw.times[-1])}')
        # self.ui.pushButton = [Delete entry] THIS is Delete
        
        #self.procfile_list = #Get the files from home folder - match against taskname 
        #self.ui.ProcFileComboBox.addItems = maybe set this as a 
        self.ui.pb_PlotTrig.clicked.connect(self.plot_trig)
        self.ui.pb_PlotData.clicked.connect(self.plot_data)
        self.ui.pb_FFT.clicked.connect(self.plot_fft)
    
    def plot_trig(self):
        tmp_ = self.raw.copy()
        tmp_.pick_types(misc=True, meg=False, eeg=False)
        tmp_.load_data()
        tmp_.plot()
        
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
  







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = GUI_MainWindow() 
    MainWindow.show()
    sys.exit(app.exec_())
