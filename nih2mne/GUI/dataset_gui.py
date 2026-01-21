#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:39:56 2025

@author: jstout

Ui_MainWindow - From QT designer
Ui_InputDatasetTile - From Qt designer (converted without -x)

This code implements a drag/drop GUI for quick data QA and event marking
Each file gets its own tile that has independent information and processing buttons
The tile also presents some status information on the file - movement / missing markerfile
Once the files have been event tagged and their MarkerFile.mrk produced, 
the bids gui can be opened to generate the correct output + Anat association

#make glob case insensitive -- for the task match on trigproc files
Make the trigger file locations environmental overidable

"""

from nih2mne.GUI.templates.file_staging_meg import Ui_MainWindow
from nih2mne.GUI.templates.input_meg_dset_tile_listWidgetBase import \
    Ui_InputDatasetTile
from nih2mne.GUI.templates.input_error_dset_tile_listWidgetBase import \
    Ui_ErrorDatasetTile

from nih2mne.config import TRIG_FILE_LOC

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
import os, os.path as op
import mne
import glob
import pandas as pd
from nih2mne.utilities.calc_hm import get_localizer_dframe, compute_movement
from nih2mne.utilities.data_crop_wrapper import get_term_time
import shutil
import copy
import numpy as np
from nih2mne.GUI.templates.bids_creator_gui_control_functions \
    import BIDS_MainWindow as BIDS_Ui_MainWindow


class GUI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        ####  Setup addon features >>>
        self.ui.FileDrop.dragEnterEvent = self.dragEnterEvent
        self.ui.FileDrop.dropEvent = self.dropEvent
        self.ui.pb_DeleteAllEntries.clicked.connect(self.clear_all_entries)
        self.ui.pb_LaunchBidsCreator.clicked.connect(self.open_bids_creator)

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
            try:
                # Instantiate a filename tile
                _tmp_tile = InputDatasetTile(fname=i)
            except BaseException as e:
                # Instantiate a NULL/ERROR tile
                _tmp_tile = ErrorDatasetTile(fname=i, 
                                             error_type=type(e), 
                                             error_code=str(e))
            
            # Add signal propagation from tile to MainWindow class
            _tmp_tile.close_clicked.connect(self.handle_close_request)
            
            # Do the weird Qt List Item/ItemWidget add
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(_tmp_tile.sizeHint())
            self.ui.scrollAreaWidgetContents.addItem(item)
            self.ui.scrollAreaWidgetContents.setItemWidget(item, _tmp_tile)
            
    def clear_all_entries(self):
        '''Delete all of the entries in the dataset list'''
        contents = self.ui.scrollAreaWidgetContents
        while contents.count():
            item = contents.takeItem(0)
    
    def get_fnames_from_list(self):
        '''Return a list of all filenames that have been dropped into list'''
        fname_list = []
        listwidget = self.ui.scrollAreaWidgetContents
        for i in range(listwidget.count()):
            item = listwidget.item(i)
            itemWidget = listwidget.itemWidget(item)
            fname_list.append(itemWidget.fname)
        return fname_list
    
    def open_bids_creator(self):
        '''Open second window and populate the dataset list'''
        fnames = self.get_fnames_from_list()
        print('Opening bids app')
        self._bids_window_open(meg_dsets = fnames)
        self.bids_gui.ui.list_fname_conversion.addItems(fnames)
        
        _meghash = self._assess_meghash(fnames)
        self.bids_gui.ui.te_meghash.setPlainText(_meghash)
    
    def _assess_meghash(self, fnames):
        try:
            _tmp = [op.basename(i).split('_')[0] for i in fnames]
            _tmp = set(_tmp)
            if (len(_tmp) > 1) or (len(_tmp)==0):
                return 'None'
            else:
                return list(_tmp)[0]
        except:
            print('Could not assess meghash')
            return 'None'
        
    
    def _bids_window_open(self, meg_dsets=None):
        '''Implement the logic to create and maintain a second main window'''
        self.bids_gui = BIDS_Ui_MainWindow(meg_dsets=meg_dsets)
        self.bids_gui.show()
        
    def handle_close_request(self, widget):
        '''If file tile "emits" a close signal, this will trigger a loop over
        filenames to identify the widget that produced the close signal'''
        item_count = self.ui.scrollAreaWidgetContents.count()
        for i in range(item_count):
            item = self.ui.scrollAreaWidgetContents.item(i)
            print(type(item))
            if self.ui.scrollAreaWidgetContents.itemWidget(item) == widget:
                self.ui.scrollAreaWidgetContents.takeItem(i)
                print(f"Parent removed item at row {i}")
                break
        
class InputDatasetTile(QtWidgets.QWidget):
    close_clicked = pyqtSignal(object)
    
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
        elif _splits[0].startswith('sub-'):
            subjid = _splits[0].replace('sub-','')
            taskname = fname.split('task-')[-1].split('_')[0]
            date = 'NA'
            if 'run-' in fname:
                run = fname.split('run-')[-1].split('_')[0]
            else:
                run = 'NA'
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
        
        ## Remove tile
        self.ui.pb_DeleteTile.clicked.connect(lambda: self.close_clicked.emit(self))
        
        
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
        del tmp_
        
        
    def set_status_label(self):
        '''
        Set information on the status bar
        # check for default head transform
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
            if len(evt_text) > 120:
                evt_text = evt_text[:120] + '...(trucated to fit)'
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
        del tmp_
    
    def plot_data(self):
        if not _can_load(self.fname): 
            self.ui.pb_PlotData.setText('PlotData: Not enough RAM')
            return 
        tmp_ = self.raw.copy()
        tmp_.pick_types(meg=True)
        tmp_.load_data()
        tmp_.plot()
        del tmp_
        
    def plot_fft(self):
        'Generate and plot fft - remove early termination zeros if present'
        # Check if data can fit in RAM
        if not _can_load(self.fname): 
            self.ui.pb_FFT.setText('FFT: Not enough RAM')
            return 
        tmp_ = self.raw.copy()
        tmp_.pick_types(meg=True)
        tmp_.load_data()
        # Remove the excess zeros produced by early termination
        if self.early_termination:
            data = tmp_._data
            sfreq = tmp_.info['sfreq']
            test_channel_idx = 100
            _, term_time = get_term_time(data[test_channel_idx,:], sfreq)
            tmp_.crop(0, term_time)
        psd_ = tmp_.compute_psd(fmin=0, fmax=100)
        psd_.plot()
        del tmp_, psd_
    
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
        if (current_trigfile == None) or (current_trigfile == ""):
            return
        cmd = f'{self.trigfile_dir}/{current_trigfile} {self.fname}'
        subprocess.run(cmd.split())  #Add errror processing
        self.load_meg() #Reload to get the newly created annotations
        self.set_events_label()
        self.set_status_label() 

        
class ErrorDatasetTile(QtWidgets.QWidget):
    '''If an error occurs in making a dataset tile, this is an override to list 
    the error'''
    close_clicked = pyqtSignal(object)
    
    def __init__(self, parent=None, fname=None, error_type=None, 
                 error_code=None):
        super().__init__(parent)
        
        # Create instance of the UI class and set it up
        self.ui = Ui_ErrorDatasetTile()
        self.ui.setupUi(self)
        
        # Determine filename info
        self.fname = fname
        base_fname = op.basename(self.fname)
        self.ui.ReadoutFilename.setText(f'(!ERROR!) File: {base_fname}')
        
        ## Info 
        self.error_code = error_code
        self.error_type = error_type
        self.set_status_label() 
        
        ## Launch Error Code
        self.ui.pb_ReviewError.clicked.connect(self.show_error)
        
        ## Remove tile
        self.ui.pb_DeleteTile.clicked.connect(lambda: self.close_clicked.emit(self))
    
    def show_error(self):
        QtWidgets.QMessageBox.critical(self, 'Error', self.error_code)
    
    def set_status_label(self):
        '''
        Set information on the status bar
        '''
        status_text = self.error_type
        self.ui.lbl_Status.setText(f'STATUS: {status_text}')
        
        
    
        
# Helper functions        
  
def _assess_ram():
    if 'SLURM_JOB_ID' in os.environ:
        _slurm = True
    else:
        _slurm = False
        
    if _slurm: 
        _jobid = os.environ.get('SLURM_JOBID')
        _uid = os.environ.get('SLURM_JOB_UID')
        usage_f = f'/sys/fs/cgroup/memory/slurm/uid_{_uid}/job_{_jobid}/memory.memsw.usage_in_bytes'
        lim_f = f'/sys/fs/cgroup/memory/slurm/uid_{_uid}/job_{_jobid}/memory.limit_in_bytes'
        
        with open(usage_f) as f: _usage=f.readline().replace("\n","")
        with open(lim_f) as f: _lim=f.readline().replace("\n","")
        
        avail_mem = int(_lim) - int(_usage)
    else:
        import psutil
        mem = psutil.virtual_memory()
        avail_mem = mem.available
    return avail_mem

def _get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip if it's a symbolic link
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def _can_load(fname):
    f_size = _get_folder_size(fname) #.ds meg files are directories
    if _assess_ram() > f_size:
        return True
    else:
        print('Cannot load due to RAM size limitations')
        return False
        
        


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # Add App Icon
    icon_img = op.join(op.dirname(__file__), 'templates', 'opposum_squid_icon.png')
    if op.exists(icon_img): app.setWindowIcon(QtGui.QIcon(icon_img))
    
    MainWindow = GUI_MainWindow() 
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

