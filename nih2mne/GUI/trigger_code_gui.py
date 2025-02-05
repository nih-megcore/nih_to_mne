#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:55:27 2025

@author: jstout

Trigger Channel Coding GUI

Layout:
    File Lookup
    Initial Trigger Procs (Identify transitions):
        UADC1-15  - Analog
        UADC16 - Projector Analog
        PPT001 - Parrallel Port
        --LIST Output Counts per line--
    Code out Line variable names:
        UADC1-15 - what are they named
        UADC16 - Projector
        PPT001:
            Code for each trigger value
    Visual Tasks:
        Correct all to projector?
        Check for projector inversion?
    Auditory Tasks:
        Add static delay to event time
    General Add static delay:
        Didn't code out projector so add 19ms delay
    Parse Marks:
        In1
        In2
        MarkOn
        MarkName
    Display  Totals:
        For each condition write out the pandas value counts
    Check Visual scan of data:
        Invoke MNE display
    WriteScript:
        Create a trigger processing script


TODO: Temporal coding on PPT -- a 1 followed by a 5 is a Y event 
Set action to invert trigger for counts    
Set all dig trigs to invert together   
Create a parsemarks list and remove all after an erase command
detect_digital currently does not support downgoing triggers

Add trigger inversion for digital trigger
"""
import glob
import mne
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, \
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel,  QComboBox, QLineEdit, QCheckBox, \
    QFileDialog, QDialog
# from functools import partial
import sys
import os, os.path as op
import numpy as np
from nih2mne.utilities.trigger_utilities import (parse_marks, detect_digital, 
                                                 check_analog_inverted, threshold_detect, 
                                                 append_conditions, correct_to_projector, 
                                                 add_event_offset)



############## Setup Main Classes ################



class trig_tile(QHBoxLayout):
    '''Trigger tile for the analogue and digital panels'''
    def __init__(self, chan_name=None, include_polarity=True, 
                 event_count=None, meg_fname=None):
        super(trig_tile, self).__init__()
        
        self.meg_fname = meg_fname
        self.trig_type = chan_name[1:4]
        self.ch_name = chan_name

        self.addWidget(QLabel(f'{chan_name} :'))  
        
        # Add Checkboxes for upgoing/Downgoing trigger polarity 
        self.b_upgoing_trigger = QCheckBox()
        self.b_upgoing_trigger.setCheckState(2)
        self.b_upgoing_trigger.clicked.connect(self.set_up_trigger_polarity)
        self.b_downgoing_trigger = QCheckBox()
        self.b_downgoing_trigger.setCheckState(0)
        self.b_downgoing_trigger.clicked.connect(self.set_down_trigger_polarity)
        
        self.addWidget(self.b_upgoing_trigger)
        self.addWidget(QLabel('Up'))
        self.addWidget(self.b_downgoing_trigger)
        self.addWidget(QLabel('Down'))
        
        # Name the event, so that it can be referenced later
        self.addWidget(QLabel('   Event Name:'))
        self.event_name = QLineEdit()
        self.addWidget(self.event_name)
        
        # Event counter display
        if self.trig_type == 'ADC':
            self.event_dframe = threshold_detect(self.meg_fname,  channel=self.ch_name)
            self.event_count = len(self.event_dframe)
        elif self.trig_type == 'PPT':
            self.event_count = event_count
        else:
            self.event_count = 'ERR'
        self.event_count_label = QLabel(f'N={self.event_count}')
        self.addWidget(self.event_count_label)

    def set_up_trigger_polarity(self):
        self.trigger_polarity = 'up'
        self.b_downgoing_trigger.setCheckState(0)
        print(self.b_upgoing_trigger.checkState())
    
    def set_down_trigger_polarity(self):
        self.trigger_polarity = 'down'
        self.b_upgoing_trigger.setCheckState(0)

class parsemarks_tile(QHBoxLayout):
    '''Parsemarks tile 
    
    | EventsList | MarkOn | EventsList | MarkOn | Time1 | Time2 | Count | Name |
    | ComboBox   | CheckB | ComboBox   | CheckB  | LineE | LineE | QLabel | LineE | 
    
    SET/DEL options will be handled by the main window
    
    '''
    def __init__(self, event_namelist=None):
        super(parsemarks_tile, self).__init__()
        
        #Assemble Tile
        self.evt_namelist = event_namelist
        
        self.b_evt1_name = QComboBox()
        self.b_evt1_name.addItems(event_namelist)
        self.addWidget(self.b_evt1_name)
        
        self.b_mark_on_lead = QCheckBox()
        self.b_mark_on_lead.setChecked(True)
        # self.b_mark_on_lead.clicked.connect(.....)
        self.addWidget(self.b_mark_on_lead)
        
        self.b_evt2_name = QComboBox()
        self.b_evt2_name.addItems(event_namelist)
        self.addWidget(self.b_evt2_name)
        
        self.b_mark_on_lag = QCheckBox()
        # self.b_mark_on_lag.clicked.connect(....)
        self.addWidget(self.b_mark_on_lag)
        
        self.b_window_t1 = QLineEdit('0')
        self.addWidget(self.b_window_t1)
        self.b_window_t2 = QLineEdit('0.5')
        self.addWidget(self.b_window_t2)
        
        self.event_count = QLabel('N=')  #Still need to compute event count and enter
        self.addWidget(self.event_count)
        
        self.static_name_label = QLabel('Name:')
        self.addWidget(self.static_name_label)
        
        self.event_name = QLineEdit()
        self.addWidget(self.event_name)

        
class grid_selector(QMainWindow):
    '''Helper function create a grid of items based on a list'''
    def __init__(self, input_list=[], title=None, 
                 gridsize_row=None, gridsize_col=None):
        super(grid_selector, self).__init__()
        self.setGeometry(100,100, 500, 500) 
        self.setWindowTitle(title)
        self.input_list = input_list
        if (gridsize_col==None) or (gridsize_row==None):
            self.gridsize_row, self.gridsize_col = self._get_rowcol()
        else:
            self.gridsize_row, self.gridsize_col = self.gridsize_row, gridsize_col
        
        _ = self.make_choice_grid()
        self.setCentralWidget(self.dialog)
        self.show()
    
    def make_choice_grid(self):
        self.dialog = QDialog() 
        self.grid_layout = QGridLayout()
        tile_idxs = np.arange(self.gridsize_row * self.gridsize_col)
        tile_idxs_grid = tile_idxs.reshape(self.gridsize_row, self.gridsize_col)
        row_idxs, col_idxs = np.unravel_index(tile_idxs, [self.gridsize_row, self.gridsize_col])
        i=0 
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            if i > len(self.input_list) -1:
                tmp_ = QLabel('')
            else:
                tmp_ = QPushButton(self.input_list[i])
                tmp_.setCheckable(True)
            self.grid_layout.addWidget(tmp_, row_idx, col_idx)
            i+=1
        
        layout = QVBoxLayout()
        layout.addLayout(self.grid_layout)
        self.b_set_selection = QPushButton('CLICK to set selection')
        layout.addWidget(self.b_set_selection)
        self.dialog.setLayout(layout)
        
        return self.grid_layout
            
    def _get_rowcol(self):
        listlen = len(self.input_list)
        if listlen < 4: 
            return 2,2
        if listlen < 16: 
            return 4,4
        if listlen < 32:
            return 4,8
        if listlen < 64: 
            return 8,8
        if listlen < 117: 
            return 9, 13
         
        
        
        
        
############# Setup Window ######################            
        
        
        

class event_coding_Window(QMainWindow):
    def __init__(self, cmdline_meg_fname=False):
        super(event_coding_Window, self).__init__()
        self.setGeometry(100,100, 1000, 1000) #250*gridsize_col, 100*gridsize_row)
        self.setWindowTitle('Event Coding GUI')
        
        self.tile_dict = {}
        
        # Finalize Widget and display
        main_layout = self.setup_full_layout()
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        ####### -- FOR TESTING ONLY -- #######
        if cmdline_meg_fname != False:   
            self.meg_fname = cmdline_meg_fname
            meg_display_name = cmdline_meg_fname.split('/')[-1]
            self.meg_display_name.setText(meg_display_name)
            self.meg_raw = mne.io.read_raw_ctf(cmdline_meg_fname, clean_names=True, 
                                               system_clock='ignore', preload=False, 
                                               verbose=False)
            self.trig_ch_names = [i for i in self.meg_raw.ch_names if i.startswith('UADC') or i.startswith('UPPT')]
            self.fill_trig_chan_layout()
            self.b_update_event_names = QPushButton('Update Event Names: Will erase below')
            self.b_update_event_names.clicked.connect(self.update_event_names)
            self.trig_parsemarks_layout.addWidget(self.b_update_event_names)
            self.trig_parsemarks_layout.addWidget(QLabel('Create New Events from Other Events'))
            self.keep_events_layout.addWidget(QLabel('Events To Write'))
        ####### -- FOR TESTING ONLY -- #######
            
    
    def setup_full_layout(self):
        main_layout = QVBoxLayout()

        # Setup File Chooser
        meg_choose_layout = QHBoxLayout()
        self.b_choose_meg = QPushButton('Select MEG')
        # self.b_choose_meg.clicked.connect(self.select_meg_dset)
        self.b_choose_meg.clicked.connect(lambda: self.select_meg_dset(meg_fname=None))
        
        meg_choose_layout.addWidget(self.b_choose_meg)
        self.meg_display_name = QLabel('')
        meg_choose_layout.addWidget(self.meg_display_name)
        main_layout.addLayout(meg_choose_layout)
        
        self.ana_trigger_layout = QVBoxLayout()
        main_layout.addLayout(self.ana_trigger_layout)
        self.dig_trigger_layout = QVBoxLayout()
        main_layout.addLayout(self.dig_trigger_layout)
        
        # Middle operations after dig/ana setup
        self.middle_operations = QHBoxLayout()
        self.b_corr2proj = QPushButton('Correct to Projector')
        self.b_corr2proj.clicked.connect(self.open_corr2proj)
        self.middle_operations.addWidget(self.b_corr2proj)
        self.b_add_fixed_delay = QPushButton('Add Fixed Offset')
        self.b_add_fixed_delay.clicked.connect(self.open_add_fixed_delay)
        self.middle_operations.addWidget(self.b_add_fixed_delay) 
        self.delay_label = QLabel('Offset (ms):')
        self.middle_operations.addWidget(self.delay_label)
        self.b_fixed_delay_edit = QLineEdit()
        self.middle_operations.addWidget(self.b_fixed_delay_edit)
        main_layout.addLayout(self.middle_operations)
        
        self.trig_parsemarks_layout = QVBoxLayout()
        main_layout.addLayout(self.trig_parsemarks_layout)
        
        self.keep_events_layout = QVBoxLayout()
        main_layout.addLayout(self.keep_events_layout)
        
        self.b_write_parser_file = QPushButton('Write Parser File')
        self.b_write_parser_file.clicked.connect(self.write_parser_script)
        main_layout.addWidget(self.b_write_parser_file)
                
        return main_layout
    
    def fill_trig_chan_layout(self):
        '''
        Iterates over analog channels and PPT conditions to fill panel
        | Trig Name | b_UpTrig | b_DownTrig | Count=  | OutName: | 
        '''
        # Setup panel headers
        self.ana_trigger_layout.addWidget(QLabel('Analogue Channels'))
        self.dig_trigger_layout.addWidget(QLabel('Digital Channel Trigger Values'))
        for i in self.trig_ch_names:
            ### Analogue Channels ###
            if i.startswith('UADC'):
                self.tile_dict[i]=trig_tile(chan_name=i,include_polarity=True,
                                            meg_fname=self.meg_fname)
                self.ana_trigger_layout.addLayout(self.tile_dict[i])
                if i=='UADC016':
                    self.tile_dict[i].event_name.setText('projector')
            ### Digital Channels - Breakout the Codes ###
            elif i.startswith('UPPT'):
                dig_dframe = detect_digital(self.meg_fname, channel=i)
                event_vals = list(dig_dframe.condition.unique())
                event_vals = sorted(event_vals, key=int)
                dig_event_counts = dig_dframe.condition.value_counts()
                for evt_name in event_vals:
                    evt_key = f'{i}_{evt_name}'
                    self.tile_dict[evt_key]= trig_tile(chan_name=f'{i} [{evt_name}]', 
                                                 include_polarity=True,
                                                 event_count=dig_event_counts[evt_name], 
                                                 meg_fname = self.meg_fname)
                    self.dig_trigger_layout.addLayout(self.tile_dict[evt_key])
            else:
                print(f'Not processing channel {i}')
    
    def update_event_names(self, events_only=False):
        '''Set action for updating event names.  This will write all of the 
        events to the an event list and update the parse_marks and evt_keep
        panels. 
        raw_event_list: directly evaluated from the raw trigger lines
        parsed_event_list: parse_marks derived events
        ## logfile_event_list: Future implementation
        ## temporal_coding_event_list : Future implementation
        
        event_list: combination of the above
        
        '''
        namelist = []
        for key in self.tile_dict.keys():
            tmp_txt = self.tile_dict[key].event_name.text()
            namelist.append(tmp_txt)
        self.event_namelist = namelist
        
        #Check for duplicated names from raw trigger panel
        for i in reversed(range(len(namelist))):
            if namelist[i]==None:
                del(namelist[i])
            if namelist[i]=='':
                del(namelist[i])        
        if len(namelist) != len(set(namelist)):
            raise ValueError('The event names cannot have duplicates') 
        
        if events_only:
            return
        
        ############# Parsemarks Layout ##########################
        self.parsemarks_tile_list = []
        self.parsemarks_full_layout_list = []
        self.add_parsemarks_line()
        
        ############# Keep Events Layout #########################
        #Empty the keep events layout list, so it doesn't append the previous
        self.update_keep_events_list(flush=True)
        
    def open_corr2proj(self):
        # Create a popup to select events that will be corrected to projector
        self.update_event_names(events_only=True)
        self.corr2proj_selector = grid_selector(self.event_namelist)  
        self.corr2proj_selector.b_set_selection.clicked.connect(self._set_correct2proj_list)
    
    def open_add_fixed_delay(self):
        # Create a popup to select the events that will be corrected to a fixed delay
        self.update_event_names(events_only=True)
        self.fixed_delay_selector = grid_selector(self.event_namelist)  
        self.fixed_delay_selector.b_set_selection.clicked.connect(self._set_fixedDelay_list)
    
    def _set_fixedDelay_list(self):
        'Fixed delay added to self.add_offset_list'
        layout = self.fixed_delay_selector.grid_layout
        self.add_offset_list = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                if hasattr(item.widget(), 'isChecked'):
                    if item.widget().isChecked():
                        self.add_offset_list.append(item.widget().text())
        print(f'Setting fixed delay to the following: {self.add_offset_list}')
    
    def _set_correct2proj_list(self):
        'Correct to projector list assigned to self.corr2proj_list'
        layout = self.corr2proj_selector.grid_layout
        self.corr2proj_list = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                if hasattr(item.widget(), 'isChecked'):
                    if item.widget().isChecked():
                        self.corr2proj_list.append(item.widget().text())
        print(f'Correcting the following to projector timing: {self.corr2proj_list}')        
        
    def update_keep_events_list(self, flush=False):
        num_keep_buttons = self.keep_events_layout.count()
        if flush==True:
            if num_keep_buttons > 1:
                for i in reversed(range(1,num_keep_buttons)):  #Skip the label - remove from the end
                    item = self.keep_events_layout.takeAt(i)
                    self.keep_events_layout.removeItem(item)
                    if item.widget():
                        item.widget().deleteLater()

        for i in set(self.event_namelist): 
            tmp=QPushButton(i)
            tmp.setCheckable(True)
            self.keep_events_layout.addWidget(tmp)
            del tmp
        self.keep_events_layout.update()
        
        
    def add_parsemarks_line(self):
        '''Add an additional line to the parsemarks panel 
        Create parsemarks tile and append ADD and SET buttons
        Finally add new name to the event list'''
        if len(self.parsemarks_tile_list)>0:
            self.set_parsemarks_line() #Add the name of the previous entry to the namelist
            
        tmp_pm_tile = parsemarks_tile(event_namelist=self.event_namelist)
        tmp_full_pm_layout = QHBoxLayout()
        tmp_full_pm_layout.addLayout(tmp_pm_tile)
        
        b_parsemarks_set = QPushButton('SET')
        b_parsemarks_set.clicked.connect(self.set_parsemarks_line) 
        tmp_full_pm_layout.addWidget(b_parsemarks_set)
        b_parsemarks_add = QPushButton('ADD')
        b_parsemarks_add.clicked.connect(self.add_parsemarks_line)
        tmp_full_pm_layout.addWidget(b_parsemarks_add)
        self.trig_parsemarks_layout.addLayout(tmp_full_pm_layout)  
        self.parsemarks_tile_list.append(tmp_pm_tile)  #Add the new entry to the list
        self.parsemarks_full_layout_list.append(tmp_full_pm_layout)  #Hate that this has to be added
          
    def set_parsemarks_line(self):
        '''Add the parsemarks name to the name list and update Keep list panel'''
        last_button = self.parsemarks_tile_list[-1]
        last_event_name = last_button.event_name.text() #event_name_widget.text()
        self.event_namelist.append(last_event_name)
        self.update_keep_events_list(flush=True)
        self.keep_events_layout.update()
                


    def select_meg_dset(self, meg_fname=None):
        if meg_fname==None:
            print('Here is the dialogue')
            meg_fname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a MEG folder (.ds)')
        self.meg_fname = meg_fname
        meg_display_name = meg_fname.split('/')[-1]
        self.meg_display_name.setText(meg_display_name)
        print(meg_fname)
        if meg_fname != None:
            self.meg_raw = mne.io.read_raw_ctf(meg_fname, clean_names=True, 
                                               system_clock='ignore', preload=False, 
                                               verbose=False)
        self.trig_ch_names = [i for i in self.meg_raw.ch_names if i.startswith('UADC') or i.startswith('UPPT')]
        self.fill_trig_chan_layout()
        
        self.b_update_event_names = QPushButton('Update Event Names: Will erase below')
        self.b_update_event_names.clicked.connect(self.update_event_names)
        self.trig_parsemarks_layout.addWidget(self.b_update_event_names)
        self.trig_parsemarks_layout.addWidget(QLabel('Create New Events from Other Events'))
        
        self.keep_events_layout.addWidget(QLabel('Events To Write'))
        
        
    def write_parser_script(self):
        '''    
        Part 1: Write Threshold Detect Options
        Part 2: Write Digital Detect components
        Part 3: Write Parse_marks components
        Part 4: Combine dataframes
        Part 5: Select Keep options
        Part 6: Write the markerfile to the input dataset
        '''
        
        ##### Finalize selections list ###### 
        self.events_to_write = []
        for i in range(self.keep_events_layout.layout().count()):
            try:
                if self.keep_events_layout.layout().itemAt(i).widget().isChecked():
                    self.events_to_write.append(self.keep_events_layout.layout().itemAt(i).widget().text())
            except:
                pass  #This is in case the widget doesn't have the isChecked tag
        print(self.events_to_write)
        
        ##### Python Header Section #####
        header=["#!/usr/bin/env python3\n",
                "# -*- coding: utf-8 -*-\n",
                "\n\n",
                "'''\n"
                "This code was generated by the nih2mne package: \n",
                "https://github.com/nih-megcore/nih_to_mne.git  \n",
                "'''\n"
                "\n\n", 
                "import sys\n"
                "import mne\n",
                "import nih2mne\n",
                '''from nih2mne.utilities.trigger_utilities import (parse_marks, detect_digital,
                    check_analog_inverted, threshold_detect, append_conditions, correct_to_projector, add_event_offset)\n''',
                '''from nih2mne.utilities.markerfile_write import main as write_markerfile\n'''
                "\n\n",
                "meg_fname = sys.argv[1]\n\n"
                ]
        fname, _ = QFileDialog.getSaveFileName(self, "Save File", "trig_parser.sh", "Text Files (*.sh);;All Files (*)")#, options=options)
        with open(fname, 'w') as f:
            f.writelines(header)
        
        ##### Make a list of dataframes ####
        init_code = []
        init_code.append('dframe_list=[]')
                         
        
        ##### Analog Triggers #####
        ana_trig_code = []
        # print(self.events_to_write)
        for i, tile in self.tile_dict.items():
            if i.startswith('UADC'):# and (tile.event_name.text() in self.events_to_write) :
                print(i, tile.event_name.text())
                markname = tile.event_name.text()
                if markname == '':
                    continue
                if tile.b_downgoing_trigger.checkState()==2:
                    invert_val = True
                else:
                    invert_val = False
                tmp_code = f"tmp_dframe = threshold_detect(dsname=meg_fname, channel='{i}', mark='{markname}', invert={invert_val})"
                ana_trig_code.append(tmp_code)
                tmp_code = f"dframe_list.append(tmp_dframe)"
                ana_trig_code.append(tmp_code)
                
        ##### Digital Triggers #####       #####!!!!!!    IF NAME does not exist - dont write
        dig_trig_code = []
        tmp_code = f"dig_dframe = detect_digital(filename=meg_fname, channel='UPPT001')" #, mark='{markname}')"
        dig_trig_code.append(tmp_code)
        for i, tile in self.tile_dict.items():
            if i.startswith('UPPT'):
                print(i, tile.event_name.text())
                markname = tile.event_name.text()
                if markname == '':
                    continue
                dig_val = i.split('_')[-1]
                tmp_code = f"dig_dframe.loc[dig_dframe.condition=='{dig_val}', 'condition']='{markname}'"
                dig_trig_code.append(tmp_code)        
        tmp_code = f"dframe_list.append(dig_dframe)"
        dig_trig_code.append(tmp_code)
        
        ##### Tidy up data   #######
        time_corr_code = []
        time_corr_code.append(f"dframe = append_conditions(dframe_list)")
                
        ##### Correct to Projector ###### 
        if len(self.corr2proj_list) > 0:
            tmp_code = f'dframe = correct_to_projector(dframe, event_list={self.corr2proj_list}, window=[-0.2,0.2])'
            time_corr_code.append(tmp_code)        
        
        ##### Add offset to events ######
        if self.b_fixed_delay_edit.text()=='':
            offset_val_ms = 0
        else: 
            offset_val_ms = float(self.b_fixed_delay_edit.text())
        if hasattr(self, 'add_offset_list'):
            if len(self.add_offset_list) > 0:
                offset_val_s = offset_val_ms / 1000 # convert to seconds
                tmp_code = f'dframe = add_event_offset(dframe, event_list={self.add_offset_list}, offset={offset_val_s})'
                time_corr_code.append(tmp_code)
        
        #####  Parsed Triggers  #######
        parsed_trig_code = []
        # Append the above triggers for parsemarks reference
        # parsed_trig_code.append(f"dframe = append_conditions(dframe_list)")
        for parse_tile in self.parsemarks_tile_list:
            markname = parse_tile.event_name.text()
            if markname in self.events_to_write:
                print(markname)
                
                lead_cond = parse_tile.b_evt1_name.currentText()
                lag_cond = parse_tile.b_evt2_name.currentText()
                window_start = float(parse_tile.b_window_t1.text())
                window_end = float(parse_tile.b_window_t2.text())
                window = [window_start, window_end]
                if parse_tile.b_mark_on_lead.isChecked():
                    on_val = 'lead'
                elif parse_tile.b_mark_on_lag.isChecked():
                    on_val = 'lag'
                else:
                    raise ValueError('Cannot interpret lead/lag for parsemarks: {markname}')
                
                tmp_code = f"dframe = parse_marks(dframe=dframe, lead_condition='{lead_cond}', lag_condition='{lag_cond}', window={window},  marker_on='{on_val}', marker_name='{markname}', append_result=True)"
                parsed_trig_code.append(tmp_code)
                tmp_code = "dframe.dropna(inplace=True)"
                parsed_trig_code.append(tmp_code)
                
        ##### Keep Events Section ######
        keep_evts_code = []
        tmp_ = \
f"""
final_dframe_list = []
for evt_name in {self.events_to_write}:
    keep_dframe=dframe[dframe.condition==evt_name]
    final_dframe_list.append(keep_dframe)
final_dframe = append_conditions(final_dframe_list)
"""
        keep_evts_code.append(tmp_)
        keep_evts_code.append('write_markerfile(dframe=final_dframe, ds_filename=meg_fname)')
        
                
        ##### Combine Initial Trigger Processing #####
        init_trig_code = init_code + ana_trig_code + dig_trig_code + time_corr_code + parsed_trig_code + keep_evts_code
        with open(fname, 'a') as f:
            for i in init_trig_code:
                f.write(i+'\n')
                f.write('\n')
        
        # tmp_code = 
        # init_trig_code.ap
                

                
# app = QApplication(sys.argv)
# win = event_coding_Window(cmdline_meg_fname='sub-ON80038_ses-01_task-sternberg_run-01_meg.ds')  
# win.tile_dict          

# i=0
# for key in win.tile_dict.keys():
#     tile = win.tile_dict[key]
#     tile.event_name.setText(f'New_{i}')
#     print(tile.event_name.text())
#     i+=1
    
        
    
    
#%%    


def window():
    app = QApplication(sys.argv)
    win = event_coding_Window() 
    win.show()
    sys.exit(app.exec_())
    
# window()



#%% CMD line section
    
def cmdline_main():
    window()

if __name__ == '__main__':
    cmdline_main()






