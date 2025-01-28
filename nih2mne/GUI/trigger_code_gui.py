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
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel,  QComboBox, QLineEdit, QCheckBox
# from functools import partial
import sys
import os, os.path as op
import numpy as np
from nih2mne.utilities.trigger_utilities import (parse_marks, detect_digital, 
                                                 check_analog_inverted, threshold_detect, 
                                                 append_conditions)



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
        
        
        
        
        
        
############# Setup Window ######################            
        
        
        

class event_coding_Window(QMainWindow):
    def __init__(self, cmdline_meg_fname=False):
        super(event_coding_Window, self).__init__()
        self.setGeometry(100,100, 1000, 1000) #250*gridsize_col, 100*gridsize_row)
        self.setWindowTitle('Event Coding GUI')
        
        self.tile_dict = {}
        
        # Finalize Widget and dispaly
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
    
    def update_event_names(self):
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
        
        ############# Parsemarks Layout ##########################
        self.parsemarks_tile_list = []
        self.parsemarks_full_layout_list = []
        self.add_parsemarks_line()
        
        ############# Keep Events Layout #########################
        #Empty the keep events layout list, so it doesn't append the previous
        self.update_keep_events_list(flush=True)

            
    def update_keep_events_list(self, flush=False):
        num_keep_buttons = self.keep_events_layout.count()
        if flush==True:
            if num_keep_buttons > 1:
                for i in reversed(range(1,num_keep_buttons)):  #Skip the label - remove from the end
                    item = self.keep_events_layout.takeAt(i)
                    self.keep_events_layout.removeItem(item)
                    if item.widget():
                        item.widget().deleteLater()

        for i in self.event_namelist: 
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
        
    # def print_event_lists(self):
    #     for i in self.tile_dict.items()
        
    #     ana_trig_code = []
    #     for i, tile in self.tile_dict.items():
    #         if i.startswith('UADC') and (tile.event_name.text() in self.events_to_write) :
    #             markname = tile.event_name.text()
    #             if tile.b_downgoing_trigger.checkState()==2:
    #                 invert_val = True
    #             else:
    #                 invert_val = False
    #             tmp_code = f"tmp_dframe = threshold_detect(dsname=meg_fname, channel='{i}', mark='{markname}', invert={invert_val})"
    #             ana_trig_code.append(tmp_code)
    #             tmp_code = f"dframe_list.append(tmp_dframe)"
    #             ana_trig_code.append(tmp_code)
                
    #     ##### Digital Triggers #####       #####!!!!!!    IF NAME does not exist - dont write
    #     dig_trig_code = []
    #     for i, tile in self.tile_dict.items():
    #         if i.startswith('UPPT') and (tile.event_name.text() in self.events_to_write):
    #             markname = tile.event_name.text()
    #             # if self.b_downgoing_trigger.checkState()==2:
    #             #     invert_val = True
    #             tmp_code = f"tmp_dframe = detect_digital(filename=meg_fname, channel='{i}', mark='{markname}')"
    #             dig_trig_code.append(tmp_code)        
    #             tmp_code = f"dframe_list.append(tmp_dframe)"
    #             dig_trig_code.append(tmp_code)
        
        
        
        
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
                    check_analog_inverted, threshold_detect, append_conditions)\n''',
                '''from nih2mne.utilities.markerfile_write import main as write_markerfile\n'''
                "\n\n",
                "meg_fname = sys.argv[1]\n\n"
                ]
        fname = "/tmp/testfile.py"
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
                dig_val = i.split('_')[-1]
                tmp_code = f"dig_dframe.loc[dig_dframe.condition=='{dig_val}', 'condition']='{markname}'"
                dig_trig_code.append(tmp_code)        
        tmp_code = f"dframe_list.append(dig_dframe)"
        dig_trig_code.append(tmp_code)
        
        #####  Parsed Triggers  #######
        parsed_trig_code = []
        # Append the above triggers for parsemarks reference
        parsed_trig_code.append(f"dframe = append_conditions(dframe_list)")
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
        init_trig_code = init_code + ana_trig_code + dig_trig_code + parsed_trig_code + keep_evts_code
        fname = "/tmp/testfile.py"
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

#%% Test 
# import pytest

# def test_window():
#     import nih2mne
#     from PyQt5.QtTest import QTest
#     from PyQt5.QtCore import Qt
#     raw_fname = op.join(nih2mne.__path__[0], 'test_data','20010101','ABABABAB_haririhammer_20010101_002.ds')
#     # raw = mne.io.read_raw_ctf(raw_fname, preload=True, system_clock='ignore',clean_names=True)
#     app = QApplication(sys.argv)
#     win = event_coding_Window(cmdline_meg_fname=raw_fname)
    
#     # Set the tile_dict labels
#     for key,tile in win.tile_dict.items():
        
#         print(key)
#         tile.event_name.setText(f'evt_{key}')
#         print(tile.event_name.text())
    
#     self = win        
#     tmp_events_to_write = ['evt_UADC016','evt_UPPT001_11', 'evt_parse_test2', 'evt_parse_test3']
    
#     # Press the Update event names to initialize parse marks panel
#     QTest.mouseClick(self.b_update_event_names, Qt.LeftButton)
    
#     # Set parse marks test event names
#     tmp_parsemarks_set_names = ['evt_parse_test1', 'evt_parse_test2', 'evt_parse_test3']
#     tmp_mark_on = ['lead', 'lag','lead']
#     lead_lag_pairs = [['evt_UADC006', 'evt_UADC007'],
#                       ['evt_UADC016', 'evt_UPPT001_12'],
#                       ['evt_UADC016', 'evt_UADC007']]
    
#     for idx, parsemarks_name in enumerate(tmp_parsemarks_set_names):
#         test_parse_marks_tile = self.parsemarks_tile_list[-1]
#         # Set the marker name for the tile
#         test_parse_marks_tile.event_name.setText(parsemarks_name)
        
#         #Set lead/lag
#         lead, lag = lead_lag_pairs[idx]
#         test_parse_marks_tile.b_evt1_name.setCurrentText(lead)
#         test_parse_marks_tile.b_evt2_name.setCurrentText(lag)
        
#         if tmp_mark_on[idx] == 'lead':
#             test_parse_marks_tile.b_mark_on_lead.setChecked(True)
#         else:
#             test_parse_marks_tile.b_mark_on_lag.setChecked(True)
#             QTest.mouseClick(test_parse_marks_tile.b_mark_on_lag, Qt.LeftButton)
        
#         # The last item is the parsemarks event name
#         evt_name_idx = self.parsemarks_full_layout_list[-1].layout().count() - 1
#         widg = self.parsemarks_full_layout_list[-1].itemAt(evt_name_idx).widget()
#         QTest.mouseClick(widg, Qt.LeftButton)
    
#     ## Test that the parsemarks names have been set correctly
#     assert len(self.parsemarks_tile_list) == len(tmp_parsemarks_set_names) + 1
#     for i in range(len(tmp_parsemarks_set_names)-1):
#         assert self.parsemarks_tile_list[i].event_name.text() == tmp_parsemarks_set_names[i]
        
#     ## Test that all names are set correctly
#     test_all_event_list = ['evt_UADC006', 'evt_UADC007', 'evt_UADC016', 'evt_UPPT001_1',
#                       'evt_UPPT001_11', 'evt_UPPT001_12', 'evt_UPPT001_21',
#                       'evt_UPPT001_22', 'evt_parse_test1', 'evt_parse_test2',
#                       'evt_parse_test3']
#     assert set(win.event_namelist) == set(test_all_event_list)
    
#     # Set the keep datasets using QTest Mouseclick
#     for i in range(self.keep_events_layout.layout().count()):
#         evt_name = self.keep_events_layout.layout().itemAt(i).widget().text()
#         print(f'Count {i}: {evt_name}')
#         if evt_name in tmp_events_to_write:
#             tmp=self.keep_events_layout.layout().itemAt(i).widget()
#             QTest.mouseClick(tmp, Qt.LeftButton)
#             try:
#                 assert tmp.isChecked()
#             except:
#                 print(tmp.text())

#     test_checked=[]             
#     for i in range(self.keep_events_layout.layout().count()):
#         try:
#             if self.keep_events_layout.layout().itemAt(i).widget().isChecked():
#                 print(f'{ self.keep_events_layout.layout().itemAt(i).widget().text()}')
#                 test_checked.append(self.keep_events_layout.layout().itemAt(i).widget().text())
#         except:
#             pass  #This is in case the widget doesn't have the isChecked tag
    
#     # Confirm that the clicking sets the appropriate checks to be written
#     assert set(test_checked) == set(tmp_events_to_write)
    
#     # Confirm that the 
#     # del self.events_to_write
#     self.write_parser_script()
    
    
    


#%% CMD line section
    
def cmdline_main():
    window()

if __name__ == '__main__':
    cmdline_main()






