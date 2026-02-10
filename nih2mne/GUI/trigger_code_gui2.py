#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/04/2026

@author: jstout

Trigger Channel Coding GUI (v2)

MainWindow w/ tabs
Channel tab -- 
   reads the trigger channels
   gen trig_tile for each ADC chan and per event on UPPT
   and adds those to trig_dict
Parse Marks tab -- 
  generate parse_marks tile

TODO: Temporal coding on PPT -- a 1 followed by a 5 is a Y event 
Set action to invert trigger for counts    
Set all dig trigs to invert together   
Create a parsemarks list and remove all after an erase command
detect_digital currently does not support downgoing triggers

Add trigger inversion for digital trigger
"""
#%% Defaults section

"""
[tab_channel_labels]
-lbl_FName
-pb_SelectMeg

scrollAreaAnaContents
-label_AnalogChannels
-list_AnalogChannels

lbl_DigitalChannels
list_DigitalChannels

[tab_parse_events]
pb_CorrectToProjector
pb_FixedOffset
te_FixedOffset
lbl_ParseMarks
list_ParseMarks

[tab_write_parser_file]
pb_CheckOutputEvents
pb_WriteProcessingScript
lbl_SelectOutputEvents
list_SelectOutputEvents
"""


import glob
import mne
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, \
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel,  QComboBox, QLineEdit, QCheckBox, \
    QFileDialog, QDialog, QListWidgetItem
import sys
import os, os.path as op
import numpy as np
from nih2mne.utilities.trigger_utilities import (parse_marks, detect_digital, 
                                                 check_analog_inverted, threshold_detect, 
                                                 append_conditions, correct_to_projector, 
                                                 add_event_offset)
from collections import OrderedDict
from PyQt5.QtCore import Qt, pyqtSignal
from nih2mne import config



############## Setup Main Classes ################
from nih2mne.GUI.templates.trigger_single_line import Ui_Form as trig_singleline_UiForm

class trig_tile(QWidget, trig_singleline_UiForm): 
    def __init__(self, chan_name=None, include_polarity=True, 
                 event_count=None, meg_fname=None):
        super().__init__()
        self.setupUi(self) 
        self.meg_fname = meg_fname
        self.trig_type = chan_name[1:4]
        self.ch_name = chan_name 
        self.lbl_ChannelName.setText(f'{chan_name} :')
        self.cb_Up.setCheckState(2)
        self.trigger_polarity = 'up' #Initialize to up
        self.cb_Up.clicked.connect(self.set_up_trigger_polarity)
        self.cb_Down.setCheckState(0)
        self.cb_Down.clicked.connect(self.set_down_trigger_polarity)

        # Event counter display
        if self.trig_type == 'ADC':
            self.event_dframe = threshold_detect(self.meg_fname,  channel=self.ch_name)
            self.event_count = len(self.event_dframe)
        elif self.trig_type == 'PPT':
            self.event_count = event_count
        else:
            self.event_count = 'ERR'
        self.lbl_EvtCount.setText(f'N={self.event_count}')
        
    def set_up_trigger_polarity(self):
        self.trigger_polarity = 'up'
        self.cb_Down.setCheckState(0)
    
    def set_down_trigger_polarity(self):
        self.trigger_polarity = 'down'
        self.cb_Up.setCheckState(0)

## ParseMarks Tiles
from nih2mne.GUI.templates.parse_marks_single_line import Ui_Form as parse_marks_singleline_UiForm
class parse_marks_tile(QWidget, parse_marks_singleline_UiForm): 
    close_clicked = pyqtSignal(object)
    def __init__(self, on_lead=True, on_lag=False, start_offset=0, stop_offset=0.5, 
                 event_name_dict=OrderedDict()):
        super().__init__()
        self.setupUi(self) 
        assert isinstance(event_name_dict, OrderedDict)
        self.possible_items = event_name_dict
        self.fill_lag_combobox()
        self.fill_lead_combobox()
        self.te_StartOffset.setText(str(start_offset))
        self.te_StopOffset.setText(str(stop_offset))
        
        self.cb_OnLead.clicked.connect(self.set_onlead_selection)
        self.cb_OnLag.clicked.connect(self.set_onlag_selection)
        # Assign Default to OnLead
        self.cb_OnLead.setCheckState(2)
        
        # Handle tile deletion
        self.pb_DeleteLine.clicked.connect(lambda: self.close_clicked.emit(self))
        
        
    def set_onlead_selection(self):
        self.mark_on = 'lead'
        self.cb_OnLag.setCheckState(0) #Cross toggle on lag button
        
    def set_onlag_selection(self):
        self.mark_on = 'lag'
        self.cb_OnLead.setCheckState(0) #Cross toggle on lead button
    
    def fill_lead_combobox(self):  #May need to cross reference lead/lag to prevent doubles
        _tmp =  [self.possible_items[i]['name'] for i in self.possible_items.keys()]   
        self.combo_LeadSelection.addItems(_tmp)
        
    def fill_lag_combobox(self):
        _tmp = [self.possible_items[i]['name'] for i in self.possible_items.keys()]      
        self.combo_LagSelection.addItems(_tmp)
    
    def get_outputs(self):
        outputs = {}
        if self.cb_OnLead==2:
            outputs['mark_on'] = 'lead'
        elif self.cb_OnLag==2:
            outputs['mark_on'] = 'lag'
        
        outputs['lead_evt'] = self.combo_LeadSelection.currentText()
        outputs['lag_evt'] = self.combo_LagSelection.currentText()
        outputs['start_offset'] = self.te_StartOffset.text()
        outputs['stop_offset'] = self.te_StopOffset.text()
        outputs['name'] = self.te_MrkName.text()
        return outputs
    
    def _disable(self):
        #Lead Combo 
        self.combo_LeadSelection.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.combo_LeadSelection.setFocusPolicy(Qt.NoFocus)
        #Lag Combo
        self.combo_LagSelection.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.combo_LagSelection.setFocusPolicy(Qt.NoFocus)
        #MarkerName
        self.te_MrkName.setReadOnly(True)
        
    
      
        
# tmp = parse_marks_tile(event_name_dict=OrderedDict(test1='test',test2=3))        


#%%
from nih2mne.GUI.templates.trigger_processing_gui import Ui_MainWindow as trigUi_mw        
class event_coding_window(QMainWindow):
    def __init__(self, meg_fname=None):
        super().__init__()
        self.ui = trigUi_mw()
        self.ui.setupUi(self)
        
        #Make the active tab the Channel Labels tab
        self.ui.tabWidget.setCurrentIndex(1) 
        self.tile_dict = {}
        
        self.ui.pb_SelectMeg.clicked.connect(self.act_pb_SelectMeg)
        self.ui.pb_AddParser.clicked.connect(self.act_pb_add_parser_line)
        
        #Initialize if meg_fname provided at commandline
        if meg_fname != None:
            self.act_pb_SelectMeg(meg_fname=meg_fname)
        else:
            self.meg_fname = meg_fname
        
        self.ui.pb_CorrectToProjector.clicked.connect(self.act_pb_corr2proj)
        self.ui.pb_FixedOffset.clicked.connect(self.act_pb_add_fixed_delay)
        
        self.ui.pb_FinalEventSelection.clicked.connect(self.act_pb_select_final_events)
        self.ui.pb_PlotData.clicked.connect(self.act_plot_data)
        
        self.ui.pb_WriteProcessingScript.clicked.connect(self.write_parser_script)
    
    def act_plot_data(self):   ##FIX !!!!!!!  no events plotted
        self.raw = mne.io.read_raw_ctf(self.meg_fname, system_clock='ignore', 
                                       clean_names=True, verbose=False)
        self.raw.pick_types(meg=False, eeg=False, misc=True)
        self.raw.load_data()
        import numpy as np
        
        event_list = self.event_namelist  #Make this the keep namelist after testing
        evts = np.zeros([len(event_list), 3])
        
        self.raw.plot() #events = evts,  #[row, [samp, duration, ID_int]]
                      #event_id = {})
        
        
            
    def extract_evt_dict(self):
        evtname_dict = {}
        # Get trig_tile names (Triggers)
        for key in self.tile_dict.keys():
            _name = self.tile_dict[key].te_EvtName.text()
            if _name in [None, '']:
                continue
            else:
                if key.startswith('UADC'):
                    _type='ADC'
                else:
                    _type='PPT'
            evtname_dict[key] = {}
            evtname_dict[key]['type'] = _type
            evtname_dict[key]['name'] = _name
            evtname_dict[key]['tile'] = self.tile_dict[key]
            
        # Get parse_marks names 
        if self.ui.list_ParseMarks.count() > 0:
            for idx in range(self.ui.list_ParseMarks.count()):
                evtname_dict[f'parse_{idx}']={}
                evtname_dict[f'parse_{idx}']['type'] = 'parse'
                
                _item = self.ui.list_ParseMarks.item(idx)
                _widget = self.ui.list_ParseMarks.itemWidget(_item)
                _outputs = _widget.get_outputs()
                evtname_dict[f'parse_{idx}']['name'] = _outputs['name']
                evtname_dict[f'parse_{idx}']['tile'] = _widget
        
        #Logic if event names coincide (parseMarks can have duplicates )
        
        return evtname_dict
    
    #>> Begin of changes
    
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
        name_dict = self.extract_evt_dict()
        for key in name_dict.keys():
            name_text = name_dict[key]['name']
            if name_text.strip() == '':
                continue
            namelist.append(name_text)
        
        # for key in self.tile_dict.keys():
        #     tmp_txt = self.tile_dict[key].te_EvtName.text()
        #     namelist.append(tmp_txt)
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
        # self.parsemarks_tile_list = []
        # self.parsemarks_full_layout_list = []
        # self.add_parsemarks_line()
        
        ############# Keep Events Layout #########################
        #Empty the keep events layout list, so it doesn't append the previous
        # self.update_keep_events_list(flush=True)
    
        
    def act_pb_corr2proj(self):
        # Create a popup to select events that will be corrected to projector
        self.update_event_names(events_only=True)
        self.corr2proj_selector = grid_selector(self.event_namelist)  
        self.corr2proj_selector.b_set_selection.clicked.connect(self._set_correct2proj_list)
        
    def act_pb_add_fixed_delay(self):
        # Create a popup to select the events that will be corrected to a fixed delay
        self.update_event_names(events_only=True)
        self.fixed_delay_selector = grid_selector(self.event_namelist)  
        self.fixed_delay_selector.b_set_selection.clicked.connect(self._set_fixedDelay_list)
        
    def act_pb_select_final_events(self):
        # Create a popup to select the events that will be used to be added to Markerfile
        self.update_event_names(events_only=True)
        self.set_final_events_selector = grid_selector(self.event_namelist)  
        self.set_final_events_selector.b_set_selection.clicked.connect(self._set_final_events_list)
        
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
        
    def _set_final_events_list(self):
        'Set the final events and assign to self.finalEvents_list'
        layout = self.set_final_events_selector.grid_layout
        self.final_events_list = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                if hasattr(item.widget(), 'isChecked'):
                    if item.widget().isChecked():
                        self.final_events_list.append(item.widget().text())
        print(f'Final events in markerfil: {self.final_events_list}') 
            
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
        
        
    ## <<<<< END of changes
            
            
    def act_pb_add_parser_line(self):
        '''Take all the items in channel labels and prior parser lines output labels
        Add them to the possible entries in the parser choice'''

        #Disable prior parser line so it cannot be edited
        _count = self.ui.list_ParseMarks.count()
        last_idx = _count - 1
        if _count > 0:
            _item = self.ui.list_ParseMarks.item(last_idx)
            _widget = self.ui.list_ParseMarks.itemWidget(_item)
            _widget._disable() 
        
        #Add parser line
        #Update the parser list with a parsemarks tile  list_ParseMarks
        evts_names = self.extract_evt_dict()
        _pm_tile = parse_marks_tile(event_name_dict=OrderedDict(evts_names))
        _pm_tile.close_clicked.connect(self.handle_close_request)
                                    
        item = QListWidgetItem(self.ui.list_ParseMarks)
        item.setSizeHint(_pm_tile.sizeHint())
        self.ui.list_ParseMarks.addItem(item)
        self.ui.list_ParseMarks.setItemWidget(item, _pm_tile)
        
        
        
    def handle_close_request(self, widget):
        '''If file tile "emits" a close signal, this will trigger a loop over
        filenames to identify the widget that produced the close signal'''
        item_count = self.ui.list_ParseMarks.count()
        for i in range(item_count):
            item = self.ui.list_ParseMarks.item(i)
            print(type(item))
            if self.ui.list_ParseMarks.itemWidget(item) == widget:
                self.ui.list_ParseMarks.takeItem(i)
                print(f"Parent removed item at row {i}")
                break
        
    def act_pb_SelectMeg(self, meg_fname=None):
        print(f'MEG fname: {meg_fname}')
        if meg_fname in [None, False]:
            meg_fname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a MEG folder (.ds)')
        self.meg_fname = meg_fname
        if meg_fname.endswith('/'): meg_fname=meg_fname[:-1]  
        
        meg_display_name = meg_fname.split('/')[-1]
        self.ui.lbl_FName.setText(meg_display_name)    
        if meg_fname not in [None, False]:
            self.meg_raw = mne.io.read_raw_ctf(meg_fname, clean_names=True, 
                                               system_clock='ignore', preload=False, 
                                               verbose=False)
        self.trig_ch_names = [i for i in self.meg_raw.ch_names if i.startswith('UADC') or i.startswith('UPPT')]
        self.fill_trig_chan_layout()
        
        
    def fill_trig_chan_layout(self):
        '''
        Iterates over analog channels and PPT conditions to fill panel
        | Trig Name | b_UpTrig | b_DownTrig | Count=  | OutName: | 
        '''
        for i in self.trig_ch_names:
            ### Analogue Channels ###
            if i.startswith('UADC'):
                self.tile_dict[i]=trig_tile(chan_name=i,include_polarity=True,
                                            meg_fname=self.meg_fname)
                if i=='UADC016':
                    self.tile_dict[i].te_EvtName.setText('projector')
                
                #Add items to list in complicated QT fashion
                item = QtWidgets.QListWidgetItem(self.ui.list_AnalogChannels)
                item.setSizeHint(self.tile_dict[i].sizeHint())
                self.ui.list_AnalogChannels.addItem(item) 
                self.ui.list_AnalogChannels.setItemWidget(item, self.tile_dict[i])

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
                    
                    #Add items to list in complicated QT fashion
                    item = QtWidgets.QListWidgetItem(self.ui.list_DigitalChannels)
                    item.setSizeHint(self.tile_dict[evt_key].sizeHint())
                    self.ui.list_DigitalChannels.addItem(item) 
                    self.ui.list_DigitalChannels.setItemWidget(item, self.tile_dict[evt_key])
            else:
                print(f'Not processing channel {i}')
    
    def get_default_scriptname(self):
        default_loc = config.TRIG_FILE_LOC
        basename = op.basename(self.meg_fname)
        if basename.split('_').__len__() > 3:
            task_name = basename.split('_')[1]
        else:
            task_name = 'taskName'
        
        all_versions = glob.glob(op.join(default_loc, f'{task_name.lower()}_v?.py'))
        all_versions = sorted(all_versions)
        last_version = all_versions[-1].split('_v')[-1].split('.')[0]
        new_version = str(int(last_version) + 1)
        default_name = f'{task_name}_v{new_version}.py'
        return op.join(default_loc, default_name)
    
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
        self.events_to_write = self.final_events_list
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
        default_fname = self.get_default_scriptname()
        fname, _ = QFileDialog.getSaveFileName(self, "Save File", default_fname, 
                                               "Text Files (*.py);;All Files (*)")#, options=options)
        with open(fname, 'w') as f:
            f.writelines(header)
        
        ##### Make a list of dataframes ####
        init_code = []
        init_code.append('dframe_list=[]')
                         
        
        ##### Analog Triggers #####
        ana_trig_code = []
        ana_trig_code.append('##### Analog Trigger Coding ######')
        for i, tile in self.tile_dict.items():
            if i.startswith('UADC'):
                print(i, tile.te_EvtName.text())
                markname = tile.te_EvtName.text()
                if markname == '':
                    continue
                if tile.cb_Down.checkState()==2:
                    invert_val = True
                else:
                    invert_val = False
                tmp_code = f"tmp_dframe = threshold_detect(dsname=meg_fname, channel='{i}', mark='{markname}', invert={invert_val})"
                ana_trig_code.append(tmp_code)
                tmp_code = f"dframe_list.append(tmp_dframe)"
                ana_trig_code.append(tmp_code)
                
        ##### Digital Triggers #####       
        dig_trig_code = []
        dig_trig_code.append('##### Digital Trigger Coding ######')
        tmp_code = f"dig_dframe = detect_digital(filename=meg_fname, channel='UPPT001')" #, mark='{markname}')"
        dig_trig_code.append(tmp_code)
        for i, tile in self.tile_dict.items():
            if i.startswith('UPPT'):
                print(i, tile.te_EvtName.text())
                markname = tile.te_EvtName.text()
                if markname == '':
                    continue
                dig_val = i.split('_')[-1]
                tmp_code = f"dig_dframe.loc[dig_dframe.condition=='{dig_val}', 'condition']='{markname}'"
                dig_trig_code.append(tmp_code)        
        tmp_code = f"dframe_list.append(dig_dframe)"
        dig_trig_code.append(tmp_code)
        
        ##### Tidy up data   #######
        time_corr_code = []
        time_corr_code.append('##### Projector Correction and Event Offset ######')
        time_corr_code.append(f"dframe = append_conditions(dframe_list)")
                
        ##### Correct to Projector ###### 
        if hasattr(self, 'corr2proj_list'):
            if len(self.corr2proj_list) > 0:
                tmp_code = f'dframe = correct_to_projector(dframe, event_list={self.corr2proj_list}, window=[-0.2,0.2])'
                time_corr_code.append(tmp_code)        
        
        ##### Add offset to events ######
        if self.ui.te_FixedOffset.text().strip()=='':
            offset_val_ms = 0
        else: 
            offset_val_ms = float(self.ui.te_FixedOffset.text().strip())
        if hasattr(self, 'add_offset_list'):
            if len(self.add_offset_list) > 0:
                offset_val_s = offset_val_ms / 1000 # convert to seconds
                tmp_code = f'dframe = add_event_offset(dframe, event_list={self.add_offset_list}, offset={offset_val_s})'
                time_corr_code.append(tmp_code)
        
        #####  Parsed Triggers  #######
        parsed_trig_code = []
        parsed_trig_code.append('##### Parse Marks Coding ######')
        # Append the above triggers for parsemarks reference
        # parsed_trig_code.append(f"dframe = append_conditions(dframe_list)")
        parsemarks_dict = {i:j for i,j in self.extract_evt_dict().items() if j['type']=='parse' }  ### --- here
        for entry in parsemarks_dict.values():
            parse_tile = entry['tile']
            markname = parse_tile.te_MrkName.text()
            if markname in self.events_to_write:
                print(markname)
                
                lead_cond = parse_tile.combo_LeadSelection.currentText() 
                lag_cond = parse_tile.combo_LagSelection.currentText()
                window_start = float(parse_tile.te_StartOffset.text().strip())
                window_end = float(parse_tile.te_StopOffset.text().strip())
                window = [window_start, window_end]
                if parse_tile.cb_OnLead.isChecked():
                    on_val = 'lead'
                elif parse_tile.cb_OnLag.isChecked():
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
##### Set Events to Keep ######
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

    
    
    
        
#%% Testing    
        

app = QApplication(sys.argv)
win = event_coding_window(meg_fname='/fast2/20250205_hv2set/JACTZJMA_MID_20250205_008.ds') 
win.show()

#%%
assert win.ui.list_AnalogChannels.count()==2
#Test that the UADC016 gets labelled as projector
test_item = win.ui.list_AnalogChannels.item(1)
assert win.ui.list_AnalogChannels.itemWidget(test_item).te_EvtName.text()=='projector'

assert win.ui.list_DigitalChannels.count() == 4 

# Automatically write out the Ana/Dig trigger names
for i in range(win.ui.list_AnalogChannels.count()):
    _item = win.ui.list_AnalogChannels.item(i)
    _widget = win.ui.list_AnalogChannels.itemWidget(_item)
    _widget.te_EvtName.setText('ana'+str(i))
    
for i in range(win.ui.list_DigitalChannels.count()):
    _item = win.ui.list_DigitalChannels.item(i)
    _widget = win.ui.list_DigitalChannels.itemWidget(_item)
    _widget.te_EvtName.setText('dig'+str(i))
    
#%%  ParseMarks test
    
win.ui.pb_AddParser.click()
_item = win.ui.list_ParseMarks.item(0)
_widget = win.ui.list_ParseMarks.itemWidget(_item)
_widget.te_MrkName.setText('parse1')
_widget.combo_LeadSelection.setCurrentIndex(1)
_widget.combo_LagSelection.setCurrentIndex(2)


win.ui.pb_AddParser.click()
# Do a check to make sure that tile1 is read only
parsemarks_count = win.ui.list_ParseMarks.count()
assert parsemarks_count == 2
#Verify that the newly created marker name is in the combobox listing
assert 'parse1' in [win.ui.list_ParseMarks.itemWidget(win.ui.list_ParseMarks.item(i)).te_MrkName.text() for i in range(win.ui.list_ParseMarks.count())]

# Verify that it fills out according to the above automated inputs
win.update_event_names()
assert win.event_namelist == ['ana0', 'ana1', 'dig0', 'dig1', 'dig2', 'dig3', 'parse1']

#%%






#%%

# _item = win.ui.list_ParseMarks.item(0)
# _widget = win.ui.list_ParseMarks.itemWidget(_item)
# #Lead Combo
# # _widget.combo_LeadSelection.setReadOnly(True)
# _widget.combo_LeadSelection.setAttribute(Qt.WA_TransparentForMouseEvents, True)
# _widget.combo_LeadSelection.setFocusPolicy(Qt.NoFocus)
# #Lag Combo
# # _widget.combo_LagSelection.setReadOnly(True)
# _widget.combo_LagSelection.setAttribute(Qt.WA_TransparentForMouseEvents, True)
# _widget.combo_LagSelection.setFocusPolicy(Qt.NoFocus)

# _widget.te_MrkName.setReadOnly(True)



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






