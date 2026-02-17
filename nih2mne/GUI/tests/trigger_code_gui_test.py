#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 10:52:29 2026

@author: jstout
"""

from nih2mne.GUI.trigger_code_gui2 import event_coding_window
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import sys



# app = QApplication(sys.argv)
# win = event_coding_window() 
# win.show()
# #sys.exit(app.exec_())



# tmp = parse_marks_tile(event_name_dict=OrderedDict(test1='test',test2=3))        

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
    if _widget.te_EvtName.text() == 'projector':
        continue
    else:
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
assert win.event_namelist == ['ana0', 'projector', 'dig0', 'dig1', 'dig2', 'dig3', 'parse1']

#%% Check add fixed offset
win.ui.pb_FixedOffset.click()
win.ui.te_FixedOffset.setText('10')

layout = win.fixed_delay_selector.grid_layout
win.add_offset_list = []

for i in range(layout.count()):
    item = layout.itemAt(i)
    if item.widget():
        if item.widget().text() in ['dig1', 'dig3']:
            item.widget().click()
win.fixed_delay_selector.b_set_selection.click()   
win.fixed_delay_selector.close()

assert win.add_offset_list == ['dig1','dig3']

#%% Check correct to projector 
win.ui.pb_CorrectToProjector.click()
layout = win.corr2proj_selector.grid_layout

for i in range(layout.count()):
    item = layout.itemAt(i)
    if item.widget():
        if item.widget().text() in ['dig0', 'dig2']:
            item.widget().click()
win.corr2proj_selector.b_set_selection.click()
win.corr2proj_selector.close()
assert win.corr2proj_list == ['dig0', 'dig2']

#%% Select output events
win.ui.pb_FinalEventSelection.click()
layout = win.set_final_events_selector.grid_layout
win.final_events_list = []

for i in range(layout.count()):
    item = layout.itemAt(i)
    if item.widget():
        if item.widget().text() in ['dig0', 'dig2','parse1']:
            item.widget().click()
win.set_final_events_selector.b_set_selection.click()
win.set_final_events_selector.close()
assert win.final_events_list == ['dig0', 'dig2', 'parse1']
assert win.ui.te_OutputEvents.toPlainText() == 'dig0 \ndig2 \nparse1 \n'

#%% Check the parsemarks Check N=? button 
    
win.ui.pb_AddParser.click()
_item = win.ui.list_ParseMarks.item(0)
_widget = win.ui.list_ParseMarks.itemWidget(_item)
_widget.te_MrkName.setText('parse1')
_widget.combo_LeadSelection.setCurrentIndex(3)
_widget.combo_LagSelection.setCurrentIndex(5)


win.handle_check_request(_widget)
assert _widget.pb_Check.text() == 'N=38'





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
