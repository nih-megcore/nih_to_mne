#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:58:37 2025

@author: jstout
"""

import pytest
import os, os.path as op
import nih2mne
import sys
#from PyQt5 import QtWidgets
#from PyQt5.QtWidgets import QApplication
#from nih2mne.GUI.trigger_code_gui import event_coding_Window

@pytest.mark.skip(reason="Pyqt import lib issue")
def test_window():
    from PyQt5.QtTest import QTest
    from PyQt5.QtCore import Qt
    raw_fname = op.join(nih2mne.__path__[0], 'test_data','20010101','ABABABAB_haririhammer_20010101_002.ds')
    app = QApplication(sys.argv)
    win = event_coding_Window(cmdline_meg_fname=raw_fname)
    
    # Set the tile_dict labels
    for key,tile in win.tile_dict.items():
        
        print(key)
        tile.event_name.setText(f'evt_{key}')
        print(tile.event_name.text())
    
    self = win        
    tmp_events_to_write = ['evt_UADC016','evt_UPPT001_11', 'evt_parse_test2', 'evt_parse_test3']
    
    # Press the Update event names to initialize parse marks panel
    QTest.mouseClick(self.b_update_event_names, Qt.LeftButton)
    
    # Set parse marks test event names
    tmp_parsemarks_set_names = ['evt_parse_test1', 'evt_parse_test2', 'evt_parse_test3']
    tmp_mark_on = ['lead', 'lag','lead']
    lead_lag_pairs = [['evt_UADC006', 'evt_UADC007'],
                      ['evt_UADC016', 'evt_UPPT001_12'],
                      ['evt_UADC016', 'evt_UADC007']]
    
    for idx, parsemarks_name in enumerate(tmp_parsemarks_set_names):
        test_parse_marks_tile = self.parsemarks_tile_list[-1]
        # Set the marker name for the tile
        test_parse_marks_tile.event_name.setText(parsemarks_name)
        
        #Set lead/lag
        lead, lag = lead_lag_pairs[idx]
        test_parse_marks_tile.b_evt1_name.setCurrentText(lead)
        test_parse_marks_tile.b_evt2_name.setCurrentText(lag)
        
        if tmp_mark_on[idx] == 'lead':
            test_parse_marks_tile.b_mark_on_lead.setChecked(True)
        else:
            test_parse_marks_tile.b_mark_on_lag.setChecked(True)
            QTest.mouseClick(test_parse_marks_tile.b_mark_on_lag, Qt.LeftButton)
        
        # The last item is the parsemarks event name
        evt_name_idx = self.parsemarks_full_layout_list[-1].layout().count() - 1
        widg = self.parsemarks_full_layout_list[-1].itemAt(evt_name_idx).widget()
        QTest.mouseClick(widg, Qt.LeftButton)
    
    ## Test that the parsemarks names have been set correctly
    assert len(self.parsemarks_tile_list) == len(tmp_parsemarks_set_names) + 1
    for i in range(len(tmp_parsemarks_set_names)-1):
        assert self.parsemarks_tile_list[i].event_name.text() == tmp_parsemarks_set_names[i]
        
    ## Test that all names are set correctly
    test_all_event_list = ['evt_UADC006', 'evt_UADC007', 'evt_UADC016', 'evt_UPPT001_1',
                      'evt_UPPT001_11', 'evt_UPPT001_12', 'evt_UPPT001_21',
                      'evt_UPPT001_22', 'evt_parse_test1', 'evt_parse_test2',
                      'evt_parse_test3']
    assert set(win.event_namelist) == set(test_all_event_list)
    
    # Set the keep datasets using QTest Mouseclick
    for i in range(self.keep_events_layout.layout().count()):
        evt_name = self.keep_events_layout.layout().itemAt(i).widget().text()
        print(f'Count {i}: {evt_name}')
        if evt_name in tmp_events_to_write:
            tmp=self.keep_events_layout.layout().itemAt(i).widget()
            QTest.mouseClick(tmp, Qt.LeftButton)
            try:
                assert tmp.isChecked()
            except:
                print(tmp.text())

    test_checked=[]             
    for i in range(self.keep_events_layout.layout().count()):
        try:
            if self.keep_events_layout.layout().itemAt(i).widget().isChecked():
                print(f'{ self.keep_events_layout.layout().itemAt(i).widget().text()}')
                test_checked.append(self.keep_events_layout.layout().itemAt(i).widget().text())
        except:
            pass  #This is in case the widget doesn't have the isChecked tag
    
    # Confirm that the clicking sets the appropriate checks to be written
    assert set(test_checked) == set(tmp_events_to_write)
    
    # Confirm that the 
    # del self.events_to_write
    self.write_parser_script()
    
    
