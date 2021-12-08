#!/usr/bin/python
# -*- coding: utf-8 -*-

from kivy.app import App
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
import kivy.clock

import webbrowser
import threading

class MainWindow(Screen):
    pass

class Optimize(Screen):
    pass

class LoadFiles(Screen):
    pass

class AboutOrganizer(Screen):
    pass

class SeeSchedule(Screen):
    pass

class WindowManager(ScreenManager):
    pass


kv = Builder.load_file('new_window.kv')

class ScheduleOrganizer(App):
    def build(self):
        return kv


if __name__ == '__main__':
    ScheduleOrganizer().run()