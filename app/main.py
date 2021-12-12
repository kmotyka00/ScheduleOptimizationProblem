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
import pathlib

class MainWindow(Screen):
    pass

class Optimize(Screen):
    pass

class LoadFiles(Screen):
    pass

class ClientFileChooser(Screen):
    def get_path(self):
        return str(pathlib.Path(__file__).parent.parent.resolve())


class InstructorFileChooser(Screen):
    def get_path(self):
        return str(pathlib.Path(__file__).parent.parent.resolve())

class AboutOrganizer(Screen):
    def github_button_on(self):
        self.ids.github_button_img.source = 'images/GitHub-Mark-Light-120px-plus_pressed.png'

    def github_button_off(self):
        self.ids.github_button_img.source = 'images/GitHub-Mark-Light-120px-plus.png'
        webbrowser.open('https://github.com/kmotyka00/ScheduleOptimizationProblem')

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