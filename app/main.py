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


class Optimize(Widget):
    neighborhood_type_lst = list()

    def checkbox_click(self, instance, value, neighborhood_type):
        if value is True:
            Optimize.neighborhood_type_lst.append(neighborhood_type)
        else:
            Optimize.neighborhood_type_lst.remove(neighborhood_type)

class MainWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass


kv = Builder.load_file('new_window.kv')

class ScheduleOrganizer(App):
    def build(self):
        return kv


if __name__ == '__main__':
    ScheduleOrganizer().run()