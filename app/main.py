#!/usr/bin/python
# -*- coding: utf-8 -*-

from kivy.app import App
from kivy.metrics import cm
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
import kivy.clock

import webbrowser
import pathlib
import threading
from schedule import Schedule
import time
import glob


try:
    client_file = glob.glob('../client_data/*.csv')[0]
# When ther is no such file the list is empty and we get index error,
# which we try to capture
except IndexError:
    client_file = str()

try:
    instructor_file = glob.glob('../instructor_data/*.csv')[0]
# When ther is no such file the list is empty and we get index error,
# which we try to capture
except IndexError:
    instructor_file = str()

# global variable
schedule_global = None


class MainWindow(Screen):
    #TODO: split logic and GUI part between .kv and .py
    my_popup = Popup(title="Error", size_hint=(0.3, 0.3), auto_dismiss=False)
    popup_content = BoxLayout(orientation="vertical")
    popup_content.add_widget(Label(size_hint=(1, 0.8), pos_hint={'top': 1}, text_size=(cm(6), cm(4)), font_size='20sp',
                                   text="TypeError: Error during updating schedule - schedule is empty."))
    popup_content.add_widget(Button(text='Close me!', size_hint=(1, 0.2), on_press=my_popup.dismiss))
    my_popup.content = popup_content

    def update_schedule_description(self):
        # try:
        global schedule_global
        lista = [time_slot[0, 0].lesson_type for time_slot in schedule_global.schedule.reshape(-1, 1, 1)]
        SeeSchedule.time_slots = lista
        print(SeeSchedule.time_slots)
        # except AttributeError:
        #     MainWindow.my_popup.open()


class Optimize(Screen):
    # TODO: co prodram ma zrobić po skończeniu optymalizacji
    # TODO: ładowanie plików
    # TODO: usunąć wywoływanie opt. w schedule.py
    def __init__(self, **kw):
        super().__init__(**kw)
        self.parameters = {
            'neighborhood_type_lst': list(),
            'initial_solution': False,
            'alpha': 0.99,
            'initial_temp': 100,
            'n_iter_one_temp': 50,
            'min_temp': 0.1,
            'epsilon': 0.01,
            'n_iter_without_improvement': 1000}

    def checkbox_click(self, instance, value, neighborhood_type):
        if value is True:
            self.parameters['neighborhood_type_lst'].append(neighborhood_type)
        else:
            self.parameters['neighborhood_type_lst'].remove(neighborhood_type)


    def on_text(self, parameter, input_parameter):
        try:
            self.parameters[parameter] = int(input_parameter)
        except ValueError:
            self.parameters[parameter] = 0
        print(self.parameters['n_iter_without_improvement'])

    def start_optimization(self):
        SM = Schedule(client_file=client_file,
                      instructor_file=instructor_file,
                      max_clients_per_training=5, time_slot_num=6)

        print("\nINITIAL SCHEDULE")
        print(SM)
        print('Initial earnings: ', SM.get_cost())
        first_cost = SM.get_cost()
        tic = time.time()

        best_cost, num_of_iter, all_costs = SM.simulated_annealing(alpha=self.parameters['alpha'],
                                                                   initial_temp=self.parameters['initial_temp'],
                                                                   n_iter_one_temp=self.parameters['n_iter_one_temp'],
                                                                   min_temp=self.parameters['min_temp'],
                                                                   epsilon=self.parameters['epsilon'],
                                                                   n_iter_without_improvement=self.parameters['n_iter_without_improvement'],
                                                                   initial_solution=self.parameters['initial_solution'],
                                                                   neighborhood_type_lst=self.parameters['neighborhood_type_lst'])
        toc = time.time()

        print("\nAFTER OPTIMIZATION")
        print(SM)
        print("Number of iterations: ", num_of_iter)

        print("Best earnings: ", best_cost)
        second_cost = best_cost
        print("Time: ", toc - tic)

        SM.improve_results()
        print("\nIMPROVED SCHEDULE")
        print(SM)
        print("Best improved earnings: ", SM.get_cost())

        third_cost = SM.get_cost()

        print(f'{first_cost} $ --> {second_cost} $ --> {third_cost} $')
        global schedule_global
        schedule_global = SM

class LoadFiles(Screen):
    pass



class ClientFileChooser(Screen):
    def get_path(self):
        return str(pathlib.Path(__file__).parent.parent.resolve()) + r'\client_data'

    def selected(self, filename):
        global client_file
        try:
            self.ids.client_path_label.text = filename[0]
            client_file = filename[0]
        except:
            pass


class InstructorFileChooser(Screen):
    def get_path(self):
        return str(pathlib.Path(__file__).parent.parent.resolve()) + r'\instructor_data'

    def selected(self, filename):
        global instructor_file
        try:
            self.ids.instructor_path_label.text = filename[0]
            instructor_file = filename[0]
        except:
            pass

class AboutOrganizer(Screen):
    def github_button_on(self):
        self.ids.github_button_img.source = 'images/GitHub-Mark-Light-120px-plus_pressed.png'

    def github_button_off(self):
        self.ids.github_button_img.source = 'images/GitHub-Mark-Light-120px-plus.png'
        webbrowser.open('https://github.com/kmotyka00/ScheduleOptimizationProblem')


class SeeSchedule(Screen):
    time_slots = [None for i in range(36)]
    def generate_schedule_layout(self):
        global schedule_global
        for time_slot in SeeSchedule.time_slots:
            self.ids.schedule_layout.add_widget(Button(text=f'{time_slot}'))

class WindowManager(ScreenManager):
    pass


kv = Builder.load_file('new_window.kv')

class ScheduleOrganizer(App):
    def build(self):
        return kv


if __name__ == '__main__':
    ScheduleOrganizer().run()

#TODO: Wyświetlanie wyników, wykresu
