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

import threading
from schedule import Schedule
import time




class MainWindow(Screen):
    pass

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
        SM = Schedule(client_file=r"C:\Users\kacpe\Desktop\StudiaS5\BO\ScheduleOptimizationProblem\form_answers.csv",
                      instructor_file=r"C:\Users\kacpe\Desktop\StudiaS5\BO\ScheduleOptimizationProblem\instructors_info.csv",
                      max_clients_per_training=5, time_slot_num=6)
        SM.generate_random_schedule(greedy=False)

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