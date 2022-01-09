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
from kivy.uix.togglebutton import ToggleButton
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
import copy

import numpy as np
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
#matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import pandas as pd


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
initial_solution = None


class MainWindow(Screen):
    pass


class Optimize(Screen):
    # TODO: usunąć wywoływanie opt. w schedule.py
    first_cost = float()
    second_cost = float()
    third_cost = float()
    all_costs = list()
    num_of_iter = int()
    clients_num = int()
    instructors_num = int()
    SM = None
    def __init__(self, **kw):
        super().__init__(**kw)
        self.parameters = {
            'neighborhood_type_lst': ['move_one'],
            # parameter unused by end user, moved to function call
            # 'initial_solution': False,
            'alpha': 0.999,
            'initial_temp': 100,
            'n_iter_one_temp': 50,
            'min_temp': 0.1,
            'epsilon': 0.01,
            'n_iter_without_improvement': 200,
            'greedy': False,
            'improve_results': True,
            'penalty_for_repeated': 250,
            'penalty_for_unmatched': 100,
            'use_penalty_method': False}

    def checkbox_greedy(self, instance, value):
        if value is True:
            self.parameters['greedy'] = True
        else:
            self.parameters['greedy'] = False

    def checkbox_improve_results(self, instance, value):
        if value is True:
            self.parameters['improve_results'] = True
        else:
            self.parameters['improve_results'] = False

    def checkbox_click(self, instance, value, neighborhood_type):
        if value is True:
            self.parameters['neighborhood_type_lst'].append(neighborhood_type)
        else:
            self.parameters['neighborhood_type_lst'].remove(neighborhood_type)

    def toggle_penalty_method(self, instance):
        if instance.state == 'down':
            self.parameters['use_penalty_method'] = True
            self.ids['change_instructor_box'].active = True
        else:
            self.parameters['use_penalty_method'] = False
            self.ids['change_instructor_box'].active = False


    def on_text(self, parameter, input_parameter):
        try:
            self.parameters[parameter] = int(input_parameter)
        except ValueError:
            self.parameters[parameter] = 0

    def update_algorithm_parameters(self):
        self.manager.get_screen('see_algorithm_parameters').ids.first_cost.text = str(Optimize.first_cost) + " $"
        self.manager.get_screen('see_algorithm_parameters').ids.second_cost.text = str(Optimize.second_cost) + " $"
        self.manager.get_screen('see_algorithm_parameters').ids.third_cost.text = str(Optimize.third_cost) + " $"
        self.manager.get_screen('see_algorithm_parameters').ids.num_of_iter.text = str(Optimize.num_of_iter)
        self.manager.get_screen('see_algorithm_parameters').ids.clients_num.text = str(Optimize.clients_num)
        self.manager.get_screen('see_algorithm_parameters').ids.instructors_num.text = str(Optimize.instructors_num)
        self.manager.get_screen('see_algorithm_parameters').ids.alg_time.text = str(format(Optimize.alg_time, '.2f')) + " s"

    def generate_initial_solution(self, force=False):
        global schedule_global
        global initial_solution
        if schedule_global is None or force is True:
            schedule_global = Schedule(client_file=client_file,
                          instructor_file=instructor_file,
                          class_num=ScheduleParameters.schedule_parameters['class_num'],
                          day_num=ScheduleParameters.schedule_parameters['day_num'],
                          time_slot_num=ScheduleParameters.schedule_parameters['time_slot_num'],
                          max_clients_per_training=ScheduleParameters.schedule_parameters['max_clients_per_training'],
                          ticket_cost=ScheduleParameters.schedule_parameters['ticket_cost'],
                          hour_pay=ScheduleParameters.schedule_parameters['hour_pay'],
                          pay_for_presence=ScheduleParameters.schedule_parameters['pay_for_presence'],
                          class_renting_cost=ScheduleParameters.schedule_parameters['class_renting_cost'],
                          use_penalty_method=self.parameters['use_penalty_method'],
                          penalty_for_repeated=self.parameters['penalty_for_repeated'],
                          penalty_for_unmatched=self.parameters['penalty_for_unmatched'])

            schedule_global.generate_random_schedule(greedy=self.parameters['greedy'])
            Optimize.clients_num = pd.read_csv(client_file, sep=";").shape[0]
            Optimize.instructors_num = pd.read_csv(instructor_file, sep=";").shape[0]
            initial_solution = copy.deepcopy(schedule_global)

    def start_optimization(self):
        global schedule_global
        self.generate_initial_solution()
        SM = schedule_global
        print("\nINITIAL SCHEDULE")
        print(self.parameters['use_penalty_method'], self.parameters['neighborhood_type_lst'])
        print(SM)
        print('Initial earnings: ', SM.get_cost())
        first_cost = SM.get_cost()
        Optimize.first_cost = first_cost

        tic = time.time()

        best_cost, num_of_iter, all_costs = SM.simulated_annealing(alpha=self.parameters['alpha'],
                                                                   initial_temp=self.parameters['initial_temp'],
                                                                   n_iter_one_temp=self.parameters['n_iter_one_temp'],
                                                                   min_temp=self.parameters['min_temp'],
                                                                   epsilon=self.parameters['epsilon'],
                                                                   n_iter_without_improvement=self.parameters['n_iter_without_improvement'],
                                                                   initial_solution=True,
                                                                   neighborhood_type_lst=self.parameters['neighborhood_type_lst'])

        Optimize.all_costs = all_costs
        Optimize.num_of_iter = num_of_iter
        toc = time.time()

        Optimize.alg_time = toc - tic

        print("\nAFTER OPTIMIZATION")
        print(SM)
        print("Number of iterations: ", num_of_iter)

        print("Best earnings: ", best_cost)
        second_cost = best_cost
        Optimize.second_cost = second_cost
        print("Time: ", toc - tic)

        if self.parameters['improve_results']:
            SM.improve_results()
            print("\nIMPROVED SCHEDULE")
            print(SM)
            print("Best improved earnings: ", SM.get_cost())

            third_cost = SM.get_cost()
            print(f'{first_cost} $ --> {second_cost} $ --> {third_cost} $')
            Optimize.third_cost = third_cost
        else:
            print(f'{first_cost} $ --> {second_cost} $')



class ScheduleOptions(Screen):
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


class ScheduleParameters(Screen):
    schedule_parameters = {
        'class_num': 1,
        'day_num': 6,
        'time_slot_num': 6,
        'max_clients_per_training': 5,
        'ticket_cost': 40,
        'hour_pay': 50,
        'pay_for_presence': 50,
        'class_renting_cost': 200}

    def on_text(self, parameter, input_parameter):
        try:
            ScheduleParameters.schedule_parameters[parameter] = int(input_parameter)
        except ValueError:
            ScheduleParameters.schedule_parameters[parameter] = 0
        global schedule_global

        if schedule_global is not None:
            setattr(schedule_global, parameter, ScheduleParameters.schedule_parameters[parameter])
            setattr(initial_solution, parameter, ScheduleParameters.schedule_parameters[parameter])

        print(ScheduleParameters.schedule_parameters['ticket_cost'])

    def update_text_input_fields(self):
        global schedule_global
        for parameter in ScheduleParameters.schedule_parameters:
            self.ids[parameter].text = str(getattr(schedule_global, parameter))

class AboutOrganizer(Screen):
    def github_button_on(self):
        self.ids.github_button_img.source = 'images/GitHub-Mark-Light-120px-plus_pressed.png'

    def github_button_off(self):
        self.ids.github_button_img.source = 'images/GitHub-Mark-Light-120px-plus.png'
        webbrowser.open('https://github.com/kmotyka00/ScheduleOptimizationProblem')

# Functions to enable assignment
# self.font_size: self.width / 8
def set_font_size(font_size_divider, *args, **kwargs):
    def wrap(instance, value, *args, **kwargs):
        instance.font_size = value / font_size_divider
    return wrap

# self.text_size: self.size
def set_text_size(instance, value):
    instance.text_size = value

class SeeSchedule(Screen):
    classroom_displayed = 0

    def chceck_content(self, instance):
        # Lesson Content Popup
        if instance.ids['lesson'] is not None:
            participants = '\n'
            for participant in instance.ids['lesson'].participants:
                participants += str(participant) + '\n'
            lesson_content = f"Instructor ID: \n{str(instance.ids['lesson'].instructor)}" \
                             f"\n\nParticipants: {participants}"
        else:
            lesson_content = "That's a FREE REAL ESTATE"
        lesson_content_popup = Popup(title=f'Lesson information')
        popup_content = BoxLayout(orientation="vertical")
        label = Label(text=lesson_content, halign='center', valign='center')
        # self.font_size: self.width / 8
        label.bind(width=set_font_size(50))
        # self.text_size: self.size
        label.bind(size=set_text_size)
        popup_content.add_widget(label)
        popup_content.add_widget(Button(text='Close', size_hint=(1, 0.2), on_press=lesson_content_popup.dismiss))
        lesson_content_popup.content = popup_content
        return lesson_content_popup.open()

    def generate_see_schedule_layout(self):
        # order is important, because it affects order of elements in GUI
        if 'hours_layout' not in self.ids.bottom_part.ids:
            self.genrate_hours_layout()
        if 'schedule_layout' not in self.ids.bottom_part.ids:
            self.generate_schedule_layout()
        if 'days_layout' not in self.ids.top_bar.ids:
            self.genrate_days_layout()

    def generate_schedule_layout(self):
        schedule_layout = GridLayout(cols=ScheduleParameters.schedule_parameters['day_num'], size_hint=(0.86, 1))
        for time_slot in range(ScheduleParameters.schedule_parameters['day_num']
                               * ScheduleParameters.schedule_parameters['time_slot_num']):
            if f'Button{time_slot}' not in self.ids.keys():
                button = Button(text=f'{None}', halign='center', valign='center')
                button.background_normal = ''
                if time_slot % 2 == 0:
                    button.background_color = [.8, .8, .9, .6]
                else:
                    button.background_color = [.8, .8, .9, .8]
                # self.font_size: self.width / 8
                button.bind(width=set_font_size(8))
                # self.text_size: self.size
                button.bind(size=set_text_size)
                button.bind(on_press=self.chceck_content)
                button.ids['lesson'] = None
                button.ids['lesson_time_slot'] = None
                schedule_layout.add_widget(button)
                self.ids[f'Button{time_slot}'] = button
        self.ids.bottom_part.ids[f'schedule_layout'] = schedule_layout
        self.ids.bottom_part.add_widget(schedule_layout)

    def genrate_hours_layout(self):
        hours_layout = GridLayout(rows=ScheduleParameters.schedule_parameters['time_slot_num'], size_hint=(0.14, 1))
        for time_slot in range(ScheduleParameters.schedule_parameters['time_slot_num']):
            if f'Button_hours{time_slot}' not in self.ids.keys():
                button = Button(text=f'{time_slot}', halign='center', valign='center')
                button.background_normal = ''
                button.background_color = [.8, .8, .9, .2]
                # self.font_size: self.width / 8
                button.bind(width=set_font_size(8))
                # self.text_size: self.size
                button.bind(size=set_text_size)
                hours_layout.add_widget(button)
                self.ids[f'Button_hours{time_slot}'] = button
        self.ids.bottom_part.ids[f'hours_layout'] = hours_layout
        self.ids.bottom_part.add_widget(hours_layout)

    def genrate_days_layout(self):
        days = ['Monday', 'Tusday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        #                                                                              +1 is for displayed class button
        days_layout = GridLayout(cols=ScheduleParameters.schedule_parameters['day_num']+1, size_hint=(0.86, 1))
        for day in range(ScheduleParameters.schedule_parameters['day_num']):
            if f'Button_days{day}' not in self.ids.keys():
                button = Button(text=f'{days[day]}', halign='center', valign='center')
                button.background_normal = ''
                if day % 2 == 0:
                    button.background_color = [.8, .8, .9, .2]
                else:
                    button.background_color = [.8, .8, .9, .3]
                # self.font_size: self.width / 8
                button.bind(width=set_font_size(8))
                # self.text_size: self.size
                button.bind(size=set_text_size)
                days_layout.add_widget(button)
                self.ids[f'Button_days{day}'] = button
        self.ids.top_bar.ids[f'days_layout'] = days_layout
        self.ids.top_bar.add_widget(days_layout)

    # Error Popup
    my_error_popup = Popup(title="Error", size_hint=(None, None), size=(300, 200), auto_dismiss=False)
    error_popup_content = BoxLayout(orientation="vertical")
    error_popup_content.add_widget(
        Label(size=(cm(6), cm(4)), pos_hint={'top': 1}, text_size=(cm(6), cm(4)), font_size='20sp',
              text='AttributeError: Error during updating schedule - schedule is empty.',
              halign='center', valign='center'))
    error_popup_content.add_widget(Button(text='Close', size_hint=(1, 0.2), on_press=my_error_popup.dismiss))
    my_error_popup.content = error_popup_content

    def update_schedule_description(self):
        try:
            global schedule_global
            # reshape and transpose schedule for convenient indexing
            temp_schedule = schedule_global.schedule[SeeSchedule.classroom_displayed].T.reshape(-1, 1, 1).squeeze()
            for time_slot in range(ScheduleParameters.schedule_parameters['day_num']
                               * ScheduleParameters.schedule_parameters['time_slot_num']):
                lesson = temp_schedule[time_slot]
                if lesson != None:
                    # Read lesson_type and convert it to user friendly string
                    new_text = f'{lesson.lesson_type}'.split('.')[1].split('_')
                    converted_text = str()
                    for i in range(len(new_text)):
                        converted_text += new_text[i] + ' '
                    converted_text += f" [{lesson.instructor.id}]"
                    # make button kind of a parent for lesson
                    self.ids[f'Button{time_slot}'].ids['lesson'] = lesson
                    self.ids[f'Button{time_slot}'].ids['lesson_time_slot'] = \
                        time_slot
                    self.ids[f'Button{time_slot}'].text = converted_text
                else:
                    self.ids[f'Button{time_slot}'].ids['lesson'] = lesson
                    self.ids[f'Button{time_slot}'].ids['lesson_time_slot'] = \
                        time_slot
                    self.ids[f'Button{time_slot}'].text = 'Free'
        except AttributeError:
            SeeSchedule.my_error_popup.open()

    def change_displayed_class_button_on_press(self, instance):
        SeeSchedule.classroom_displayed = instance.ids['classroom_id']
        self.ids['change_class_button'].text = f"Displayed classroom id: {instance.ids['classroom_id']}"
        self.update_schedule_description()


    # Pick Classroom Popup
    def pick_classroom_popup(self):
        pick_classroom_popup = Popup(title="Pick Classroom", size_hint=(0.6, 0.6), auto_dismiss=False)
        pick_classroom_popup_main_layout = GridLayout(cols=1, size_hint=(1, 1))
        pick_classroom_popup_classroom_buttons = GridLayout(cols=3, size_hint=(1, 0.8))
        for classroom in range(ScheduleParameters.schedule_parameters['class_num']):
            button = ToggleButton(group='classroom_displayed', text=f'Classroom:{classroom}', halign='center', valign='center')
            button.ids['classroom_id'] = classroom
            # self.font_size: self.width / 8
            button.bind(width=set_font_size(12))
            # self.text_size: self.size
            button.bind(size=set_text_size)
            button.bind(on_press=self.change_displayed_class_button_on_press)
            pick_classroom_popup_classroom_buttons.add_widget(button)

        pick_classroom_popup_main_layout.add_widget(pick_classroom_popup_classroom_buttons)
        pick_classroom_popup_main_layout.add_widget(Button(text='Close', size_hint=(1, 0.2),
                                                       on_press=pick_classroom_popup.dismiss))

        pick_classroom_popup.content = pick_classroom_popup_main_layout
        pick_classroom_popup.open()


class GoalFunction(Screen):
    box = None

    def draw_plot(self):
        box = self.ids.box
        box.clear_widgets()
        plt.clf()
        plt.plot(Optimize.all_costs)
        plt.title('Goal function over number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('Earnings [$]')
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def reset_solution(self):
        global initial_solution
        global schedule_global
        schedule_global = copy.deepcopy(initial_solution)
        if initial_solution is not None:
            ScheduleParameters.schedule_parameters = {'class_num': initial_solution.class_num,
                                                      'day_num': initial_solution.day_num,
                                                      'time_slot_num': initial_solution.time_slot_num,
                                                      'max_clients_per_training': initial_solution.max_clients_per_training,
                                                      'ticket_cost': initial_solution.ticket_cost,
                                                      'hour_pay': initial_solution.hour_pay,
                                                      'pay_for_presence': initial_solution.pay_for_presence,
                                                      'class_renting_cost': initial_solution.class_renting_cost}
        ScheduleParameters.update_text_input_fields(self.manager.get_screen('schedule_parameters'))

class SeeAlgorithmParameters(Screen):
    pass

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file('new_window.kv')

class ScheduleOrganizer(App):
    def build(self):
        return kv

if __name__ == '__main__':
    ScheduleOrganizer().run()

