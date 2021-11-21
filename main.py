import numpy as np
import pandas as pd
from typing import List
from enum import Enum
import copy
import random
import cv2

class LessonType(Enum):
    """ Class enumerating lesson types,
    enumeration numbers are important,
    because they are used in a questionnaire for customers """
    CELLULITE_KILLER = 0
    ZUMBA = 1
    ZUMBA_ADVANCED = 2
    FITNESS = 3
    CROSSFIT = 4
    BRAZILIAN_BUTT = 5
    PILATES = 6
    CITY_PUMP = 7
    STRETCHING = 8
    YOGA = 9


class Client:
    def __init__(self, id, selected_training: List[LessonType] = None):
        self.id = id
        if selected_training is None:
            self.selected_training = np.array(list())
        else:
            self.selected_training = np.array(selected_training)

    def __str__(self) -> str:
        return f"id: {self.id}, selected_training: {self.selected_training}"


class Instructor:
    def __init__(self, id, qualifications: List[LessonType] = None):
        self.id = id
        if qualifications is None:
            self.qualifications = np.array(list())
        else:
            self.qualifications = np.array(qualifications)


class Lesson:
    def __init__(self, instructor: Instructor, lesson_type: LessonType, participants: List[Client] = None):
        self.instructor = instructor
        self.lesson_type = lesson_type
        if participants is None:
            self.participants = np.array(list())
        else:
            self.participants = np.array(participants)

    def __str__(self):
        lesson_type = str(self.lesson_type)
        lt = lesson_type.split(sep='.')
        lesson_type = lt[1]
        return f"Instructor Id: {self.instructor.id}, Type: {lesson_type}"


class Schedule:
    def __init__(self, client_file: str = 'form_answers.csv', instructor_file: str = 'instructors_info.csv',
                 class_num=1, day_num=6, time_slot_num=6, max_clients_per_training=5,
                 ticket_cost=40, hour_pay=50, pay_for_presence=50, class_renting_cost=200):

        self.class_id = np.arange(class_num)
        self.day = np.arange(day_num)  # monday - saturday
        self.time_slot = np.arange(time_slot_num)
        self.max_clients_per_training = max_clients_per_training

        self.clients = list()
        df = pd.read_csv(client_file, sep=";")
        for index, client in df.iterrows():
            self.clients.append(Client(client['Client_ID'],
                                       [LessonType(int(elem)) for elem in client['Lesson_Types'].split(sep=" ")]))
        self.clients = np.array(self.clients)

        self.instructors = list()
        df = pd.read_csv(instructor_file, sep=";")
        for index, instructor in df.iterrows():
            self.instructors.append(Instructor(instructor['Instructor_ID'],
                                               [LessonType(int(elem)) for elem in
                                                instructor['Lesson_Types'].split(sep=" ")]))
        self.instructors = np.array(self.instructors)
        self.schedule = np.array([None for x in
                                  range(self.class_id.shape[0] * self.day.shape[0] * self.time_slot.shape[0])])
        self.schedule = self.schedule.reshape((self.class_id.shape[0], self.day.shape[0], self.time_slot.shape[0]))

        # economy
        self.ticket_cost = ticket_cost
        self.hour_pay = hour_pay
        self.pay_for_presence = pay_for_presence
        self.class_renting_cost = class_renting_cost

    def generate_random_schedule(self, greedy=False):

        self.schedule = self.schedule.reshape((-1, 1))  # temp reshaping for easier indexing
        i = 0
        free_ts = list(np.arange(self.schedule.shape[0]))
        for lesson_type in LessonType:
            all_participants = [client for client in self.clients if lesson_type in client.selected_training]
            num_of_trainings = int(np.ceil(len(all_participants) / self.max_clients_per_training))
            for training in range(num_of_trainings):
                participants = all_participants[training * self.max_clients_per_training:
                                                training * self.max_clients_per_training + self.max_clients_per_training]

                if greedy:
                    instructor = self.instructors[i % self.instructors.shape[0]]
                    self.schedule[i] = Lesson(instructor, lesson_type, participants)
                    i += 1

                else:
                    random_id = free_ts.pop(random.randint(0, len(free_ts) - 1))
                    instructor = self.instructors[random_id % self.instructors.shape[0]]
                    self.schedule[random_id] = Lesson(instructor, lesson_type, participants)

        self.schedule = self.schedule.reshape((self.class_id.shape[0], self.day.shape[0], self.time_slot.shape[0]))

    def get_cost(self, current_solution=None):
        # dla każdej komórki w current_solution policzyć liczbę uczestników,
        # pomnożyć przez cenę zajęć,
        # odjąć koszt przyjścia trenera w danym dniu
        # i odjąć koszt wynajęcia sali w danym dniu

        if current_solution is None:
            current_solution = self.schedule

        participants_sum = 0
        instructors_hours = np.zeros(shape=(self.instructors.shape[0], self.day.shape[0]))
        class_per_day = np.zeros(shape=(self.class_id.shape[0], self.day.shape[0]))

        for c in range(current_solution.shape[0]):
            for d in range(current_solution.shape[1]):
                for ts in range(current_solution.shape[2]):
                    if current_solution[c, d, ts] is not None:
                        participants_sum += current_solution[c, d, ts].participants.shape[0]
                        instructors_hours[current_solution[c, d, ts].instructor.id, d] += 1
                        class_per_day[c, d] = 1

        return self.ticket_cost * participants_sum - \
               self.hour_pay * instructors_hours.sum() - \
               self.pay_for_presence * (instructors_hours > 0).sum() - \
               self.class_renting_cost * class_per_day.sum()


    def get_neighbor(self, current_solution):

        not_none_id_list = np.argwhere(current_solution != None)
        random_not_none_id = tuple(random.choice(not_none_id_list))

        none_id_list = np.argwhere(current_solution == None)
        random_none_id = tuple(random.choice(none_id_list))

        current_solution[random_none_id] = current_solution[random_not_none_id]
        current_solution[random_not_none_id] = None
        return current_solution

    def simulated_annealing(self, alpha=0.999, initial_temp=1000, n_iter_one_temp=5, min_temp=0.1, epsilon=0.1,
                            n_iter_without_improvement=1000, initial_solution=False):

        if not initial_solution:
            self.generate_random_schedule()  # self.schedule initialized

        current_solution = copy.deepcopy(self.schedule)
        best_solution = copy.deepcopy(current_solution)

        current_temp = initial_temp

        current_cost = self.get_cost(current_solution)
        best_cost = self.get_cost(best_solution)

        counter = 0
        total_counter = 0

        while current_temp > min_temp and counter < n_iter_without_improvement:
            for j in range(0, n_iter_one_temp):
                total_counter += 1
                neighbor_solution = self.get_neighbor(current_solution)
                neighbor_cost = self.get_cost(neighbor_solution)
                delta = neighbor_cost - current_cost
                if delta >= 0:
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost
                    if current_cost > best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost
                else:
                    s = random.uniform(0, 1)
                    if s < np.exp(delta/current_temp):
                        current_solution = neighbor_solution
                        current_cost = neighbor_cost

            current_temp = alpha * current_temp

            if (abs(current_cost - best_cost)) < epsilon:
                counter += 1
            else:
                counter = 0

        self.schedule = best_solution
        return best_cost, total_counter

    def __str__(self):

        result = str()
        days = ["-- MONDAY --", "-- TUESDAY --", "-- WEDNESDAY --", "-- THURSDAY --", "-- FRIDAY --", "-- SATURDAY --"]
        hour = ["16:00 - 17:00", "17:00 - 18:00", "18:00 - 19:00", "19:00 - 20:00", "20:00 - 21:00", "21:00 - 22:00"]
        for c in range(self.schedule.shape[0]):
            for d in range(self.schedule.shape[1]):
                result += "\n" + days[d] + "\n\n"
                for ts in range(self.schedule.shape[2]):
                    result += hour[ts] + "\t"
                    if self.schedule[c, d, ts] is None:
                        result += "Free\n"
                    else:
                        result += str(self.schedule[c, d, ts]) + "\n"

        return result


SM = Schedule(max_clients_per_training=2)

SM.generate_random_schedule(greedy=False)
print("INITIAL SCHEDULE")
print(SM)
print('Initial cost: ', SM.get_cost())

best_cost, num_of_iter = SM.simulated_annealing(alpha=0.9999, initial_temp=1000, n_iter_one_temp=10, min_temp=0.1,
                                                epsilon=0, n_iter_without_improvement=1000, initial_solution=True)

print("AFTER OPTIMIZATION")
print(SM)
print("Number of iterations: ", num_of_iter)
print("Best cost", best_cost)

print("\nEssa")

#  TODO: - poprawienie rozwiązania podczas działania algorytmu SA (przenoszenie względem prowadzących)
#  TODO: - dodanie listy kompetencji i mniej losowe przydzielanie prowadzących do zajęć (może jako prawdopodobieństwo)
#  TODO: - ograniczenia
#  TODO: - ogarnąć co już zaimplemetnowaliśmy (jakie ograniczenia już są)
