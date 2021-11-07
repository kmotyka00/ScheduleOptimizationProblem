import numpy as np
import pandas as pd
from typing import List
from enum import Enum


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
            self.selected_training = np.array(list())
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
        return f"I: {self.instructor.id}, L: {self.lesson_type}"


class Schedule:
    def __init__(self, client_file: str = 'form_answers.csv', instructor_file: str = 'instructors_info.csv',
                 class_num=1, day_num=6, time_slot_num=6, max_clients_per_training=5):
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
                                        [LessonType(int(elem)) for elem in instructor['Lesson_Types'].split(sep=" ")]))
        self.instructors = np.array(self.instructors)
        self.schedule = np.array([None for x in
                                  range(self.class_id.shape[0] * self.day.shape[0] * self.time_slot.shape[0])])
        self.schedule = self.schedule.reshape((self.class_id.shape[0], self.day.shape[0], self.time_slot.shape[0]))


    def generate_random_schedule(self):
        # picked = [lesson_type for client in self.clients for lesson_type in client.selected_training]
        # type_and_count = {}
        # for lesson_type in LessonType:
        #     type_and_count[lesson_type] = picked.count(lesson_type)

        # temp reshaping for easier indexing
        self.schedule = self.schedule.reshape((-1, 1))
        i = 0
        for lesson_type in LessonType:
            all_participants = [client for client in self.clients if lesson_type in client.selected_training]
            num_of_trainings = int(np.ceil(len(all_participants) / self.max_clients_per_training))
            for training in range(num_of_trainings):
                instructor = self.instructors[i % self.instructors.shape[0]]
                participants = all_participants[training*self.max_clients_per_training:
                                                training*self.max_clients_per_training+self.max_clients_per_training]
                self.schedule[i] = Lesson(instructor, lesson_type, participants)
                i += 1
        self.schedule = self.schedule.reshape((self.class_id.shape[0], self.day.shape[0], self.time_slot.shape[0]))
        i = 0

    def goal_function(self):
        # dla każdej komórki w self.schedule policzyć liczbę uczestników,
        # pomnożyć przez cenę zajęć,
        # odjąć koszt przyjścia trenera w danym dniu
        # i odjąć koszt wynajęcia sali w danym dniu
        participants_sum = 0
        instructors_per_day = np.zeros(shape=(self.instructors.shape[0], self.day.shape[0]))
        class_per_day = np.zeros(shape=(self.class_id.shape[0], self.day.shape[0]))

        for c in range(self.schedule.shape[0]):
            for d in range(self.schedule.shape[1]):
                for ts in range(self.schedule.shape[2]):
                    if self.schedule[c, d, ts] is not None:
                        participants_sum += self.schedule[c, d, ts].participants.shape[0]
                        instructors_per_day[self.schedule[c, d, ts].instructor.id, d] = 1
                        class_per_day[c, d] = 1

        return 20 * participants_sum - 50 * instructors_per_day.sum() - 200 * class_per_day.sum()

    def __str__(self):
        return str(self.schedule)


SM = Schedule()
SM.generate_random_schedule()
print(SM.goal_function())
print("Essa")

