import numpy as np
import pandas as pd
from typing import List
from enum import Enum
import copy
import random
import time
import matplotlib.pyplot as plt


class LessonType(Enum):
    """ 
    Class enumerating lesson types,
    enumeration numbers are important,
    because they are used in a questionnaire for customers 
    """
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
    """
    A class used to represent a Client

    ...
    Atributes
    ---------
    id : int
        a number which represents the client 
    selected_training: List[LessonType], default=None
        list of the trainings selected by a client

    Methods
    -------
    __str__()
        Helps to print information about the client prettier and cleaner
    """

    def __init__(self, id: int, selected_training: List[LessonType] = None):
        self.id = id

        # checks if list of selected trainings was given, if not, creates empty array
        if selected_training is None:
            self.selected_training = np.array(list())
        else:
            self.selected_training = np.array(selected_training)

    def __str__(self) -> str:
        """
        A method which helps to print information about the client

        ...
        Returns
        -------
        str
            string to be printed
        """
        return f"id: {self.id}, selected_training: {self.selected_training}"


class Instructor:
    """
    A class used to represent an Instructor

    ...
    Atributes
    ---------
    id : int
        a number which represents the instructor 
    qualifications: List[LessonType], default=None
        list of the trainings which can be tought by the instructor

    Methods
    -------
    __str__()
        Helps to print information about the instructor prettier and cleaner
    """

    def __init__(self, id, qualifications: List[LessonType] = None):
        self.id = id

        # checks if qualifications list was given, if not, creates empty array
        if qualifications is None:
            self.qualifications = np.array(list())
        else:
            self.qualifications = np.array(qualifications)

    def __str__(self) -> str:
        """
        A method which helps to print information about the instructor

        ...
        Returns
        -------
        str
            string to be printed
        """
        qualification_str = str()
        for elem in self.qualifications:
            temp = str(elem).split('.')[1].split('_')
            converted_text = str()
            for i in range(len(temp)):
                converted_text += temp[i] + ' '
            qualification_str += converted_text + '\n'
        return f"id: {self.id}, qualifications: {qualification_str}"


class Lesson:
    """
    A class used to represent a Lesson

    ...
    Atributes
    ---------
    instructor : Instructor
        an instructor which conducts classes
    lesson_type: LessonType
        type of the conducted classes
    participiants: List[Client], default=None
        a list of clients which take part in the classes

    Methods
    -------
    __str__()
        Helps to print information about the lesson prettier and cleaner
    """

    def __init__(self, instructor: Instructor, lesson_type: LessonType, participants: List[Client] = None):
        self.instructor = instructor
        self.lesson_type = lesson_type

        # checks if participants list was given, if not, creates empty array
        if participants is None:
            self.participants = np.array(list())
        else:
            self.participants = np.array(participants)

    def __str__(self):
        """
        A method which helps to print information about the lesson

        ...
        Returns
        -------
        str
            string to be printed
        """

        # changes "LessonType.Type" representation to "Type" and prints it with instructor id
        lesson_type = str(self.lesson_type)
        lt = lesson_type.split(sep='.')
        lesson_type = lt[1]
        return f"I: {self.instructor.id}, L: {lesson_type}"


class Schedule:
    """
    A class used to represent our Schedule 

    ...
    Atributes
    ---------
    client_file : str, default='form_answers.csv'
        the path to the file which contains information about the trainings selected by
        clients (ids of the clients and selected training stored in a *.csv file)
    instructor_file: str, 'instructors_info.csv'
        the path to the file which contains information about the qualifications and ids
        of the instructors stored in the *.csv file
    class_num: int, default=1
        the number of classrooms in the building
    day_num: int, default=6
        the number of days on which the classes are held
    time_slot_num: int, default=6
        the number of time slots during a day on which the classes are held
    max_clients_per_training: int, default=5
        maximum number of clients which can participate in the classes
    ticket_cost: int, default=40
        cost of a class ticket
    hour_pay: int, default=50
        instructor's hour pay
    pay_for_presence: int, default=50
        amount of money which instructor revieves for coming to the workplace
    class_renting_cost: int, default=200
        cost of renting a class (per day)
    use_penalty_method: bool, default=False
        if True, constrains are not absolute, but penalty function is applied

    Methods
    -------
    generate_random_schedule(greedy=False)
        Based on parameters of our schedule generate random schedule
    get_cost(current_solution=None)
        Calculate cost of classes for a current schedule
    get_neighbor(current_solution)
        Move one lesson from current schedule to different timeslot
    simulated_annealing(self, alpha=0.9999, initial_temp=1000, n_iter_one_temp=50, min_temp=0.1,
                        epsilon=0.01, n_iter_without_improvement=1000, initial_solution=True)
        Simulated Annealing algorithm which optimizes arranging of a schedule
    improve_results()
        Minimizes days of presence for each instructor
    __str__()
        Helps to print a current schedule preety and intuitive
    """

    def __init__(self, client_file: str = './client_data/form_answers.csv',
                 instructor_file: str = './instructor_data/instructors_info.csv',
                 class_num=1, day_num=6, time_slot_num=6, max_clients_per_training=5,
                 ticket_cost=40, hour_pay=50, pay_for_presence=50, class_renting_cost=500,
                 use_penalty_method=False, penalty_for_repeated=250, penalty_for_unmatched=100):
        self.class_num = class_num
        self.day_num = day_num  # monday - saturday
        self.time_slot_num = time_slot_num
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
        self.instructors = np.array(self.instructors)  # lista na początku żeby móc appendować na essie
        self.schedule = np.array([None for x in
                                  range(self.class_num * self.day_num * self.time_slot_num)])
        self.schedule = self.schedule.reshape((self.class_num, self.day_num, self.time_slot_num))

        # economy
        self.ticket_cost = ticket_cost
        self.hour_pay = hour_pay
        self.pay_for_presence = pay_for_presence
        self.class_renting_cost = class_renting_cost

        # penalty function
        self.use_penalty_method = use_penalty_method
        self.penalty_for_repeated = penalty_for_repeated
        self.penalty_for_unmatched = penalty_for_unmatched

    def generate_random_schedule(self, greedy=False):
        """
        A method which generates random schedule with respect to parameters of Schedule class.
        
        ...
        Parameters
        ----------
        greedy: bool, default=False
            Decides about the method of generating random schedule. It greedy=True
            then all classes are stored time slot by time slot in our schedule (all
            classes are held in the row). If greedy=False classes are initialized
            totally randomly.
        """

        # reshaping to a vector [1 x classes * days * time slots] to facilitate iterating 
        self.schedule = self.schedule.reshape((-1, 1, 1)).squeeze()
        i = 0

        # list of free time slots in particular days 
        free_ts = list(np.arange(self.schedule.shape[0]))

        # iterating over lesson types
        for lesson_type in LessonType:

            # list of all participants and instructor who have chosen particular lesson type
            all_participants = [client for client in self.clients if lesson_type in client.selected_training]
            all_instructors = {instructor.id: instructor for instructor in self.instructors if
                               lesson_type in instructor.qualifications}

            # number of trainings which need to be coducted to please all clients
            num_of_trainings = int(np.ceil(len(all_participants) / self.max_clients_per_training))

            # itereting over trainings which need to be conducted
            for training in range(num_of_trainings):

                # dividing all participiants to particular groups
                participants = all_participants[training * self.max_clients_per_training:
                                                training * self.max_clients_per_training + self.max_clients_per_training]

                # choosing a way to allocate lessons in schedule
                if greedy:
                    lesson_id = i
                    i += 1
                else:
                    lesson_id = free_ts.pop(random.randint(0, len(free_ts) - 1))
                    # TODO obsłużyć zbyt dużą liczbę zajęć - np. poprzez dodanie nowej sali

                # interval in vector self.schedule which represents a break between same time slots
                # and days in different classes
                interval = self.day_num * self.time_slot_num

                # iterating over same days and time slots in different classes
                for ts in range(lesson_id % interval, self.schedule.shape[0], interval):

                    # checking if lesson is taking place 
                    if self.schedule[ts] != None:
                        if self.schedule[ts].instructor.id in all_instructors.keys():
                            # removing inaccessible instructors from all_instructors list
                            all_instructors.pop(self.schedule[ts].instructor.id)  # TODO: TEST

                # random choice of instructor from all_instructors list for new lesson
                instructor = random.choice(list(all_instructors.values()))

                # putting new lesson to schedule
                self.schedule[lesson_id] = Lesson(instructor, lesson_type, participants)

        # reshaping self.schedule back to matrix
        self.schedule = self.schedule.reshape((self.class_num, self.day_num, self.time_slot_num))

    def get_cost(self, current_solution=None):
        """
        A method which calculates cost of current solution.
        
        ...
        Parameters
        ----------
        current_solution: np.array, default=None
            If current_solution is None method computes cost for self.schedule,
            if given, method computes cost for given parameter.

        Returns
        -------
        float
            Cost of given solution.
        """
        # initialize optional parameter current_solution
        if current_solution is None:
            current_solution = self.schedule

        participants_sum = 0
        instructors_hours = np.zeros(shape=(self.instructors.shape[0], self.day_num))
        class_per_day = np.zeros(shape=(self.class_num, self.day_num))
        repeated_instructors = 0
        unmatched_instructors = 0

        # for every lesson in solution count number of participants, instructors' hours and classrooms used

        for d in range(current_solution.shape[1]):
            for ts in range(current_solution.shape[2]):
                used_instructors = list()
                for c in range(current_solution.shape[0]):
                    if current_solution[c, d, ts] is not None:
                        if current_solution[c, d, ts].lesson_type not in \
                                current_solution[c, d, ts].instructor.qualifications:
                            unmatched_instructors += 1
                        used_instructors.append(current_solution[c, d, ts].instructor.id)
                        participants_sum += current_solution[c, d, ts].participants.shape[0]
                        instructors_hours[current_solution[c, d, ts].instructor.id, d] += 1
                        class_per_day[c, d] = 1
                repeated_instructors += len(used_instructors) - len(set(used_instructors))

        # count cost function
        cost = self.ticket_cost * participants_sum - \
               self.hour_pay * instructors_hours.sum() - \
               self.pay_for_presence * (instructors_hours > 0).sum() - \
               self.class_renting_cost * class_per_day.sum()
        if self.use_penalty_method:
            cost += unmatched_instructors * self.penalty_for_unmatched + \
                    repeated_instructors * self.penalty_for_repeated
        return cost

    def get_neighbor(self, current_solution, neighborhood_type_lst: List[str]):
        """
        A method which generates new solution which differ from previous one
        only by one classes.
        
        Method randomly chooses one lesson and moves it to different time slot.
        
        ...
        Parameters
        ----------
        current_solution: np.array
            Parameter which contains solution for which we want to find a neighbor.

        neighborhood_type_lst: List[str]
            Parameter specify method of choosing neighborhood

        Returns
        -------
        np.array
            Generated neighbor.
        """
        for neighborhood_type in neighborhood_type_lst:
            if neighborhood_type == 'move_one' or neighborhood_type == 'move_two':
                # reshaping current_solution to make sure it's a matrix
                current_solution = current_solution.reshape((-1, 1, 1))
                if neighborhood_type == 'move_one':
                    i_max = 1
                else:
                    i_max = 2
                for i in range(i_max):
                    # get list of indices where lesson is scheduled
                    # using != None is necessary, because we are interested in
                    # checking if value in array is None, not if array is None
                    not_none_id_list = np.argwhere(current_solution != None)
                    # choose random index from not_none_id_list
                    random_not_none_id = tuple(random.choice(not_none_id_list))

                    # get list of indices where timeslot is free
                    # using == None is necessary, because we are interested in
                    # checking if value in array is None, not if array is None
                    none_id_list = np.argwhere(current_solution == None)
                    random_none_id = tuple(random.choice(none_id_list))

                    # swap values contained in drawn indices
                    current_solution[random_none_id] = current_solution[random_not_none_id]
                    current_solution[random_not_none_id] = None

            if neighborhood_type == 'move_to_most_busy' or neighborhood_type == 'swap_with_most_busy':
                # reshaping current_solution to make sure it's a matrix
                current_solution = current_solution.reshape((self.class_num, self.day_num, self.time_slot_num))

                most_busy = None
                most_trainings = 0
                least_busy = None
                least_trainings = self.time_slot_num
                for c in range(self.schedule.shape[0]):
                    for d in range(self.schedule.shape[1]):
                        trainings_num = np.count_nonzero(current_solution[c, d, :] != None)
                        if neighborhood_type == 'move_to_most_busy':
                            condition = most_trainings <= trainings_num < self.time_slot_num
                        else:
                            condition = most_trainings <= trainings_num

                        if condition:
                            most_trainings = trainings_num
                            most_busy = (c, d)
                        if least_trainings >= trainings_num >= 1:
                            least_trainings = trainings_num
                            least_busy = (c, d)

                c_least, d_least = least_busy
                id_least = random.choice(np.argwhere(current_solution[c_least, d_least, :] != None))
                c_most, d_most = most_busy
                if neighborhood_type == 'move_to_most_busy':
                    id_most = random.choice(np.argwhere(current_solution[c_most, d_most, :] == None))
                    current_solution[c_most, d_most, id_most] = current_solution[c_least, d_least, id_least]
                    current_solution[c_least, d_least, id_least] = None
                elif neighborhood_type == 'swap_with_most_busy':
                    id_most = random.choice(np.argwhere(current_solution[c_most, d_most, :] != None))
                    current_solution[c_most, d_most, id_most], current_solution[c_least, d_least, id_least] \
                        = current_solution[c_least, d_least, id_least], current_solution[c_most, d_most, id_most]

            if neighborhood_type == 'change_instructor':
                current_solution = current_solution.reshape((-1, 1, 1))

                not_none_id_list = np.argwhere(current_solution != None)
                random_not_none_id = tuple(random.choice(not_none_id_list))

                type_of_selected_lesson = current_solution[random_not_none_id].lesson_type

                if self.use_penalty_method:
                    available_instructors = self.instructors
                else:
                    available_instructors = [instructor for instructor in self.instructors
                                             if type_of_selected_lesson in instructor.qualifications and
                                             instructor.id != current_solution[random_not_none_id].instructor.id]

                if len(available_instructors) == 0:
                    new_instructor = current_solution[random_not_none_id].instructor
                else:
                    new_instructor = random.choice(available_instructors)

                current_solution[random_not_none_id].instructor = new_instructor

        current_solution = current_solution.reshape((self.class_num, self.day_num, self.time_slot_num))
        return current_solution

    def simulated_annealing(self, alpha=0.999, initial_temp=100, n_iter_one_temp=50, min_temp=0.1,
                            epsilon=0.01, n_iter_without_improvement=1000, initial_solution=True,
                            neighborhood_type_lst=None, greedy=False):
        """
        Simulated Annealing is a probabilistic technique for approximating the global optimum
        of a given function.
        
        ...
        Parameters
        ----------
        alpha: float, default=0.999
            Parameter between (0, 1). Used to change pace of lowering the temperature.
            When closer to 1 temperature will decrease slower.
        initial_temp: float, default=1000
            Initial temeprature.
        n_iter_one_temp: int, default=50
            Number of iterations with the same value of the temperature.
        min_temp: float, default=0.1
            The temperature witch algorithm seeks to.
        epsilon: float, default=0.01
            Value of the accteptalbe absolute error.
        n_iter_without_improvement: int, default=1000
            Number of iterations needed to be processed without crossing the epsilon to 
            accept the solution.
        initial_solution: bool, default=True
            If initial_solution solution is False algorithm will generate random schedule.
            If True, algortihm will optimize self.schedule.
        neighborhood_type_lst: List[str]
            Parameter specify method of choosing neighborhood
        greedy: bool, defalut=False
            Create initial solution packing trainings next to each other

        Returns
        -------
        best_cost: float
            Best earnings achieved while algorithm was running.
        total_counter: int
            Number of iterations processed when the algorithm was running.

        """
        if neighborhood_type_lst is None:
            neighborhood_type_lst = ['move_one']

        if not initial_solution:
            self.generate_random_schedule(greedy)  # self.schedule initialized

        # copy existing schedule to prevent unwanted changes
        current_solution = copy.deepcopy(self.schedule)
        best_solution = copy.deepcopy(current_solution)

        current_temp = initial_temp

        # count current cost
        current_cost = self.get_cost(current_solution)
        best_cost = self.get_cost(best_solution)

        # counter for n_iter_without_improvement
        counter = 0
        # counter of total number of iterations
        total_counter = 0

        # list containing all costs
        all_costs = list()

        # loop while non of stopping criteria is fulfilled
        while current_temp > min_temp and counter < n_iter_without_improvement:
            for j in range(0, n_iter_one_temp):
                total_counter += 1
                neighbor_solution = self.get_neighbor(current_solution, neighborhood_type_lst)
                neighbor_cost = self.get_cost(neighbor_solution)
                # delta - value to evaluate quality of new solution
                delta = neighbor_cost - current_cost
                # if ne solution is better than current one - take it
                if delta >= 0:
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost
                    if current_cost > best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost
                # else if s is small enough take it anyway
                else:
                    s = random.uniform(0, 1)
                    if s < np.exp(delta / current_temp):
                        current_solution = neighbor_solution
                        current_cost = neighbor_cost

                all_costs.append(current_cost)
            # decrease temperature
            current_temp = alpha * current_temp
            # check if new solution gives different cost - 2nd stopping criteria
            if (abs(current_cost - best_cost)) < epsilon:
                counter += 1
            else:
                counter = 0

        self.schedule = best_solution
        return best_cost, total_counter, all_costs

    def improve_results(self):
        # TODO: dodać wyświtlanie przed i po w GUUI
        """
        Minimizes days of presence for each instructor.
        
        Method which looks for days when instructor teaches only one class and tries
        to move this lesson to a day when instructor provides more classes. 
        
        ...
        Parameters
        ----------

        """
        # loop repeating until there is no change to ensure
        # that instructor 1 can be moved in space freed by instructor 3
        changed = True
        iter = 0
        while changed and iter < 1000:
            iter += 1
            changed = False

            # loop for each instructor
            for instructor in self.instructors:

                # pick all trainings taught by instructor
                # in format
                # [day, list of timeslots of lessons taught by instructor, list of timeslots free in that day, class]
                trainings = list()
                for c in range(self.schedule.shape[0]):
                    for d in range(self.schedule.shape[1]):
                        timeslots = list()
                        free_ts = list()
                        for ts in range(self.schedule.shape[2]):
                            if self.schedule[c, d, ts] != None:
                                if self.schedule[c, d, ts].instructor.id == instructor.id:
                                    timeslots.append(ts)
                            else:
                                free_ts.append(ts)
                        if len(timeslots) > 0:
                            trainings.append([d, timeslots, free_ts, c])

                # sorting trainings by number of lessons taught by instructor in that day
                trainings = [v for v in sorted(trainings, key=lambda item: len(item[1]))]

                # for each day check if trainings can be reassigned
                # to some day with more trainings thought by that instructor
                for i in range(len(trainings) - 1):
                    for j in range(i + 1, len(trainings)):
                        timeslots_taken = list()
                        changed = False
                        REASSIGNMENT_INVALID_FLAG = False
                        # check if lessons from trainings[i] can be reassigned
                        # to the day represented by trainings[j]
                        if len(trainings[i][1]) <= len(trainings[j][2]):
                            for ts_iter in range(len(trainings[i][1])):
                                for c in range(self.schedule.shape[0]):
                                    if self.schedule[c, trainings[j][0], trainings[j][2][ts_iter]] is not None:
                                        if self.schedule[c, trainings[j][0], trainings[j][2][ts_iter]].instructor.id \
                                                == instructor.id:
                                            REASSIGNMENT_INVALID_FLAG = True
                                if REASSIGNMENT_INVALID_FLAG:
                                    break
                                # reassign
                                self.schedule[trainings[j][3], trainings[j][0], trainings[j][2][ts_iter]] = \
                                    self.schedule[trainings[i][3], trainings[i][0], trainings[i][1][ts_iter]]

                                self.schedule[trainings[i][3], trainings[i][0], trainings[i][1][ts_iter]] = None
                                # add to taken sluts in j
                                trainings[j][1].append(trainings[j][2][ts_iter])
                                # add to free slots in i
                                trainings[i][2].append(trainings[i][1][ts_iter])
                                # memorize to delete from free in j
                                timeslots_taken.append(trainings[j][2][ts_iter])

                                # mark that changes have been made
                                changed = True

                            if not REASSIGNMENT_INVALID_FLAG:
                                # delete memorized slots from free in j
                                for ts_iter in timeslots_taken:
                                    trainings[j][2].remove(ts_iter)
                                # clear list of taken slots in i
                                trainings[i][1] = []

                        # if previous statement is untrue check if opposite reassignment can be performed,
                        # i. e. check if lessons from trainings[j] can be reassigned
                        # to the day represented by trainings[i]
                        elif len(trainings[i][2]) > len(trainings[j][1]):
                            for ts_iter in range(len(trainings[j][1])):
                                for c in range(self.schedule.shape[0]):
                                    if self.schedule[c, trainings[i][0], trainings[i][2][ts_iter]] is not None:
                                        if self.schedule[c, trainings[i][0], trainings[i][2][ts_iter]].instructor.id \
                                                    == instructor.id:
                                            REASSIGNMENT_INVALID_FLAG = True
                                if REASSIGNMENT_INVALID_FLAG:
                                    break
                                self.schedule[trainings[i][3], trainings[i][0], trainings[i][2][ts_iter]] = \
                                    self.schedule[trainings[j][3], trainings[j][0], trainings[j][1][ts_iter]]

                                self.schedule[trainings[j][3], trainings[j][0], trainings[j][1][ts_iter]] = None
                                # add to taken slots in i
                                trainings[i][1].append(trainings[i][2][ts_iter])
                                # add to free sluts in j
                                trainings[j][2].append(trainings[j][1][ts_iter])
                                # memorize to delete from free in i
                                timeslots_taken.append(trainings[i][2][ts_iter])

                                # mark that changes have been made
                                changed = True

                            if not REASSIGNMENT_INVALID_FLAG:
                                # delete memorized slots from free in i
                                for ts_iter in timeslots_taken:
                                    trainings[i][2].remove(ts_iter)
                                    # clear list of taken slots in i
                                trainings[j][1] = []

                        # if changes have been made
                        if changed:
                            # sort trainings to keep ascending order of
                            # number of lessons thought by instructor in that day
                            trainings = [v for v in sorted(trainings, key=lambda item: len(item[1]))]
                            # go to next iteration of i - next iterations for that i are unnecessary,
                            # because trainings[i] have been reassigned
                            break

    def __str__(self):
        """
        Helps to print a current schedule preety and intuitive.

        ...
        Returns
        -------
        str
            string to be printed
        """
        result = str()
        days = ["----- MONDAY -----", "----- TUESDAY -----", "----- WEDNESDAY -----", "----- THURSDAY -----",
                "----- FRIDAY -----", "----- SATURDAY -----"]
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

#
# SM = Schedule(client_file='./client_data/form_answers_2.csv',
#                 instructor_file='./instructor_data/instructors_info_2.csv',
#                 max_clients_per_training=5, time_slot_num=6, class_num=2)
# SM.generate_random_schedule(greedy=False)
#
# print("\nINITIAL SCHEDULE")
# print(SM)
# print('Initial earnings: ', SM.get_cost())
# first_cost = SM.get_cost()
# tic = time.time()
# best_cost, num_of_iter, all_costs = SM.simulated_annealing(alpha=0.999, initial_temp=1000, n_iter_one_temp=10,
#                                                            min_temp=0.1,
#                                                            epsilon=0.01, n_iter_without_improvement=10,
#                                                            initial_solution=True, neighborhood_type_lst=['move_one',
#                                                                                                          'change_instructors'])
# toc = time.time()
#
# print("\nAFTER OPTIMIZATION")
# print(SM)
# print("Number of iterations: ", num_of_iter)
#
# print("Best earnings: ", best_cost)
# second_cost = best_cost
# print("Time: ", toc - tic)
#
# SM.improve_results()
# print("\nIMPROVED SCHEDULE")
# print(SM)
# print("Best improved earnings: ", SM.get_cost())
#
# third_cost = SM.get_cost()
#
# print(f'{first_cost} $ --> {second_cost} $ --> {third_cost} $')
#
# plt.figure()
# plt.plot(all_costs)
# plt.title('Goal function over number of iterations')
# plt.xlabel('Number of iterations')
# plt.ylabel('Earnings [$]')
# plt.show()


#  TODO: - dodanie listy kompetencji i mniej losowe przydzielanie prowadzących do zajęć (może jako prawdopodobieństwo)
#  TODO: - ograniczenia - chwiliowo pomijamy ograniczenie 6) i 7)
#  TODO: - z 7) można zrobić tak, że po ułożeniu już planu sprawdzamy dla każdego użytkownika ile razy w tygodniu
#   trenuje i jeśli jego liczba treningów jest większa niż max to przenosimy go do innej grupy. Można założyć na
#   początku działania algorytmu limit np. 12 (zamiast 10) żeby mieć jakieś pole manewru. Takie podejście może
#   okazać się lepsze bo nie utrudnia działania algorytmu a takich przypadków nie powinno być dużo
#  TODO: - dodać dokumentację
