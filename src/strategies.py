import abc
import random
from copy import copy

class Strategy(metaclass=abc.ABCMeta):
    def __init__(self):
       self.__paddle = None

    def __setpaddle(self, p_paddle):
        self.__paddle = p_paddle

    def __getpaddle(self):
        return self.__paddle

    @abc.abstractmethod
    def next_pos(self, p_pos, p_vel, p_dir_up):
       pass 

    def notify_score(self, p_score):
        pass

    def new_state(self, p_state, p_is_first_state=False):
        pass

    paddle = property(__getpaddle,__setpaddle)

class DumbStrat(Strategy):
    def __init__(self):
        self.__up_switch = True

    def next_pos(self, p_paddle, p_dir_up):

        if p_paddle.up_moveable == False:
            self.__up_switch = False
        elif p_paddle.down_moveable == False:
            self.__up_switch = True

        if self.__up_switch:
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        else:
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity

class ManualStrat(Strategy):

    def next_pos(self, p_paddle, p_dir_up):

        if p_dir_up == True and p_paddle.up_moveable:
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        elif p_dir_up == False and p_paddle.down_moveable:
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity

class ReinforcedStrat(Strategy):
    def __init__(self):
        self.__policy_network = None
        self.__target_network = None

        #list of tuples of state, action, reward+1, state+1
        self.__replay_mem = list()
        self.__n_of_mem = 6

        self.__s = None
        self.__a = None
        self.__r_1 = None
        self.__s_1 = None

        self.__exploration_rate = 1
        self.__max_exploration_rate = 1
        self.__min_exploration_rate = 0.01
        self.__exploration_decay_rate = 0.001

        self.__learning_rate = 0.1
        self.__discount_rate = 0.99

    
    def next_pos(self, p_paddle, p_dir_up):
        exploration_rate_threshold = random.uniform(0,1)
        
        if exploration_rate_threshold > self.__exploration_rate:
            #choose best action by running through TN
            pass
        else:
            up_switch = random.randint(0,1)

        if up_switch:
            self.__a = "UP"
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        else:
            self.__a = "DOWN"
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity
        
        
    def notify_score(self, p_score):
        if p_score == 1:
            self.__r_1 = p_score
        elif p_score == 0:
            self.__r_1 = -1

    def new_state(self, p_state, p_is_first_state=False):
        if p_is_first_state:
            self.__s = p_state
        else:
            self.__s_1 = p_state
            s_copy = copy(self.__s)
            a_copy = copy(self.__a)

            if self.__r_1 == None:
                r_1_copy = 0
            else:
                r_1_copy = copy(self.__r_1)
                
            s_1_copy = copy(self.__s_1)

            new_tuple = (s_copy, a_copy, r_1_copy, s_1_copy)

            self.__replay_mem.append(new_tuple)

            if len(self.__replay_mem)>self.__n_of_mem:
                self.__replay_mem.pop(0)

            self.__s = p_state
            self.__s_1 = None