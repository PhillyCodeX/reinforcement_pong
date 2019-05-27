import abc
import random
from copy import copy
from src.rl_objects import ReplayMemory, Experience, DQN

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
        self.__replay_mem = ReplayMemory()
        self.__last_exp = Experience()

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
            #TODO Implement Exploitation
            pass
        else:
            up_switch = random.randint(0,1)

        if up_switch:
            self.__last_exp.a = "UP"
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        else:
            self.__last_exp.a = "DOWN"
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity
        
        
    def notify_score(self, p_score):
        if p_score == 1:
            self.__last_exp.r_1 = p_score
        elif p_score == 0:
            self.__last_exp.r_1 = -1

    def new_state(self, p_state, p_is_first_state=False):
        if p_is_first_state:
            self.__last_exp.s = p_state
        else:
            self.__last_exp.s_1 = p_state

            if self.__last_exp.r_1 == None:
                self.__last_exp.r_1 = 0
            else:
                self.__last_exp.r_1 = copy(self.__last_exp.r_1)

            self.__last_exp.a = copy(self.__last_exp.a)

            self.__replay_mem.push(copy(self.__last_exp))

            #Reset for next State
            self.__last_exp.s = p_state
            self.__last_exp.s_1 = None
            self.__last_exp.r_1 = None
            self.__last_exp.a = None