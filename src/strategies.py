import abc
import random

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

    @abc.abstractmethod
    def notify_score(self, p_score):
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

    def notify_score(self, p_score):
        pass

class ManualStrat(Strategy):

    def next_pos(self, p_paddle, p_dir_up):

        if p_dir_up == True and p_paddle.up_moveable:
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        elif p_dir_up == False and p_paddle.down_moveable:
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity

    def notify_score(self, p_score):
        pass

class ReinforcedStrat(Strategy):
    def __init__(self):
        self.__policy_network = None
        self.__target_network = None

        #list of tuples of state, action, reward+1, state+1
        self.__replay_mem = list()
        
        self.__s = None
        self.__a = None
        self.__r_1 = None
        self.__s_1 = None

    def next_pos(self, p_paddle, p_dir_up):
        learning_rate = 0.1
        discount_rate = 0.99

        exploration_rate = 1
        max_exploration_rate = 1
        min_exploration_rate = 0.01
        exploration_decay_rate = 0.001

        exploration_rate_threshold = random.uniform(0,1)
        
        if exploration_rate_threshold > exploration_rate:
            #choose best action
            pass
        else:
            #choose random action
            pass

        up_switch = True

        if up_switch:
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        else:
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity
        
        
    def notify_score(self, p_score):
        self.__r_1 = p_score

    def new_state(self, p_state, p_is_first_state=False):
        if p_is_first_state:
            self.__s = p_state
        else:
            self.__s = self.__s_1
            self.__s = p_state