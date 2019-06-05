import abc
import random
import pickle
import numpy as np
np.seterr(invalid='ignore')

from copy import copy
from src.rl_objects import ReplayMemory, Experience, DQN
from PIL import Image

import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

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

class RandomStrat(Strategy):
    def __init__(self):
        self.__up_switch = True

    def next_pos(self, p_paddle, p_dir_up):

        up_switch = random.randint(0,1)

        if self.__up_switch and p_paddle.up_moveable:
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        elif self.__up_switch == False and p_paddle.down_moveable:
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity


class ManualStrat(Strategy):

    def next_pos(self, p_paddle, p_dir_up):

        if p_dir_up == True and p_paddle.up_moveable:
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        elif p_dir_up == False and p_paddle.down_moveable:
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity

class ReinforcedStrat(Strategy):
    def __init__(self, p_width, p_height, p_name, p_resume=False ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__resize_factor = 40

        if p_width > p_height:
            adapted_width = 4*p_width / p_height * self.__resize_factor 
            adapter_heigth = self.__resize_factor
        else:
            adapted_width = self.__resize_factor
            adapter_heigth = p_height / 4*p_width * self.__resize_factor 

        self.__width = int(adapted_width)
        self.__heigth = int(adapter_heigth)

        if p_resume:
            self.__policy_network = pickle.load(open('models/'+p_name+'_save_pn.p', 'rb'))
            self.__target_network = pickle.load(open('models/'+p_name+'_save_tn.p', 'rb'))
        else:
            self.__policy_network = DQN(self.__heigth, self.__width, 2).to(device).double()
            self.__target_network = DQN(self.__heigth, self.__width, 2).to(device).double()

        self.__target_network.load_state_dict(self.__policy_network.state_dict())
        self.__target_network.eval()

        self.__optimizer = optim.RMSprop(self.__policy_network.parameters())

        self.__steps_done = 0
        self.__TARGET_THRESHOLD = 3000

        #list of tuples of state, action, reward+1, state+1
        self.__replay_mem = ReplayMemory(1000)
        self.__last_exp = Experience()

        self.__exploration_rate = 1
        self.__max_exploration_rate = 1
        self.__min_exploration_rate = 0.01
        self.__exploration_decay_rate = 0.000001

        self.__learning_rate = 1e-3
        self.__discount_rate = 0.999

        self.__avg_loss = 0
        self.__loss_list = np.zeros([1])

        self.__sum_reward = 0
        self.__reward_list = np.zeros([1])

    def safe_model(self, p_name):
        pickle.dump(self.__policy_network, open('models/'+p_name+'_save_pn.p', 'wb'))
        pickle.dump(self.__target_network, open('models/'+p_name+'_save_tn.p', 'wb'))

    def next_pos(self, p_paddle, p_dir_up):
        exploration_rate_threshold = random.uniform(0,1)
        up_switch = True

        if exploration_rate_threshold > self.__exploration_rate and self.__replay_mem.memory:
            with torch.no_grad():
                current_state = self._ReinforcedStrat__replay_mem.memory[-1].s
                up_switch = torch.max(self._ReinforcedStrat__policy_network(current_state),0)[0].argmin().item()
        else:
            up_switch = random.randint(0,1)

        if up_switch and p_paddle.up_moveable:
            self.__last_exp.a = "UP"
            p_paddle.y_pos = p_paddle.y_pos-p_paddle.velocity
        elif up_switch == False and p_paddle.down_moveable:
            self.__last_exp.a = "DOWN"
            p_paddle.y_pos = p_paddle.y_pos+p_paddle.velocity

        if self.__replay_mem.memory:
            self.optimize()    
            self.__steps_done += 1

            self.__exploration_rate -= self.__exploration_decay_rate 

            if self.__steps_done >= self.__TARGET_THRESHOLD:
                self._ReinforcedStrat__target_network.load_state_dict(self._ReinforcedStrat__policy_network.state_dict())
                self.steps_done = 0


    def __update_loss(self, p_loss):
        self.__loss_list = np.append(self.__loss_list, p_loss.item())
        self.__avg_loss = np.median(self.__loss_list)

    
    def reset(self):
        self.__loss_list = list()
        self.__reward_list = list()
        self.__avg_loss = 0
        self.__sum_reward = 0


    def optimize(self):
        experience = random.choice(self.__replay_mem.memory)
        exp_s = experience.s
        exp_s_1 = experience.s_1
        exp_r_1 = experience.r_1

        q_value = self._ReinforcedStrat__policy_network(exp_s).max(1)[0]
        q_value_target = exp_r_1 + self.__discount_rate * self._ReinforcedStrat__target_network(exp_s_1).max(1)[0].detach()

        loss = F.smooth_l1_loss(q_value, q_value_target)
        self.__update_loss(loss)

        self.__optimizer.zero_grad()

        loss.backward()
        for param in self._ReinforcedStrat__policy_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.__optimizer.step()

    def notify_score(self, p_score):
        if p_score == 0:
            self.__last_exp.r_1 = -1
            self.__reward_list = np.append(self.__reward_list, -1)
        else:
            self.__last_exp.r_1 = p_score
            self.__reward_list = np.append(self.__reward_list, p_score)

        self.__sum_reward = np.sum(self.__reward_list)

    def __img_processing(self, p_img_matrix):
        np_img = np.ascontiguousarray(p_img_matrix, dtype=np.float32)
        np_img[(np_img > 0) & (np_img < np.amax(np_img))] = 0
        np_normalized = np_img / np.amax(np_img)
        np_stacked_state = np.vstack((np_normalized[0],np_normalized[1],np_normalized[2],np_normalized[3]))

        processed_state = torch.from_numpy(np_stacked_state)

        resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

        return resize(processed_state).type('torch.DoubleTensor').unsqueeze(1)

    def new_state(self, p_state, p_is_first_state=False):
        
        processed_state = self.__img_processing(p_state)

        if p_is_first_state:
            self.__last_exp.s = processed_state
        else:
            self.__last_exp.s_1 = processed_state

            if self.__last_exp.r_1 == None:
                self.__last_exp.r_1 = 0
            else:
                self.__last_exp.r_1 = copy(self.__last_exp.r_1)

            self.__last_exp.a = copy(self.__last_exp.a)

            self.__replay_mem.enqueue(copy(self.__last_exp))

            #Reset for next State
            self.__last_exp.s = processed_state
            self.__last_exp.s_1 = None
            self.__last_exp.r_1 = None
            self.__last_exp.a = None

    def __getavg_loss(self):
        return self.__avg_loss

    def __getsum_reward(self):
        return self.__sum_reward 

    avg_loss = property(__getavg_loss)
    sum_reward = property(__getsum_reward)
    
class GodStrat(Strategy):

    def next_pos(self, p_paddle, p_dir_up):
        return
