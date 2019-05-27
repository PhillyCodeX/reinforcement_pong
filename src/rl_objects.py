import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class ReplayMemory:
    def __init__(self, p_cap=6):
        self.__capacity = p_cap
        self.__memory = list()

    def enqueue(self, p_new_state):
        self.__memory.append(p_new_state)

        if len(self.__memory)>self.__capacity:
            self.dequeue()

    def dequeue(self):
        self.__memory.pop(0)

    def __getmemory(self):
        return self.__memory
        
    memory = property(__getmemory)

class Experience:
    def __init__(self):
        self.__s = None
        self.__a = None
        self.__r_1 = None
        self.__s_1 = None

    def __gets(self):
        return self.__s 
    
    def __sets(self, p_s):
        self.__s = p_s

    def __geta(self):
        return self.__a 
    
    def __seta(self, p_a):
        self.__a = p_a

    def __getr_1(self):
        return self.__r_1 
    
    def __setr_1(self, p_r_1):
        self.__r_1 = p_r_1

    def __gets_1(self):
        return self.__s_1 
    
    def __sets_1(self, p_s_1):
        self.__s_1 = p_s_1

    s = property(__gets, __sets)
    a = property(__geta, __seta)
    r_1 = property(__getr_1, __setr_1)
    s_1 = property(__gets_1, __sets_1)