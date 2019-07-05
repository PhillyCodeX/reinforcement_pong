import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
        TODO Dokumentieren
    """

    def __init__(self, h, w, outputs):
        """
            TODO Dokumentieren
        """

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2).cuda()
        self.bn1 = nn.BatchNorm2d(16).cuda()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2).cuda()
        self.bn2 = nn.BatchNorm2d(32).cuda()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2).cuda()
        self.bn3 = nn.BatchNorm2d(32).cuda()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            """
                TODO Dokumentieren
            """

            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs).cuda()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """
            TODO Dokumentieren
        """

        x = F.relu(self.bn1(self.conv1(x.cuda())))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class ReplayMemory:
    """
        TODO Dokumentieren
    """
    
    def __init__(self, p_cap=6):
        """
            TODO Dokumentieren
        """

        self.__capacity = p_cap
        self.__memory = list()

    def enqueue(self, p_new_state):
        """
            TODO Dokumentieren
        """

        self.__memory.append(p_new_state)

        if len(self.__memory)>self.__capacity:
            self.dequeue()

    def dequeue(self):
        """
            TODO Dokumentieren
        """

        self.__memory.pop(0)

    def __getmemory(self):
        """
            TODO Dokumentieren
        """

        return self.__memory
        
    memory = property(__getmemory)

class Experience:
    """
        TODO Dokumentieren
    """
    
    def __init__(self):
        """
            TODO Dokumentieren
        """

        self.__s = None
        self.__a = None
        self.__r_1 = None
        self.__s_1 = None

    def __gets(self):
        """
            TODO Dokumentieren
        """

        return self.__s 
    
    def __sets(self, p_s):
        """
            TODO Dokumentieren
        """

        self.__s = p_s

    def __geta(self):
        """
            TODO Dokumentieren
        """

        return self.__a 
    
    def __seta(self, p_a):
        """
            TODO Dokumentieren
        """
        
        self.__a = p_a

    def __getr_1(self):
        """
            TODO Dokumentieren
        """

        return self.__r_1 
    
    def __setr_1(self, p_r_1):
        """
            TODO Dokumentieren
        """

        self.__r_1 = p_r_1

    def __gets_1(self):
        """
            TODO Dokumentieren
        """

        return self.__s_1 
    
    def __sets_1(self, p_s_1):
        """
            TODO Dokumentieren
        """
        
        self.__s_1 = p_s_1

    s = property(__gets, __sets)
    a = property(__geta, __seta)
    r_1 = property(__getr_1, __setr_1)
    s_1 = property(__gets_1, __sets_1)