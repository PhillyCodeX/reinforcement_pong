from abc import ABC

class Strategy(ABC):
    
   @abc.abstractmethod
   def next_pos(p_pos):
       pass 