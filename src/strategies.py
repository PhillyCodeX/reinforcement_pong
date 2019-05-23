import abc

class Strategy(metaclass=abc.ABCMeta):
    
   @abc.abstractmethod
   def next_pos(self, p_pos):
       pass 
       
class DumbStrat(Strategy):
    
    def next_pos(self, pos):
        pass

class ManualStrat(Strategy):
    
    def next_pos(self, pos):
        pass

class ReinforcedStrat(Strategy):
    
    def next_pos(self, pos):
        pass