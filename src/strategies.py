import abc

class Strategy(metaclass=abc.ABCMeta):
    
   @abc.abstractmethod
   def next_pos(self, p_pos, p_vel, p_dir_up):
       pass 
       
class DumbStrat(Strategy):
    
    def next_pos(self, p_pos, p_vel, p_dir_up):
        pass

class ManualStrat(Strategy):
    inputMap = [False, False] 

    def next_pos(self, p_pos, p_vel, p_dir_up):
        if p_dir_up is None:
            return p_pos
        elif p_dir_up == True:
            return p_pos+p_vel
        else:
            return p_pos-p_vel

class ReinforcedStrat(Strategy):
    
    def next_pos(self, p_pos, p_vel, p_dir_up):
        pass