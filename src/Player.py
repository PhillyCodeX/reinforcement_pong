class Player(object):
    def __init__(self, p_name, p_strategy, p_paddle):
        self.__name = p_name
        self.__strategy = p_strategy
        self.__paddle = p_paddle
    
    def __setname(self, p_name):
        self.__name = p_name
    
    def __getname(self):
        return self.__name 
    
    def __setstrategy(self, p_strategy):
        self.__strategy = p_strategy
    
    def __getstrategy(self):
        return self.__strategy

    def __setpaddle(self, p_paddle):
        self.__paddle = p_paddle

    def __getpaddle(self):
        return self.__paddle 

    def next_pos(self):
        cur_pos = self.__paddle.y_pos
        self.__strategy.next_pos(cur_pos)
        print("Player moved")