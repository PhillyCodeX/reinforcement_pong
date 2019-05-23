from src.strategies import Strategy, ManualStrat, DumbStrat, ReinforcedStrat

class Paddle(object):
    def __init__(self, p_y_pos, p_x_pos, p_length = 100):
        self.__length = p_length
        self.__y_pos = p_y_pos
        self.__x_pos = p_x_pos
        self.__up_moveable = True
        self.__down_moveable = True
        self.__velocity = 10

    def __getlength(self):
        return self.__length

    def __setlength(self, p_length):
        self.__length = p_length

    def __gety_pos(self):
        return self.__y_pos

    def __sety_pos(self, p_y_pos):
        self.__y_pos = p_y_pos

    def __getx_pos(self):
        return self.__x_pos

    def __setx_pos(self, p_x_pos):
        self.__x_pos = p_x_pos

    def __getup_moveable(self):
        return self.__up_moveable 
    
    def __setup_moveable(self, p_bool):
        self.__up_moveable = p_bool

    def __getdown_moveable(self):
        return self.__down_moveable 
    
    def __setdown_moveable(self, p_bool):
        self.__down_moveable = p_bool
    
    def __getvelocity(self):
        return self.__velocity 

    def __setvelocity(self, p_vel):
        self.__velocity = p_vel

    length = property(__getlength, __setlength)
    y_pos = property(__gety_pos, __sety_pos)
    x_pos = property(__getx_pos, __setx_pos)
    up_moveable = property(__getup_moveable, __setup_moveable)
    down_moveable = property(__getdown_moveable, __setdown_moveable)
    velocity = property(__getvelocity,__setvelocity)

class Ball(object):
    def __init__(self, p_x_pos, p_y_pos, p_x_dir, p_y_dir, p_velocity=10):
        self.__velocity = p_velocity
        self.__x_pos = p_x_pos
        self.__y_pos = p_y_pos
        self.__x_dir = p_x_dir
        self.__y_dir = p_y_dir

    def __getvelocity(self):
        return self.__velocity

    def __setvelocity(self, p_velocity):
        self.__velocity = p_velocity

    def __getx_pos(self):
        return self.__x_pos

    def __setx_pos(self, p_x_pos):
        self.__x_pos = p_x_pos

    def __gety_pos(self):
        return self.__y_pos

    def __sety_pos(self, p_y_pos):
        self.__y_pos = p_y_pos

    def __getx_dir(self):
        return self.__x_dir

    def __setx_dir(self, p_x_dir):
        self.__x_dir = p_x_dir

    def __gety_dir(self):
        return self.__y_dir

    def __sety_dir(self, p_y_dir):
        self.__y_dir = p_y_dir

    velocity = property(__getvelocity, __setvelocity)
    x_pos = property(__getx_pos, __setx_pos)
    y_pos = property(__gety_pos, __sety_pos)
    x_dir = property(__getx_dir, __setx_dir)
    y_dir = property(__gety_dir, __sety_dir)

class Area(object):
    def __init__(self, p_height=300, p_width=300):
        self.__height = p_height
        self.__width = p_width

        y_middle = p_height / 2
        x_middle = p_width / 2

        self.__paddle1 = Paddle(y_middle,10)
        self.__paddle2 = Paddle(y_middle,p_width-10)
        self.__ball = Ball(x_middle,y_middle,x_middle+10,y_middle+5)

    def __getheight(self):
        return self.__height
    
    def __setheight(self, p_height):
        self.__height = p_height

    def __getwidth(self):
        return self.__width
    
    def __setwidth(self, p_width):
        self.__width = p_width

    def __getpaddle1(self):
        return self.__paddle1 

    def __setpaddle1(self, p_paddle):
        self.__paddle1 = p_paddle  

    def __getpaddle2(self):
        return self.__paddle2 
        
    def __setpaddle2(self, p_paddle):
        self.__paddle2 = p_paddle

    def __getball(self):
        return self.__ball 
    
    def __setball(self, p_ball):
        self.__ball = p_ball
    
    height = property(__getheight, __setheight)
    width = property(__getwidth, __setwidth)
    paddle1 = property(__getpaddle1, __setpaddle1)
    paddle2 = property(__getpaddle2, __setpaddle2)
    ball = property(__getball, __setball)

class Player(object):
    def __init__(self, p_name, p_strategy):
        self.__name = p_name
        self.__strategy = p_strategy
        self.__paddle = None
        self.__points = 0
    
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

    def __setpoints(self, p_points):
        self.__points = p_points
    
    def __getpoints(self):
        return self.__points 

    def next_pos(self, p_event):
        cur_pos = self.__paddle.y_pos
        self.__strategy.next_pos(cur_pos, p_event)
        print("Player moved")

    name = property(__getname, __setname)
    strategy = property(__getstrategy, __setstrategy)
    paddle = property(__getpaddle, __setpaddle)
    points = property(__getpoints, __setpoints)

class Game(object):
    def __init__(self):
        self.__area = Area()
        self.__player1 = None
        self.__player2 = None

    def newPlayer(self):
        name = input("Enter your name: ")
        
        chosen_strategy = input("What's your strategy?[manual|dumb|rl] :")

        if chosen_strategy == 'manual':
            strategy = ManualStrat()
        elif chosen_strategy == 'dumb':
            strategy = DumbStrat()
        elif chosen_strategy == 'rl':
            strategy = ReinforcedStrat()
        
        player = Player(name,strategy)
        
        return player

    def setPlayers(self):
        self.player1 = self.newPlayer()
        self.player1.paddle = self.__area.paddle1

        self.player2 = self.newPlayer()
        self.player2.paddle = self.__area.paddle2 

    def play(self):
        return ""

    def __setarea(self, p_area):
        self.__area = p_area

    def __getarea(self):
        return self.__area

    area = property(__getarea, __setarea)