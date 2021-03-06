from src.strategies import Strategy, ManualStrat, DumbStrat, ReinforcedStrat, RandomStrat, GodStrat, FollowTheBallStrat
import numpy as np
import pickle
import threading
import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import pygame 
import pygame.surfarray

class Paddle(object):
    """This class is a plain object representing a Paddle"""

    def __init__(self, p_y_pos, p_x_pos, p_length = 100):
        """ 
            TODO: Comment
        """

        self.__length = p_length
        self.__y_pos = p_y_pos
        self.__x_pos = p_x_pos
        self.__up_moveable = True
        self.__down_moveable = True
        self.__velocity = 20

    def __getlength(self):
        """ 
            TODO: Comment
        """

        return self.__length

    def __setlength(self, p_length):
        """ 
            TODO: Comment
        """

        self.__length = p_length

    def __gety_pos(self):
        """ 
            TODO: Comment
        """

        return self.__y_pos

    def __sety_pos(self, p_y_pos):
        """ 
            TODO: Comment
        """

        self.__y_pos = p_y_pos

    def __getx_pos(self):
        """ 
            TODO: Comment
        """

        return self.__x_pos

    def __setx_pos(self, p_x_pos):
        """ 
            TODO: Comment
        """

        self.__x_pos = p_x_pos

    def __getup_moveable(self):
        """ 
            TODO: Comment
        """

        return self.__up_moveable 
    
    def __setup_moveable(self, p_bool):
        """ 
            TODO: Comment
        """

        self.__up_moveable = p_bool

    def __getdown_moveable(self):
        """ 
            TODO: Comment
        """

        return self.__down_moveable 
    
    def __setdown_moveable(self, p_bool):
        """ 
            TODO: Comment
        """

        self.__down_moveable = p_bool
    
    def __getvelocity(self):
        """ 
            TODO: Comment
        """

        return self.__velocity 

    def __setvelocity(self, p_vel):
        """ 
            TODO: Comment
        """

        self.__velocity = p_vel

    length = property(__getlength, __setlength)
    y_pos = property(__gety_pos, __sety_pos)
    x_pos = property(__getx_pos, __setx_pos)
    up_moveable = property(__getup_moveable, __setup_moveable)
    down_moveable = property(__getdown_moveable, __setdown_moveable)
    velocity = property(__getvelocity,__setvelocity)

class Ball(object):
    """ 
        TODO: Comment
    """

    def __init__(self, p_x_pos, p_y_pos, p_x_dir, p_y_dir, p_velocity=5):
        """ 
            TODO: Comment
        """

        self.__velocity = p_velocity
        self.__x_pos = p_x_pos
        self.__y_pos = p_y_pos
        self.__x_dir = p_x_dir
        self.__y_dir = p_y_dir

    def next_pos(self, p_time_sec):
        """ 
            TODO: Comment
        """
        
        self.x_pos += self.velocity * self.x_dir * p_time_sec
        self.y_pos += self.velocity * self.y_dir * p_time_sec

    def __getvelocity(self):
        """ 
            TODO: Comment
        """

        return self.__velocity

    def __setvelocity(self, p_velocity):
        """ 
            TODO: Comment
        """

        self.__velocity = p_velocity

    def __getx_pos(self):
        """ 
            TODO: Comment
        """

        return self.__x_pos

    def __setx_pos(self, p_x_pos):
        """ 
            TODO: Comment
        """

        self.__x_pos = p_x_pos

    def __gety_pos(self):
        """ 
            TODO: Comment
        """

        return self.__y_pos

    def __sety_pos(self, p_y_pos):
        """ 
            TODO: Comment
        """

        self.__y_pos = p_y_pos

    def __getx_dir(self):
        """ 
            TODO: Comment
        """

        return self.__x_dir

    def __setx_dir(self, p_x_dir):
        """ 
            TODO: Comment
        """

        self.__x_dir = p_x_dir

    def __gety_dir(self):
        """ 
            TODO: Comment
        """

        return self.__y_dir

    def __sety_dir(self, p_y_dir):
        """ 
            TODO: Comment
        """

        self.__y_dir = p_y_dir

    velocity = property(__getvelocity, __setvelocity)
    x_pos = property(__getx_pos, __setx_pos)
    y_pos = property(__gety_pos, __sety_pos)
    x_dir = property(__getx_dir, __setx_dir)
    y_dir = property(__gety_dir, __sety_dir)

class Area(object):
    """ 
        TODO: Comment
    """
    
    def __init__(self, p_height=360, p_width=640):
        """ 
            TODO: Comment
        """

        self.__height = p_height
        self.__width = p_width

        y_middle = p_height / 2
        x_middle = p_width / 2

        self.__paddle1 = Paddle(y_middle,0)
        self.__paddle2 = Paddle(y_middle,p_width-10)
        self.__ball = Ball(x_middle,y_middle,5,7)

    def check_paddle_moveable(self, p_paddle):
        """ 
            TODO: Comment
        """

        if p_paddle.y_pos < 0:
            p_paddle.up_moveable = False
        else:
            p_paddle.up_moveable   = True

        if p_paddle.y_pos + p_paddle.length > self.height:
            p_paddle.down_moveable = False
        else:
            p_paddle.down_moveable = True

    def resolve_collisions(self):
        """ 
            TODO: Comment
        """

        if self.ball.y_pos > self.height or self.ball.y_pos < 0:
            self.ball.y_dir = -self.ball.y_dir
        
        if self.ball.x_pos >= self.paddle2.x_pos:
            if self.ball.y_pos >= self.paddle2.y_pos and self.ball.y_pos <= self.paddle2.y_pos + self.paddle2.length:
                self.ball.x_dir = -self.ball.x_dir
                return
        elif self.ball.x_pos <= self.paddle1.x_pos:
            if self.ball.y_pos >= self.paddle1.y_pos and self.ball.y_pos <= self.paddle1.y_pos + self.paddle1.length:
                self.ball.x_dir = -self.ball.x_dir
                return

        score_for = 0

        if self.ball.x_pos > self.width:
            score_for = 1
            self.ball.x_pos = self.width / 2
            self.ball.y_pos = self.height / 2

        if self.ball.x_pos < 0:
            score_for = 2
            self.ball.x_pos = self.width / 2
            self.ball.y_pos = self.height / 2

        return score_for

    def __getheight(self):
        """ 
            TODO: Comment
        """

        return self.__height
    
    def __setheight(self, p_height):
        """ 
            TODO: Comment
        """

        self.__height = p_height

    def __getwidth(self):
        """ 
            TODO: Comment
        """

        return self.__width
    
    def __setwidth(self, p_width):
        """ 
            TODO: Comment
        """

        self.__width = p_width

    def __getpaddle1(self):
        """ 
            TODO: Comment
        """

        return self.__paddle1 

    def __setpaddle1(self, p_paddle):
        """ 
            TODO: Comment
        """

        self.__paddle1 = p_paddle  

    def __getpaddle2(self):
        """ 
            TODO: Comment
        """

        return self.__paddle2 
        
    def __setpaddle2(self, p_paddle):
        """ 
            TODO: Comment
        """

        self.__paddle2 = p_paddle

    def __getball(self):
        """ 
            TODO: Comment
        """

        return self.__ball 
    
    def __setball(self, p_ball):
        """ 
            TODO: Comment
        """

        self.__ball = p_ball

   
    
    height = property(__getheight, __setheight)
    width = property(__getwidth, __setwidth)
    paddle1 = property(__getpaddle1, __setpaddle1)
    paddle2 = property(__getpaddle2, __setpaddle2)
    ball = property(__getball, __setball)

class Player(object):
    """ 
        TODO: Comment
    """

    def __init__(self, p_name, p_strategy):
        """ 
            TODO: Comment
        """

        self.__name = p_name
        self.__strategy = p_strategy
        self.__paddle = None
        self.__points = 0
    
    def __setname(self, p_name):
        """ 
            TODO: Comment
        """

        self.__name = p_name
    
    def __getname(self):
        """ 
            TODO: Comment
        """

        return self.__name 
    
    def __setstrategy(self, p_strategy):
        """ 
            TODO: Comment
        """

        self.__strategy = p_strategy
    
    def __getstrategy(self):
        """ 
            TODO: Comment
        """

        return self.__strategy

    def __setpaddle(self, p_paddle):
        """ 
            TODO: Comment
        """

        self.__paddle = p_paddle
        self.__strategy.paddle = p_paddle

    def __getpaddle(self):
        """ 
            TODO: Comment
        """

        return self.__paddle 

    def __setpoints(self, p_points):
        """ 
            TODO: Comment
        """

        self.__points = p_points
    
    def __getpoints(self):
        """ 
            TODO: Comment
        """

        return self.__points 

    def next_pos(self, p_dir_up):
        """ 
            TODO: Comment
        """

        self.strategy.next_pos(self.paddle, p_dir_up)

    name = property(__getname, __setname)
    strategy = property(__getstrategy, __setstrategy)
    paddle = property(__getpaddle, __setpaddle)
    points = property(__getpoints, __setpoints)

class Game(object):
    """ 
       TODO: Comment
    """

    def __init__(self):
        """ 
            TODO: Comment
        """

        self.__area = Area()
        self.__player1 = None
        self.__player2 = None
        self.__winner = None
        self.__winning_score = 10

        self.__cur_rgb_matrix = None
        self.__last_states = list()
        self.__n_of_images = 4


    def newPlayer(self,  p_name, p_train_mode = False, p_resume = False):
        """ 
            TODO: Comment
        """

        if p_train_mode:
            name = "Miles Davis"

            if p_resume:
                strategy = pickle.load(open('models/' + p_name + '.p', 'rb'))
            else:
                strategy = ReinforcedStrat(self.__area.width, self.__area.height, p_name)
        else:
            name = input("Enter your name: ")
            
            chosen_strategy = input("What's your strategy?[manual|dumb|rl] :")
            
            if chosen_strategy == 'manual':
                strategy = ManualStrat()
            elif chosen_strategy == 'dumb':
                strategy = DumbStrat()
            elif chosen_strategy == 'rl':
                if p_resume:
                    strategy = pickle.load(open('models/' + p_name + '.p', 'rb'))
                else:
                    strategy = ReinforcedStrat(self.__area.width, self.__area.height, p_name)

            elif chosen_strategy == 'random':
                strategy = RandomStrat()
            elif chosen_strategy == 'follow':
                strategy = FollowTheBallStrat(self.__area.ball)
            elif chosen_strategy == 'god':
                strategy = GodStrat()
            else:
                strategy = DumbStrat()



        player = Player(name,strategy)
        
        return player

    def setPlayers(self, p_train_mode = False, p_resume = False):
        """ 
            TODO: Comment
        """

        self.player1 = self.newPlayer("p1", p_train_mode, p_resume)
        self.player1.paddle = self.__area.paddle1

        if self.player1.strategy.__class__.__name__ == 'GodStrat':
            self.player1.paddle.y_pos = 0
            self.player1.paddle.length = self.area.height

        self.player2 = self.newPlayer("p2", p_train_mode, p_resume)
        self.player2.paddle = self.__area.paddle2

        if self.player2.strategy.__class__.__name__ == 'GodStrat':
            self.player2.paddle.y_pos = 0
            self.player2.paddle.length = self.area.height

    def __thread_p1(self, p_dir_up, p_screen):
        """ 
            TODO: Comment
        """

        self.player1.next_pos(p_dir_up)
        pygame.draw.rect(p_screen, (255, 255, 255),[self.area.paddle1.x_pos, self.area.paddle1.y_pos, 10, self.area.paddle1.length])

    def __thread_p2(self, p_dir_up, p_screen):
        """ 
            TODO: Comment
        """

        self.player2.next_pos(p_dir_up)
        pygame.draw.rect(p_screen, (255, 255, 255),[self.area.paddle2.x_pos, self.area.paddle2.y_pos, 10, self.area.paddle2.length])

    def __thread_ball(self, p_screen):
        """ 
            TODO: Comment
        """

        self.area.ball.next_pos(1)
        pygame.draw.rect(p_screen, (255,255,255),[self.area.ball.x_pos,self.area.ball.y_pos,20,20])

    def play(self):
        """ 
            TODO: Comment
        """

        pygame.init()

        resolution = (self.area.width, self.area.height)
        screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("Smart Pong")
        score_font = pygame.font.SysFont("Clear Sans Regular", 30)

        cancel = False
        dir_up = None

        while not cancel:
            pressed_down = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cancel = True
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        dir_up = False
                    if event.key == pygame.K_UP:
                        dir_up = True

                if event.type == pygame.KEYUP:
                    dir_up = None

            pygame.display.flip()
            screen.fill((0, 0, 0))

            self.area.check_paddle_moveable(self.player1.paddle)
            self.area.check_paddle_moveable(self.player2.paddle)

            p1_thread = threading.Thread(target=self.__thread_p1, args=(dir_up,screen,))
            p2_thread = threading.Thread(target=self.__thread_p2, args=(dir_up,screen,))
            ball_thread = threading.Thread(target=self.__thread_ball, args=(screen,))
            
            p1_thread.start()
            p2_thread.start()
            ball_thread.start()

            p1_thread.join()
            p2_thread.join()
            ball_thread.join()

            score_for = self.area.resolve_collisions()

            if score_for == 1:
                self.player1.points += 1
                self.player1.strategy.notify_score(1)
                self.player2.strategy.notify_score(0)
            elif score_for == 2:
                self.player2.points += 1
                self.player1.strategy.notify_score(0)
                self.player2.strategy.notify_score(1)

            screen.blit(score_font.render(str(self.player1.points), True, (255, 0, 255)), (self.area.width / 4, 50))
            screen.blit(score_font.render(str(self.player2.points), True, (255, 0, 255)),
                        (self.area.width / 1.25, 50))

            if self.player1.points == self.winning_score:
                self.winner = self.player1
                return
            elif self.player2.points == self.winning_score:
                self.winner = self.player2
                return

            score_for = 0

            surface_array = pygame.surfarray.array2d(pygame.display.get_surface())
            self.cur_matrix = surface_array

    def reset(self):
        """ 
            TODO: Comment
        """

        self.player1.strategy.reset()
        self.player2.strategy.reset()
        self.player1.points = 0
        self.player2.points = 0
        self.winner = None
        self.__last_states = list()

    def __setarea(self, p_area):
        """ 
            TODO: Comment
        """

        self.__area = p_area

    def __getarea(self):
        """ 
            TODO: Comment
        """

        return self.__area

    def __setwinner(self, p_player):
        """ 
            TODO: Comment
        """

        self.__winner = p_player

    def __getwinner(self):
        """ 
            TODO: Comment
        """

        return self.__winner

    def __getcur_matrix(self):
        """ 
            TODO: Comment
        """

        return self.__cur_rgb_matrix

    def __setcur_matrix(self, p_matrix):
        """ 
            TODO: Comment
        """

        self.__cur_rgb_matrix = p_matrix
        self.__last_states.append(self.__cur_rgb_matrix)

        if len(self.__last_states) > self.__n_of_images:
            self.__last_states.pop(0)
            self.player1.strategy.new_state(self.__last_states)
            self.player2.strategy.new_state(self.__last_states)
        elif len(self.__last_states) == self.__n_of_images:
            self.player1.strategy.new_state(self.__last_states, p_is_first_state = True)
            self.player2.strategy.new_state(self.__last_states, p_is_first_state = True)
    
    def __getn_of_images(self):
        """ 
            TODO: Comment
        """

        return self.__n_of_images

    def __setn_of_images(self, p_n):
        """ 
            TODO: Comment
        """

        self.__n_of_images = p_n

    def __getwinning_score(self):
        """ 
            TODO: Comment
        """

        return self.__winning_score

    def __setwinning_score(self, p_int):
        """ 
            TODO: Comment
        """
        
        self.__winning_score = p_int

    area = property(__getarea, __setarea)
    winner = property(__getwinner, __setwinner)
    cur_matrix = property(__getcur_matrix, __setcur_matrix)
    n_of_images = property(__getn_of_images, __setn_of_images)
    winning_score = property(__getwinning_score, __setwinning_score)