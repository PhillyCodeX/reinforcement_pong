import src.Paddle as Paddle
import src.Ball as Ball

class Area(object):
    def __init__(self, p_height, p_width):
        self.__height = p_height
        self.__width = p_width

        y_middle = p_height / 2
        x_middle = p_width / 2

        self.__paddle1 = Paddle(y_middle)
        self.__paddle2 = Paddle(y_middle)
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
    