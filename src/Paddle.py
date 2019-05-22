class Paddle(object):
    def __init__(self, p_y_pos, p_length = 10):
        self.__length = p_length
        self.__y_pos = p_y_pos

    def __getlength(self):
        return self.__length

    def __setlength(self, p_length):
        self.__length = p_length

    def __gety_pos(self):
        return self.__y_pos

    def __sety_pos(self, p_y_pos):
        self.__y_pos = p_y_pos