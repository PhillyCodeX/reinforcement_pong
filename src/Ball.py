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