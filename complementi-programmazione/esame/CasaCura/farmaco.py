class Farmaco():
    def __init__(self, name, freq, start_date):
        self.name = name
        self.__freq_ = freq
        self.__start_date_ = start_date

    @property
    def freq(self):
        return self.__freq_
    
    @freq.setter
    def freq(self, value):
        self.__freq_ = value

    @property
    def start_date(self):
        return self.__start_date_