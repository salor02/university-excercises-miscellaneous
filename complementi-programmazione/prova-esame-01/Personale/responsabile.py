from Personale.utente import Utente

class Responsabile(Utente):
    def __init__(self, name, hourly_pay):
        super().__init__(name, hourly_pay)
        self.__subs_ = []

    @property
    def subs(self):
        return self.__subs_

    def add_sub(self, sub):
        self.__subs_.append(sub)