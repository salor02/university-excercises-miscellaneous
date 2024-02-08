from Banca.transazione import *

class Utente:
    def __init__(self, name):
        self.name = name
        self.__txs_list_ = []

    @property
    def txs_list(self):
        return self.__txs_list_
    
    def add_tx(self, date, amount):
        if amount < 0:
            self.__txs_list_.append(Prelievo(date, amount))
        else:
            self.__txs_list_.append(Versamento(date, amount))