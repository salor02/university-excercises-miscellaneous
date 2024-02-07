from Personale.responsabile import Responsabile
from Personale.utente import Utente
from Personale.richiesta import Richiesta

class Subordinato(Utente):
    def __init__(self, name, hourly_pay, responsabile):
        super().__init__(name, hourly_pay)
        self.__responsabile_ = responsabile
        self.__requests_ = []

    def pagamento(self):
        sum_hours = 0

        for request in self.__requests_:
            if request.status == 'Accepted':
                sum_hours = sum_hours + int(request.hours)
        return sum_hours

    def new_richiesta(self, hours):
        self.__requests_.append(Richiesta(hours))


    @property
    def requests(self):
        return self.__requests_
