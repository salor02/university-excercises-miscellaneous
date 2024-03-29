#Trattare l’attributo “malato” come descrittore (che considera anche l’attributo
#“spavaldo”).

class MalatoDesc:
    def __init__(self):
        self._malato = False

    def __get__(self, instance, owner):
        return self._malato
    
    def __set__(self, instance, value):
        if instance.spavaldo:
            self._malato = False
        else:
            self._malato = value

    def __delete__(self, instance):
        pass

class Cane:
    count = 0
    malato = MalatoDesc()

    def __init__(self, sesso, zampe = 4, coda = True, eta = 0, malato = False, spavaldo = False):
        self.__spavaldo_ = spavaldo
        self.__malato_ = malato
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta
        #Cane.count += 1 #equivalente
        self.__class__.count += 1
    
    @property
    def spavaldo(self):
        return self.__spavaldo_

    @spavaldo.setter
    def spavaldo(self, new_val):
        self.__spavaldo_ = new_val

    def whoami(self):
        print("Ciao! Sono " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")
        if(self.malato):
            print("Sono anche malato purtroppo!")
        if(self.__spavaldo_):
            print("Sono anche spavaldo LESGOO")

    def abbaia(self):
        print("WOOF WOOF")

    def cammina(self):
        print("STO CAMMINANDO")

    def corri(self):
        print("STO CORRENDO")

    @classmethod
    def instance_size(cls):
        print("La classe è stata istanziata " + str(cls.count) + " volte")

pluto = Cane("maschio", 4, True, 5, True, True)
pluto.whoami()
pluto.abbaia()
pluto.cammina()
pluto.corri()

franco = Cane("femmina", eta=54, malato=True, spavaldo=True)
franco.whoami()

Cane.instance_size()