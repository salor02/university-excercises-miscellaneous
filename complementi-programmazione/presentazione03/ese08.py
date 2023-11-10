from abc import ABC, abstractmethod

class Malato:
    def __init__(self):
        pass

    def __set__(self, instance, value):
        if instance.spavaldo:
            instance.malato = False
        else:
            instance.malato = value

    def __get__(self, instance, owner):
        return instance.malato

class Animale(ABC):
    count = 0

    malato = Malato()

    def __init__(self, sesso, zampe = 4, coda = True, eta = 0, malato = False, spavaldo = False):
        self.__spavaldo_ = spavaldo
        Animale.malato = malato
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta
        #Cane.count += 1 #equivalente
        Animale.count += 1

    def whoami(self):
        print("Ciao! Mi chiamo " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")
        if(self.malato):
            print("Sono anche malato purtroppo!")
        if(self.__spavaldo_):
            print("Sono anche spavaldo LESGOO")

    def cammina(self):
        print("STO CAMMINANDO WUAGLIU")

    def corri(self):
        print("STO CURREN WUAGLIU")

    @property
    def spavaldo(self):
        return self.__spavaldo_
    
    @spavaldo.setter
    def spavaldo(self, value):
        self.__spavaldo_ = value

    @abstractmethod
    def fai_verso(self):
        pass

    @classmethod
    def instance_size(cls):
        print("La classe è stata istanziata " + str(cls.count) + " volte")

class Cane(Animale):

    def fai_verso(self):
        print("WOOF WOOF")

    

pluto = Cane("maschio", 4, True, 5, malato=True, spavaldo=True)

pluto.whoami()

pluto.spavaldo = True
if pluto.malato == True:
    print("Sono malato cavoletti")
if pluto.spavaldo == True:
    print("Sono spavaldo ragazzi niente paura, non posso ammalarmi")
