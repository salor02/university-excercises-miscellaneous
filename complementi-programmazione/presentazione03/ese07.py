from abc import ABC, abstractmethod

class Animale(ABC):
    count = 0

    def __init__(self, sesso, zampe = 4, coda = True, eta = 0):
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

    def cammina(self):
        print("STO CAMMINANDO WUAGLIU")

    def corri(self):
        print("STO CURREN WUAGLIU")

    @abstractmethod
    def fai_verso(self):
        pass

    @classmethod
    def instance_size(cls):
        print("La classe è stata istanziata " + str(cls.count) + " volte")

    def __add__(self, other_animale):
        if self.sesso != other_animale.sesso:
            if type(self) == type(other_animale):
                return self.__class__("maschio", 4, True, 0)
            else:
                raise Exception("Due animali di specie diverse non possono fare un cucciolo!")
        else:
            raise Exception("Due animali dello stesso sesso non possono fare un cucciolo!")

class Cane(Animale):

    def fai_verso(self):
        print("WOOF WOOF")

class Gatto(Animale):

    def fai_verso(self):
        print("MIAOOOO MIAOOO")

    

pluto = Cane("maschio", 4, True, 5)
peppa = Cane("femmina", 4, True, 5)
tommasa = Gatto("femmina", 4, True, 21)

pluto.whoami()
pluto.fai_verso()
pluto.cammina()
pluto.corri()

tommasa.whoami()
tommasa.fai_verso()
tommasa.cammina()
tommasa.corri()

##### addizione

try:
    cucciolo = pluto + peppa
    print("E' nato un cucciolo!!")
    cucciolo.whoami()
except Exception as e:
    print(e)

#####


Animale.instance_size()