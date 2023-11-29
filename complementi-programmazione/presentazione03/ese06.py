#Fare in modo che la classe Cane erediti da una classe Animale
#Creare anche una classe Gatto che eredita da Animale
#Capire quali metodi mettere nelle classe Animale e quali in Cane/Gatto

class Animale:
    count = 0

    def __init__(self, sesso, zampe = 4, coda = True, eta = 0):
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta
        #Cane.count += 1 #equivalente
        Animale.count += 1

    def whoami(self):
        print("Ciao! Sono " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")

    def cammina(self):
        print("STO CAMMINANDO")

    def corri(self):
        print("STO CORRENDO")

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

    def abbaia(self):
        print("WOOF WOOF")

class Gatto(Animale):

    def miagola(self):
        print("MIAOOOO MIAOOO")

    

pluto = Cane("maschio", 4, True, 5)
peppa = Cane("femmina", 4, True, 5)
tommasa = Gatto("femmina", 4, True, 21)

pluto.whoami()
pluto.abbaia()
pluto.cammina()
pluto.corri()

tommasa.whoami()
tommasa.miagola()
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