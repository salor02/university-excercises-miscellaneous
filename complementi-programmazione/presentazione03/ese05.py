class Cane:
    count = 0

    def __init__(self, sesso, zampe = 4, coda = True, eta = 0, malato = False, spavaldo = False):
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta
        #Cane.count += 1 #equivalente
        self.__class__.count += 1

    def __add__(self, other_cane):
        if self.sesso != other_cane.sesso:
            return Cane("maschio", 4, True, 0)
        
        raise Exception("Due cani dello stesso sesso non possono fare un cucciolo!")

    def whoami(self):
        print("Ciao! Mi chiamo " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")

    def abbaia(self):
        print("WOOF WOOF")

    def cammina(self):
        print("STO CAMMINANDO WUAGLIU")

    def corri(self):
        print("STO CURREN WUAGLIU")

    @classmethod
    def instance_size(cls):
        print("La classe è stata istanziata " + str(cls.count) + " volte")

pluto = Cane("maschio", 4, True, 5)
pluto.whoami()
pluto.abbaia()
pluto.cammina()
pluto.corri()

gina = Cane("femmina", eta=54)

#####
try:
    cucciolo = pluto + gina
    print("E' nato un cucciolo!!")
    cucciolo.whoami()
except Exception as e:
    print(e)


#####


Cane.instance_size()