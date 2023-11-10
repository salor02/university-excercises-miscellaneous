class Cane:
    count = 0

    def __init__(self, sesso, zampe = 4, coda = True, eta = 0, malato = False, spavaldo = False):
        self.__malato_ = malato
        self.__spavaldo_ = spavaldo
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta
        #Cane.count += 1 #equivalente
        self.__class__.count += 1

    def get_malato(self):
        return self.__malato_
    
    def get_spavaldo(self):
        return self.__spavaldo_
    
    def set_malato(self, new_val):
        self.__malato_ = new_val

    def set_spavaldo(self, new_val):
        self.__spavaldo_ = new_val

    def whoami(self):
        print("Ciao! Mi chiamo " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")
        if(self.__malato_):
            print("Sono anche malato purtroppo!")
        if(self.__spavaldo_):
            print("Sono anche spavaldo LESGOO")

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

pluto.set_malato(True)
pluto.set_spavaldo(True)
pluto.whoami()

franco = Cane("femmina", eta=54)

Cane.instance_size()