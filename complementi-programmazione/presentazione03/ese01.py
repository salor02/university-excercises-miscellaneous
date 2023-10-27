class Cane:
    
    def __init__(self, sesso, zampe = 4, coda = True, eta = 0):
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta

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

pluto = Cane("maschio", 4, True, 5)
pluto.whoami()
pluto.abbaia()
pluto.cammina()
pluto.corri()