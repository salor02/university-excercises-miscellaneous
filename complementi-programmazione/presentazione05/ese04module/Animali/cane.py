class Cane:
    
    def __init__(self, sesso, razza, zampe = 4, coda = True, eta = 0):
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta
        self.razza = razza

    def whoami(self):
        print("Ciao! Sono " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")

    def abbaia(self):
        print("WOOF WOOF")

    def cammina(self):
        print("STO CAMMINANDO")

    def corri(self):
        print("STO CORRENDO")

    @classmethod
    def stessoLuogo(cls, cane1, cane2):
        if cane1.razza == cane2.razza:
            return True
        else:
            return False
        
if __name__ == '__main__':
    pluto = Cane('M','dalmata')
    pluto.abbaia()