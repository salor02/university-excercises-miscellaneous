class Gatto:
    
    def __init__(self, sesso, razza, zampe = 4, coda = True, eta = 0):
        self.sesso = sesso
        self.zampe = zampe
        self.coda = coda
        self.eta = eta
        self.razza = razza

    def whoami(self):
        print("Ciao! Mi chiamo " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")

    def miagola(self):
        print("MIAOO MIAOO")

    def cammina(self):
        print("STO CAMMINANDO MIAOO")

    def corri(self):
        print("STO CURREN MIAOAOAO")

    @classmethod
    def stessoLuogo(cls, gatto1, gatto2):
        if gatto1.razza == gatto2.razza:
            return True
        else:
            return False