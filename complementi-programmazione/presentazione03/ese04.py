#Modificare i getters e setters usando @property
#
#Bis: Definire un comportamento speciale per l’attributo malato:
#ritorno il valore di malato MA se spavaldo è True ritorna sempre False

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

    @property
    def malato(self):
        if self.__spavaldo_:
            return False
        return self.__malato_
    
    @property
    def spavaldo(self):
        return self.__spavaldo_
    
    @malato.setter
    def malato(self, new_val):
        self.__malato_ = new_val

    @spavaldo.setter
    def spavaldo(self, new_val):
        self.__spavaldo_ = new_val

    def whoami(self):
        print("Ciao! Sono " + self.sesso + " e ho " + str(self.zampe) + " zampe, inoltre ho " + str(self.eta) + " anni")
        if(self.coda):
            print("Ho anche la coda!")
        if(self.__malato_):
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

pluto = Cane("maschio", 4, True, 5)
pluto.whoami()
pluto.abbaia()
pluto.cammina()
pluto.corri()

#####

pluto.malato = True
pluto.spavaldo = True
if pluto.malato == True:
    print("Sono malato cavoletti")
if pluto.spavaldo == True:
    print("Sono spavaldo ragazzi niente paura, non posso ammalarmi")

#####

franco = Cane("femmina", eta=54)

Cane.instance_size()