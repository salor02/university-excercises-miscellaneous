#Creare un package Animali che contenga i moduli creati in precedenza.
#Fare in modo che con
#from Animali import *
#vengano importati tutti i moduli

from Animali import *

pluto = cane.Cane('M','dalmata')
john = cane.Cane('F','dalmata')
tom = gatto.Gatto('F','siberiano')

pluto.abbaia()
tom.miagola()

print(cane.Cane.stessoLuogo(pluto, john))