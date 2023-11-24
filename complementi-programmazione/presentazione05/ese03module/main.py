from Animali import *

pluto = cane.Cane('M','dalmata')
john = cane.Cane('F','dalmata')
tom = gatto.Gatto('F','siberiano')

pluto.abbaia()
tom.miagola()

print(cane.Cane.stessoLuogo(pluto, john))