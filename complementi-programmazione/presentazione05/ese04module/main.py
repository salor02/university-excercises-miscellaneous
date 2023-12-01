#Fare in modo che ogni modulo possa essere eseguito indipendentemente.
#Se eseguito indipendentemente istanzierà la classe dell’animale e gli farà fare il
#verso.
#Questo comportamento non deve avvenire se il modulo viene importato

from Animali import *

pluto = cane.Cane('M','dalmata')
john = cane.Cane('F','dalmata')
tom = gatto.Gatto('F','siberiano')

pluto.abbaia()
tom.miagola()

print(cane.Cane.stessoLuogo(pluto, john))