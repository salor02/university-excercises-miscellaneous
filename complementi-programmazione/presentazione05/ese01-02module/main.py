#Creare un applicativo diviso in moduli.
#Riprendiamo i codici sugli animali e facciamo un modulo per animale.
#Il modulo deve comprendere la classe dell’animale e delle funzioni utili:
#- es. stessoLuogo(cane1, cane2) -> ritorna se possono vivere nello stesso luogo

#Testare il codice dell’esercizio utilizzando uno script esterno in cui
#vengono importati i moduli.

import cane
import gatto

pluto = cane.Cane('M','dalmata')
leonessa = cane.Cane('F','dalmata')
tom = gatto.Gatto('F','siberiano')

print(cane.Cane.stessoLuogo(pluto,leonessa))
pluto.abbaia()
tom.miagola()