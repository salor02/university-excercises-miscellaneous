#Partendo da una stringa molto lunga di caratteri minuscoli scrivere una sequenza di
#espressioni che tengano solo le consonanti e le renda maiuscole.

text = "ciao come va la vita io sto bene"

#applica prima il filtro e poi il map (elimina anche gli spazi)
res = map(lambda l: l.upper(), filter(lambda l:(l not in ['a','e','i','o','u',' ']), text))

print(list(res))