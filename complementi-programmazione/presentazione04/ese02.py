#Provare a contare la frequenza di ogni lettera filtrata
from functools import reduce

text = "ciao come va la vita io sto bene"
text = text.replace(" ","")

#applica prima il filtro e poi il map (elimina anche gli spazi)
res = map(lambda l: l.upper(), filter(lambda l:(l not in ['a','e','i','o','u',' ']), text))

#**freq_dict spacchetta il dizionario del giro precedente e aggiunge una nuova coppia di valori in modo che la lambda function ritorni un nuovo dizionario
freq = reduce(lambda freq_dict, letter: {**freq_dict, letter: freq_dict.setdefault(letter,0)+1}, res, dict())

print(freq)