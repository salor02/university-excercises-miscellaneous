#Considerare ora 10 stringhe molto lunghe e contare la frequenza cumulativa di ogni
#lettera.
#Utilizzare il paradigma map/reduce

from functools import reduce

book = list()

book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")
book.append("ciao come va la vita io sto bene")

#normalizzazione delle stringhe contenute nella lista (book)
book = [text.upper().replace(" ","") for text in book]

def sum_dict(x, y):
    #crea un set costituito dalle chiavi dei due dict (non duplicate) e per ogni chiave prende il valore di entrambi e li somma
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}

#**freq_dict spacchetta il dizionario del giro precedente e aggiunge una nuova coppia di valori in modo che la lambda function ritorni un nuovo dizionario
freq = reduce(sum_dict, map(lambda text: reduce(lambda freq_dict, letter: {**freq_dict, letter: freq_dict.get(letter,0)+1}, text, dict()), book), dict())

print(dict(freq))