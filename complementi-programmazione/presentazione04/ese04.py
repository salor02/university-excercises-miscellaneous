#Scrivere un programma che consenta all’utente di scrivere delle espressioni
#matematiche e di ottenere il loro risultato.
#Un po’ come una calcolatrice

expr = input("Espressione da valutare: ")
res = eval(expr)
print(res)