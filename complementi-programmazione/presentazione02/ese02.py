#Modificare il programma precedente in modo che l’utente possa inserire solo
#numeri.
#Gestire gli altri casi tramite eccezioni.

#eleva al quadrato solamente i numeri pari
def modify(num):
    if num % 2 == 0:
        return num * num
    else:
        return num

val_list = []
new_val = input("Inserire valore: ")

#viene sollevata un'eccezzzzzione se non viene inserito un numero intero
while new_val != '#':
    try:
        new_val = int(new_val)
        val_list.append(new_val)
    except ValueError as e:
        print("E' possibile inserire solo numeri interi!")
    new_val = input("Inserire valore: ")

print("Lista creata: ")
print(val_list)

#applicazione funzione a tutti gli elementi della lista con map
print("Lista modificata: ")
new_list = list(map(modify, val_list))
print(new_list)