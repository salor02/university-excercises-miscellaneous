#Modificare il programma precedente utilizzando una lambda function.

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
#applicazione lamda function elevamento a potenza per tutti i numeri della lista
new_list = list(map(lambda num: num**2, val_list))
print(new_list)