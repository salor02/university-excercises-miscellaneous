#Scrivere un programma che chieda all’utente di immettere una serie di valori
#terminata dal simbolo ‘#’
#Dopodichè applica una funzione a vostro piacere su tutti gli elementi.
#Non usare un ciclo per l’applicazione della funzione

#eleva al quadrato solamente i numeri pari
def modify(num):
    num = int(num)
    if num % 2 == 0:
        return num * num
    else:
        return num

val_list = []
new_val = input("Inserire valore: ")

while new_val != '#':
    val_list.append(new_val)
    new_val = input("Inserire valore: ")

print("Lista creata: ")
print(val_list)

#applicazione funzione a tutti gli elementi della lista con map
print("Lista modificata: ")
new_list = list(map(modify, val_list))
print(new_list)

