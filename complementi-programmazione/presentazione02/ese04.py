#Scrivere un decoratore che arricchisca una funzione come segue:
#1) Oltre a ritornare il risultato della funzione stessa, lo stampa anche a video.
#2) Prima di eseguire chiede all’utente se vuole modificare il parametro in
#ingresso (opzione non possibile se vi sono più parametri)

def decoratore(function):
    def new_function(*args, **kwargs):
        if len(args) == 1:
            if input("Vuoi cambiare il parametro in input? (y/n)") == 'y':
                args = (input("Inserire nuovo valore: "),)
        
        value = function(*args, **kwargs)
        print("Somma dei valori: " + str(value))
        return value
    return new_function

@decoratore
def cubo(num):
    return int(num)**3

num = 5
res = cubo(num)
