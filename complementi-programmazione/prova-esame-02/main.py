from menu import Menu
from Banca import *
import pickle
import datetime

users = []
selected_user = None #tiene traccia dell'utente loggato attualmente nel sistema

# Function to convert string to datetime
def convert(date_time):
    format = '%d-%m-%y'
    datetime_str = datetime.datetime.strptime(date_time, format)
    return datetime_str

def versamento():
    date = convert(input("Inserire data nel formato DD-MM-YY: "))
    amount = int(input("Importo: "))
    selected_user.add_tx(date, amount)
    return True

def prelievo():
    date = convert(input("Inserire data nel formato DD-MM-YY: "))
    amount = int(input("Importo: "))
    selected_user.add_tx(date, amount*(-1))
    return True

def lista_operazioni():
    start_date = convert(input("Inserire data iniziale periodo nel formato DD-MM-YY: "))
    end_date = convert(input("Inserire data finale periodo nel formato DD-MM-YY: "))

    total_amount = 0
    for tx in selected_user.txs_list:
        if tx.date >= start_date and tx.date<=end_date:
            print(f'[{tx.type}]\t {tx.date} : {tx.amount}')
            total_amount = total_amount + tx.amount

    print(f'Totale: {total_amount}')
    
    return True

def save_ops():
    with open(selected_user.name + '_txs.txt','w') as out_file:
        for tx in selected_user.txs_list:
            out_file.write(f'[{tx.type}]\t {tx.date} : €{tx.amount}\n')
    return True

def new_user():
    name = input("Nome: ")
    users.append(utente.Utente(name))
    return True

def select_user():
    global selected_user

    for idx, user in enumerate(users):
        print(f'{idx} - {user.name}')
    
    selected_idx = int(input("ID Utente: "))

    #controllo utente non esistente
    if selected_idx < 0 or selected_idx >= len(users):
        return False
    
    selected_user = users[selected_idx]
    
    #se utente esiste mostro il menu utente
    user_menu.display()
    result = user_menu.select()
    while result == 'REPEAT':
        user_menu.display()
        print('Operazione non esistente, riprovare')
        result = user_menu.select()

    if result == 'EXIT': return 'EXIT' #gestione uscita (lo comunica al main)

    return result #comunica al main l'esito dell'operazione


#salvataggio su file binario
def save_file():
    with open('sav','wb') as out_file:
        pickle.dump(users, out_file)
    return True

#caricamento da file binario
def upload_file():
    global users
    with open('sav','rb') as in_file:
        users = pickle.load(in_file)
    return True

MAIN_MENU = {
    1:('Nuovo utente', new_user),
    2:('Seleziona utente', select_user),
    3:('Salva dati su file', save_file),
    4:('Carica dati da file', upload_file)
}

USER_MENU = {
    1:('Versamento', versamento),
    2:('Prelievo', prelievo),
    3:('Lista operazioni', lista_operazioni),
    4:('Salva operazioni su file di testo', save_ops)
}

main_menu = Menu('Menu principale', MAIN_MENU)
user_menu = Menu('Menu utente registrato', USER_MENU)

if __name__ == '__main__':

    while True:
        main_menu.display()

        result = main_menu.select()
        while result == 'REPEAT': #gestione input errato
            main_menu.display()
            print('Operazione non esistente, riprovare')
            result = main_menu.select()

        if result == 'EXIT': break #gestione uscita

        if result != True:
            print('[ERROR] operazione non eseguita')
        else:
            print('[SUCCESS] Operazione eseguita con successo')
