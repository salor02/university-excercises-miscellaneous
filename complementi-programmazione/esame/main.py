from menu import Menu
from CasaCura import *
import pickle
from datetime import date, timedelta

pazienti = []
selected_paziente = None #tiene traccia del paziente attualmente selezionato
terapia = terapia.Terapia()

#creazione nuovo paziente
def new_paziente():
    name = input('Nome paziente: ')
    pazienti.append(paziente.Paziente(name))
    #se tutto ok comunica al main che il paziente è stato inserito
    return True

#selezione paziente esistente
def select_paziente():
    #stampa lista pazienti
    for idx, paziente in enumerate(pazienti):
        print(f'{idx} - {paziente.name}')
    
    #controllo e gestione eventuali errori sull'input
    try:
        choice = int(input('ID paziente: '))
    except ValueError:
        print('[ERROR] Inserire un numero intero')
        return False
    else:
        if choice < 0 or choice > len(pazienti):
            print('[ERROR] Paziente non esistente')
            return False
    
    #aggiorna la viariabile relativa al paziente selezionato
    global selected_paziente
    selected_paziente = pazienti[choice]

    #display del menu dedicato al paziente loggato
    paziente_menu.display()
    result = paziente_menu.select()
    while result == 'REPEAT':
        paziente_menu.display()
        print('Operazione non esistente, riprovare')
        result = paziente_menu.select()

    if result == 'EXIT': return 'EXIT' #gestione uscita e comunicazione al main

    return result #comunica al main l'esito dell'operazione

#salvataggio su file di backup
def save_file():

    #"compatta" la lista di pazienti e l'oggetto terapia in una lista per il salvataggio su file
    data = [pazienti, terapia]
    
    #gestione eventuali errori
    try:
        with open('backup','wb') as out_file:
            pickle.dump(data, out_file)
    except Exception:
        print('[ERROR] Errore generico in fase di salvataggio')
        return False
    
    return True

#caricamento da file di backup
def upload_file():

    #gestione eventuali errori
    try:
        with open('backup','rb') as in_file:
            data = pickle.load(in_file)
    except Exception:
        print('[ERROR] Errore generico in fase di salvataggio')
        return False
    
    #caricamento effettivo dei dati tramite "scompattamento" della lista
    global pazienti
    pazienti = data[0]
    global terapia
    terapia = data[1]
    
    return True

#inserimento farmaco per paziente selezionato
def new_farmaco():

    #input dati necessari alla creazione dell'oggetto farmaco
    name = input('Nome farmaco: ')
    try:
        freq = int(input('Frequenza di assunzione: '))
    except ValueError:
        print('[ERROR] Inserire un numero intero')
        return False
    
    #inserimento farmaco nella terapia del paziente
    terapia.add_farmaco(name, freq, date.today(), selected_paziente)
    return True

#eliminazione farmaco per paziente selezionato
def del_farmaco():
    farmaci = terapia.get_terapia(selected_paziente)

    #check per controllare che il paziente abbia dei farmaci
    if len(farmaci) == 0:
        print('Il paziente non ha ancora nessun farmaco')
        return False
    
    #mostra lista completa dei farmaci del paziente
    for idx, farmaco in enumerate(farmaci):
        print(f'{idx} - {farmaco.name}')
    
    #inserimento id farmaco e gestione degli errori
    try:
        choice = int(input("ID farmaco da eliminare: "))
    except ValueError:
        print('[ERROR] Inserire un numero intero ')
        return False
    else:
        if choice < 0 or choice > len(farmaci):
            print('[ERROR] Farmaco non esistente')
            return False
    
    #se tutto ok elimina farmaco dalla terapia e restituisce il controllo al main
    terapia.del_farmaco(choice, selected_paziente)
    return True

#visualizzazione terapia giornaliera
def terapia_daily():
    
    farmaci = terapia.get_terapia_day(selected_paziente)
    print(f'Terapia [{date.today()}]')
    for farmaco in farmaci:
        print(f'{farmaco.name}')
    return True

#calcolo e recupero terapie di giorni successivi (non viene chiamata direttamente dal main)
def terapia_next():

    try:
        num_days = int(input('Numero giorni da visualizzare: '))
    except ValueError:
        print('[ERROR] Inserire un numero intero')
        return False
    else:
        if num_days<0:
            print('[ERROR] Inserire numero positivo di gioni')
            return False

    calendario = dict()
    #include il giorno corrente quindi restituisce la data di oggi e altri N giorni
    for day in range(num_days+1):
        end_date = date.today() + timedelta(days=day)
        calendario[end_date] = terapia.get_terapia_day(selected_paziente, end_date)

    return calendario

#stampa su standard output la terapia dei giorni specificati utilizzando terapia_next()
def stampa_terapia():
    calendario = terapia_next()

    for item in calendario.items():
        print(f'Data - [{item[0]}]')
        for farmaco in item[1]:
            print(f' - {farmaco.name}')
        print()
    
    return True
    
#stampa su file di testo la terapia dei giorni specificati urilizzando terapia_next()
def esporta_terapia():
    calendario = terapia_next()

    with open(selected_paziente.name + '_terapia.txt','w') as out_file:
        for item in calendario.items():
            out_file.write(f'Data - [{item[0]}]')
            for farmaco in item[1]:
                out_file.write(f' - {farmaco.name}')
            out_file.write('\n')
    
    return True

#menu principale
MAIN_MENU = {
    1:('Nuovo paziente', new_paziente),
    2:('Seleziona paziente', select_paziente),
    3:('Salva dati su file', save_file),
    4:('Carica dati da file', upload_file)
}

#menu per paziente registrato
PAZIENTE_MENU = {
    1:('Nuovo farmaco', new_farmaco),
    2:('Elimina farmaco', del_farmaco),
    3:('Visualizza terapia giornaliera', terapia_daily),
    4:('Visualizza terapia giorni successivi', stampa_terapia),
    5:('Esporta terapia', esporta_terapia)
}

main_menu = Menu('Menu principale', MAIN_MENU)
paziente_menu = Menu('Menu paziente registrato', PAZIENTE_MENU)

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
