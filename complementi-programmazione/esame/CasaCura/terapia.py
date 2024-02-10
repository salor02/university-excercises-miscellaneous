from CasaCura.farmaco import Farmaco
from datetime import date, timedelta
import argparse
import pickle

class Terapia():
    def __init__(self):
        self.lista_farmaci = dict()

    #aggiunge al record del paziente il farmaco inserito come parametro
    def add_farmaco(self, name, freq, start_date, paziente):
        #se il paziente non ha ancora nessun farmaco inizializza la lista
        if not paziente in self.lista_farmaci:
            self.lista_farmaci[paziente] = []
        
        #aggiunta del farmaco alla lista del paziente
        self.lista_farmaci[paziente].append(Farmaco(name, freq, start_date))

    #elimina il farmaco di indice idx_farmaco dalla lista dei farmaci della terapia del paziente
    def del_farmaco(self, idx_farmaco, paziente):
        del self.lista_farmaci[paziente][idx_farmaco]

    #restituisce la terapia del paziente inserito come parametro
    def get_terapia(self, paziente):
        #se il paziente non ha ancora nessun farmaco inizializza la lista
        if not paziente in self.lista_farmaci:
            self.lista_farmaci[paziente] = []
        return self.lista_farmaci[paziente]
    
    #recupera i farmaci da prendere in un dato giorno
    def get_terapia_day(self, paziente, end_date = date.today()):
        #se il paziente non ha ancora nessun farmaco inizializza la lista
        if not paziente in self.lista_farmaci:
            self.lista_farmaci[paziente] = []

        farmaci = []

        for farmaco in self.lista_farmaci[paziente]:
            delta = end_date - farmaco.start_date
            if delta.days % farmaco.freq == 0:
                farmaci.append(farmaco)

        return farmaci
    
###START PUNTO 7
if __name__ == '__main__':
    #parsing argomenti
    parser = argparse.ArgumentParser(description='Esportazione terapia da file di backup')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('num_days', type=int)
    parser.add_argument('paziente', type=str)
    arguments = parser.parse_args()

    #importazione dati da file di backup
    try:
        with open(arguments.input_file,'rb') as in_file:
            data = pickle.load(in_file)
        pazienti = data[0]
        terapia = data[1]
    except FileNotFoundError:
        print('File di backup non esistente')
        exit()

    #selezione paziente corretto
    for paziente in pazienti:
        if paziente.name == arguments.paziente:
            selected_paziente = paziente

    try:
        #selezione farmacia per il numero di giorni inserito
        calendario = dict()
        #include il giorno corrente quindi restituisce la data di oggi e altri N giorni
        for day in range(arguments.num_days+1):
            end_date = date.today() + timedelta(days=day)
            calendario[end_date] = terapia.get_terapia_day(selected_paziente, end_date)
    except NameError:
        print('Paziente non esistente')
        exit()

    #esportazione su file di testo
    with open(arguments.output_file,'w') as out_file:
        for item in calendario.items():
            out_file.write(f'Data - [{item[0]}]')
            for farmaco in item[1]:
                out_file.write(f' - {farmaco.name}')
            out_file.write('\n')

    print('[SUCCESS] Esportazione completata')
### END PUNTO 7
