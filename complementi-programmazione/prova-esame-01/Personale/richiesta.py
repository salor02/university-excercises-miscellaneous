import argparse
import pickle

class Richiesta:
    def __init__(self, hours):
        self.__hours_ = int(hours)
        self.__status_ = 'Waiting'

    @property
    def hours(self):
        return self.__hours_
    
    @property
    def status(self):
        return self.__status_
    
    @status.setter
    def status(self, value):
        self.__status_ = value

    @staticmethod
    def esporta_richieste(sub, filename):
        with open(filename,'w') as file:
            for idx, request in enumerate(sub.requests):
                file.write(str(idx) + "\t-\t" + request.status + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Esportazione richieste da file di backup')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('user_name', type=str)
    arguments = parser.parse_args()

    #caricamento da file
    with open(arguments.input_file,'rb') as input_file:
        registered_user = pickle.load(input_file)
    
    #la lista dei subordinati è in seconda posizione della lista di utenti generale
    registered_subs = registered_user[1]

    for sub in registered_subs:
        if sub.name == arguments.user_name:
            Richiesta.esporta_richieste(sub, arguments.output_file)
            print("Esportazione completata")