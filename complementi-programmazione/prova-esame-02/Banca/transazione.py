import argparse
import pickle
import datetime

# Function to convert string to datetime
def convert(date_time):
    format = '%d-%m-%y'
    datetime_str = datetime.datetime.strptime(date_time, format)
    return datetime_str

class Transazione:
    def __init__(self, date, amount):
        self.date = date
        self.amount = amount

    @staticmethod
    def save_ops(users, filename, start_date, end_date):
        with open(filename,'w') as out_file:
            for user in users:
                out_file.write('Utente: ' + user.name + '\n')
                for tx in user.txs_list:
                    if tx.date >= start_date and tx.date <= end_date:
                        out_file.write(f'[{tx.type}]\t {tx.date} : €{tx.amount}\n')

class Versamento(Transazione):
    def __init__(self, date, amount):
        if amount < 0:
            raise ValueError
        super().__init__(date, amount)
        self.type = 'VERSAMENTO'

class Prelievo(Transazione):
    def __init__(self, date, amount):
        if amount > 0:
            raise ValueError
        super().__init__(date, amount)
        self.type = 'PRELIEVO'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Esportazione transazioni da file di backup')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('start_date', type=str)
    parser.add_argument('end_date', type=str)
    arguments = parser.parse_args()

    #caricamento da file
    with open(arguments.input_file,'rb') as input_file:
        users = pickle.load(input_file)
    
    Transazione.save_ops(users, arguments.output_file, convert(arguments.start_date), convert(arguments.end_date))

    