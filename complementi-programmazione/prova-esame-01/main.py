from Personale import *
import pickle

def save(filename, registered_resp, registered_sub):
    with open(filename,'wb') as file:
        registered_user = [registered_resp,registered_sub]
        pickle.dump(registered_user, file)

def upload(filename):
    with open(filename,'rb') as file:
        registered_user = pickle.load(file)
        return registered_user

if __name__ == '__main__':
    registered_resp = []
    registered_sub = []

    while True:
        print("1 -\tCreazione nuovo utente")
        print("2 -\tSelezionare utente esistente")
        print("3 -\tSalvataggio su file")
        print("4 -\tCaricamento dati da file")
        print("0 -\tEsci")
        print("-"*40)

        choice = int(input("Scegliere l'operazione da effettuare:\t"))
        
        #creazione nuovo utente
        if choice == 1:

            #input dati comuni
            hourly_pay = input("Stipendio orario:\t")
            name = input("Nome:\t")
            role = input("Responsabile(R) o Subordinato(S):\t").upper()

            #creazione utente responsabile
            if role == 'R':
                registered_resp.append(responsabile.Responsabile(name, hourly_pay))
                print('Responsabile creato con successo!')

            #creazione utente subordinato
            elif role == 'S':
                #stampa lisat responsabili tra cui scegliere
                print('Lista responsabili:\t')
                for idx, resp in enumerate(registered_resp):
                    print(str(idx) + '\t-\t' + resp.name)
                idx = int(input("Inserire ID responsabile:\t"))

                #controllo sull'ID inserito
                if idx >= 0 and idx < len(registered_resp):
                    new_sub = subordinato.Subordinato(name, hourly_pay, registered_resp[idx])
                    registered_sub.append(new_sub)
                    registered_resp[idx].add_sub(new_sub)
                    print('Subordinato creato con successo')
                else:
                    print('ID responsabile non esistente')
                    continue

            #ruolo specificato non esistente, non salva nulla e torna al menu
            else:
                print("Ruolo non esistente")
                continue

        #selezione utente esistente
        elif choice == 2:

            selected_idx = -1
            selected_user = None
            role = input("Responsabile(R) o Subordinato(S):\t").upper()

            #selezione utente responsabile
            if role == 'R':
                print(' *** Responsabili *** ')
                for idx, resp in enumerate(registered_resp):
                    print(str(idx) + '\t-\t' + resp.name)
                selected_idx = int(input("Inserire ID:\t"))
                if selected_idx >= 0 and selected_idx < len(registered_resp):
                    selected_user = registered_resp[selected_idx]

            #selezione utente subordinato
            elif role == 'S':
                print(' *** Subordinati *** ')
                for idx, sub in enumerate(registered_sub):
                    print(str(idx) + '\t-\t' + sub.name)
                selected_idx = int(input("Inserire ID:\t"))
                if selected_idx >= 0 and selected_idx < len(registered_sub):
                    selected_user = registered_sub[selected_idx]

            #ruolo specificato non esistente, non salva nulla e torna al menu
            else:
                print("Ruolo non esistente")
                continue

            if isinstance(selected_user, subordinato.Subordinato):
                print("1 -\t Effettua nuova richiesta di pagamento")
                print("2 -\t Visualizza richieste di pagamento")
                print("3 -\t Conteggio ore accettate e importo dovuto")
                print("4 -\t Esportazione richieste su file di testo")
                choice = int(input("Selezionare operazione:\t"))
                if choice == 1:
                    hours = input("Inserire numero ore da richiedere:\t ")
                    selected_user.new_richiesta(hours)
                elif choice == 2:
                    print("Lista richieste effettuate: ")
                    for i, request in enumerate(selected_user.requests):
                        print(str(i) + '\t-\t' + request.status)
                elif choice == 3:
                    sum_hours = selected_user.pagamento()
                    print("Ore accettate: " + str(sum_hours) + ", pagamento dovuto: " + str(sum_hours*int(selected_user.hourly_pay)))
                elif choice == 4:
                    richiesta.Richiesta.esporta_richieste(selected_user, selected_user.name+"_requests.txt")
                    print("Esportazione completata")
                else:
                    print("Operazione non esistente")
                    continue

            elif isinstance(selected_user, responsabile.Responsabile):
                print("Lista utenti subordinati")
                for i, sub in enumerate(selected_user.subs):
                    print(str(i) + '\t-\t' + sub.name)
                choice = int(input("Selezionare ID subordinato:\t"))
                
                sub_requests = selected_user.subs[choice].requests
                for i, request in enumerate(sub_requests):
                    print(str(i) + '\t-\t' + request.status)

                request_idx = int(input("Selezionare richiesta:\t"))
                accept = input("Accettare (A) o Rifiutare(R)").upper()
                if accept == 'A':
                    selected_user.subs[choice].requests[request_idx].status = 'Accepted'
                elif accept == 'R':
                    selected_user.subs[choice].requests[request_idx].status = 'Rejected'
                else:
                    print("Operazione non esistente")
                    continue

            else:
                print('Utente non esistente')
                continue

        #salvtaggio su file
        elif choice == 3:
            save("sav", registered_resp, registered_sub)
            print("Salvataggio completato")
        
        #caricamento da file
        elif choice == 4:
            registered_user = upload("sav")
            registered_resp = registered_user[0]
            registered_sub = registered_user[1]
            print("Caricamento completato")
        
        elif choice == 0:
            break
