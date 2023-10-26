/*
TODO: prova con altri automi

NB:
1.  Specificare nome file di input come parametro da terminale
2.  Se si vuole ottenere una stampa su file specificare nome file di output come secondo parametro, altrimenti la stampa verrà
    effettuata su standard output
3.  Il programma funziona solo se gli stati dell'automa non deterministico in input sono numeri interi 
    in sequenza da 0 a N. Per supportare numeri interi non sequenziale sarebbe stato necessario inserire una lista
    all'interno della struct AFND.
4.  Il programma funziona solo se l'alfabeto dei possibili simboli di input è rappresentabile tramite char, in caso di alfabeto costituito
    da stringhe il programma crusherebbe.
*/

#include <iostream>
#include <utility>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

const char EPS = '@'; //epsilon per indicare transizione con input non necessario

//struttura rappresentante automi finiti non deterministici
struct AFND{
    map<pair<int, char>, set<int>> data; //"tabella" delle transizioni cioè (stato, input) -> stato_dest
    string alpha; //alfabeto contenuto in un array di char, cioe una stringa
    int state_number; //numero di stati, adatta in questo caso solo al salvataggio di stati interi sequenziali
    set<int> final_state; //set degli stati finali
};

//struttura rappresentante automi finiti deterministici
struct AFD{
    map<pair<int, char>, int> data; /*"tabella" delle transizioni cioè (stato, input) -> stato_dest (il numero intero di ogni stato in questo caso rappresenta
                                        un set di stati provenienti dall'automa non deterministico)*/
    string alpha;   //alfabeto contenuto in un array di char, cioe una stringa
    vector<set<int>> state_list; //lista degli stati, in cui ogni stato corrisponde ad un set di interi corrispondenti a stati dell'automa ND
    set<int> final_state; //set degli stati finali
};

//scansione dello stream di input e conseguente salvataggio dell'automaND su struttura dati
void upload_AFND(AFND &automaND, ifstream &f){

    string line;

    //lettura simboli dell'alfabeto
    getline(f, line);
    line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
    automaND.alpha = EPS + line;
    
    //lettura degli stati finali
    getline(f, line);

    istringstream iss(line);
    int final_state;

    while(iss>>final_state){
        automaND.final_state.insert(final_state);
    }

    //costruzione della struttura dati
    int stato;
    char input;
    int dest_state;

    for(int i = 0; getline(f, line); i++){
        //definizione stato e input corrente in base al numero di riga e alla cardinalità dell'alfabeto
        stato = i / automaND.alpha.size(); 
        input = automaND.alpha[i % automaND.alpha.size()];

        //utile per la lettura degli stati di destinazione
        istringstream iss(line);

        //costituenti della map
        pair<int, char> transition(stato, input);

        set<int> dest;
        while(iss>>dest_state){
            dest.insert(dest_state);
        }

        //assegnazione risultato lettura linea corrente alla struttura dati
        automaND.data[transition] = dest;
    }

    //assegnazione numero di stati trovati
    automaND.state_number = stato;

    //stampa debug
    /*for(auto item:automaND.data){
        cout<<item.first.first;
        cout<<'\t';
        cout<<item.first.second;
        cout<<'\t';
        for(auto dest:item.second){
            cout<<dest;
            cout<<' ';
        }
        cout<<endl;
    }*/
}

//cerca lo stato iniziale dell'automa, da cui dovrà partire poi la conversione in automa deterministico
bool get_initial_state(const AFND &automaND, int &initial_state){
    
    //creazione di un set degli stati
    set<int> state;
    for(int i = 0; i < automaND.state_number; i++){
        state.insert(i);
    }

    //stato iniziale ottenuto come differenza tra il set degli stati e gli stati di destinazione (in uno stato iniziale non arriva nessuna transizione)
    for(auto item:automaND.data){
        for(auto d_state:item.second){
            state.erase(d_state);
        }
    }

    //se tutto è andato bene il set dovrebbe avere un solo elemento corrisponendente proprio allo stato inziale
    if(state.size() == 1){
        for(auto item:state){
            initial_state = item;
            return true;
        }
    }

    return false;
}

/*"esplora" tutte le epsilon transizioni a partire dallo stato passato come parametro, serve a costruire in maniera 
ricorsiva gli stati dell'automa deterministico come insieme di stati dell'automa non deterministico*/
void explore_eps(const AFND &automaND, int initial_state, set<int> &state){
    
    //inserisce nello stato dell'automaD lo stato da cui inizia la ricerca
    state.insert(initial_state);
    
    //costruisce un set formato da tutte le destinazioni raggiungibili dallo stato iniziale tramite un'epsilon transizione
    set<int> epsilon_set = automaND.data.at(make_pair(initial_state, EPS));

    //nessuna destinazione raggiungibile tramite EPS per questo punto iniziale
    if(epsilon_set.size() < 1) return;

    //richiama la funzione su tutte le nuove destinazioni trovate, finchè non si trova uno stato da cui non parte nessuna eps-transizione
    for(auto item:epsilon_set){
        explore_eps(automaND, item, state);
    }
}

//applica il vero e proprio algoritmo di conversione utilizzando anche la funzione explore_eps
bool conversion(const AFND &automaND, AFD &automaD, const int initial_state){

    //i due alfabeti dei simboli sono uguali eccetto che per la epsilon
    automaD.alpha = automaND.alpha;
    automaD.alpha.erase(0,1);
    
    //coda degli stati da analizzare
    queue<set<int>> stateD_queue;

    //set di appoggio per gestire gli stati
    set<int> current_state; //stato analizzato correntemente dal ciclo di seguito
    set<int> discovered_state; //stato scoperto percorrendo l'automa ND

    //costruzione dello stato iniziale dell'automa D
    explore_eps(automaND, initial_state, discovered_state);
    stateD_queue.push(discovered_state);
    automaD.state_list.push_back(discovered_state);

    //ciclo fino a che la coda degli stati da analizzare non si svuota
    while(!stateD_queue.empty()){
        current_state = stateD_queue.front();
        
        //vengono analizzate tutte le transizioni di tutti gli elementi di current_state
        for(int i = 0; i < automaD.alpha.size(); i++){
            discovered_state.clear();
            char current_input = automaD.alpha[i];

            for(auto item:current_state){
                set<int> current_output = automaND.data.at(make_pair(item, current_input));
                for(auto output:current_output){
                    /*viene eseguita questa procedura su tutti gli stati dell'automa ND in modo tale da creare un nuovo stato 
                    per l'automaD che deve comprendere, quindi, anche le epsilon transizioni (analizzate ricorsivamente in questo caso)*/
                    explore_eps(automaND, output, discovered_state);
                }
            }

            /*lo stato scoperto viene aggiunto alla coda solo se non è gia presente nella lista degli stati dell'automa D (marked) oppure se è uno stato vuoto
            questo implica che la transizione analizzata non esiste*/
            auto discovered_itr = find(automaD.state_list.begin(), automaD.state_list.end(), discovered_state);
            int discovered_output;//identifica lo stato di arrivo della transizione

            if(discovered_state.size() > 0){
                //stato scoperto non presente nella lista degli stati trovati
                if(discovered_itr == automaD.state_list.end()){
                    stateD_queue.push(discovered_state);
                    automaD.state_list.push_back(discovered_state);
                    discovered_output = automaD.state_list.size() - 1;
                }
                else{
                    discovered_output = distance(automaD.state_list.begin(), discovered_itr);
                }

                /*viene registrata una transizione dallo stato corrente allo stato discovered. Se quest'ultimo è già presente nella lista degli stati
                verrà cercato e ottenuto il suo indice altrimenti viene preso come indice la size della lista - 1 dato che è sicuramente l'ultimo stato inserito*/
                auto current_itr = find(automaD.state_list.begin(), automaD.state_list.end(), current_state);
                automaD.data[make_pair(distance(automaD.state_list.begin(), current_itr), current_input)] = discovered_output;  
            }
        }

        //dequeue dello stato esaminato
        stateD_queue.pop();
    }

    //definizione stati finali dell'automa D
    for(int i = 0; i < automaD.state_list.size(); i++){
        for(auto state:automaD.state_list[i]){
            /*se almeno uno degli elementi dello stato D appartiene al set degli stati finali di ND allora
            lo stato D diventerà uno stato finale di D*/
            if(automaND.final_state.find(state) != automaND.final_state.end()){
                automaD.final_state.insert(i);
            }
        }
    }

    return true;
}

//stampa l'automa deterministico appena costurito sullo stream passato come parametro
void printAFD(const AFD &automaD, ostream &os){
    //stampa degli stati
    for(auto state:automaD.state_list){
        for(auto state_item:state){
            os<<state_item<<" ";
        }
        os<<endl;
    }

    //stampa degli stati finali
    for(auto state:automaD.final_state){
        os<<state<<" ";
    }
    os<<endl;

    //stampa delle transizioni
    for(int state = 0; state < automaD.state_list.size(); state++){
        for(auto input:automaD.alpha){

            //blocco try catch utilizzato perche at() solleva un'eccezione se la chiave fornita non è a associata a nessun valore
            try{
                os<<automaD.data.at(make_pair(state,input))<<endl;
            }
            catch(out_of_range exception){
                os<<endl;
            }
        }
    }
}

int main(int argc, char *argv[]){

    AFND automaND;
    AFD automaD;

    if(argc > 1){

        //INPUT FILE OPENING
        string file_name = argv[1];
        cout<<"Trying to open file -- " + file_name<<endl;
        ifstream f;
        f.open(file_name);
        if(!f){
            cout<<"File '" + file_name + "' not found!\n";
            return 1;
        }
        cout<<"File '" + file_name + "' opened successfully!\n";

        //UPLOAD AFND FROM FILE
        upload_AFND(automaND, f);

        f.close();

        int initial_state;
        if(!get_initial_state(automaND, initial_state)){
            cout<<"No starting point has been found\n";
            return 1;
        };

        cout<<"Initial state:\t"<<initial_state<<endl;

        //CONVERSIONE
        cout<<"Starting conversion...\n";
        if(!conversion(automaND, automaD, initial_state)){
            cout<<"An error occurred during conversion from ND to D\n";
            return 1;
        }

        cout<<"Conversion completed!\n";

        //STAMPA SU FILE
        if(argc > 2){
            string file_name = argv[2];
            cout<<"Trying to open file -- " + file_name<<endl;
            ofstream f;
            f.open(file_name);

            cout<<"Saving on file '"<<file_name<<"' ...\n";
            printAFD(automaD, f);
        }
        else{
            cout<<"No output file provided, will write result on default output stream\n";
            printAFD(automaD, cout);
        }
        

        cout<<"Save complete!\n";
        return 0;
    }

    cout<<"No file has been provided\n";
    return 1;
}