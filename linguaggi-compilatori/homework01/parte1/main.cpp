/*
0. trovare stato iniziale altrimenti non so da dove iniziare
1. esaminare input in ordine
2. costruire set degli stati del D seguendo ricorsivamente epsilon
3. una volta costurito il primo set fare punto 2 per ogni simbolo dell'alfabeto
4. salvare ogni nuovo stato in un set e ogni transazione in un map<set,char> = set

TODO: leggere alfabeto anche non char e capire perche mi legge una riga in meno

NB:
1.  Il programma funziona solo se gli stati dell'automa non deterministico in input sono numeri interi 
    in sequenza da 0 a N. Per supportare numeri interi non sequenziale sarebbe stato necessario inserire una lista
    all'interno della struct AFND.
*/

#include <iostream>
#include <bits/stdc++.h>
#include <utility>
#include <set>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <time.h>
#include <unistd.h>

using namespace std;

const char EPS = '@'; //epsilon per indicare transizione con input non necessario

struct AFND{
    map<pair<int, char>, set<int>> data;
    string alpha;
    int state_number;
    int final_state_number; //non utilizzato
};

struct AFD{
    map<pair<set<int>, char>, int> data;
    string alpha;
    vector<set<int>> state_list;
    int final_state_number; //non utilizzato
};

void upload_AFND(AFND &automaND, ifstream &f){

    string line;

    //lettura simboli dell'alfabeto
    getline(f, line);
    line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
    automaND.alpha = EPS + line;
    
    //lettura del numero degli stati finali, inutile al momento
    getline(f, line);

    //costruzione della struttura dati
    int stato;
    char input;
    string dest_state;

    for(int i = 0; getline(f, line); i++){
        //definizione stato e input corrente in base al numero di riga e alla cardinalità dell'alfabeto
        stato = i / automaND.alpha.size(); 
        input = automaND.alpha[i % automaND.alpha.size()];

        //utile per la lettura degli stati di destinazione
        istringstream iss(line);

        //costituenti della map
        pair<int, char> transition(stato, input);

        set<int> dest;
        while(getline(iss, dest_state, ' ')){
            dest.insert(stoi(dest_state));
        }

        //assegnazione risultato lettura linea corrente alla struttura dati
        automaND.data[transition] = dest;
    }

    //assegnazione numero di stati trovati
    automaND.state_number = stato;

    //stampa
    for(auto item:automaND.data){
        cout<<item.first.first;
        cout<<'\t';
        cout<<item.first.second;
        cout<<'\t';
        for(auto dest:item.second){
            cout<<dest;
            cout<<' ';
        }
        cout<<endl;
    }
}

bool get_initial_state(AFND &automaND, int &initial_state){
    
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
            return true;
        }
    }

    return false;
}

void explore_eps(AFND &automaND, int initial_state, set<int> &state){
    
    state.insert(initial_state);
    
    set<int> epsilon_set = automaND.data[make_pair(initial_state, EPS)];

    if(epsilon_set.size() < 1) return; //nessuna destinazione raggiungibile tramite EPS per questo punto iniziale

    for(auto item:epsilon_set){
        explore_eps(automaND, item, state);
    }
}

bool conversion(AFND &automaND, AFD &automaD, int initial_state){

    automaD.alpha = automaND.alpha;
    
    //crea set con tutti gli stati raggiungibili senza input (epsilon)
    queue<set<int>> stateD_queue;

    set<int> current_state;
    set<int> discovered_state;
    explore_eps(automaND, initial_state, current_state);
    stateD_queue.push(current_state);

    while(!stateD_queue.empty()){
        current_state = stateD_queue.front();
        

        for(int i = 1; i < automaD.alpha.size(); i++){
            char current_input = automaD.alpha[i];
            for(auto item:current_state){
                set<int> current_output = automaND.data[make_pair(item, current_input)];
                for(auto output:current_output){
                    explore_eps(automaND, output, discovered_state);
                }
            }
            for(auto item:discovered_state) cout<<item<<" ";
            cout<<endl;
            if(find(automaD.state_list.begin(), automaD.state_list.end(), discovered_state) == automaD.state_list.end() && discovered_state.size() > 0){
                stateD_queue.push(discovered_state);
                automaD.state_list.push_back(discovered_state);
            }

            discovered_state.clear();
                
        }
        cout<<stateD_queue.size()<<endl;
        //sleep(1);
        stateD_queue.pop();
        
    }

    cout<<"Discovered states number: "<<automaD.state_list.size()<<endl;
    for(auto state:automaD.state_list){
        for(auto state_item:state){
            cout<<state_item<<" ";
        }
        cout<<endl;
    }
    return true;
}

int main(int argc, char *argv[]){

    AFND automaND;
    AFD automaD;

    if(argc > 1){

        //FILE OPENING
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
        return 0;
    }

    cout<<"No file has been provided\n";
    return 1;
}