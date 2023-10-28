/*
NB:
1.  Specificare nome file di input come parametro da terminale
2.  Se si vuole ottenere una stampa su file specificare nome file di output come ultimo parametro, altrimenti la stampa verrà
    effettuata su standard output
3.  Dato che l'automa viene costruito sulla base di 3 array paralleli (che non possono contenere celle vuote), 
    per definire una destinazione non esistente viene salvato come valore -1
4.  Rispetto al template di partenza ho aggiunto i metodi pubblici: parallel_push_back, parallel_set, convert_to_AFND. Quest'ultimo 
    è, ovviamente, una rielaborazione del metodo visit già implementato.
5.  Rispetto al template di partenza ho aggiunto la procedura: print_AFND
*/

#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <cstring>
#include <vector>
#include <utility>
using namespace std;

// Definizione di macro per l'output verboso
#define VERBOSE(msg) if(verbose) { cout << msg << endl; }

set<char> operators = {'*','|','.'}; // Il punto indica la concatenazione

const char EPS = '@'; //epsilon per indicare transizione con input non necessario

//struttura dati per la rappresentazione dell'automa da stampare
struct AFND{
    string symbols; //simboli dell'alfabeto (include anche epsilon in posizione 0)
    /*  i tre array di seguito sono stati implementati per realizzare la struttura dati proposta
        a lezione. l'inidice dell'array corrisponde allo stato corrente, input corrisponde al simbolo in input
        per avviare la transizione, dest1 e dest2 rappresentano le destinazioni delle transizioni. NB: dest1 non può mai 
        essere vuoto eccetto per il caso dello stato finale. Per inizializzare una cella dell'array senza dare un valore valido
        ho scelto di usare -1.*/
    vector<char> input;
    vector<int> dest1;
    vector<int> dest2;
    pair<int, int> terminal_states; //stati iniziale e finale
};

class AST {
private:
    char symb;
    AST* left;
    AST* right;
  
public:
    AST(char symb, AST* left, AST* right): symb(symb), left(left), right(right) {};
    AST(char symb): symb(symb), left(nullptr), right(nullptr) {};
    string visit() {
        /*Algoritmo di visita dell'albero in post-ordine.
        Usando l'informazione ai nodi (un operatore o un carattere dell'alfabeto)
        la visita semplicemente risostruisce la rappresentazione esterna (come stringa)
        dell'albero che a sua volta rappresenta un'espressione regolare.
        Il metodo volutamente non è generico per evitare complicazioni con il C++.
        I tre casi (foglia, un solo figlio, due figli) vengono qui tenuti
        distinti "a scopo didattico" perché (in generale) le
        operazioni da fare possono essere completamente distinte.
        */
        string treerep = "";               // treerep conterrà la rappresentazione del
                        // sottoalbero radicato nel nodo corrente
        if (left != nullptr) {
            string ltree = left->visit();    // Visitiamo il sottoalbero di sx
            if (right != nullptr) {
                // Caso di nodo con due figli
                string rtree = right->visit(); // Visitiamo il sottoalbero di dx
                treerep.push_back(symb);       // 
                treerep.push_back('(');        // 
                treerep.append(ltree);         //
                treerep.push_back(')');        // Calcoli al nodo dopo le due visite 
                treerep.push_back('(');        //  
                treerep.append(rtree);         //
                treerep.push_back(')');        //
            } 
            else {
                // Caso di nodo con un solo figlio
                treerep.push_back(symb);       //
                treerep.push_back('(');        // Calcoli al nodo dopo la visita a sx
                treerep.append(ltree);         //
                treerep.push_back(')');        //
            }
        } 
        else{
            treerep.push_back(symb);         // Calcolo nel caso di foglia
        }
        return treerep;
    }

    //procedura ausiliare per l'inserimento in coda nei vettori paralleli
    void parallel_push_back(AFND& automaND, const char input, const int dest1, const int dest2){
        automaND.input.push_back(input);
        automaND.dest1.push_back(dest1);
        automaND.dest2.push_back(dest2);
    }

    //procedura ausiliare per la modifica di un elemento all'interno dei vettori paralleli
    void parallel_set(AFND& automaND, const int idx, const char input, const int dest1, const int dest2){
        automaND.input[idx] = input;
        automaND.dest1[idx] = dest1;
        automaND.dest2[idx] = dest2;
    }

    /*  visita l'AST in postorder e restituisce l'indice di stato inziale e finale riferito agli array paralleli.
        In sostanza, questa funzione restituisce al chiamante lo stato iniziale e finale del "sottoautoma" costruito sulla base delle
        informazioni contenute nei sottoalberi di sinistra e destra. L'automa finale viene quindi costruito ricorsivamente seguendo le regole di priorità
        definite dalla visita postorder. NB: I nodi inizializzati con tutti i campi "nulli" (' ', -1, -1) rappresentano stati in attesa di essere
        collegati ad altri stati, oppure, nel caso dell'ultima ricorsione, lo stato finale.*/
    pair<int, int> convert_to_AFND(AFND &automaND){
        //vengono memorizzati l'inidice dello stato iniziale e quello dello stato finale della transizione gestita
        int start_idx, end_idx;

        if (left != nullptr) {
            pair<int, int> left_child_id = left->convert_to_AFND(automaND);    // Visitiamo il sottoalbero di sx
            if (right != nullptr) {
                // Caso di nodo con due figli (concatenazione o unione)
                pair<int, int> right_child_id = right->convert_to_AFND(automaND); // Visitiamo il sottoalbero di dx

                //concatenazione
                if(symb == '.'){
                    //collegamento stato finale ricavato dall'albero sinistro con stato iniziale ricavato dall'albero destro
                    this->parallel_set(automaND, left_child_id.second, EPS, right_child_id.first, -1);
                    start_idx = left_child_id.first;
                    end_idx = right_child_id.second;
                }

                //unione
                if(symb == '|'){
                    //nuovo stato iniziale e collegamento a stati iniziali ricavati da entrambi i sottoalberi
                    this->parallel_push_back(automaND, EPS, left_child_id.first, right_child_id.first);
                    start_idx = automaND.input.size() - 1;
                    //nuovo stato finale
                    this->parallel_push_back(automaND,' ', -1, -1);
                    end_idx = automaND.input.size() - 1;
                    //collegamento vecchio stato finale ricavato da albero sinistro con nuovo stato finale
                    this->parallel_set(automaND, left_child_id.second, EPS, end_idx, -1);
                    //collegamento vecchio stato finale ricavato da albero destro con nuovo stato finale
                    this->parallel_set(automaND, right_child_id.second, EPS, end_idx, -1);
                }
            } 
            else{
                // Caso di nodo con un solo figlio (chiusura riflessiva)

                //nuovo stato finale
                this->parallel_push_back(automaND,' ', -1, -1);
                end_idx = automaND.input.size() - 1;
                //collegamento vecchio stato finale a nuovo stato finale e a vecchio stato iniziale
                this->parallel_set(automaND, left_child_id.second, EPS, left_child_id.first, end_idx);
                //nuovo stato iniziale e collegamento a vecchio stato iniziale e a nuovo stato finale
                this->parallel_push_back(automaND, EPS, left_child_id.first, end_idx);
                start_idx = automaND.input.size() - 1;
            }
        } 
        else{
            //Nodo foglia (simbolo dell'alfabeto)
            /*  Viene inserita nella struttura dati la transizione che riconosce un singolo carattere, essa ha come
                input il carattere del nodo foglia e ha un solo stato di destinazione*/
            
            //stato di "attesa" dell'input (nuovo stato iniziale)
            this->parallel_push_back(automaND, symb, automaND.input.size()+1, -1);
            start_idx = automaND.input.size()-1;
            //stato di destinazione dopo aver ricevuto l'input (nuovo stato finale)
            this->parallel_push_back(automaND, ' ', -1, -1);
            end_idx = automaND.input.size()-1;
        }
        return make_pair(start_idx, end_idx);
    }
};

int skipblank(string S, int j) {
  /* Restituisce il primo non blank a partire dalla
     posizione j (inclusa)
  */
  while (j<S.length() and S[j] == ' ') j++;
  return j;
}

string removeblank(string S) {
  /* Restituisce la stringa S "compattata", in cui cioè
     sono rimossi tutti i caratteri "blank"
  */
  int n = S.length();
  int j = skipblank(S,0);
  string compact = "";
  while (j<n) {
    compact.push_back(S.at(j));
    j = skipblank(S,++j);
  }
  return compact;
}

int getsubtree(string S, int j) {
  /* Restituisce la lunghezza della stringa che, a partire dalla
     posizione j, bilancia correttamente le parentesi
     Si aspetta S.at(j) == '('
  */
  int s = j;  // Ricorda la starting position
  j++;
  int numpar = 1;
  while (numpar>0) {
    if (S.at(j)=='(') numpar++;
    if (S.at(j)==')') numpar--;
    j++;
  }
  return j-s;
}

AST* create(string linearrep, set<char> alphabet, bool verbose) {
  /* Costruisce la rappresentazione interna dell'AST
     corrispondente alla rappresentazione lineare linearrep.
     Posto n=lunghezza di linearrep, i casi n=1 e n=3
     indicano che linearrep rappresenta un nodo foglia.
     La differenza è che, se n=1, l'intero albero di cui
     dobbiamo costruire la rappresentazione è formato da un
     solo nodo (radice e foglia simultaneamente).
     Il caso n=3 implica invece che siamo in un passaggio ricorsivo.
     Se n>3 allora linerrep descrive un (sotto) albero composto da
     una radice, con l'operatore, e da almeno un figlio. L'algoritmo
     ricerca nella stringa la rappresentazione del figlio/dei figli,
     ricorsivamente ne costruisce la/le rappresentazione/i, e la/le usa
     per costruire l'albero da restituire al chiamante.
   */
  int n = linearrep.length();
  if (n==1 or n==3) { // linearrep = 'c' oppure "(c)"
    char c = linearrep.at(max(0,n-2));  // Unifica i due casi recuperando la lettera
    if (alphabet.count(c) == 0) throw invalid_argument("Il simbolo "+string(1,c)+" non fa parte dell'alfabeto");
    return new AST(c);
  }
  char op = linearrep.at(1);
  if (operators.count(op) == 0) throw invalid_argument("Il simbolo "+string(1,op)+" non è un operatore legale");
  int start = 2;
  int stop = getsubtree(linearrep, start);
  string left = linearrep.substr(start,stop);
  start += stop;
  if (op=='*') {  // L'albero andrà costruito con un solo figlio
    if (verbose) cout << "op:" << op << "\tleft:" << left << endl;
    AST* lt = create(left,alphabet,verbose);
    return new AST(op,lt,nullptr);
  } else {       // Operatore binario -> due figli
    stop = getsubtree(linearrep, start);
    string right = linearrep.substr(start,stop);
    if (verbose) {
      cout << "op:" << op  << "\tleft:" << left;
      cout << "\tright:" << right << endl;
    }
    AST* lt = create(left,alphabet,verbose);
    AST* rt = create(right,alphabet,verbose);
    return new AST(op,lt,rt);
  }
}

void printAFND(const AFND& automaND, ostream& os){
    
    //simboli dell'alfabeto (in posizione zero c'è epsilon che non viene stampato)
    for(int i = 1; i < automaND.symbols.size(); i++){
        os<<automaND.symbols[i]<<" ";
    }
    os<<endl;

    //stato finale
    os<<automaND.terminal_states.second<<endl;

    //stampa per debug
    /*for(int i = 0; i < automaND.input.size(); i++){
        cout<<"ID:"<<i<<"\t"<<automaND.input[i]<<"\t"<<automaND.dest1[i]<<"\t"<<automaND.dest2[i]<<endl;
    }*/

    //stampa dell'automa nel formato corretto
    for(int i = 0; i < automaND.input.size(); i++){
        for(int j = 0; j < automaND.symbols.size() ; j++){
            if(automaND.input[i] == automaND.symbols[j]){
                os<<automaND.dest1[i];
                if(automaND.dest2[i] != -1){
                    os<<" "<<automaND.dest2[i];
                }
            }
            os<<endl;
        }
    }
}

int main(int argc, char** argv) {
    bool verbose = false;
    string fn;
    if (argc==1) {
        cout << "Missing file name\n";
        return 1;
    } else if (strcmp(argv[1],"-v")==0) {
        verbose = true;
        fn = argv[2];
    } else {
        fn = argv[1];
    }
    set<char> symbols;           // Conterrà i simboli dell'alfabeto
    ifstream f(fn);              // Apriamo il file con alfabeto e con la rapp. esterna dell'albero
    string line;                 // La prima riga deve contenere l'alfabeto
    getline(f, line);
    
    stringstream alphabet(line); // Ci prepariamo per scan e tokenizzazionea
    char symb;
    while(alphabet>>symb) {      // Se il primo parametro è un oggetto di tipo
                                // stringstream, getline è di fatto uno scanner
            symbols.insert(symb);
    }

    getline(f, line);           // La seconda riga contiene la rappresentazione
                                // esterna dell'AST che, a sua volta, rappresenta
                                // l'espressione regolare.
                                // Per semplicità, non contempliamo l'uso di
                                // espressioni regolari che includono epsilom

    AST *root = create(removeblank(line), symbols, verbose);
    //cout << root->visit() << endl; // Verifichiamo di aver internamente ricostruito l'AST
                                    // in modo corretto

    //CONVERSIONE
    AFND automaND;

    //conversione dell'alfabeto da set a stringa perchè più comodo nella fase di stampa
    automaND.symbols = EPS;
    for(char symb:symbols){
        automaND.symbols += symb;
    }

    VERBOSE("Starting conversion...");                                
    automaND.terminal_states = root->convert_to_AFND(automaND);
    VERBOSE("Conversion completed!");

    //STAMPA DEL RISULTATO DELLA CONVERSIONE
    if((verbose && argc == 4) || (!verbose && argc == 3)){
        string file_name = argv[argc - 1];
        VERBOSE("Trying to open file -- " + file_name);
        ofstream f;
        f.open(file_name);

        VERBOSE("Saving on file '"<<file_name<<"' ...");
        printAFND(automaND, f);
        f.close();
    }
    else{
        VERBOSE("No output file provided, will write result on default output stream");
        printAFND(automaND, cout);
    }

}
