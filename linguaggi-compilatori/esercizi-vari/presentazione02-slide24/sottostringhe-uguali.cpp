#include <string>
#include <iostream>

using namespace std;

int main(int argc, char **argv){
    if(argc > 1){
        
        string str = argv[1];

        //esclude le stringhe di lunghezza dispari dato che sarebbe impossibile avere due sottostringhe uguali
        if(str.length() % 2 != 0){
            cout<<"Rejected - Definizione non rispettata (stringa di lunghezza dispari)\n";
            return 0;
        }

        int mid = str.length()/2;

        //controlla che la prima parte della stringa sia uguale alla seconda da metà in poi
        for(int i = 0; i < mid; i++){
            if(str[i] != str[mid + i]){
                cout<<"Rejected - Definizione non rispettata\n";
                return 0;
            }
        }

        cout<<"Accepted\n";
        return 0;
        
    }

    cout<<"Missing argument!\n";
    return 1;
}