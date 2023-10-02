#include <cstring>
#include <iostream>

using namespace std;

int main(int argc, char **argv){
    if(argc > 1){
        char* str = argv[1];

        for(int i = 0; i < strlen(str) - 1; i++){

            //reject per stringa non appartenente all'alfabeto
            if(str[i] != '0' && str[i] != '1'){
                cout<<"Rejected - Stringa non appartenente all'alfabeto\n";
                return 0;
            }

            //reject per definizione non rispettata
            if(str[i] == '0' && str[i+1] != '1'){
                cout<<"Rejected - Definizione non rispettata\n";
                return 0;
            }
        }

        //reject per definzione non rispettata (ultimo carattere è zero)
        if(str[strlen(str) - 1] == '0'){
            cout<<"Rejected - Definzione non rispettata (ultimo carattere è zero)\n";
            return 0;
        }
        else{
            cout<<"Accepted\n";
            return 0;
        }
        
    }

    cout<<"Missing argument!\n";
    return 1;
}