#include <cstring>
#include <cmath>
#include <iostream>

using namespace std;

int main(int argc, char **argv){
    if(argc > 1){
        
        char* str = argv[1];
        int ones_count = 0;

        for(int i = 0; i < strlen(str); i++){

            //reject per stringa non appartenente all'alfabeto
            if(str[i] != '0' && str[i] != '1'){
                cout<<"Rejected - Stringa non appartenente all'alfabeto\n";
                return 0;
            }

            if(str[i] == '1')
                ones_count++;
        }

        int min_ones = ceil(strlen(str) / 2.0);

        if(ones_count >= min_ones){
            cout<<"Accepted\n";
            return 0;
        }
        else{
            cout<<"Rejected - numero di 1 insufficiente\n";
            return 0;
        }
        
        
    }

    cout<<"Missing argument!\n";
    return 1;
}