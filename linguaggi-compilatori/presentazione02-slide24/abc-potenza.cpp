#include <iostream>
#include <cstring>
#include <set>

using namespace std;

int main(int argc, char **argv){
    set<char> S = {'a','b','c'};

    if(argc > 1){

        char* str = argv[1];

        for(int i = 0; i < strlen(str); i++){

            if(S.count(str[i]) == 0){
                cout<<"Rejected - Definzione non rispettata\n";
                return 0;
            }

            //se si presenta una b toglie dal set l'a perchè non si può più presentare
            if(str[i] == 'b') 
                S.erase('a');

            //se si presenta una c toglie dal set a e b perchè non si possono più presentare
            if(str[i] == 'c'){
                S.erase('a');
                S.erase('b');
            }
        }

        cout<<"Accepted\n";
        return 0;
    }

    cout<<"Missing argument!\n";
    return 1;
    
}
