#include <string>
#include <iostream>

using namespace std;

int main(int argc, char **argv){
    if(argc > 1){
        
        string str = argv[1];

        for(int i = 0; i < str.length()/2; i++){
            if(str.begin()[i] != str.rbegin()[i]){
                cout<<"Rejected - Stringa non palindroma\n";
                return 0;
            }
        }

        cout<<"Accepted\n";
        return 0;
        
    }

    cout<<"Missing argument!\n";
    return 1;
}