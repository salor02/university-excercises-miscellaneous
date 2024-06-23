#include <stdio.h>

#define N 3

void loopfusion(int a[N][N]){
    for (int i=0; i<N; i++)
        for(int j = 0; j<N; j++){
            a[j][i]=j*i;
        }
}

int main() {
    // Dichiarazione e inizializzazione della matrice a
    int a[N][N] = {{1,2,3},
                {4,5,6},
                {7,8,9}};

    loopfusion(a);

    return 0;
}

