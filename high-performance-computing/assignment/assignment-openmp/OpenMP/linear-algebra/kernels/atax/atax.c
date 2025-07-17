#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "atax.h"

#include <omp.h>
/* Funzione di inizializzazione degli array. */
static void init_array(int nx, int ny,
                       DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), // Matrice A di dimensione nx x ny
                       DATA_TYPE POLYBENCH_1D(x, NY, ny))         // Vettore x di dimensione ny
{
  int i, j;

  // Inizializza il vettore x con valori che vanno da 0 a (ny-1) moltiplicati per PI
  for (i = 0; i < ny; i++)
    x[i] = i * M_PI; // Ogni elemento di x è il suo indice moltiplicato per PI

  // Inizializza la matrice A con valori calcolati tramite la formula (i * (j +  1)) / nx
  // i: indice di riga, j: indice di colonna
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      A[i][j] = ((DATA_TYPE)i * (j + 1)) / nx; // Imposta A[i][j] come un valore normalizzato
}

/* Funzione per stampare l'array y (output del calcolo). */
static void print_array(int nx,
                        DATA_TYPE POLYBENCH_1D(y, NX, nx)) // Vettore y di dimensione nx
{
  int i;

  // Stampa i valori dell'array y, uno alla volta
  for (i = 0; i < nx; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]); // Stampa un valore di y usando un formato specifico
    if (i % 20 == 0)                             // Aggiunge una nuova riga ogni 20 valori per migliorare la leggibilità
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

/* Funzione principale di calcolo, che implementa l'algoritmo ATAx. */
static void kernel_atax(int nx, int ny,
                        DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), // Matrice A di dimensione nx x ny
                        DATA_TYPE POLYBENCH_1D(x, NY, ny),         // Vettore x di dimensione ny
                        DATA_TYPE POLYBENCH_1D(y, NY, ny),         // Vettore y di dimensione ny (output)
                        DATA_TYPE POLYBENCH_1D(tmp, NX, nx))       // Vettore temporaneo tmp di dimensione nx
{

  //___________SOLUZIONE SEQUENZIALE___________

#if defined SEQUENTIAL

  // Inizializza l'array y a zero
  for (int i = 0; i < _PB_NY; i++)
    y[i] = 0;

  // Calcola il prodotto matrice-vettore: A * x
  // tmp[i] conterrà il risultato della moltiplicazione della riga i della matrice A per il vettore x
  for (int i = 0; i < _PB_NX; i++) // Ciclo sulle righe della matrice A
  {
    tmp[i] = 0;                         // Inizializza tmp[i] a zero
    for (int j = 0; j < _PB_NY; j++)        // Ciclo sulle colonne della matrice A (lunghezza di x)
    	tmp[i] = tmp[i] + A[i][j] * x[j]; // Somma il prodotto A[i][j] * x[j] nel vettore tmp[i]

    // Ora aggiorna il vettore y con il risultato della moltiplicazione riga di A * tmp
    for (int j = 0; j < _PB_NY; j++)
    	y[j] = y[j] + A[i][j] * tmp[i]; // Somma il prodotto A[i][j] * tmp[i] nel vettore y
  }

// _________SOLUZIONE PARALLEL FOR___________
#elif defined PARALLEL
// Inizializza l'array y a zero
#pragma omp parallel for
for (int i = 0; i < _PB_NY; i++)
	y[i] = 0;

// Calcolo parallelo di tmp[i]
#pragma omp parallel for
for (int i = 0; i < _PB_NX; i++)
{
tmp[i] = 0;
for (int j = 0; j < _PB_NY; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
}

// Aggiornamento parallelo di y[j] DA NOTARE IL CAMBIO DI VARIABILI D'ITERAZIONE
#pragma omp parallel for
for (int j = 0; j < _PB_NY; j++)
{
for (int i = 0; i < _PB_NX; i++)
    y[j] = y[j] + A[i][j] * tmp[i];
}

#elif defined PARALLEL2
#pragma omp parallel
{
    #pragma omp for schedule(static)
    for (int i = 0; i < _PB_NY; i++)
        y[i] = 0;

    #pragma omp barrier

    #pragma omp for schedule(static)
    for (int i = 0; i < _PB_NX; i++){
        tmp[i] = 0;

        for (int j = 0; j < _PB_NY; j++)      
            tmp[i] += A[i][j] * x[j]; 
    }
    int TILE = 32;

    double y_private[_PB_NY];// Copia privata per thread
    memset(y_private, 0, sizeof(double) * _PB_NY);

    #pragma omp for schedule(static)
    for(int i = 0; i < _PB_NX; i++){
        for(int jj = 0; jj < _PB_NY; jj+=TILE){
            int jmax = (jj+TILE < _PB_NY) ? (jj+TILE) : _PB_NY;
            for(int j = jj; j < jmax; j++){
                y_private[j] += A[i][j] * tmp[i];
            }
        }
    }

    #pragma omp critical
    for(int j = 0; j < _PB_NX; j++){
        y[j] += y_private[j];
    }
}


 
// _________SOLUZIONE TARGET___________
#elif defined TARGET

  #pragma omp target data map(to: A[0:nx][0:ny], x[0:ny]) \
                          map(tofrom: y[0:ny]) \
                          map(alloc: tmp[0:nx])
  {
    // Inizializza l'array y a zero
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < ny; i++)
      y[i] = 0;

    // Calcola il prodotto matrice-vettore: A * x
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < _PB_NX; i++) {
      tmp[i] = 0;
      for (int j = 0; j < _PB_NY; j++)
        tmp[i] += A[i][j] * x[j];
    }

    // Ora aggiorna il vettore y con il risultato della moltiplicazione riga di A * tmp
    #pragma omp target teams distribute parallel for
    for (int j = 0; j < _PB_NY; j++) {
      for (int i = 0; i < _PB_NX; i++) {
        y[j] += A[i][j] * tmp[i];
      }
    }
  }

  #endif
}


void print_matrix_stderr(double** A, int nx, int ny) {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            fprintf(stderr, "%.2f ", A[i][j]);
            if ((j + 1) % 20 == 0) // Vai a capo ogni 20 colonne
                fprintf(stderr, "\n");
        }
        if (ny % 20 != 0)
            fprintf(stderr, "\n"); // Vai a capo a fine riga se necessario
    }
}


int main(int argc, char **argv)
{
  /* Recupera la dimensione del problema (in questo caso, NX e NY sono predefiniti). */
  int nx = NX; // Numero di righe della matrice A
  int ny = NY; // Numero di colonne della matrice A

  /* Dichiarazione e allocazione degli array necessari. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny); // Matrice A
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);         // Vettore x
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);         // Vettore y (output)
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);       // Vettore temporaneo tmp

  
  /* Inizializza gli array con i dati appropriati. */
  init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Avvia il timer per misurare il tempo di esecuzione del calcolo. */
  polybench_start_instruments;

  /* Esegui il kernel ATAx (calcolo principale). */
  kernel_atax(nx, ny,
              POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(x),
              POLYBENCH_ARRAY(y),
              POLYBENCH_ARRAY(tmp));

  /* Ferma il timer e stampa i risultati delle misurazioni. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Previene l'eliminazione del codice morto (DCE) e stampa i risultati finali di y. */
  polybench_prevent_dce(print_array(nx, POLYBENCH_ARRAY(y)));

  /* Dealloca la memoria per gli array per evitare perdite di memoria. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}