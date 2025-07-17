#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2000  // Matrici 4096x4096: ~67 milioni di elementi

int main() {
    double *A = (double*) malloc(N*N * sizeof(double));
    double *B = (double*) malloc(N*N * sizeof(double));
    double *C_cpu = (double*) malloc(N*N * sizeof(double));
    double *C_gpu = (double*) malloc(N*N * sizeof(double));
    int i, j, k;
    double t_cpu, t_gpu;

    // Inizializzazione matrici
    #pragma omp parallel for collapse(2)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i*N + j] = (double)(i + j);
            B[i*N + j] = (double)(i - j);
        }

    // --- Calcolo su CPU ---
    t_cpu = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C_cpu[i*N + j] = sum;
        }
    t_cpu = omp_get_wtime() - t_cpu;

    // --- Calcolo su GPU ---
    t_gpu = omp_get_wtime();
    #pragma omp target teams distribute parallel for simd collapse(2) \
        map(to:A[0:N*N], B[0:N*N]) map(from:C_gpu[0:N*N])
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C_gpu[i*N + j] = sum;
        }
    t_gpu = omp_get_wtime() - t_gpu;

    // Verifica correttezza (controlla solo alcuni elementi)
    int err = 0;
    for (i = 0; i < N; i += N/10)
        for (j = 0; j < N; j += N/10)
            if (C_cpu[i*N + j] != C_gpu[i*N + j])
                err = 1;

    printf("Tempo CPU: %f secondi\n", t_cpu);
    printf("Tempo GPU: %f secondi\n", t_gpu);
    if (!err)
        printf("Risultato: OK, i risultati coincidono\n");
    else
        printf("Risultato: ERRORE nei risultati\n");

    free(A); free(B); free(C_cpu); free(C_gpu);
    return 0;
}
