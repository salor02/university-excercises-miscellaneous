#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "atax.h"

//print array per debug
static void print_array(int ny, DATA_TYPE* y)
{
    for (int i = 0; i < ny; i++){
        fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]);
        if ((i+1) % 20 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

//inizializzazione A e x
static void init_array(int nx, int ny, DATA_TYPE* A, DATA_TYPE* x)         // Vettore x di dimensione ny
{
  int i, j;

    // Inizializza il vettore x con valori che vanno da 0 a (ny-1) moltiplicati per PI
    for (i = 0; i < ny; i++)
        x[i] = i * M_PI; // Ogni elemento di x è il suo indice moltiplicato per PI

    // Inizializza la matrice A con valori calcolati tramite la formula (i * (j +  1)) / nx
    // i: indice di riga, j: indice di colonna
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            A[i*ny + j] = ((DATA_TYPE)i * (j + 1)) / nx; // Imposta A[i][j] come un valore normalizzato
}

#ifdef SHARED
const int TILE_DIM = 16;

__global__ void atax_kernel_tmp(DATA_TYPE* A, DATA_TYPE* x, 
    DATA_TYPE* tmp, int nx, int ny) {
    
    //viene allocato in shmem lo spazio per una tile della matrice A. Ha una colonna in più in modo da ridurre bank conflict
    __shared__ DATA_TYPE smem_A[TILE_DIM][TILE_DIM+1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //questa è la row di A che il thread si deve occupare di moltiplicare per x
    int row = blockIdx.y * blockDim.y + ty;
    
    //somma dei risultati parziali, calcolati per ogni tile, alla fine conterrà il valore di tmp[row]
    DATA_TYPE sum = 0.0;
    
    //questo for divide A in tile orizzontalmente
    for (int tile = 0; tile < (ny + TILE_DIM - 1) / TILE_DIM; tile++) {
        int col_start = tile * TILE_DIM;
        
        //ogni thread contribuisce a caricare shmem. Ogni thread accede ad una colonna diversa
        if (row < nx && (col_start + tx) < ny) {
            smem_A[ty][tx] = A[row * ny + col_start + tx];
        } else {
            smem_A[ty][tx] = 0.0;
        }
        __syncthreads();
        
        //ogni thread calcola il contributo a tmp[row] e accede alla shared mem per riga
        if (row < nx) {
            for (int k = 0; k < TILE_DIM && (col_start + k) < ny; k++) {
                sum += smem_A[ty][k] * x[col_start + k];
            }
        }
        __syncthreads();
    }
    
    //aggiornamento valore tmp completamente calcolato
    if (row < nx) {
        tmp[row] = sum;
    }
}
#else
//calcolo di tmp, ogni thread moltiplica un'intera riga di A per tutto il vettore x, calcolando quindi interamente una cella di tmp
__global__ void atax_kernel_tmp(DATA_TYPE* A, DATA_TYPE* x, 
    DATA_TYPE* tmp, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    DATA_TYPE sum = 0.0;

    if (i < nx) {
        for (int j = 0; j < ny; j++) {
            sum += A[i * ny + j] * x[j];
        }
        tmp[i] = sum;
    }
}
#endif

//calcolo di y, ogni thread moltiplica un'intera colonna di A per tutto il vettore tmp, calcolando quindi interamente una cella di y
__global__ void atax_kernel_y(DATA_TYPE* A, DATA_TYPE* tmp, 
    DATA_TYPE* y, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    DATA_TYPE sum = 0.0;
    
    if (j < ny) {
        for (int i = 0; i < nx; i++) {
            sum += A[i * ny + j] * tmp[i];
        }
        y[j] = sum;
    }
}


int main(int argc, char **argv)
{
    int nx = NX;
    int ny = NY;

    //allocazione array e matrice A, tutti su memoria pinned
    DATA_TYPE *A, *x, *y, *tmp;
    cudaMallocHost((void**)&A, nx * ny * sizeof(DATA_TYPE));
    cudaMallocHost((void**)&x, ny * sizeof(DATA_TYPE));  
    cudaMallocHost((void**)&y, ny * sizeof(DATA_TYPE));
    cudaMallocHost((void**)&tmp, nx * sizeof(DATA_TYPE));   

    //inizializzazione array
    init_array(nx, ny, A, x);
    memset(y,   0, ny * sizeof(DATA_TYPE));
    memset(tmp, 0, nx * sizeof(DATA_TYPE));

    //registrazione eventi per time benchmark
    cudaEvent_t start, 
                stop, 
                startKernel_tmp, 
                stopKernel_tmp, 
                startKernel_y, 
                stopKernel_y, 
                startMemHostToDevice, 
                stopMemHostToDevice, 
                startMemDeviceToHost, 
                stopMemDeviceToHost;
    cudaEventCreate(&startKernel_tmp);
    cudaEventCreate(&stopKernel_tmp);
    cudaEventCreate(&startKernel_y);
    cudaEventCreate(&stopKernel_y);
    cudaEventCreate(&startMemHostToDevice);
    cudaEventCreate(&stopMemHostToDevice);
    cudaEventCreate(&startMemDeviceToHost);
    cudaEventCreate(&stopMemDeviceToHost);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //*******INIZIO SEZIONE OFFLOAD ********/
    cudaEventRecord(start);

    //******* TRASFERIMENTO HTD ********/
    cudaEventRecord(startMemHostToDevice);
    DATA_TYPE *d_A, *d_x, *d_y, *d_tmp;
    cudaMalloc((void**)&d_A, nx * ny * sizeof(DATA_TYPE));
    cudaMalloc((void**)&d_x, ny * sizeof(DATA_TYPE));
    cudaMalloc((void**)&d_y, ny * sizeof(DATA_TYPE));
    cudaMalloc((void**)&d_tmp, nx * sizeof(DATA_TYPE));

    cudaMemcpy(d_A, A, nx * ny * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, ny * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaEventRecord(stopMemHostToDevice);

    //******* ESECUZIONE KERNEL ********/
    #ifdef SHARED
        dim3 block_size_tmp(16,16);
        dim3 grid_size_tmp((ny + block_size_tmp.x - 1) / block_size_tmp.x, (nx + block_size_tmp.y) / block_size_tmp.y);
    #else
        dim3 block_size_tmp(256);
        dim3 grid_size_tmp((ny + block_size_tmp.x - 1) / block_size_tmp.x);
    #endif

    dim3 block_size_y(256);
    dim3 grid_size_y((ny + block_size_y.x - 1) / block_size_y.x);

    cudaEventRecord(startKernel_tmp);
    atax_kernel_tmp<<<grid_size_tmp, block_size_tmp>>>(d_A, d_x, d_tmp, nx, ny);
    cudaDeviceSynchronize();
    cudaEventRecord(stopKernel_tmp);

    cudaEventRecord(startKernel_y);
    atax_kernel_y<<<grid_size_y, block_size_y>>>(d_A, d_tmp, d_y, nx, ny);
    cudaDeviceSynchronize();
    cudaEventRecord(stopKernel_y);
    
    //*******TRASFERIMENTO DTH ********/
    cudaEventRecord(startMemDeviceToHost);
    cudaMemcpy(y, d_y, ny * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopMemDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //*******FINE ESECUZIONE ********/


    //******* BENCHMARK ********/

    print_array(ny, y);

    float total_time = 0, mem_HtD_time = 0, mem_DtH_time = 0, kernel_tmp_time = 0, kernel_y_time = 0;

    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&mem_HtD_time, startMemHostToDevice, stopMemHostToDevice);
    cudaEventElapsedTime(&mem_DtH_time, startMemDeviceToHost, stopMemDeviceToHost);
    cudaEventElapsedTime(&kernel_tmp_time, startKernel_tmp, stopKernel_tmp);
    cudaEventElapsedTime(&kernel_y_time, startKernel_y, stopKernel_y);

    printf("total_time: %f ms\n", total_time);
    printf("mem_HtD_time: %f %%\n", (mem_HtD_time/total_time)*100);
    printf("mem_DtH_time: %f %%\n", (mem_DtH_time/total_time)*100);
    printf("kernel_tmp_time: %f %%\n", (kernel_tmp_time/total_time)*100);
    printf("kernel_y_time: %f %%\n", (kernel_y_time/total_time)*100);

    //******* FREE DELLE RISORSE ********/
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startKernel_tmp);
    cudaEventDestroy(stopKernel_tmp);
    cudaEventDestroy(startKernel_y);
    cudaEventDestroy(stopKernel_y);
    cudaEventDestroy(startMemHostToDevice);
    cudaEventDestroy(stopMemHostToDevice);
    cudaEventDestroy(startMemDeviceToHost);
    cudaEventDestroy(stopMemDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_tmp);
  
    cudaFreeHost(A);
    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFreeHost(tmp);

    return 0;
}
