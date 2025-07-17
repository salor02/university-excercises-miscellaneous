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

__global__ void atax_kernel_tmp(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* tmp, int nx, int ny) {
    __shared__ DATA_TYPE smem_A[16][16];
    __shared__ DATA_TYPE smem_x[16];

    // Coordinate del thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Coordinate globali
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    //ogni thread mette in shared la cella di A a cui è interessato
    if (row < nx && col < ny) {
        smem_A[ty][tx] = A[row * ny + col];
    } else {
        smem_A[ty][tx] = 0.0f;  // Padding per boundary
    }

    // Caricamento del vettore x (solo la prima riga di thread)
    if (ty == 0 && col < ny) {
        smem_x[tx] = x[col];
    } else if (ty == 0) {
        smem_x[tx] = 0.0f;  // Padding
    }

    // Sincronizzazione essenziale
    __syncthreads();
    
    // // Processa x a blocchi
    // for (int start = 0; start < ny; start += blockDim.x) {
    //     // Carica chunk di x in shared memory
    //     if (start + tid < ny) {
    //         shared_x[tid] = x[start + tid];
    //     } else {
    //         shared_x[tid] = 0.0;
    //     }
    //     __syncthreads();
        
    //     // Calcola per questo chunk
    //     if (i < nx) {
    //         int end = min(blockDim.x, ny - start);
    //         for (int j = 0; j < end; j++) {
    //             sum += A[i * ny + (start + j)] * shared_x[j];
    //         }
    //     }
    //     __syncthreads();
    // }
    
    // if (i < nx) {
    //     tmp[i] = sum;
    // }
}

__global__ void atax_kernel_y(DATA_TYPE* A, DATA_TYPE* tmp, DATA_TYPE* y, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory per tmp
    __shared__ DATA_TYPE shared_tmp[256];  // Dimensione del blocco
    
    DATA_TYPE sum = 0.0;
    
    // Processa tmp a blocchi
    for (int start = 0; start < nx; start += blockDim.x) {
        // Carica chunk di tmp in shared memory
        if (start + tid < nx) {
            shared_tmp[tid] = tmp[start + tid];
        } else {
            shared_tmp[tid] = 0.0;
        }
        __syncthreads();
        
        // Calcola per questo chunk
        if (j < ny) {
            int end = min(blockDim.x, nx - start);
            for (int i = 0; i < end; i++) {
                sum += A[(start + i) * ny + j] * shared_tmp[i];
            }
        }
        __syncthreads();
    }
    
    if (j < ny) {
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
    dim3 blockSize(16,16);
    dim3 grid_size_tmp((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);
    dim3 grid_size_y((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(startKernel_tmp);
    atax_kernel_tmp<<<grid_size_tmp, blockSize>>>(d_A, d_x, d_tmp, nx, ny);
    cudaDeviceSynchronize();
    cudaEventRecord(stopKernel_tmp);

    cudaEventRecord(startKernel_y);
    atax_kernel_y<<<grid_size_y, blockSize>>>(d_A, d_tmp, d_y, nx, ny);
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
    printf("mem_HtD_time: %f ms\n", mem_HtD_time);
    printf("mem_DtH_time: %f ms\n", mem_DtH_time);
    printf("kernel_tmp_time: %f ms\n", kernel_tmp_time);
    printf("kernel_y_time: %f ms\n", kernel_y_time);

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
