/*
 * BSD 2-Clause License
 * 
 * Copyright (c) 2020, Alessandro Capotondi
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file exercise2.cu
 * @author Alessandro Capotondi
 * @date 5 May 2020
 * @brief Exercise 2 - CUDA MATMUL Optimized
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

extern "C"
{
#include "utils.h"
}

#define TWO02 (1 << 2)
#define TWO04 (1 << 4)
#define TWO08 (1 << 8)

#ifndef N
#define N (1 << 10)
#endif
#ifndef TILE_W
#define TILE_W 128
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

void gemm(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int n)
{

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < n; ++k)
            {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

__global__ void gemm_v1(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int n)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    float sum = 0.0;
    for (int k = 0; k < n; ++k)
    {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

__global__ void gemm_v2(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int n)
{
    //TODO Shared memory used to store Asub and Bsub respectively

    //TODO Block row and column

    //TODO Thread row and column within Csub

    //TODO Each thread computes one element of Csub
    // by accumulating results into Cvalue

    //TODO Loop over all the sub-matrices of A and B that are
    // required to compute Csub.
    // Multiply each pair of sub-matrices together
    // and accumulate the results.
    for (int kb = 0; kb < (n / BLOCK_SIZE); ++kb)
    {
        //TODO Get the starting address (a_offset) of Asub
        // (sub-matrix of A of dimension BLOCK_SIZE x BLOCK_SIZE)
        // Asub is located i_block sub-matrices to the right and
        // k_block sub-matrices down from the upper-left corner of A
        //TODO Get the starting address (b_offset) of Bsub

        //TODO Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix

        //TODO Synchronize to make sure the sub-matrices are loaded
        // before starting the computation

        //TODO Multiply As and Bs together

        //TODO Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
    }

    //TODO Each thread block computes one sub-matrix Csub of C
}

__global__ void gemm_v3(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int n)
{
    //TODO Shared memory used to store Asub and Bsub respectively

    //TODO Block row and column

    //TODO Thread row and column within Csub

    //TODO Each thread computes one element of Csub
    // by accumulating results into Cvalue

    //TODO Loop over all the sub-matrices of A and B that are
    // required to compute Csub.
    // Multiply each pair of sub-matrices together
    // and accumulate the results.
    for (int kb = 0; kb < (n / BLOCK_SIZE); ++kb)
    {
        //TODO Get the starting address (a_offset) of Asub
        // (sub-matrix of A of dimension BLOCK_SIZE x BLOCK_SIZE)
        // Asub is located i_block sub-matrices to the right and
        // k_block sub-matrices down from the upper-left corner of A
        //TODO Get the starting address (b_offset) of Bsub (Coalesced Access)

        //TODO Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix

        //TODO Synchronize to make sure the sub-matrices are loaded
        // before starting the computation

        //TODO Multiply As and Bs together

        //TODO Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
    }

    //TODO Each thread block computes one sub-matrix Csub of C
}

int main(int argc, char *argv[])
{
    int n = N, iret = 0;
    float *a, *b, *c, *g;
    struct timespec rt[2];
    double wt; // walltime

    if (argc > 1)
        n = atoi(argv[1]);

    if (NULL == (a = (float *)malloc(sizeof(*a) * n * n)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (b = (float *)malloc(sizeof(*b) * n * n)))
    {
        printf("error: memory allocation for 'y'\n");
        iret = -1;
    }
    if (NULL == (c = (float *)malloc(sizeof(*c) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }
    if (NULL == (g = (float *)malloc(sizeof(*g) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }

    if (0 != iret)
    {
        free(a);
        free(b);
        free(c);
        free(g);
        exit(EXIT_FAILURE);
    }

    //Init Data
    int _b = rand() % TWO04;
    int _c = rand() % TWO08;
#pragma omp parallel for
    for (int i = 0; i < n * n; i++)
    {
        a[i] = _b / (float)TWO02;
        b[i] = _c / (float)TWO04;
        c[i] = g[i] = 0.0;
    }

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gemm(a, b, g, n);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    //CUDA Buffer Allocation
    float *d_a, *d_b, *d_c;
    gpuErrchk(cudaMalloc((void **)&d_a, sizeof(float) * n * n));
    gpuErrchk(cudaMalloc((void **)&d_b, sizeof(float) * n * n));
    gpuErrchk(cudaMalloc((void **)&d_c, sizeof(float) * n * n));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gpuErrchk(cudaMemcpy(d_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE));
    gemm_v1<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM-v1 (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    for (int i = 0; i < n * n; i++)
    {
        iret = *(int *)(g + i) ^ *(int *)(c + i);
        assert(iret == 0);
    }

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gpuErrchk(cudaMemcpy(d_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid((n + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE));
    gemm_v2<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM-v2 (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    for (int i = 0; i < n * n; i++)
    {
        iret = *(int *)(g + i) ^ *(int *)(c + i);
        assert(iret == 0);
    }

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gpuErrchk(cudaMemcpy(d_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid((n + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE));
    gemm_v3<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM-v3 (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * n * n * n / (1.0e9 * wt));

    for (int i = 0; i < n * n; i++)
    {
        iret = *(int *)(g + i) ^ *(int *)(c + i);
        assert(iret == 0);
    }
    free(a);
    free(b);
    free(c);
    free(g);
    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_c));

    return 0;
}
