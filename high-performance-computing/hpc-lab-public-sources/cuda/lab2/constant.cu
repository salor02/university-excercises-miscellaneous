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
 * @file constant.cu
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 2
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

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
#define N (1 << 27)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (128)
#endif

float K[4098];
//TODO declare constant K
__constant__ float cK[4098];

/*
 * Filering 
 */
void filter(float * __restrict__ y, int n)
{
#pragma omp parallel for simd schedule(simd: static)
    for (int i = 0; i < n; i++)
    {
        y[i] = y[i] - K[i%4098];
    }
}

//TODO GPU Filter implementation
__global__ void filter_v1(float * __restrict__ y, int n)
{
}

//TODO GPU Filter implementation without constant mem
__global__ void filter_v2(float * __restrict__ y, float * __restrict__ k, int n)
{
}

int main(int argc, const char **argv)
{
    int iret = 0;
    int n = N;
    float *h_y, *d_y;
    float *h_x, *d_x, *d_k;
    float *h_z;

    if (argc > 1)
        n = atoi(argv[1]);

    if (NULL == (h_x = (float *)malloc(sizeof(float) * n)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (h_y = (float *)malloc(sizeof(float) * n)))
    {
        printf("error: memory allocation for 'y'\n");
        iret = -1;
    }
    if (NULL == (h_z = (float *)malloc(sizeof(float) * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }
    if (0 != iret)
    {
        free(h_y);
        free(h_z);
        exit(EXIT_FAILURE);
    }

    //Init Data
    float b = rand() % TWO04;
    float c = rand() % TWO08;

    for (int i = 0; i < 4098; i++)
    {
        K[i] = b;
    }
    for (int i = 0; i < n; i++)
    {
        h_x[i] = h_y[i] = h_z[i] = c / (float)TWO04;
    }

    start_timer();
    filter(h_z, n);
    stop_timer();
    printf("Filter (Host): %9.3f sec %9.1f MFLOPS\n", elapsed_ns() / 1.0e9, n / ((1.0e6 / 1e9) * elapsed_ns()));

    //CUDA Buffer Allocation
    gpuErrchk(cudaMalloc((void **)&d_y, sizeof(float) * n));
    //TODO: Load Device Constant using cudaMemcpyToSymbol
    gpuErrchk(cudaMemcpyToSymbol(cK, K, sizeof(float)*4098));

    start_timer();
    //TODO Add Code here for calling filter_v1
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(h_y, d_y, sizeof(float) * n, cudaMemcpyDeviceToHost));
    stop_timer();
    printf("Filter-v1 (GPU): %9.3f sec %9.1f MFLOPS\n", elapsed_ns() / 1.0e9, n / ((1.0e6 / 1e9) * elapsed_ns()));

    //Check Matematical Consistency
    for (int i = 0; i < n; ++i)
    {
        iret = *(int *)(h_y + i) ^ *(int *)(h_z + i);
        assert(iret == 0);
    }

    //-- No-Constant version --
    gpuErrchk(cudaMalloc((void **)&d_x, sizeof(float) * n));
    gpuErrchk(cudaMalloc((void **)&d_k, sizeof(float) * 4098));

    start_timer();
    //TODO Add Code here for calling filter_v2ù

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(h_x, d_x, sizeof(float) * n, cudaMemcpyDeviceToHost));
    stop_timer();
    printf("Filter-v2 (GPU): %9.3f sec %9.1f MFLOPS\n", elapsed_ns() / 1.0e9, n / ((1.0e6 / 1e9) * elapsed_ns()));

    //Check Matematical Consistency
    for (int i = 0; i < n; ++i)
    {
        iret = *(int *)(h_y + i) ^ *(int *)(h_x + i);
        assert(iret == 0);
    }

    //CUDA Buffer Allocation
    free(h_x);
    gpuErrchk(cudaFree(d_x));
    free(h_y);
    gpuErrchk(cudaFree(d_y));
    free(h_z);
    return 0;
}
