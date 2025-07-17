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
 * @file saxpy.c
 * @author Alessandro Capotondi
 * @date 12 May 2020
 * @brief Saxpy
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

#define TWO02 (1 << 2)
#define TWO04 (1 << 4)
#define TWO08 (1 << 8)
#ifndef N
#define N (1 << 27)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (512)
#endif

/*
 *SAXPY (host implementation)
 * y := a * x + y
 */
void host_saxpy(float * __restrict__ y, float a, float * __restrict__ x, int n)
{
#pragma omp parallel for simd schedule(simd: static)
    for (int i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void gpu_saxpy(float * __restrict__ y, float a, float * __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(int argc, const char **argv)
{
    int iret = 0;
    int n = N;
    float *h_x, *d_x;
    float *h_y, *d_y;
    float *h_z;
    float a = 101.0f / TWO02,
          b, c;
    struct timespec rt[2];
    double wt; // walltime

    if (argc > 1)
        n = atoi(argv[1]);

    //TODO Update malloc to cudaMallocHost or cudaMallocManaged (if necessary)
    if (NULL == (h_x = (float *)malloc(sizeof(float) * n)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    //TODO Update malloc to cudaMallocHost or cudaMallocManaged (if necessary)
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
        //TODO Update cudaFreeHost or cudaFree (if necessary)
        free(h_x);
        //TODO Update cudaFreeHost or cudaFree (if necessary)
        free(h_y);
        free(h_z);
        exit(EXIT_FAILURE);
    }

    //Init Data
    b = rand() % TWO04;
    c = rand() % TWO08;
    for (int i = 0; i < n; i++)
    {
        h_x[i] = b / (float)TWO02;
        h_y[i] = h_z[i] = c / (float)TWO04;
    }

    //TODO Remove if unecessary
    gpuErrchk(cudaMalloc((void **)&d_x, sizeof(float) * n));
    gpuErrchk(cudaMalloc((void **)&d_y, sizeof(float) * n));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    //TODO Remove if unecessary
    gpuErrchk(cudaMemcpy(d_x, h_x, sizeof(float) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, h_y, sizeof(float) * n, cudaMemcpyHostToDevice));
    gpu_saxpy<<<((n + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(d_y, a, d_x, n);
    gpuErrchk(cudaPeekAtLastError());
    //TODO Remove if unecessary
    gpuErrchk(cudaMemcpy(h_y, d_y, sizeof(float) * n, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("saxpy (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2 * n / wt);

    //Check Matematical Consistency
    clock_gettime(CLOCK_REALTIME, rt + 0);
    host_saxpy(h_z, a, h_x, n);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("saxpy (Host): %9.3f sec %9.1f GFLOPS\n", wt, 2 * n / wt);
    for (int i = 0; i < n; ++i)
    {
        iret = *(int *)(h_y + i) ^ *(int *)(h_z + i);
        assert(iret == 0);
    }

    //TODO Update cudaFreeHost or cudaFree (if necessary)
    free(h_x);
    //TODO Remove if unecessary
    gpuErrchk(cudaFree(d_x));
    //TODO Update cudaFreeHost or cudaFree (if necessary)
    free(h_y);
    //TODO Remove if unecessary
    gpuErrchk(cudaFree(d_y));
    free(h_z);

    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();
    return 0;
}
