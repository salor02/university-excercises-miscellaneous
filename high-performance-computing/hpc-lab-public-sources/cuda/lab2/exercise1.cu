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
 * @file exercise1.cu
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 1
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

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
#define N (1LL << 28)
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE (1024)
#endif

/**
 * @brief  EX 1 - Offset and Strided Accesses
 *
 * a) Measure the bandwidth accessing the memory using an offset = {1,2,4,8,16,32} (mem_update v1)
 * b) Measure the bandwidth accessing the memory using a stride = {1,2,4,8,16,32} (mem_update v2)
 * 
 * @return void
 */

#ifndef STRIDE
#define STRIDE 0
#endif

// mem_update v1 - Offseted Accesses
__global__ void mem_udpate(float * __restrict__ y, float a)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    y[(i+STRIDE)%N] = a;
}

// mem_update v2 - Strided Accesses
// __global__ void mem_udpate(float * __restrict__ y, float a)
// {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     y[(i*STRIDE)%N] = a;
// }

int main(int argc, const char **argv)
{
    int iret = 0;
    float *h_y, *d_y;
    float a = 101.0f / TWO02;
    
    if (NULL == (h_y = (float *)malloc(sizeof(float) * N)))
    {
        printf("error: memory allocation for 'y'\n");
        iret = -1;
    }
    if (0 != iret)
    {
        free(h_y);
        exit(EXIT_FAILURE);
    }

    //CUDA Buffer Allocation
    gpuErrchk(cudaMalloc((void **)&d_y, sizeof(float) * N));
    gpuErrchk(cudaMemcpy(d_y, h_y, sizeof(float) * N, cudaMemcpyHostToDevice));

    start_timer();
    mem_udpate<<<128*BLOCK_SIZE,BLOCK_SIZE>>>(d_y, a);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    stop_timer();
    
    gpuErrchk(cudaMemcpy(h_y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost));
    printf("mem_udpate (GPU): %9.3f sec %9.1f MB/s\n", elapsed_ns() / 1.0e9, (4 * 128*BLOCK_SIZE*BLOCK_SIZE) / ((1.0e6 / 1e9) * elapsed_ns()));

    //CUDA Buffer Allocation
    free(h_y);
    gpuErrchk(cudaFree(d_y));

    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();
    return 0;
}
