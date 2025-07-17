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
 * @file exercise3.cu
 * @author Alessandro Capotondi
 * @date 5 May 2020
 * @brief Exercise 3 - Image Luminance Histogram
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

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

#define NBINS 256

void hist(unsigned char *__restrict__ im, int *__restrict__ hist, int width, int height)
{
#pragma omp parallel for
    for (int i = 0; i < width * height; i++)
    {
        int val = im[i];
#pragma omp atomic
        hist[val]++;
    }
}

__global__ void hist_v1(unsigned char *__restrict__ im, int *__restrict__ hist, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height)
    {
        int value;
        value = im[(j * width) + i];
        atomicAdd(&(hist[value]), 1);
        //hist[value]++;
    }
}

__global__ void hist_v2(unsigned char *__restrict__ im, int *__restrict__ hist, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int blockIndex = (threadIdx.y * blockDim.y) + threadIdx.x;
    __shared__ int tmpHist[NBINS];

    if (blockIndex < NBINS)
    {
        tmpHist[blockIndex] = 0;
    }
    __syncthreads();

    if (i < width && j < height)
    {
        int value;
        value = im[(j * width) + i];
        atomicAdd(&(tmpHist[value]), 1);
    }
    __syncthreads();

    if (blockIndex < NBINS)
        atomicAdd(&(hist[blockIndex]), tmpHist[blockIndex]);
}

int main(int argc, char *argv[])
{
    int iret = 0;
    struct timespec rt[2];
    double wt; // walltime
    int hist_host[NBINS], hist_gpu[NBINS];

    string filename("data/buzz.jpg");

    if (argc > 1)
        filename = argv[1];

    // Load Image
    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    int width = image.size().width;
    int height = image.size().height;

    memset(hist_host, 0, NBINS * sizeof(int));
    memset(hist_gpu, 0, NBINS * sizeof(int));

    // Compute CPU Version - Golden Model
    clock_gettime(CLOCK_REALTIME, rt + 0);
    hist(image.ptr(), hist_host, width, height);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Hist (Host) : %9.6f sec\n", wt);

    //CUDA Buffer Allocation
    int *d_hist_gpu;
    unsigned char *d_image;
    gpuErrchk(cudaMalloc((void **)&d_hist_gpu, sizeof(int) * NBINS));
    gpuErrchk(cudaMalloc((void **)&d_image, sizeof(unsigned char) * width * height));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gpuErrchk(cudaMemcpy(d_image, image.ptr(), sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hist_v1<<<dimGrid, dimBlock>>>(d_image, d_hist_gpu, width, height);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(hist_gpu, d_hist_gpu, sizeof(int) * NBINS, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Hist (GPU) : %9.6f sec\n", wt);

    for (int i = 0; i < NBINS; i++)
    {
        iret = *(int *)(hist_host + i) ^ *(int *)(hist_gpu + i);
        assert(iret == 0);
    }
    // Reset Output
    gpuErrchk(cudaMemset(d_hist_gpu, 0, NBINS * sizeof(unsigned int)));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gpuErrchk(cudaMemcpy(d_image, image.ptr(), sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice));
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(width/BLOCK_SIZE, height/BLOCK_SIZE);
    hist_v2<<<dimGrid, dimBlock>>>(d_image, d_hist_gpu, width, height);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(hist_gpu, d_hist_gpu, sizeof(int) * NBINS, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Hist-2 (GPU) : %9.6f sec\n", wt);

    for (int i = 0; i < NBINS; i++)
    {
        iret = *(int *)(hist_host + i) ^ *(int *)(hist_gpu + i);
        assert(iret == 0);
    }

    gpuErrchk(cudaFree(d_hist_gpu));
    gpuErrchk(cudaFree(d_image));

    return iret;
}
