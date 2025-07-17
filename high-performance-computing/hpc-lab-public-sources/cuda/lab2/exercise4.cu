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
 * @file exercise4.cu
 * @author Alessandro Capotondi
 * @date 5 May 2020
 * @brief Exercise 4 - Stencil 2d - Sobel
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

void sobel_host(unsigned char *__restrict__ orig, unsigned char *__restrict__ out, int width, int height)
{
#pragma omp parallel for simd collapse(2)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int dx = (-1 * orig[(y - 1) * width + (x - 1)]) + (-2 * orig[y * width + (x - 1)]) + (-1 * orig[(y + 1) * width + (x - 1)]) +
                     (orig[(y - 1) * width + (x + 1)]) + (2 * orig[y * width + (x + 1)]) + (orig[(y + 1) * width + (x + 1)]);
            int dy = (orig[(y - 1) * width + (x - 1)]) + (2 * orig[(y - 1) * width + x]) + (orig[(y - 1) * width + (x + 1)]) +
                     (-1 * orig[(y + 1) * width + (x - 1)]) + (-2 * orig[(y + 1) * width + x]) + (-1 * orig[(y + 1) * width + (x + 1)]);
            out[y * width + x] = sqrt((float)((dx * dx) + (dy * dy)));
        }
    }
}

//TODO Each thread compute one pixel out reading from global memory
__global__ void sobel_v1(unsigned char *__restrict__ orig, unsigned char *__restrict__ out, int width, int height)
{
}

#ifdef V2
//TODO Each thread compute one pixel out reading from shared memory (corner case readed from global memory)
__global__ void sobel_v2(unsigned char *__restrict__ orig, unsigned char *__restrict__ out, int width, int height)
{
    //TODO Declare i and j: global output indexes


    //TODO Declare it and jt: Thread row and column of output matrix

    //TODO Declare shared input patch

    //TODO Load input patch
    // Each thread loads one element of the patch

    //TODO Synchronize to make sure the sub-matrices are loaded
    // before starting the computation

    //TODO if block boundary do
    if (jt > 0 && it > 0 && jt < BLOCK_SIZE - 1 && it < BLOCK_SIZE - 1 && j > 0 && i > 0 && j < width - 1 && i < height - 1)
    {

    }
    else if (j > 0 && i > 0 && j < width - 1 && i < height - 1)
    {
        //TODO if not-block boundary do (tip check global boundaries)
    }
}
#endif

#ifdef V3
//TODO Each thread compute one pixel out reading from shared memory.
__global__ void sobel_v3(unsigned char *__restrict__ orig, unsigned char *__restrict__ out, int width, int height)
{
    //TODO Declare i and j: global output indexes (tip: use BLOCK_SIZE-2)

    //TODO Declare it and jt: Thread row and column of output matrix

    //TODO Check if i and j are out of memory
    if (i >= width && j >= height)
        return;

    //TODO Declare shared input patch

    //TODO Load input patch
    // Each thread loads one element of the patch

    //TODO Synchronize to make sure the sub-matrices are loaded
    // before starting the computation

    //TODO Update block and bound checks
    if (jt > 0 && it > 0 && jt < BLOCK_SIZE - 1 && it < BLOCK_SIZE - 1 && j > 0 && i > 0 && j < width - 1 && i < height - 1)
    {
    }
}
#endif

#ifdef V4
//TODO Each thread compute one pixel out reading from shared memory. Avoid thread under-usage
__global__ void sobel_v4(unsigned char *__restrict__ orig, unsigned char *__restrict__ out, int width, int height)
{
    //TODO Declare i and j: global output indexes (tip: use BLOCK_SIZE)

    //TODO Declare it and jt: Thread row and column of output matrix

    //TODO Declare shared input patch (tip: use BLOCK_SIZE+2)

    //TODO Load input patch
    // Each thread loads one element of the patch

    //TODO Check condition and load remaining elements
    if ((it + BLOCK_SIZE) < BLOCK_SIZE + 2 && (jt) < BLOCK_SIZE + 2 && (i + BLOCK_SIZE) < width && (j) < height)
        s_in[it + BLOCK_SIZE][jt] = orig[(i + BLOCK_SIZE) * width + j];

    if ((it) < BLOCK_SIZE + 2 && (jt + BLOCK_SIZE) < BLOCK_SIZE + 2 && (i) < width && (j + BLOCK_SIZE) < height)
        s_in[it][jt + BLOCK_SIZE] = orig[i * width + j + BLOCK_SIZE];

    if ((it + BLOCK_SIZE) < BLOCK_SIZE + 2 && (jt + BLOCK_SIZE) < BLOCK_SIZE + 2 && (i + BLOCK_SIZE) < width && (j + BLOCK_SIZE) < height)
        s_in[it + BLOCK_SIZE][jt + BLOCK_SIZE] = orig[(i + BLOCK_SIZE) * width + j + BLOCK_SIZE];

    //TODO Synchronize to make sure the sub-matrices are loaded
    // before starting the computation

    //TODO Update all idx adding y +1 and x +1
    if (jt < BLOCK_SIZE && it < BLOCK_SIZE && j < (width - 2) && i < (height - 2))
    {
    }
}
#endif

int main(int argc, char *argv[])
{
    int iret = 0;
    struct timespec rt[2];
    double wt; // walltime
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

    // Create Output Images
    Mat out1 = image.clone();
    Mat out2 = image.clone();
    Mat result = image.clone();
    memset(out1.ptr(), 0, sizeof(unsigned char) * width * height);
    memset(out2.ptr(), 0, sizeof(unsigned char) * width * height);
    memset(result.ptr(), 0, sizeof(unsigned char) * width * height);

    // Compute CPU Version - Golden Model
    clock_gettime(CLOCK_REALTIME, rt + 0);
    sobel_host(image.ptr(), out1.ptr(), width, height);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Sobel (Host) : %9.6f sec\n", wt);

    //CUDA Buffer Allocation
    unsigned char *d_image_in;
    unsigned char *d_image_out;
    gpuErrchk(cudaMalloc((void **)&d_image_in, sizeof(unsigned char) * width * height));
    gpuErrchk(cudaMalloc((void **)&d_image_out, sizeof(unsigned char) * width * height));
    gpuErrchk(cudaMemset(d_image_out, 0, sizeof(unsigned char) * width * height));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    //TODO Copy Image to the device
    
    //TODO Define Grid and Block
    
    //TODO Launch Kernel sobel_v1

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(out2.ptr(), d_image_out, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Sobel-v1 (GPU) : %9.6f sec\n", wt);

    //Check results
    absdiff(out1, out2, result);
    int percentage = countNonZero(result);

#ifdef V2
    //Reset Output image
    memset(out2.ptr(), 0, sizeof(unsigned char) * width * height);
    gpuErrchk(cudaMemset(d_image_out, 0, sizeof(unsigned char) * width * height));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    //TODO Copy Image to the device
    
    //TODO Define Grid and Block
    
    //TODO Launch Kernel sobel_v2
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Sobel-v2 (GPU) : %9.6f sec\n", wt);

    //Check results
    absdiff(out1, out2, result);
    percentage = countNonZero(result);
    if (percentage)
    {
        printf("Divergence %d\n", percentage);
        imshow("Output GPU", out2);
        imshow("error diff", result);
        waitKey(0);
    }
    assert(percentage == 0);
#endif

#ifdef V3
    //Reset Output image
    memset(out2.ptr(), 0, sizeof(unsigned char) * width * height);
    gpuErrchk(cudaMemset(d_image_out, 0, sizeof(unsigned char) * width * height));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gpuErrchk(cudaMemcpy(d_image_in, image.ptr(), sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice));
    //TODO Copy Image to the device
    
    //TODO Define Grid and Block
    
    //TODO Launch Kernel sobel_v3
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(out2.ptr(), d_image_out, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Sobel-v3 (GPU) : %9.6f sec\n", wt);

    //Check results
    absdiff(out1, out2, result);
    percentage = countNonZero(result);
    if (percentage)
    {
        printf("Divergence %d\n", percentage);
        imshow("Output GPU", out2);
        imshow("error diff", result);
        waitKey(0);
    }
    assert(percentage == 0);
#endif
#ifdef V4
    //Reset Output image
    memset(out2.ptr(), 0, sizeof(unsigned char) * width * height);
    gpuErrchk(cudaMemset(d_image_out, 0, sizeof(unsigned char) * width * height));

    clock_gettime(CLOCK_REALTIME, rt + 0);
    //TODO Copy Image to the device
    
    //TODO Define Grid and Block
    
    //TODO Launch Kernel sobel_v4
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(out2.ptr(), d_image_out, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("Sobel-v4 (GPU) : %9.6f sec\n", wt);

    //Check results
    absdiff(out1, out2, result);
    percentage = countNonZero(result);
    if (percentage)
    {
        printf("Divergence %d\n", percentage);
        imshow("Output GPU", out2);
        imshow("error diff", result);
        waitKey(0);
    }
    assert(percentage == 0);
#endif
    gpuErrchk(cudaFree(d_image_out));
    gpuErrchk(cudaFree(d_image_in));

    return iret;
}
