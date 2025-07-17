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
 * @file sobel.cu
 * @author Alessandro Capotondi
 * @date 12 May 2020
 * @brief Sobel Filtering
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

int FILTER_HOST[3][3] = {{-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1}};

void sobel_host(unsigned char *__restrict__ orig, unsigned char *__restrict__ out, int width, int height)
{
#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int dx = 0, dy = 0;
            for (int k = -1; k <= 1; k++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    dx += FILTER_HOST[k + 1][z + 1] * orig[(y + k) * width + x + z];
                    dy += FILTER_HOST[z + 1][k + 1] * orig[(y + k) * width + x + z];
                }
            }
            out[y * width + x] = sqrt((float)((dx * dx) + (dy * dy)));
        }
    }
}

__constant__ int FILTER_GPU[3][3] = {{-1, 0, 1},
                                     {-2, 0, 2},
                                     {-1, 0, 1}};

__global__ void sobel_v1(unsigned char *__restrict__ orig, unsigned char *__restrict__ out, int width, int height)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j > 0 && i > 0 && j < width - 1 && i < height - 1)
    {
        int dx = 0, dy = 0;
        for (int k = -1; k <= 1; k++)
        {
            for (int z = -1; z <= 1; z++)
            {
                dx += FILTER_GPU[k + 1][z + 1] * orig[(i + k) * width + j + z];
                dy += FILTER_GPU[z + 1][k + 1] * orig[(i + k) * width + j + z];
            }
        }
        out[i * width + j] = sqrt((float)((dx * dx) + (dy * dy)));
    }
}

int main(int argc, char *argv[])
{
    int iret = 0;
    struct timespec rt[2];
    string filename("data/sample.avi");

    if (argc > 1)
        filename = argv[1];

    //Open Video Example
    VideoCapture cap(filename);
    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int nCh = 3;

    // Frame Buffers
    Mat frameRGB = Mat::zeros(height, width, CV_8UC3);
    Mat frameIn = Mat::zeros(height, width, CV_8UC1);
    Mat frameOut = Mat::zeros(height, width, CV_8UC1);

    int nFrames = 0;
    double time_cnt = 0.0;
    while (1)
    {
        bool lastFrame = cap.read(frameRGB); // read a new frame from video
        if (!lastFrame)
            break;

        cvtColor(frameRGB, frameIn, COLOR_BGR2GRAY);

        // Compute CPU Version - Golden Model
        clock_gettime(CLOCK_REALTIME, rt + 0);
        sobel_host(frameIn.ptr(), frameOut.ptr(), width, height);
        clock_gettime(CLOCK_REALTIME, rt + 1);
        time_cnt+= (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        nFrames++;

#ifdef DISPLAY
        // Show frames
        imshow("frameIn", frameIn);
        imshow("frameOut", frameOut);
        waitKey(1);
#endif
    }
    printf("Sobel (Host) : %d frames, %9.6f s per-frame (%9.6f fps)\n", nFrames, time_cnt/nFrames, 1/(time_cnt/nFrames));

    // CUDA VERSION --------------------------------------------------
    //Open Video Example
    cap = VideoCapture(filename);
    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    unsigned char *d_image_in;
    unsigned char *d_image_out;
    gpuErrchk(cudaMalloc((void **)&d_image_in, sizeof(unsigned char) * width * height));
    gpuErrchk(cudaMalloc((void **)&d_image_out, sizeof(unsigned char) * width * height));
    gpuErrchk(cudaMemset(d_image_out, 0, sizeof(unsigned char) * width * height));
    
    nFrames = 0;
    time_cnt = 0.0;
    while (1)
    {
        bool lastFrame = cap.read(frameRGB); // read a new frame from video
        if (!lastFrame)
            break;

        cvtColor(frameRGB, frameIn, COLOR_BGR2GRAY);

        // Compute CPU Version - Golden Model
        clock_gettime(CLOCK_REALTIME, rt + 0);
        gpuErrchk(cudaMemcpy(d_image_in, frameIn.ptr(), sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice));
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sobel_v1<<<dimGrid, dimBlock>>>(d_image_in, d_image_out, width, height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpy(frameOut.ptr(), d_image_out, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost));
        clock_gettime(CLOCK_REALTIME, rt + 1);
        time_cnt+= (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        nFrames++;

#ifdef DISPLAY
        // Show frames
        imshow("frameIn", frameIn);
        imshow("frameOut", frameOut);
        waitKey(1);
#endif
    }
    printf("Sobel (GPU) : %d frames, %9.6f s per-frame (%9.6f fps)\n", nFrames, time_cnt/nFrames, 1/(time_cnt/nFrames));

    gpuErrchk(cudaFree(d_image_out));
    gpuErrchk(cudaFree(d_image_in));
    frameOut.release();
    frameIn.release();
    frameRGB.release();
    cap.release();

    return iret;
}
