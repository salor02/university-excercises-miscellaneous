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
 * @file colorTracking.cpp
 * @author Alessandro Capotondi
 * @date 5 May 2020
 * @brief Color Tracking
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#ifdef CHECK
#include "data/sample_ground_truth.h"
#endif

extern "C"
{
#include "utils.h"
}

void ConversionRgb2Hsv(uint8_t *__restrict__ out, uint8_t *__restrict__ in, int width, int height, int nch)
{
    // Get Raw Image Data pointer
    int pixels = width * height * nch;

    //Convert each pixel of input RGB image to HSV
    for (int idx = 0; idx < pixels; idx += nch)
    {
        uint8_t V, S, H;
        uint8_t rgb_min, rgb_max, rgb_mm;

        //Read color channels
        uint8_t red = in[idx];
        uint8_t green = in[idx + 1];
        uint8_t blue = in[idx + 2];

        rgb_max = max(max(red, green), blue);
        rgb_min = min(min(red, green), blue);
        rgb_mm = rgb_max - rgb_min;

        //Value computation
        V = rgb_max;
        if (V == 0)
            H = S = 0;
        else
        {
            //Saturation computation
            S = (int)(((long)255 * (long)(rgb_mm)) / (long)V);
            if (S == 0)
                H = 0;
            else
            {
                //Hue computation
                if (rgb_max == red)
                    H = 0 + 43 * (green - blue) / rgb_mm;
                else if (rgb_max == green)
                    H = 85 + 43 * (blue - red) / rgb_mm;
                else
                    H = 171 + 43 * (red - green) / rgb_mm;
            }
        }

        //Write HSV
        out[idx] = (uint8_t)H;
        out[idx + 1] = (uint8_t)S;
        out[idx + 2] = (uint8_t)V;
    }
}

void ImgThreashold(uint8_t *__restrict__ out, uint8_t *__restrict__ in, int32_t thLow[3], int32_t thHi[3], int width, int height, int nch)
{
    int pixels = width * height;
    uint8_t lb1 = (uint8_t)thLow[0];
    uint8_t lb2 = (uint8_t)thLow[1];
    uint8_t lb3 = (uint8_t)thLow[2];
    uint8_t ub1 = (uint8_t)thHi[0];
    uint8_t ub2 = (uint8_t)thHi[1];
    uint8_t ub3 = (uint8_t)thHi[2];

    for (int idx = 0; idx < pixels; ++idx)
    {
        out[idx] = ((in[idx * nch] >= lb1) && (in[idx * nch] <= ub1) &&
                    (in[idx * nch + 1] >= lb2) && (in[idx * nch + 1] <= ub2) &&
                    (in[idx * nch + 2] >= lb3) && (in[idx * nch + 2] <= ub3))
                       ? 255
                       : 0;
    }
}

void ImgCenterbyMoments(int *y, int *x, uint8_t *__restrict__ in, int width, int height, int nch)
{
    uint64_t m_00 = 0, m_01 = 0, m_10 = 0;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (in[i * width + j] > 0)
            {
                m_00++;
                m_01 += j;
                m_10 += i;
            }
        }
    }
    *y = m_00 ? (double)m_01 / m_00 : 0;
    *x = m_00 ? (double)m_10 / m_00 : 0;
}

void ImgMerge(uint8_t *__restrict__ out, uint8_t *__restrict__ in1, uint8_t *__restrict__ in2, int width, int height, int nch)
{
    // Get Raw Image Data pointer
    int pixels = width * height * nch;

    for (int idx = 0; idx < pixels; ++idx)
    {
        if(in2[idx])
            out[idx] = in2[idx];
        else
            out[idx] = in1[idx];
    }
}

int main(int argc, char *argv[])
{
    struct timespec rt[2];
    double wt;

    //Open Video Example
    VideoCapture cap("data/sample.avi");
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int nCh = 3;

    // Upper and Lower Color Threasholds
    int32_t thHi[3], thLow[3];
    Scalar _thHi, _thLow;

    // Frame Buffers
    Mat frameRGB = Mat::zeros(height, width, CV_8UC3);
    Mat frameHVS = Mat::zeros(height, width, CV_8UC3);
    Mat frameTrack = Mat::zeros(height, width, CV_8UC3);
    Mat frameMask = Mat::zeros(height, width, CV_8UC1);

    //Check Video
    if (!cap.isOpened())
    {
        cout << "[Error] Cannot open the video file" << endl;
        return -1;
    }

    if (argc > 1)
    {
        if (argc == 7)
        {
            thLow[0] = atoi(argv[1]);
            thLow[1] = atoi(argv[2]);
            thLow[2] = atoi(argv[3]);
            thHi[0] = atoi(argv[4]);
            thHi[1] = atoi(argv[5]);
            thHi[2] = atoi(argv[6]);
        }
        else
        {
            cout << "[Error] Invalid arguments: usage ./cTracking [thLow(H S V) thHi(H S V)]" << endl;
            return -1;
        }
    }
    else
    {
        //Default Values
        thLow[0] = 160;
        thLow[1] = 100;
        thLow[2] = 100;
        thHi[0] = 180;
        thHi[1] = 255;
        thHi[2] = 255;
    }
    _thHi = Scalar(thHi[0], thHi[1], thHi[2]);

    //Print Information
    printf("--------------------------------------\n");
    printf("Video Info\n");
    printf("--------------------------------------\n");
    printf("width :\t%d\n", (int)cap.get(CAP_PROP_FRAME_WIDTH));
    printf("height:\t%d\n", (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    printf("fps   :\t%d\n", (int)cap.get(CAP_PROP_FPS));
    printf("--------------------------------------\n");

    int lastX = 0;
    int lastY = 0;
    int posX = 0;
    int posY = 0;
#ifdef CHECK
    int check_id = 0;
#endif
    int nFrames = 0;
    double time_cnt = 0.0;
    while (1)
    {
        bool lastFrame = cap.read(frameRGB); // read a new frame from video
        if (!lastFrame)
            break;

        clock_gettime(CLOCK_REALTIME, rt + 0);
        ConversionRgb2Hsv(frameHVS.ptr(), frameRGB.ptr(), width, height, nCh);
        ImgThreashold(frameMask.ptr(), frameHVS.ptr(), thLow, thHi, width, height, nCh);
        ImgCenterbyMoments(&posY, &posX, frameMask.ptr(), width, height, nCh);

        // We want to draw a line only if its a valid position
        if (lastX > 0 && lastY > 0 && posX > 0 && posY > 0)
            line(frameTrack, Point(lastY, lastX), Point(posY, posX), _thHi, 2, CV_8UC3, 0);

        ImgMerge(frameRGB.ptr(), frameRGB.ptr(), frameTrack.ptr(), width, height, nCh);
                clock_gettime(CLOCK_REALTIME, rt + 1);
        time_cnt+= (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        nFrames++;

#ifdef DISPLAY
        // Show frames
        imshow("frameRGB", frameRGB);
        imshow("frameMask", frameMask);
        imshow("frameTrack", frameTrack);
        waitKey(1);
#endif

#ifdef CHECK
        assert(ground_truth_YX[check_id++]==posY);
        assert(ground_truth_YX[check_id++]==posX);
#endif
        lastX = posX;
        lastY = posY;
    }

    printf("ColorTracking: %d frames, %9.6f s per-frame (%9.6f fps)\n", nFrames, time_cnt/nFrames, 1/(time_cnt/nFrames));

    //Release Memory - Avoid Memory Leak!
    frameRGB.release();
    frameHVS.release();
    frameTrack.release();
    frameMask.release();
    cap.release();

    return 0;
}
