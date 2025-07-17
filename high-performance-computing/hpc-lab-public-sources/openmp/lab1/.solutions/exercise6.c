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
 * @file exercise6.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 6
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

#if !defined(W)
#define W (1 << 15)
#endif

/**
 * @brief EX 6 - Task Parallelism w/tasks
 * 
 * a) Create a parallel region with 4 threads. Use SINGLE directive to allow only one thread to execute the loop. Use TASK directive to outline tasks.
 * b) Change number of iterations to 1024 and W to 1000000. Parallelize with TASK directive.
 * c) Same setup as b): parallelize with SINGLE instead of TASK. Comment on ease of coding and performance of the various parallelization schemes.
 * 
 * @return void
 */
void exercise()
{
    unsigned int i;

#if 0 //Wrong!
#pragma omp parallel num_threads(4)
    for(i=0; i<4; i++)
    {
#pragma omp single nowait
        {
	        DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
        	work((i+1)*W);
	}
    }

/*
 *./exercise6.exe
 *
 *
 *============================
 *Test - Iteration 0...
 *============================
 *0: I am executing iteration 0!
 *3: I am executing iteration 3!
 *2: I am executing iteration 2!
 *1: I am executing iteration 1!
 *
 *
 *============================
 *Test - Iteration 1...
 *============================
 *0: I am executing iteration 2!
 *3: I am executing iteration 2!
 *2: I am executing iteration 3!
 *1: I am executing iteration 3!
 */
#endif

#if 1
#pragma omp parallel num_threads(4)
#pragma omp single nowait
    for (i = 0; i < 4; i++)
    {
#pragma omp task
        {
            DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
            work((i + 1) * W);
        }
    }
#endif
}
