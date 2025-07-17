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
 * @file exercise7.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 7
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

#if !defined(W)
#define W (1000)
#endif

#if !defined(T)
#define T (20)
#endif

/**
 * @brief EX 7 - Task Parallelism w/tasks
 * 
 * a) Parallelize with TASK directive.
 * b) Parallelize with SINGLE NOWAIT directive.
 * c) Compare Results with a for loop.
 * @return void
 */
void exercise()
{
    unsigned int i;

#if 1
#pragma omp parallel
#pragma omp single nowait
    for (i = 0; i < 16384; i++)
    {
#pragma omp task
        {
            DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
            work(W);
        }
    }
#endif

#if 0
#pragma omp parallel
    for (i = 0; i < 16384; i++)
    {
#pragma omp single nowait
        {
            DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
            work(W);
        }
    }
#endif

#if 0
#pragma omp parallel for
    for (i = 0; i < 16384; i++)
    {
        DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
        work(W);
    }
#endif
}
