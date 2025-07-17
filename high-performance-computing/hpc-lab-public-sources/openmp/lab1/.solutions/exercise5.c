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
 * @file exercise5.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 5
 * 
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

#if !defined(W)
#define W (1 << 15)
#endif

/* Dummy Tasks */
void task1();
void task2();
void task3();
void task4();

/**
 * @brief EX 5 - Task Parallelism w/sections
 * 
 * a) Create a parallel region with 4 threads. Use thread IDs to execute
 *    different WORK functions on different threads.
 * b) Create a parallel region with 4 threads. Achieve the same work partitioning
 *    as a) using SECTIONS.
 * 
 * @return void
 */
void exercise()
{
#if 0 //5a
#pragma omp parallel num_threads(4)
{
    if(omp_get_thread_num()==0)
        task1();
    
    if(omp_get_thread_num()==1)
    task2();

    if(omp_get_thread_num()==2)
    task3();

    if(omp_get_thread_num()==3)
    task4();
}
#endif

#if 1 //5b
#pragma omp parallel sections
    {
#pragma omp section
        task1();

#pragma omp section
        task2();

#pragma omp section
        task3();

#pragma omp section
        task4();
    }
#endif
}

void task1()
{
    DEBUG_PRINT("%hu: exec task1!\n", omp_get_thread_num());
    work((1 * W));
}

void task2()
{
    DEBUG_PRINT("%hu: exec task2!\n", omp_get_thread_num());
    work((2 * W));
}

void task3()
{
    DEBUG_PRINT("%hu: exec task3!\n", omp_get_thread_num());
    work((3 * W));
}

void task4()
{
    DEBUG_PRINT("%hu: exec task4!\n", omp_get_thread_num());
    work((4 * W));
}
