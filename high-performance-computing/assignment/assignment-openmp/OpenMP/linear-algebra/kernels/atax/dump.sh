#!/bin/bash

make EXT_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DMINI_DATASET -D"$1"" CC=clang clean all
./atax_acc 2> ./dump/mini/"$1"
make EXT_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DSMALL_DATASET -D"$1"" CC=clang clean all
./atax_acc 2> ./dump/small/"$1"
make EXT_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DSTANDARD_DATASET -D"$1"" CC=clang clean all
./atax_acc 2> ./dump/standard/"$1"
make EXT_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -D"$1"" CC=clang clean all
./atax_acc 2> ./dump/large/"$1"
make EXT_CFLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DEXTRALARGE_DATASET -D"$1"" CC=clang clean all
./atax_acc 2> ./dump/extralarge/"$1"

