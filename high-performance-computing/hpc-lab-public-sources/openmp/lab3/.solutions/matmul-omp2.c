#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef N
#define N (1 << 10)
#endif

#pragma omp declare target
#define SM 64

#define NTHRDS7 (1 << 0x7) /* 2^{7}  */
#define NTHRDS8 (1 << 0x8) /* 2^{8}  */
#define NTHRDS9 (1 << 0x9) /* 2^{9}  */

#define LTEAMSD (1 << 0xD) /* 2^{13} */
#define LTEAMSE (1 << 0xE) /* 2^{14} */
#define LTEAMSF (1 << 0xF) /* 2^{15} */
#define LTEAMSG (1 << 020) /* 2^{16} */

#define BLKROW (512) /* 4x number of threads in each team */
#define BLKDIM (16)

void gemm_accel_opt2(float *restrict a, float *restrict b, float *restrict c, int n)
{
/*
 * - jik-loop
 * - 2^7 threads per team and 2^13 teams
 * - collapse(3)
 * - 4x j-loop unrolling (stride of 1   col )
 * - 4x i-loop unrolling (stride of 2^7 rows)
 * - 4x k-loop unrolling
 * - rb: 4x data re-use
 * - ra: 4x data re-use
 * - register blocking
 */
#pragma omp target data                           \
    map(to                                        \
        : n, a [0:n * n], b [0:n * n]) map(tofrom \
                                           : c [0:n * n])
    {
#pragma omp target teams num_teams(LTEAMSD) thread_limit(NTHRDS7) \
    map(to                                                        \
        : n, a [0:n * n], b [0:n * n]) map(tofrom                 \
                                           : c [0:n * n]) default(none) shared(a, b, c, n)
#pragma omp distribute parallel for num_threads(NTHRDS7) \
    dist_schedule(static, NTHRDS7) collapse(3) default(none) shared(a, b, c, n)
        for (int j = 0; j < n; j += 4)
        { /* 4x unrolling */
            for (int iblk = 0; iblk < n / BLKROW; ++iblk)
            {
                for (int i = 0; i < NTHRDS7; ++i)
                { /* 4x unrolling */
                    /* register for c: 4x j-loop * 4x i-loop */
                    float rc0, rc1, rc2, rc3,
                        rc4, rc5, rc6, rc7,
                        rc8, rc9, rca, rcb,
                        rcc, rcd, rce, rcf;
                    rc0 = c[j * n + iblk * BLKROW + i];
                    rc1 = c[j * n + iblk * BLKROW + i + NTHRDS7];
                    rc2 = c[j * n + iblk * BLKROW + i + NTHRDS7 * 2];
                    rc3 = c[j * n + iblk * BLKROW + i + NTHRDS7 * 3];
                    rc4 = c[(j + 1) * n + iblk * BLKROW + i];
                    rc5 = c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7];
                    rc6 = c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 2];
                    rc7 = c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 3];
                    rc8 = c[(j + 2) * n + iblk * BLKROW + i];
                    rc9 = c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7];
                    rca = c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 2];
                    rcb = c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 3];
                    rcc = c[(j + 3) * n + iblk * BLKROW + i];
                    rcd = c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7];
                    rce = c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 2];
                    rcf = c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 3];
                    for (int k = 0; k < n; k += 4)
                    { /* 4x unrolling */
                        /* register for b: 4x j-loop * 4x k-loop */
                        float rb0, rb1, rb2, rb3,
                            rb4, rb5, rb6, rb7,
                            rb8, rb9, rba, rbb,
                            rbc, rbd, rbe, rbf;
                        rb0 = b[j * n + k];
                        rb1 = b[j * n + k + 1];
                        rb2 = b[j * n + k + 2];
                        rb3 = b[j * n + k + 3];
                        rb4 = b[(j + 1) * n + k];
                        rb5 = b[(j + 1) * n + k + 1];
                        rb6 = b[(j + 1) * n + k + 2];
                        rb7 = b[(j + 1) * n + k + 3];
                        rb8 = b[(j + 2) * n + k];
                        rb9 = b[(j + 2) * n + k + 1];
                        rba = b[(j + 2) * n + k + 2];
                        rbb = b[(j + 2) * n + k + 3];
                        rbc = b[(j + 3) * n + k];
                        rbd = b[(j + 3) * n + k + 1];
                        rbe = b[(j + 3) * n + k + 2];
                        rbf = b[(j + 3) * n + k + 3];
                        /* register for a: 4x i-loop * 4x k-loop */
                        float ra0, ra1, ra2, ra3,
                            ra4, ra5, ra6, ra7,
                            ra8, ra9, raa, rab,
                            rac, rad, rae, raf;
                        ra0 = a[k * n + iblk * BLKROW + i];
                        ra1 = a[k * n + iblk * BLKROW + i + NTHRDS7];
                        ra2 = a[k * n + iblk * BLKROW + i + NTHRDS7 * 2];
                        ra3 = a[k * n + iblk * BLKROW + i + NTHRDS7 * 3];
                        ra4 = a[(k + 1) * n + iblk * BLKROW + i];
                        ra5 = a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7];
                        ra6 = a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7 * 2];
                        ra7 = a[(k + 1) * n + iblk * BLKROW + i + NTHRDS7 * 3];
                        ra8 = a[(k + 2) * n + iblk * BLKROW + i];
                        ra9 = a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7];
                        raa = a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7 * 2];
                        rab = a[(k + 2) * n + iblk * BLKROW + i + NTHRDS7 * 3];
                        rac = a[(k + 3) * n + iblk * BLKROW + i];
                        rad = a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7];
                        rae = a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7 * 2];
                        raf = a[(k + 3) * n + iblk * BLKROW + i + NTHRDS7 * 3];
                        /*
     * register blocking
     */
                        // col 1 of c:
                        rc0 += ra0 * rb0;
                        rc0 += ra4 * rb1;
                        rc0 += ra8 * rb2;
                        rc0 += rac * rb3;
                        rc1 += ra1 * rb0;
                        rc1 += ra5 * rb1;
                        rc1 += ra9 * rb2;
                        rc1 += rad * rb3;
                        rc2 += ra2 * rb0;
                        rc2 += ra6 * rb1;
                        rc2 += raa * rb2;
                        rc2 += rae * rb3;
                        rc3 += ra3 * rb0;
                        rc3 += ra7 * rb1;
                        rc3 += rab * rb2;
                        rc3 += raf * rb3;
                        // col 2 of c:
                        rc4 += ra0 * rb4;
                        rc4 += ra4 * rb5;
                        rc4 += ra8 * rb6;
                        rc4 += rac * rb7;
                        rc5 += ra1 * rb4;
                        rc5 += ra5 * rb5;
                        rc5 += ra9 * rb6;
                        rc5 += rad * rb7;
                        rc6 += ra2 * rb4;
                        rc6 += ra6 * rb5;
                        rc6 += raa * rb6;
                        rc6 += rae * rb7;
                        rc7 += ra3 * rb4;
                        rc7 += ra7 * rb5;
                        rc7 += rab * rb6;
                        rc7 += raf * rb7;
                        // col 3 of c:
                        rc8 += ra0 * rb8;
                        rc8 += ra4 * rb9;
                        rc8 += ra8 * rba;
                        rc8 += rac * rbb;
                        rc9 += ra1 * rb8;
                        rc9 += ra5 * rb9;
                        rc9 += ra9 * rba;
                        rc9 += rad * rbb;
                        rca += ra2 * rb8;
                        rca += ra6 * rb9;
                        rca += raa * rba;
                        rca += rae * rbb;
                        rcb += ra3 * rb8;
                        rcb += ra7 * rb9;
                        rcb += rab * rba;
                        rcb += raf * rbb;
                        // col 4 of c:
                        rcc += ra0 * rbc;
                        rcc += ra4 * rbd;
                        rcc += ra8 * rbe;
                        rcc += rac * rbf;
                        rcd += ra1 * rbc;
                        rcd += ra5 * rbd;
                        rcd += ra9 * rbe;
                        rcd += rad * rbf;
                        rce += ra2 * rbc;
                        rce += ra6 * rbd;
                        rce += raa * rbe;
                        rce += rae * rbf;
                        rcf += ra3 * rbc;
                        rcf += ra7 * rbd;
                        rcf += rab * rbe;
                        rcf += raf * rbf;
                    }
                    c[j * n + iblk * BLKROW + i] = rc0;
                    c[j * n + iblk * BLKROW + i + NTHRDS7] = rc1;
                    c[j * n + iblk * BLKROW + i + NTHRDS7 * 2] = rc2;
                    c[j * n + iblk * BLKROW + i + NTHRDS7 * 3] = rc3;
                    c[(j + 1) * n + iblk * BLKROW + i] = rc4;
                    c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7] = rc5;
                    c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 2] = rc6;
                    c[(j + 1) * n + iblk * BLKROW + i + NTHRDS7 * 3] = rc7;
                    c[(j + 2) * n + iblk * BLKROW + i] = rc8;
                    c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7] = rc9;
                    c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 2] = rca;
                    c[(j + 2) * n + iblk * BLKROW + i + NTHRDS7 * 3] = rcb;
                    c[(j + 3) * n + iblk * BLKROW + i] = rcc;
                    c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7] = rcd;
                    c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 2] = rce;
                    c[(j + 3) * n + iblk * BLKROW + i + NTHRDS7 * 3] = rcf;
                } /* end i-loop */
            }     /* end iblk-loop */
        }         /* end j-loop */
    }
}

void gemm_cublas(float *restrict a, float *restrict b, float *restrict c, int n)
{
    cublasHandle_t handle;
    float alfa = 1.0f,
          beta = 1.0f,
          *a_dev = NULL,
          *b_dev = NULL,
          *c_dev = NULL;
    /*
 * cublasSgemm in CUBLAS
 */
    if (CUBLAS_STATUS_SUCCESS != cublasCreate(&handle))
    {
        printf("error: initialization (CUBLAS)\n");
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaMalloc((void **)&a_dev, sizeof(*a) * n * n) ||
        cudaSuccess != cudaMalloc((void **)&b_dev, sizeof(*b) * n * n) ||
        cudaSuccess != cudaMalloc((void **)&c_dev, sizeof(*c) * n * n))
    {
        printf("error: memory allocation (CUDA)\n");
        cudaFree(a_dev);
        cudaFree(b_dev);
        cudaFree(c_dev);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    if (CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*a), a, n, a_dev, n) ||
        CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*b), b, n, b_dev, n) ||
        CUBLAS_STATUS_SUCCESS != cublasSetMatrix(n, n, sizeof(*c), c, n, c_dev, n))
    {
        printf("error: host --> accl (CUBLAS)\n");
        cudaFree(a_dev);
        cudaFree(b_dev);
        cudaFree(c_dev);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    if (CUBLAS_STATUS_SUCCESS != cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             n, n, n, &alfa, a_dev, n, b_dev, n, &beta, c_dev, n))
    {
        printf("error: cublasSgemm (CUBLAS)\n");
        cudaFree(a_dev);
        cudaFree(b_dev);
        cudaFree(c_dev);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    if (cudaSuccess != cudaDeviceSynchronize())
    {
        printf("error: device synchronization (CUDA)\n");
        cudaFree(a_dev);
        cudaFree(b_dev);
        cudaFree(c_dev);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    if (CUBLAS_STATUS_SUCCESS != cublasGetMatrix(n, n, sizeof(*c), c_dev, n, c, n))
    {
        printf("error: accl --> host (CUBLAS)\n");
        cudaFree(a_dev);
        cudaFree(b_dev);
        cudaFree(c_dev);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cublasDestroy(handle);
}

static void reorder2(float *restrict a, float *restrict b, int n)
{
    for (int i = 0; i < SM; i++)
        for (int j = 0; j < SM; j++)
            b[i * SM + j] = a[i * n + j];
}

static void kernel(float *restrict a, float *restrict b, float *restrict c, int n)
{
    for (int i = 0; i < SM; i++)
    {
        for (int k = 0; k < SM; k++)
        {
            for (int j = 0; j < SM; j++)
            {
                c[i * n + j] += a[i * n + k] * b[k * SM + j];
            }
        }
    }
}

void gemm_accel_opt(float *restrict a, float *restrict b, float *restrict c, int n)
{
#pragma omp target teams distribute parallel for collapse(3) map(to                                      \
                                                                 : n, a [0:n * n], b [0:n * n]) map(from \
                                                                                                    : c [0:n * n]) schedule(static, 1)
    for (int i = 0; i < n / SM; i++)
    {
        for (int j = 0; j < n / SM; j++)
        {
            for (int k = 0; k < n / SM; k++)
            {
                float b2[SM * SM];
                reorder2(&b[SM * (k * n + j)], b2, n);
                kernel(&a[SM * (i * n + k)], b2, &c[SM * (i * n + j)], n);
            }
        }
    }
}

#pragma omp end declare target

void gemm_opt(float *restrict a, float *restrict b, float *restrict c, int n)
{
    int bk = n / SM;
#pragma omp parallel
    {
        float b2[SM * SM];
#pragma omp for collapse(3)
        for (int i = 0; i < bk; i++)
        {
            for (int j = 0; j < bk; j++)
            {
                for (int k = 0; k < bk; k++)
                {
                    reorder2(&b[SM * (k * n + j)], b2, n);
                    kernel(&a[SM * (i * n + k)], b2, &c[SM * (i * n + j)], n);
                }
            }
        }
    }
}

void gemm(float *restrict a, float *restrict b, float *restrict c, int n)
{
    int i, j, k;
#pragma omp parallel for simd collapse(2) schedule(simd \
                                                   : static)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < n; ++k)
            {
                sum += a[i + k * n] * b[k + j * n];
            }
            c[i * n + j] += sum;
        }
    }
}

int main(int argc, char *argv[])
{
    int i, n = N,
           iret = 0;
    float *a, *b, *c, *g;
    struct timespec rt[2];
    double wt; // walltime

    if (argc > 1)
        n = atoi(argv[1]);

    /*
   * 0. prepare x, y, and z
   *
   * y := a * x + y (on host)
   * z := a * x + z (on accel)
   */
    if (NULL == (a = (float *)malloc(sizeof(*a) * n * n)))
    {
        printf("error: memory allocation for 'x'\n");
        iret = -1;
    }
    if (NULL == (b = (float *)malloc(sizeof(*b) * n * n)))
    {
        printf("error: memory allocation for 'y'\n");
        iret = -1;
    }
    if (NULL == (c = (float *)malloc(sizeof(*c) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }
    if (NULL == (g = (float *)malloc(sizeof(*g) * n * n)))
    {
        printf("error: memory allocation for 'z'\n");
        iret = -1;
    }

    if (0 != iret)
    {
        free(a);
        free(b);
        free(c);
        free(g);
        exit(EXIT_FAILURE);
    }

    if (n <= 1024)
    {
        clock_gettime(CLOCK_REALTIME, rt + 0);
        gemm(a, b, c, n);
        clock_gettime(CLOCK_REALTIME, rt + 1);
        wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        printf("gemm on host : %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));
    }

    if (n <= 4096)
    {
        clock_gettime(CLOCK_REALTIME, rt + 0);
        gemm_opt(a, b, c, n);
        clock_gettime(CLOCK_REALTIME, rt + 1);
        wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        printf("gemm_opt on host : %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));
    }

#if 0
#pragma omp target teams distribute parallel for map(to                                   \
                                                     : a [0:n * n], b [0:n * n]) map(from \
                                                                                     : c [0:n * n]) collapse(2)
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < n; ++j){
                float sum = 0.0;
                for(int k = 0; k < n; ++k){

                    sum += a[i+k*n]*b[k+j*n];
                }
                c[i*n+j] += sum;
            }
        }
#endif

    if (n <= 4096)
    {
        clock_gettime(CLOCK_REALTIME, rt + 0);
        gemm_accel_opt(a, b, c, n);
        clock_gettime(CLOCK_REALTIME, rt + 1);
        wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
        printf("GEMM-opt1 on accel: %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));

        for (i = 0; i < n; i++)
        {
            iret = *(int *)(g + i) ^ *(int *)(c + i);
            assert(iret == 0);
        }
    }
    clock_gettime(CLOCK_REALTIME, rt + 0);
    gemm_accel_opt2(a, b, c, n);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("GEMM-opt2 on accel: %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));

    if (n <= 4096)
        for (i = 0; i < n; i++)
        {
            iret = *(int *)(g + i) ^ *(int *)(c + i);
            assert(iret == 0);
        }

    clock_gettime(CLOCK_REALTIME, rt + 0);
    gemm_cublas(a, b, c, n);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("CUBLAS on accel: %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n * n * n / (1.0e6 * wt));

    if (n <= 4096)
        for (i = 0; i < n; i++)
        {
            iret = *(int *)(g + i) ^ *(int *)(c + i);
            assert(iret == 0);
        }

    free(a);
    free(b);
    free(c);
    free(g);

    return 0;
}
