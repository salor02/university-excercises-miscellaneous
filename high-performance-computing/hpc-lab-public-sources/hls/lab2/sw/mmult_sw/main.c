#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xtmrctr.h"
#include "xparameters.h"
#include "xil_cache.h"
#include "xil_io.h"

#define MAX_SIZE 64

void mmult_software(
		int* in1,   // Input matrix 1
		int* in2,   // Input matrix 2
		int* out,   // Output matrix (out = A x B)
		int dim     // Size of one dimension of matrix
              )
{
    //Performs matrix multiplication out = in1 x in2
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < dim; j++){
            for (int k = 0; k < dim; k++){
                out[i * dim + j] += in1[i * dim + k] * in2[k * dim  + j];
            }
        }
    }
}

int main()
{
    init_platform();

    printf("MMULT benchmark size: %d\n", MAX_SIZE);

    int in1[MAX_SIZE*MAX_SIZE];
    int in2[MAX_SIZE*MAX_SIZE];
    int out[MAX_SIZE*MAX_SIZE];

    printf("in1: %p\n", in1);
    printf("in2: %p\n", in2);
    printf("out: %p\n", out);

    for(int i = 0; i < MAX_SIZE*MAX_SIZE; i++) {
    	in1[i] = i;
    	in2[i] = i;
    	out[i] = 0;
    }

    XTmrCtr timer;
    XTmrCtr_Initialize(&timer, XPAR_AXI_TIMER_0_DEVICE_ID);

    XTmrCtr_Start(&timer, 0);
    mmult_software(in1, in2, out, MAX_SIZE);
    XTmrCtr_Stop(&timer, 0);

    printf("software: %d\n", XTmrCtr_GetValue(&timer, 0));

//    printf("out\n");
//    for(int i = 0; i < MAX_SIZE*MAX_SIZE; i++) {
//    	printf("%d \n", out[i]);
//    }
//    printf("\n");

    cleanup_platform();

    return 0;
}
