#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xmmult.h"
#include "xil_cache.h"
#include "xtmrctr.h"


int main()
{
    init_platform();

    XTmrCtr timer;
    XTmrCtr_Initialize(&timer, XPAR_AXI_TIMER_0_DEVICE_ID);

    const int MAX_DIM = 64;

    int in1[MAX_DIM*MAX_DIM];
    int in2[MAX_DIM*MAX_DIM];
    int out[MAX_DIM*MAX_DIM];

    for(int i = 0; i < MAX_DIM*MAX_DIM; i++) {
    	in1[i] = i;
    	in2[i] = i;
    	out[i] = 0;
    }

    XTmrCtr_Start(&timer, 0);
    Xil_DCacheFlush();

    XMmult mmult;
    XMmult_Initialize(&mmult, XPAR_MMULT_0_DEVICE_ID);

    XMmult_Set_in1(&mmult, (u32)in1);
    XMmult_Set_in2(&mmult, (u32)in2);
    XMmult_Set_out_r(&mmult, (u32)out);
    XMmult_Set_dim(&mmult, MAX_DIM);

    XMmult_Start(&mmult);

    while(!XMmult_IsDone(&mmult));

    Xil_DCacheInvalidate();

    XTmrCtr_Stop(&timer, 0);

//    for(int i = 0; i < MAX_DIM*MAX_DIM; i++) {
//    	printf("%d\n", out[i]);
//    }

    printf("clocks: %d\n", XTmrCtr_GetValue(&timer, 0));

    cleanup_platform();

    return 0;
}
