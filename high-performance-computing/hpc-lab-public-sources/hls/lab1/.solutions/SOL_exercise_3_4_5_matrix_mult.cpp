#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 64

const unsigned int max_size = MAX_SIZE;

void mmult( int *in1,
            int *in2,
            int *out,
            int dim
              )
{
#pragma HLS INTERFACE m_axi port=in1 offset=slave bundle=in1_mem
#pragma HLS INTERFACE m_axi port=in2 offset=slave bundle=in2_mem
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=out_mem

#pragma HLS INTERFACE s_axilite port=dim bundle=params
#pragma HLS INTERFACE s_axilite port=return bundle=params

	int in_1_loc [MAX_SIZE][MAX_SIZE];
	int in_2_loc [MAX_SIZE][MAX_SIZE];
	int  out_loc [MAX_SIZE][MAX_SIZE];

	#pragma HLS ARRAY_PARTITION variable=in_1_loc dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=in_2_loc dim=1 complete

	memcpy_1: memcpy(in1, in_1_loc, MAX_SIZE*MAX_SIZE*sizeof(int));
	memcpy_2: memcpy(in2, in_2_loc, MAX_SIZE*MAX_SIZE*sizeof(int));

    loop_1: for (int i = 0; i < dim; i++){
		#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size
    	loop_2: for (int j = 0; j < dim; j++){

			#pragma HLS PIPELINE
    		#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size

    		loop_3: for (int k = 0; k < MAX_SIZE; k++){

				#pragma HLS LOOP_TRIPCOUNT max=max_size min=max_size

    			out_loc[i][j] += in_1_loc[i][k] * in_2_loc[k][j];
            }
        }
    }
    memcpy_3: memcpy(out_loc, out, MAX_SIZE*MAX_SIZE*sizeof(int));
}
