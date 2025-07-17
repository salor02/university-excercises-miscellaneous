#define TEST_DATA_SIZE 4194304 // 2^22

const unsigned int c_dim = TEST_DATA_SIZE;

void vadd(int *a, int *b, int *c, const int len)
{
    //TODO: split bundles on three different AXI4 bus
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=mem
    #pragma HLS INTERFACE s_axilite port=len bundle=params
    #pragma HLS INTERFACE s_axilite port=return bundle=params

    loop: for(int i = 0; i < len; i++) {
        //TODO: insert pipeline directive
	    #pragma HLS LOOP_TRIPCOUNT min=c_dim max=c_dim
        c[i] = a[i] + b[i];
    }
}
