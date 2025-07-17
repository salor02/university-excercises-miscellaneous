#define SIZE 	128
#define N 		10

void fir(int * input, int * output) {

#pragma HLS INTERFACE m_axi port=input  offset=slave  bundle=input_mem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=output_mem

#pragma HLS INTERFACE s_axilite port=return bundle=params

	int coeff[N] = {13, -2, 9, 11, 26, 18, 95, -43, 6, 74};

	for (int n = 0; n < SIZE; n++) {
		int acc = 0;
		for (int i= 0; i< N; i++ ) {
			if  (n - i >= 0)
				acc += coeff[i] * input[n-i];
		}
		output[n] = acc;
	}
}
