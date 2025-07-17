#include <string.h>
#include <limits.h>

#define NUM_DIMS 	5
#define NUM_POINTS 	512

const unsigned int num_dims   = NUM_DIMS;
const unsigned int num_points = NUM_POINTS;
const unsigned int max_iterations = num_dims * num_points;

void nearest_neighbor(int *out, const int *points,
        const int *search_point, const int len,
        const int dim){

	#pragma HLS INTERFACE m_axi port=out 		  offset=slave bundle=out_mem
	#pragma HLS INTERFACE m_axi port=points 	  offset=slave bundle=points_mem
	#pragma HLS INTERFACE m_axi port=search_point offset=slave bundle=search_point_mem

	#pragma HLS INTERFACE s_axilite port=len 	bundle=param
	#pragma HLS INTERFACE s_axilite port=dim 	bundle=param
	#pragma HLS INTERFACE s_axilite port=return bundle=param

    int best_i = 0;
    int best_dist = INT_MAX;

    int dist = 0;
    int iterations = len * dim;

    find_best: for (int p = 0, c = 0, itr = 0; itr < iterations; itr++) {

        	#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT max=max_iterations min=max_iterations

            int dx = points[dim * p + c] - search_point[c];
            dist += dx * dx;
            if (c == dim - 1) {
                if (dist < best_dist) {
                    best_i = p;
                    best_dist = dist;
                }
                c = 0;
                dist = 0;
                p++;
            } else {
                c++;
            }
        }
        write_best:
        for (int c = 0; c < dim; ++c) {

        	#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT max=num_dims min=num_dims

            out[c] = points[best_i * dim + c];
        }
}
