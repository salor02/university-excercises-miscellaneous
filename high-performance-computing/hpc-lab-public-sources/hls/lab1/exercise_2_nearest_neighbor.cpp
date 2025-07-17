#include <string.h>
#include <limits.h>

#define NUM_DIMS 	5
#define NUM_POINTS 	512

const unsigned int num_dims   = NUM_DIMS;
const unsigned int num_points = NUM_POINTS;

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

	// TODO: merge upper_loop and lower loop
	// TODO: insert pipeline directive
	upper_loop:for(int p = 0 ; p < len; ++p){

		#pragma HLS LOOP_TRIPCOUNT max=num_points min=num_points

		int dist = 0;

		lower_loop:for(int c = 0 ; c < dim ; c++){

			#pragma HLS LOOP_TRIPCOUNT max=num_dims min=num_dims

        	int dx = points[dim*p + c] - search_point[c];
        	dist += dx * dx;
        }

        if (dist < best_dist){
        	best_i = p;
        	best_dist = dist;
        }
	}
	//TODO: insert pipeline directive
	write_best: for (int c = 0; c < dim; ++c) {
		#pragma HLS LOOP_TRIPCOUNT max=num_dims min=num_dims
		out[c] = points[best_i * dim + c];
	}
}
