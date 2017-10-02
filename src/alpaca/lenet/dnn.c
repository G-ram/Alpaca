#include <msp430.h>
#include <stdlib.h>

#include <libio/log.h>
#include <libalpaca/alpaca.h>
#include <libmspbuiltins/builtins.h>
#include <libmsp/mem.h>
#include <libmsp/periph.h>
#include <libmsp/clock.h>
#include <libmsp/watchdog.h>
#include <libmsp/gpio.h>
#include <libmspmath/msp-math.h>

#include "types.h"
#include "fixed.h"
#include "mat.h"
#include "blas.h"

#include "headers/conv1.h"
// #include "headers/conv2.h"
#include "headers/pr.h"
#include "headers/pred.h"
#include "headers/input.h"

#include "dnn.h"

GLOBAL_SB(fixed, data1, 20 * 24 * 24);
GLOBAL_SB(fixed, data2, 100 * 8 * 8);
GLOBAL_SB(mat, buf1);
GLOBAL_SB(mat, buf2);
GLOBAL_SB(mat *, src);
GLOBAL_SB(mat *, dest);
GLOBAL_SB(uint, layer);
GLOBAL_SB(uint, label);

TASK(1, task_init);
TASK(2, task_infer);
TASK(3, task_conv1_layer);
TASK(4, task_conv2_layer);
TASK(5, task_pr_layer);
TASK(6, task_pred_layer);
TASK(7, task_end);

static void init_hw() {
	msp_watchdog_disable();
	msp_gpio_unlock();
	msp_clock_setup();
}

void init() {
	init_hw();

	INIT_CONSOLE();

	__enable_interrupt();

    PRINTF(".%u.\r\n", curctx->task->idx);
}

void task_init() {
	GV(src) = &GV(buf1);
	GV(dest) = &GV(buf2);
	GV(src)->data = GV(data1);
	GV(dest)->data = GV(data2);
	GV(src) = &GV(buf1);
	GV(dest) = &GV(buf2);
	GV(layer) = 0;
	GV(label) = 0;
	// unsigned long base_addr = 0x20000;
	// PRINTF("Val: %u\r\n", read_addr(base_addr));
	TRANSITION_TO(task_infer);
}

void task_infer() {
	// Init two buffers
	if(GV(layer) == 0) {
		GV(layer)++;
		MAT_RESHAPE(GV(dest), 28, 28);
		for(uint i = 0; i < 28; i ++) {
			for(uint j = 0; j < 28; j ++) {
				MAT_SET(GV(dest), F_LIT(input[0][i][j]), i, j);
			}
		}
		TRANSITION_TO(task_infer);
	}
	else if(GV(layer) == 1) {
		GV(layer)++;
		SWAP_BUF(GV(src), GV(dest));
		MAT_RESHAPE(GV(dest), 20, 24, 24);
		TRANSITION_TO(task_conv1_layer);
	}
	else if(GV(layer) == 2) {
		GV(layer)++;
		SWAP_BUF(GV(src), GV(dest));
		MAT_RESHAPE(GV(dest), 20, 12, 12);
		// PROBLEM
		pool(GV(src), 2, 2, GV(dest));
		TRANSITION_TO(task_infer);
	}
	else if(GV(layer) == 3) {
		GV(layer)++;
		SWAP_BUF(GV(src), GV(dest));
		MAT_RESHAPE(GV(dest), 100, 8, 8);
		TRANSITION_TO(task_conv2_layer);
	}
	else if(GV(layer) == 4) {
		GV(layer)++;
		SWAP_BUF(GV(src), GV(dest));
		MAT_RESHAPE(GV(dest), 100, 4, 4);
		// PROBLEM
		pool(GV(src), 2, 2, GV(dest));
		TRANSITION_TO(task_infer);
	}
	else if(GV(layer) == 5) {
		GV(layer)++;
		SWAP_BUF(GV(src), GV(dest));
		MAT_RESHAPE(GV(dest), 500);
		TRANSITION_TO(task_pr_layer);
	}
	else if(GV(layer) == 6) {
		GV(layer)++;
		SWAP_BUF(GV(src), GV(dest));
		MAT_RESHAPE(GV(dest), 500);
		// PROBLEM
		relu(GV(src), GV(dest));
		TRANSITION_TO(task_infer);
	}
	else if(GV(layer) == 7) {
		GV(layer)++;
		SWAP_BUF(GV(src), GV(dest));
		MAT_RESHAPE(GV(dest), 10);
		TRANSITION_TO(task_pred_layer);
	} else {
		GV(layer)++;
		float max = 0.;
		uint idx = 0;
		for(uint i = 0; i < 10; i ++) {
			float prob = F_TO_FLOAT(MAT_GET(GV(dest), i));
			if(max < prob) {
				idx = i;
				max = prob;
			}
			LOG("%u => %f\n", i, prob);
		}
		GV(label) = idx;
		TRANSITION_TO(task_end);
	}
}

void task_conv1_layer() {
	fixed data[24 * 24];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 24, 24);
	for(uint i = 0; i < 20; i ++) {
		convolve2d(GV(src), 5, conv1_w[i], inter);
		MAT_CONSTRAIN(GV(dest), i);
		bias2d(inter, conv1_b[i], GV(dest));
		MAT_UNCONSTRAIN(GV(dest));
	}
	TRANSITION_TO(task_infer);
}

void task_conv2_layer() {
	fixed data[8 * 8];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 8, 8);
	for(uint i = 0; i < 100; i ++) {
		// convolve3d(GV(src), 5, conv2_w[i], inter);
		MAT_CONSTRAIN(GV(dest), i);
		// bias2d(inter, conv2_b[i], GV(dest));
		MAT_UNCONSTRAIN(GV(dest));
	}
	TRANSITION_TO(task_infer);
}

void task_pr_layer() {
	// First we need to collapse src to a 1D vector
	MAT_RESHAPE(GV(src), 100 * 4 * 4);

	fixed data[500];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 500);
	sparse_mul_vector(500, pr_w, pr_idx, pr_ptr, GV(src), inter);

	bias1d(inter, pr_b,GV(dest));
	TRANSITION_TO(task_infer);
}

void task_pred_layer() {
	fixed data[10];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 10);
	mul_vector(10, 500, pred_w, GV(src), inter);

	bias1d(inter, pred_b, GV(dest));
	TRANSITION_TO(task_infer);
}

void task_end() {
	PRINTF("Label: %d Actual: %d\r\n", GV(label), 5);
	PRINTF("===================\r\n");
	exit(0);
}

ENTRY_TASK(task_init)
INIT_FUNC(init)



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////BLAS_C//////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void convolve2d(mat *src, uint size, const fixed filter[][size], mat *dest) {
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	uint drows = MAT_GET_DIM(dest, 0);
	uint dcols = MAT_GET_DIM(dest, 1);
	for(uint i = 0; i < drows; i ++) {
		for(uint j = 0; j < dcols; j ++) {
			MAT_SET(dest, F_LIT(0.0), i, j);
			for(int k = 0; k < size; k ++) {
				int irow_idx = i + k;
				for(int l = 0; l < size; l ++) {
					int icol_idx = j + l;
					if(irow_idx >= 0 && irow_idx < rows && icol_idx >= 0 && 
						icol_idx < cols) {
						fixed w = F_ADD(MAT_GET(dest, i, j),
							F_MUL(MAT_GET(src, irow_idx, icol_idx), filter[k][l]));
						MAT_SET(dest, w, i, j);
					}
				}
			}
		}
	}
}

void convolve3d(mat *src, uint size, const fixed filter[][size][size], mat *dest) {
	uint layers = MAT_GET_DIM(src, 0);
	uint drows = MAT_GET_DIM(dest, 0);
	uint dcols = MAT_GET_DIM(dest, 1);
	for(uint i = 0; i < drows; i ++) {
		for(uint j = 0; j < dcols; j ++) {
			MAT_SET(dest, F_LIT(0.0), i, j);
		}
	}
	fixed data[drows * dcols];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, drows, dcols);
	for(uint i = 0; i < layers; i++) {
		MAT_CONSTRAIN(src, i);
		convolve2d(src, size, filter[i], inter);
		MAT_UNCONSTRAIN(src);
		for(uint j = 0; j < drows; j ++) {
			for(uint k = 0; k < dcols; k ++) {
				fixed w = F_ADD(MAT_GET(dest, j, k), MAT_GET(inter, j, k));
				MAT_SET(dest, w, j, k);
			}
		}
	}
}

void mul_vector(uint rows, uint cols, const fixed mat_data[][cols], mat *vector, 
	mat *dest) {
	for(uint i = 0; i < rows; i ++) {
		MAT_SET(dest, F_LIT(0.0), i);
		for(uint j = 0; j < cols; j ++) {
			fixed w = F_ADD(MAT_GET(dest, i), F_MUL(mat_data[i][j], MAT_GET(vector, j))); 
			MAT_SET(dest, w, i);
		}
	}
}

void sparse_mul_vector(uint rows, const fixed mat_data[], const uint mat_idx[], 
	const uint mat_ptr[], mat *vector, mat *dest) {
	for(uint i = 0; i < rows; i ++) {
		MAT_SET(dest, F_LIT(0.0), i);
		for(uint j = mat_ptr[i]; j < mat_ptr[i + 1]; j ++) {
			fixed w = F_ADD(MAT_GET(dest, i), F_MUL(mat_data[j], MAT_GET(vector, mat_idx[j]))); 
			MAT_SET(dest, w, i);
		}
	}
}

void bias2d(mat *src, const fixed bias, mat *dest) {
	uint rows = MAT_GET_DIM(src, 0); 
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = 0; i < rows; i ++) {
		for(uint j = 0; j < cols; j ++) {
			MAT_SET(dest, F_ADD(MAT_GET(src, i, j), bias), i, j);
		}
	}
}

void bias1d(mat *src, const fixed bias[], mat *dest) {
	uint rows = MAT_GET_DIM(src, 0); 
	for(uint i = 0; i < rows; i ++) {
		MAT_SET(dest, F_ADD(MAT_GET(src, i), bias[i]), i);
	}
}

void pool(mat *src, uint size, uint stride, mat *dest) {
	uint layers = MAT_GET_DIM(src, 0);
	uint rows = MAT_GET_DIM(src, 1);
	uint cols = MAT_GET_DIM(src, 2);
	for(uint i = 0; i < layers; i ++) {
		for(uint j = 0; j < rows; j += stride) {
			for(uint k = 0; k < cols; k += stride) {
				fixed max = MAT_GET(src, i, j, k);
				for(uint l = 0; l < size; l ++) {
					for(uint m = 0; m < size; m ++) {
						fixed val = MAT_GET(src, i, j + l, k + m);
						if(F_LT(max, val))
							max = val;
					}
				}
				MAT_SET(dest, max, i, j / stride, k / stride);
			}
		}
	}
}

void relu(mat *src, mat *dest) {
	uint rows = MAT_GET_DIM(src, 0);
	fixed max = F_LIT(0.0);
	for(uint i = 0; i < rows; i ++) {
		max = MAT_GET(src, i);
		MAT_SET(dest, max, i);
		if(F_LT(max, F_LIT(0.0)))
			MAT_SET(dest, F_LIT(0.0), i);
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////MAT_C//////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

uint mat_get_dim(mat *m, uint axis) {
	return m->dims[axis + m->len_constraints];
}

void mat_reshape(mat *m, uint len, uint dims[]) {
	m->len_dims = len;
	m->len_constraints = 0;
	m->constraints_offset = 0;
	for(uint i = 0; i < len; i ++) {
		m->dims[i] = dims[i];
	}
}

void mat_constrain(mat *m, uint len, uint idxs[]) {
	for(uint i = 0; i < len; i ++) {
		m->constraints[i] = idxs[i];
	}	
	m->len_constraints = len;
	uint offset = 0;
	for(uint i = 0; i < len; i ++) {
		uint factor = 1;
		uint factor_idx = m->len_dims - i;
		for(short j = factor_idx - 1; j > 0; j --) {
			factor *= m->dims[j];
		}
		offset += factor * idxs[i];
	}
	m->constraints_offset = offset;
}

void mat_unconstrain(mat *m) {
	m->len_constraints = 0;
	m->constraints_offset = 0;
}

uint _offset_calc(void *_m, uint len, uint idxs[]) {
	mat *m = (mat *)_m;
	uint offset = 0;
	for(uint i = 0; i < len; i ++) {
		uint factor = 1;
		uint factor_idx = m->len_dims - i;
		for(short j = factor_idx - 1; j > m->len_constraints; j --) {
			factor *= m->dims[j];
		}
		offset += factor * idxs[i];
	}
	return offset;
}

fixed mat_get(mat *m, uint len, uint idxs[]) {
	return *(m->data + _offset_calc(m, len, idxs) + m->constraints_offset);
}

void mat_set(mat *m, fixed val, uint len, uint idxs[]) {
	*(m->data + _offset_calc(m, len, idxs) + m->constraints_offset) = val;
}