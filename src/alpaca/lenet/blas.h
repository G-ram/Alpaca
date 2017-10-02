#ifndef BLAS_H
#define BLAS_H
#include "types.h"

void convolve2d(mat *src, uint size, const fixed filter[][size],mat *dest);

void convolve3d(mat *src, uint size, const fixed filter[][size][size], mat *dest);

void mul_vector(uint rows, uint cols, const fixed mat_data[][cols], mat *vector, 
	mat *dest);

void sparse_mul_vector(uint rows, const fixed mat_data[], const uint mat_idx[], 
	const uint mat_ptr[], mat *vector, mat *dest);

void bias2d(mat *src, const fixed bias, mat *dest);

void bias1d(mat *src, const fixed bias[], mat *dest);

void pool(mat *src, uint size, uint stride, mat *dest);

void relu(mat *src, mat *dest);

#endif