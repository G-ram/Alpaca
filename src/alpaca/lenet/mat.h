#ifndef MAT_H
#define MAT_H

#include "types.h"

#define NUMARGS(...) (sizeof((uint[]){__VA_ARGS__})/sizeof(uint))
#define MAT_RESHAPE(m, ...) (mat_reshape(m, NUMARGS(__VA_ARGS__), 			\
								(uint[]){__VA_ARGS__}))
#define MAT_CONSTRAIN(m, ...) (mat_constrain(m, NUMARGS(__VA_ARGS__), 		\
								(uint[]){__VA_ARGS__}))
#define MAT_UNCONSTRAIN(m) (mat_unconstrain(m))
#define MAT_GET(m, ...) (													\
	(NUMARGS(__VA_ARGS__) == 1) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0]) :	\
	(NUMARGS(__VA_ARGS__) == 2) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] * 	\
			m->dims[m->len_dims - 1] + ((uint[]){__VA_ARGS__})[1]):			\
	(NUMARGS(__VA_ARGS__) == 3) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] *	\
			m->dims[m->len_dims - 1] * m->dims[m->len_dims - 2] +			\
			m->dims[m->len_dims - 1] * ((uint[]){__VA_ARGS__})[1] + 		\
			((uint[]){__VA_ARGS__})[2]):									\
	mat_get(m, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))
#define MAT_SET(m, val, ...) (												\
	(NUMARGS(__VA_ARGS__) == 1) ? 											\
		*(m->data + + m->constraints_offset +								\
			((uint[]){__VA_ARGS__})[0]) = val :								\
	(NUMARGS(__VA_ARGS__) == 2) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] * 	\
			m->dims[m->len_dims - 1] + ((uint[]){__VA_ARGS__})[1]) = val :	\
	(NUMARGS(__VA_ARGS__) == 3) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] *	\
			m->dims[m->len_dims - 1] * m->dims[m->len_dims - 2] +			\
			m->dims[m->len_dims - 1] * ((uint[]){__VA_ARGS__})[1] + 		\
			((uint[]){__VA_ARGS__})[2]) = val :								\
	mat_set(m, val, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))

#define MAT_GET_DIM(m, axis) (mat_get_dim(m, axis))

uint mat_get_dim(mat *m, uint axis);

void mat_reshape(mat *m, uint len, uint dims[]);

void mat_constrain(mat *m, uint len, uint idxs[]);

void mat_unconstrain(mat *m);

fixed mat_get(mat *m, uint len, uint idxs[]);

void mat_set(mat *m, fixed val, uint len, uint idxs[]);

#endif