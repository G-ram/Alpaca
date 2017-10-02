#ifndef TYPES_H
#define TYPES_H

typedef signed int fixed;
typedef signed long lint;
typedef unsigned int uint;

typedef struct {
	uint dims[10];
	uint len_dims;
	uint constraints[10];
	uint len_constraints;
	uint constraints_offset;
	fixed *data;
} mat;

#endif