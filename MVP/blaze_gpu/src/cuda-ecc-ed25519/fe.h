#ifndef FE_H
#define FE_H

#include "fixedint.h"


/*
    fe means field element.
    Here the field is \Z/(2^255-19).
    An element t, entries t[0]...t[9], represents the integer
    t[0]+2^26 t[1]+2^51 t[2]+2^77 t[3]+2^102 t[4]+...+2^230 t[9].
    Bounds on each t[i] vary depending on context.
*/


typedef int32_t fe[10];


void __host__ __device__ fe_0(fe h);
void __device__ __host__ fe_1(fe h);

void __device__ __host__ fe_frombytes(fe h, const unsigned char *s);
void __device__ __host__ fe_tobytes(unsigned char *s, const fe h);

void __host__ __device__ fe_copy(fe h, const fe f);
int __host__ __device__ fe_isnegative(const fe f);
int __device__ __host__ fe_isnonzero(const fe f);
void fe_cmov(fe f, const fe g, unsigned int b);
void fe_cswap(fe f, fe g, unsigned int b);

void __device__ __host__ fe_neg(fe h, const fe f);
void __device__ __host__ fe_add(fe h, const fe f, const fe g);
void __device__ __host__ fe_invert(fe out, const fe z);
void __device__ __host__ fe_sq(fe h, const fe f);
void __host__ __device__ fe_sq2(fe h, const fe f);
void __device__ __host__ fe_mul(fe h, const fe f, const fe g);
void fe_mul121666(fe h, fe f);
void __device__ __host__ fe_pow22523(fe out, const fe z);
void __device__ __host__ fe_sub(fe h, const fe f, const fe g);

#endif
