#pragma once
#include <cuda_runtime.h>
#include <math_functions.h>

#include "core/basic_macros.h"
#include "core/basic_functions.h"

CU_BEGIN
//------------util functions------------
DEVICE floating_type cuAbs(floating_type a);
DEVICE floating_type add_base_device(floating_type a, floating_type b);
GLOBAL void kernelAddVec(floating_type* c,  floating_type* a,  floating_type* b, int n,bool is_subtract);
GLOBAL void kernelAddBase(floating_type* c,  floating_type* a,  floating_type* b, int n, bool is_subtract);

GLOBAL void kernnelMatAdd(floating_type* C, floating_type* A, floating_type* B,  int rows, int cols);
GLOBAL void copySub(floating_type* sub, floating_type* A, int rows_A, int cols_A, int i, int j, int rows_sub, int cols_sub);
DEVICE floating_type getElement(floating_type* A, int i, int j);
GLOBAL void setElement(floating_type* A, int row_A, int i, int j, floating_type val);
GLOBAL void kernelCopy(floating_type* data, floating_type* A, int begin_index, int stride,int n);
GLOBAL void kernelScanCoefficient(floating_type* a, int n, floating_type coefficient);
//------------MAT------------
GLOBAL void multiply(floating_type* A, floating_type* B, floating_type* C, int rows_A, int cols_A, int rows_B, int cols_B);
GLOBAL void multiplyAccelerate(floating_type* A, floating_type* B, floating_type* C, int rows_A, int cols_A, int rows_B, int cols_B);

GLOBAL void pivotU(floating_type* U, int max_row, int i, int j, int rows_sub, int cols_sub);
GLOBAL void pivotL(floating_type* L, int max_row, int i, int j, int rows_L, int cols_L);
GLOBAL void pivotP(floating_type* L, int max_row, int i, int j, int rows_U, int cols_U);
GLOBAL void kernelSwapRow(floating_type* A, int i1, int i2, int j,int rows_A, int cols_A);
GLOBAL void eliminateLU2(floating_type* L, floating_type* U, int i, int j, int rows_U, int cols_U);
GLOBAL void updateL(floating_type* L, floating_type* U, int i, int j, int rows_U, int cols_U);

GLOBAL void kernelTranspose(floating_type* B, floating_type* A, int rows_A, int cols_A);
GLOBAL void kernelSetRow(floating_type* A, int i, int cols_A);
GLOBAL void kernelEliminate(floating_type* A, int i, int j, int rows_A, int cols_A);

GLOBAL void kernelUpdateLColesky(floating_type* L, floating_type* Lt, floating_type* A, int i, int j, int rows_A, int cols_A);
GLOBAL void kernelUpdateSubAColesky(floating_type* L, floating_type* Lt, floating_type* A, int i, int j, int rows_A, int cols_A);

GLOBAL void kernelAccumulateMultiply(floating_type* out, const floating_type* in, int N);

GLOBAL void kernelGetCol(floating_type* vec, floating_type* A, int j, int rows_A,int cols_A);
GLOBAL void kernelGetEye(floating_type* vec, floating_type* A, int rows_A, int cols_A);



//reduce kernel
GLOBAL void kernelFindMaxCol(floating_type* U, floating_type* arr_val, int* arr_index, int i, int j, int rows_U, int cols_U);
GLOBAL void kernelDot(floating_type* a, floating_type* b, floating_type* c, int n);
GLOBAL void kernelDotAtomic(floating_type* c, floating_type* a, floating_type* b,  int n);
GLOBAL void kernelNorm2dAtomic(floating_type* b, floating_type* a, int n);
CU_END