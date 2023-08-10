#include "common.h"
#include <iostream>
//#include <math_functions.h>

CU_BEGIN

//------------util functions------------
DEVICE floating_type add_base_device(floating_type a, floating_type b)
{
	return a + b;
}

GLOBAL void kernelAddVec(floating_type* c,  floating_type* a, floating_type* b, int n, bool is_subtract)
{
	int i = INDEX_X;
	int tid = TID_X;
	__shared__ floating_type cache_a[TILE_SIZE];
	__shared__ floating_type cache_b[TILE_SIZE];
	floating_type coeff = is_subtract ? -1 : 1;

	if (i < n)
	{
		cache_a[tid] = a[i];
		cache_b[tid] = b[i];
		//tid += blockDim.x * gridDim.x;
	}
	__syncthreads();

	if (i < n)
	{
		floating_type  val = cache_a[tid] + cache_b[tid] * coeff;
		c[i] = cuAbs(val) > EPSILON ? val : 0.f;
	}
		
	__syncthreads();
}

GLOBAL void kernelAddBase(floating_type* c,  floating_type* a, floating_type* b, int n, bool is_subtract)
{
	int i = INDEX_X;
	int tid = TID_X;
	floating_type coeff = is_subtract ? -1 : 1;
	if (i < n)
		c[i] = a[i] + b[i] * coeff;
	__syncthreads();
}

DEVICE floating_type atomicAddM(floating_type* address, floating_type val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}


DEVICE floating_type getElement(floating_type* A, int i, int j)
{
	return 0;
}
GLOBAL void setElement(floating_type* A, int rows_A,int i, int j,floating_type val)
{
	int location = rows_A * i + j;
	A[location] = val;
}
GLOBAL void kernelScanCoefficient(floating_type* a, int n,floating_type coefficient)
{
	int index_global = INDEX_X;
	int tid = TID_X;
	__shared__ floating_type cache[TILE_SIZE];
	cache[tid] = index_global < n ? a[index_global] : 0;
	__syncthreads();
	if (index_global < n)
		a[index_global] = cache[tid] * coefficient;
	__syncthreads();
}

//------------Mat------------
GLOBAL void multiply(floating_type* A, floating_type* B, floating_type* C, int rows_A, int cols_A, int rows_B, int cols_B)
{
	
	int rows_C = rows_A;
	int cols_C = cols_B;
	//printf("blockDim.y=%d\n", blockDim.y);
	int i = INDEX_Y;	//equal:for(int i=0;i<rows_A;i++)
	int j = INDEX_X;	//equal:for(int j=0;j<cols_B;j++)
	if (i < rows_C && j < cols_C)					//rows_C=rows_A cols_C=cols_B
	{
		floating_type sum_dot = 0;
		for (int j_temp = 0; j_temp < cols_A; j_temp++)
		{
			int location_a = i * cols_A + j_temp;
			int location_b = j_temp * cols_B + j;
			sum_dot += A[location_a] * B[location_b];
			
		}
		C[i * cols_C + j] = cuAbs(sum_dot)>EPSILON? sum_dot:0.0;
	}
}

GLOBAL void multiplyAccelerate(floating_type* A, floating_type* B, floating_type* C, int rows_A, int cols_A, int rows_B, int cols_B)
{
	int rows_C = rows_A;
	int cols_C = cols_B;
	__shared__ floating_type BLOCK_A[THREADS_NUM][THREADS_NUM];
	__shared__ floating_type BLOCK_B[THREADS_NUM][THREADS_NUM];
	int row = INDEX_Y;
	int col = INDEX_X;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	int dim = blockDim.x;										//dim=THREADS_NUM,¿ÉÌæ»»
	
	floating_type sum_dot = 0;
	int BLOCK_index_n = (cols_A + dim - 1) / dim;
	for (int i = 0; i < BLOCK_index_n; i++)
	{
		BLOCK_A[thread_y][thread_x] = (row < rows_A && ((i* dim + thread_x) < cols_A)) ? 
										A[row*cols_A+i*dim+thread_x]:0.0;	//(row<rows_A&&(i * dim + thread_x<cols_A))?
		BLOCK_B[thread_y][thread_x] = (col<cols_B && ((i * dim + thread_y) < rows_B)) ? 
										B[i * dim * cols_B + thread_y * cols_B + col]:0.0;
		__syncthreads();
		for (int j = 0; j < dim; ++j)
		{
			sum_dot += BLOCK_A[thread_y][j] * BLOCK_B[j][thread_x];
		}
		__syncthreads();
	}
	if (row < rows_C && col < cols_C)
		C[row * cols_C + col] = cuAbs(sum_dot) > EPSILON ? sum_dot : 0.0;
}

GLOBAL void kernnelMatAdd(floating_type* C, floating_type* A, floating_type* B,  int rows, int cols)
{
	int i = INDEX_Y;	//equal:for(int i=0;i<rows_A;i++)
	int j = INDEX_X;	//equal:for(int j=0;j<cols_B;j++)
	if (i < rows && j < cols)
	{
		int location = i * cols + j;
		C[location] = A[location] + B[location];
	}
}
GLOBAL void copySub(floating_type* sub, floating_type* A,int rows_A,int cols_A, int i, int j, int rows_sub, int cols_sub)
{
	int row = INDEX_Y;	
	int col = INDEX_X;	
	if (row < rows_sub && col < cols_sub)
	{
		int location_A_col = j + col;
		int location_A_row = i + row;
		sub[row*cols_sub+col] = A[location_A_row * cols_A + location_A_col];
	}

}

DEVICE floating_type cuAbs(floating_type a)
{
	floating_type abs_a = a >= 0.0 ? a : -a;
	return abs_a;
}
GLOBAL void kernelSwapRow(floating_type* A, int i1, int i2, int j, int rows_A, int cols_A)
{
	
}

GLOBAL void kernelSetRow(floating_type* A, int i, int cols_A)
{
	int thread_x = threadIdx.x;
	int col = INDEX_X;
	if (col < cols_A)
		A[cols_A * i + col] = 0.f;
}

GLOBAL void pivotU(floating_type* U, int max_row, int i, int j, int rows_sub, int cols_sub)
{
	int thread_x = threadIdx.x;
	int location_in_global = INDEX_X;

	__shared__ floating_type TILE_i_row[TILE_SIZE];
	__shared__ floating_type TILE_max_row[TILE_SIZE];

	int location_in_U_i_row = i * (cols_sub+j) + j + location_in_global;
	int location_in_U_max_row = max_row * (cols_sub+j) + j + location_in_global;
	if (location_in_global < cols_sub)
	{
		TILE_i_row[thread_x] = U[location_in_U_i_row];
		TILE_max_row[thread_x] = U[location_in_U_max_row];
	}
	__syncthreads();
	if (location_in_global < cols_sub)
	{
		U[max_row * (cols_sub + j) + j + location_in_global] = TILE_i_row[thread_x];
		U[i * (cols_sub + j) + j + location_in_global] = TILE_max_row[thread_x];
	}
	
}
GLOBAL void pivotL(floating_type* L, int max_row, int i, int j, int rows_L, int cols_L)
{
	int rows_sub = rows_L - i;
	int cols_sub = j;
	int thread_x = threadIdx.x;
	int location_in_global = INDEX_X;
	__shared__ floating_type TILE_i_row[TILE_SIZE];
	__shared__ floating_type TILE_max_row[TILE_SIZE];
	int location_in_L_i_row = i * (cols_L) + location_in_global;
	int location_in_L_max_row = max_row * (cols_L) + location_in_global;
	if (location_in_global < j)
	{
		TILE_i_row[thread_x] = L[location_in_L_i_row];
		TILE_max_row[thread_x] = L[location_in_L_max_row];
	}
	__syncthreads();
	if (location_in_global < j)
	{
		L[max_row * (cols_L) + location_in_global] = TILE_i_row[thread_x];
		L[i * (cols_L) + location_in_global] = TILE_max_row[thread_x];
	}
	
}
GLOBAL void pivotP(floating_type* P, int max_row, int i, int j, int rows_P, int cols_P)
{
	int thread_x = threadIdx.x;
	int location_in_global = INDEX_X;
	__shared__ floating_type TILE_i_row[TILE_SIZE];
	__shared__ floating_type TILE_max_row[TILE_SIZE];
	int location_in_P_i_row = i * (cols_P)+location_in_global;
	int location_in_P_max_row = max_row * (cols_P)+location_in_global;
	if (location_in_global < cols_P)
	{
		TILE_i_row[thread_x] = P[location_in_P_i_row];
		TILE_max_row[thread_x] = P[location_in_P_max_row];
	}
	__syncthreads();
	if (location_in_global < cols_P)
	{
		P[max_row * (cols_P)+location_in_global] = TILE_i_row[thread_x];
		P[i * (cols_P)+location_in_global] = TILE_max_row[thread_x];
	}
}

GLOBAL void kernelEliminate(floating_type* A,int i,int j, int rows_A, int cols_A)
{
	int rows_sub_U_below = rows_A - i -1;
	int cols_sub_U = cols_A - j;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	int row_in_sub = INDEX_Y;
	int col_in_sub = INDEX_X;

	__shared__ floating_type TILE_U_below[TILE_SIZE][TILE_SIZE];					//below lines except first line
	__shared__ floating_type TILE_U_top_line[TILE_SIZE];							//first line
	__shared__ floating_type TILE_L[TILE_SIZE];										//below lines except first line

	if (row_in_sub + i < rows_A && col_in_sub + j < cols_A)
	{
		TILE_U_top_line[thread_x] = A[i * cols_A + j + col_in_sub];
		TILE_U_below[thread_y][thread_x] = A[(i + 1 + row_in_sub) * cols_A + j + col_in_sub];
		//TILE_L[thread_y] = A[(i + 1 + row_in_sub) * cols_A + j] / A[i * cols_A + j];
		TILE_L[thread_y] = TILE_U_below[thread_y][0] / TILE_U_top_line[0];
	}
	else
	{
		TILE_U_top_line[thread_x] = 0.0;
		TILE_U_below[thread_y][thread_x] = 0.0;
		TILE_L[thread_y] = 0.0;
	}
	__syncthreads();

	if (row_in_sub + 1 + i < rows_A && col_in_sub + j < cols_A)
	{
		floating_type val = TILE_U_below[thread_y][thread_x] - TILE_L[thread_y] * TILE_U_top_line[thread_x];

		__syncthreads();
		int location_in_U = (i + 1 + row_in_sub) * cols_A + j + col_in_sub;
		A[location_in_U] = cuAbs(val) > EPSILON ? val : 0.0;
		__syncthreads();
	}

}

GLOBAL void updateL(floating_type* L, floating_type* U, int i, int j, int rows_U, int cols_U)
{
	int location = INDEX_X;
	int location_in_U = (i + location) * rows_U + j;	//location_in_U=location_in_L
	floating_type div_bottom = U[rows_U * i + j];
	if ((i + location) > i && (i + location) < rows_U)
	{
		floating_type coeff= U[location_in_U] / div_bottom;
		L[location_in_U] = coeff;// cuAbs(coeff) > EPSILON ? coeff : 0.0;
	}
}

//below lines except first line
GLOBAL void eliminateLU2(floating_type* L, floating_type* U, int i, int j, int rows_U, int cols_U)
{
	int rows_sub_U_below = rows_U - i - 1;
	int cols_sub_U = cols_U - j;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	int row_in_sub = INDEX_Y;
	int col_in_sub = INDEX_X;
	__shared__ floating_type TILE_U_below[TILE_SIZE][TILE_SIZE];					//below lines except first line
	__shared__ floating_type TILE_U_top_line[TILE_SIZE];							//first line
	__shared__ floating_type TILE_L[TILE_SIZE];										//below lines except first line

	if (row_in_sub + i < rows_U && col_in_sub + j < cols_U)
	{
		TILE_U_top_line[thread_x] = U[i*cols_U+j+ col_in_sub];
		TILE_L[thread_y] = L[(i  +1+row_in_sub) * cols_U + j];
		TILE_U_below[thread_y][thread_x] = U[(i + 1 + row_in_sub) * cols_U + j + col_in_sub];
	}
	else
	{
		TILE_U_top_line[thread_x] = 0.0;
		TILE_L[thread_y] = 0.0;
		TILE_U_below[thread_y][thread_x] = 0.0;
	}
	__syncthreads();

	if (row_in_sub +1+ i < rows_U && col_in_sub + j < cols_U)
	{
		floating_type val = TILE_U_below[thread_y][thread_x] - TILE_L[thread_y] * TILE_U_top_line[thread_x];
		__syncthreads();
		int location_in_U = (i + 1+row_in_sub) * cols_U + j+col_in_sub;
		U[location_in_U] =  cuAbs(val) > EPSILON ? val : 0.0;
		__syncthreads();

	}
}

GLOBAL void kernelTranspose(floating_type* B, floating_type* A, int rows_A, int cols_A)
{
	__shared__ floating_type tile[TILE_SIZE][TILE_SIZE+1];
	int col= INDEX_X;
	int row= INDEX_Y;
	int location_in_A = row * cols_A + col;
	int location_in_B = col * rows_A + row;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	if (col < cols_A && row < rows_A)
	{
		tile[thread_y][thread_x] = A[location_in_A];
		__syncthreads();
		B[location_in_B] = tile[thread_y][thread_x];
	}
}

GLOBAL void kernelUpdateLColesky(floating_type* L, floating_type* Lt, floating_type* A, int i, int j, int rows_A, int cols_A)
{
	int rows_sub = rows_A - i - 1;
	int row = INDEX_X;
	int thread_x = threadIdx.x;
	__shared__ floating_type TILE [TILE_SIZE];
	if (row < rows_sub && j < cols_A - 1)
		TILE[thread_x] = A[(i + 1 + row) * cols_A + j]/L[i * cols_A + j];
	else
		TILE[thread_x] = 0;
	__syncthreads();
	if (row < rows_sub && j < cols_A - 1)
	{
		L[(i + 1 + row) * cols_A + j] = TILE[thread_x];
		Lt[i * cols_A + (j + 1 + row)] = TILE[thread_x];
	}
	__syncthreads();
}

GLOBAL void kernelUpdateSubAColesky(floating_type* L, floating_type* Lt, floating_type* A, int i, int j, int rows_A, int cols_A)
{
	int rows_sub = rows_A - i;
	int cols_sub = cols_A - j;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	int row_in_sub = INDEX_Y;
	int col_in_sub = INDEX_X;

	__shared__ floating_type TILE_row[TILE_SIZE];
	__shared__ floating_type TILE_col[TILE_SIZE];
	__shared__ floating_type TILE_sub[TILE_SIZE][TILE_SIZE];

	if (row_in_sub < rows_sub && col_in_sub < cols_sub)
	{
		TILE_row[thread_y] = L[(i + 1 + row_in_sub) * cols_A + j];
		TILE_row[thread_x] = Lt[i * cols_A + j + 1 + col_in_sub];
		TILE_sub[thread_y][thread_x] = A[(i + 1 + row_in_sub) * cols_A + j + 1 + col_in_sub];
	}
	else
	{
		TILE_row[thread_x] = 0;
		TILE_row[thread_y] = 0;
		TILE_sub[thread_y][thread_x] = 0;
	}
	__syncthreads();

	if (row_in_sub < rows_sub && col_in_sub < cols_sub)
	{
		A[(i + 1 + row_in_sub) * cols_A + j + 1 + col_in_sub] = TILE_sub[thread_y][thread_x] - TILE_row[thread_x] * TILE_row[thread_y];
	}
	__syncthreads();
}

GLOBAL void kernelAccumulateMultiply(floating_type* out, const floating_type* in, int n)
{
	int location_in_arr = INDEX_X;
	int thread_x = threadIdx.x;
	__shared__ floating_type TILE[TILE_SIZE];
	if (location_in_arr < n)
		TILE[thread_x] = in[location_in_arr];
	else
		TILE[thread_x] = floating_type(1);
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (thread_x < stride)										
			TILE[thread_x] *= TILE[thread_x + stride];
		__syncthreads();
	}
	if (thread_x == 0)
		out[blockIdx.x] = TILE[0];
}

GLOBAL void kernelGetCol(floating_type* vec,floating_type* A, int j, int rows_A,int cols_A)
{
	__shared__ floating_type TILE[TILE_SIZE];
	int thread_x = threadIdx.x;
	int row = INDEX_X;

	if (row < rows_A)
		TILE[thread_x] = A[row*cols_A+j];
	__syncthreads();
	vec[row] = TILE[thread_x];
	__syncthreads();
}

GLOBAL void kernelGetEye(floating_type* vec, floating_type* A, int rows_A, int cols_A)
{
	__shared__ floating_type TILE[TILE_SIZE];
	int thread_x = threadIdx.x;
	int row = INDEX_X;

	if (row < rows_A)
		TILE[thread_x] = A[row * cols_A + row];
	__syncthreads();
	vec[row] = TILE[thread_x];
	__syncthreads();
}

GLOBAL void kernelFindMaxCol(floating_type* U, floating_type* arr_val, int* arr_index, int i, int j, int rows_sub, int cols_sub)
{
	int location_in_global = INDEX_X;
	int thread_x = threadIdx.x;
	__shared__ floating_type TILE[TILE_SIZE];
	__shared__ int TILE_index[TILE_SIZE];
	int location_in_U = (i + location_in_global) * (cols_sub + j) + j;
	if (location_in_global < rows_sub)
	{
		TILE[thread_x] = cuAbs(U[location_in_U]);
		TILE_index[thread_x] = i + location_in_global;
		//printf("\nindex_row_sub=%d; ", TILE_index[thread_x]);
		//printf(",val_row_sub=%f; ", TILE[thread_x]);
	}
	else
	{
		TILE[thread_x] = floating_type(0);
		TILE_index[thread_x] = -1;
	}
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (thread_x < stride && thread_x + stride < blockDim.x && location_in_global < rows_sub)										//stride=2; thread_x=0-1; 0
		{
			if (TILE[thread_x] < TILE[thread_x + stride])
			{
				TILE[thread_x] = TILE[thread_x + stride];
				TILE_index[thread_x] = TILE_index[thread_x + stride];
				//printf("\nindex_stride=%d; ", TILE_index[thread_x + stride]);
				//printf("val_stride=%f; ", TILE[thread_x + stride]);
			}
		}
		__syncthreads();
	}
	if (thread_x == 0)
	{
		arr_val[blockIdx.x] = TILE[0];
		arr_index[blockIdx.x] = TILE_index[0];
		//if (TILE_index[0] < 0 || TILE_index[0] >= 7)
		{
			//printf("\n,kernel error; ");
			//printf("kernel_index=%d; ", TILE_index[0]);
			//printf("arr_val=%f\n", arr_val[blockIdx.x]);
		}
	}
}

GLOBAL void kernelDot(floating_type* a, floating_type* b, floating_type* c, int n)
{
	__shared__ floating_type cache[TILE_SIZE];

	int tid = INDEX_X;
	int cacheIdx = threadIdx.x;
	floating_type sum = 0;
	while (tid < n) {
		sum += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;

	}
	cache[cacheIdx] = sum;

	__syncthreads();
	int mid = blockDim.x / 2;
	while (mid != 0) {
		if (cacheIdx < mid) cache[cacheIdx] += cache[cacheIdx + mid];
		__syncthreads();
		mid /= 2;
	}
	if (cacheIdx == 0)
		c[blockIdx.x] = cache[0];
}

GLOBAL void kernelDotAtomic(floating_type* c, floating_type* a, floating_type* b, int n)
{
	__shared__ floating_type cache[TILE_SIZE];

	int tid = INDEX_X;
	int cacheIdx = threadIdx.x;
	floating_type sum = 0;
	while (tid < n) {
		sum += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;

	}
	cache[cacheIdx] = sum;

	__syncthreads();
	int mid = blockDim.x / 2;
	while (mid != 0) {
		if (cacheIdx < mid) cache[cacheIdx] += cache[cacheIdx + mid];
		__syncthreads();
		mid /= 2;
	}
	if (cacheIdx == 0)
		atomicAddM(c, cache[0]);
}

GLOBAL void kernelNorm2dAtomic(floating_type* b, floating_type* a, int n)
{
	__shared__ floating_type cache[TILE_SIZE];
	int tid = INDEX_X;
	int cacheIdx = threadIdx.x;
	floating_type sum = 0;
	while (tid < n) {
		sum += a[tid]*a[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIdx] = sum;

	__syncthreads();
	int mid = blockDim.x / 2;
	while (mid != 0) {
		if (cacheIdx < mid) 
			cache[cacheIdx] += cache[cacheIdx + mid];
		__syncthreads();
		mid /= 2;
	}
	if (cacheIdx == 0)
		atomicAddM(b, cache[0]);
}

CU_END

























