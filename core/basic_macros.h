#pragma once
#include <string>
#define CU_BEGIN namespace cual{
#define CU_END }

CU_BEGIN

//------------CUDA GENERAL------------
#define HOST_TO_DEVICE cudaMemcpyHostToDevice
#define DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define DEVICE __device__
#define GLOBAL __global__
#define CONSTANT __constant__

//------------CUDA PARAM ------------
#define INDEX_X threadIdx.x + blockIdx.x * blockDim.x
#define INDEX_Y threadIdx.y + blockIdx.y * blockDim.y
#define TID_X threadIdx.x
#define TID_Y threadIdx.y

//------------GENNERAL PARAM------------
#define floating_type double
constexpr floating_type EPSILON = 5e-15;
constexpr int THREADS_NUM = 32;
constexpr int TILE_SIZE = THREADS_NUM;
constexpr int BLOCKS_NUM = 32;

//------------ENUM CLASS------------
enum class COPY_MODE
{
    host_to_device,
    device_to_host,
    device_to_device
};
enum class ERROR_STATUS
{
    DEMENSION_ERROR,
    INDEX_ERROR,
    ZERO_DIV_BOTTOM,
    OUT_RANGE_ERROR
};

namespace ERROR_MESSAGE
{
    const std::string ROWS_COLS_NO_EQUAL = "ROWS_COLS_NO_EQUAL";
};

//------------MACROS FUNCTIONS------------
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

#define ERROR_LINE printf("ERROR: %s:%d,",__FILE__,__LINE__)

//#define THREAD_NUM(n) n>=32?32:16
#define BLOCK_NUM(data_size,thread_num) (data_size - 1) / thread_num + 1





//#define DEVICE_TO_HOST cudaMemcpyDeviceToHost
//#define MALLOC_DEVICE(dp,size) cudaMalloc((void**)&dp,size)
//#define MALLOC_HOST(dp,size) cudaMallocHost((void**)&dp,size)
//#define MEMCPY_HOST_TO_DEVICE(dp,sp,size,flag) cudaMemcpy(dp, sp, size, flag);
//#define MEMCPY_DEVICE_TO_HOST(dp,sp,size,flag) cudaMemcpy(dp, sp, size, flag);
//#define CUDA_FREE(dp) cudaFree(dp)
//#define FREE_HOST(dp) cudaFreeHost(dp)
////
//#define SAFE_DELETE(p)       { if(p) { delete (p);     (p)=nullptr; } }
//#define SAFE_DELETE_ARRAY(p) { if(p) { delete[] (p);   (p)=nullptr; } }

CU_END