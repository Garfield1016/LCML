#pragma once

#include <cuda_runtime.h>
#include "basic_macros.h"

//------------DEVICE------------
template<typename T>
T* allocateDevice(int n)
{
	T* data = nullptr;
	CHECK(cudaMalloc(&data, n * sizeof(T)));
	return data;
}
template<typename T>
T* allocateDevice(int n,T val)
{
	T* data = nullptr;
	CHECK(cudaMalloc(&data, n * sizeof(T)));
	CHECK(cudaMemset(data, val, n * sizeof(T)));
	return data;
}

template<typename T>
void allocateDevice(T** data, int n)
{
	CHECK(cudaMalloc(data, n * sizeof(T)));
}
template<typename T>
void allocateDevice(T** data, int n,floating_type val)
{
	CHECK(cudaMalloc(data, n * sizeof(T)));
	//CHECK(cudaMemset((*data), val, n * sizeof(T)));
	std::vector<T> temp(n, val);
	CHECK(cudaMemcpy(*data, temp.data() ,n * sizeof(T), HOST_TO_DEVICE));
}

template<typename T>
void deallocateDevice(T** data)
{
	CHECK(cudaFree(*data));
	*data = nullptr;
}

//------------HOST------------
template<typename T>
T* allocateHost(int n)
{
	T* data = (T*)malloc(&data, n * sizeof(T));
	return data;
}
template<typename T>
T* allocateHost(int n,T val)
{
	T* data = (T*)malloc(n * sizeof(T));
	memset(data, val, n * sizeof(T));
	return data;
}

template<typename T>
void allocateHost(T** data, int n)
{
	*data = (T*)malloc(n * sizeof(T));
}
template<typename T>
void allocateHost(T** data, int n,floating_type val)
{
	*data = (T*)malloc(n * sizeof(T));
	memset(*data, val, n * sizeof(T));
}

template<typename T>
void deallocateHost(T **data)
{
	free(*data);
	*data = nullptr;
}

//------------MEMCPY------------
template<typename T>
void memcpyHostToDevice(T* dst, T* src, unsigned int n)
{
	CHECK(cudaMemcpy(dst, src, n*sizeof(T), HOST_TO_DEVICE));
}

template<typename T>
void memcpyDeviceToDevice(T* dst, T* src, unsigned int n)
{
	CHECK(cudaMemcpy(dst, src, n * sizeof(T), DEVICE_TO_DEVICE));
}

template<typename T>
void memcpyDeviceToHost(T* dst, T* src, unsigned int n)
{
	CHECK(cudaMemcpy(dst, src, n * sizeof(T), DEVICE_TO_HOST));
}


