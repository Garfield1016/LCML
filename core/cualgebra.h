#pragma once
#include <iostream>
#include <fstream>
#include <random>
#include <cuda_runtime.h>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>
#include <thrust/extrema.h>
#include "basic_macros.h"
#include "basic_functions.h"

CU_BEGIN
static std::default_random_engine rand_engine;
static std::uniform_int_distribution<int> unifIntDis(-10, 10);
static std::uniform_real_distribution<float> unifFloatDis(1.0, 20.0);

//------------BaseArray------------
class BaseArray
{
public:
	int size_{0};
	floating_type* data_device_{nullptr};
	bool flag_release_explicit_ = false;

public:
	BaseArray() = default;
	~BaseArray();
	BaseArray(int size);
	BaseArray(int size,floating_type val);
	BaseArray(floating_type* src_data, int size, COPY_MODE copy_mode);
	BaseArray(const BaseArray& arr);
	BaseArray(BaseArray&& arr);
	void releaseExplicit();
public:
	floating_type& operator[](int i);
	floating_type at(int i);
	void set(int i,floating_type val);
	void resize(int size);
	floating_type* data();
	BaseArray clone();
	int size() const;
	void toHost(floating_type* data_host) const;
};

//------------Vec------------

class Vec:public BaseArray
{
public:
	Vec() = default;
	~Vec() = default;
	Vec(int size);
	Vec(int size, floating_type val);
	Vec(floating_type* host_data, int size, COPY_MODE copy_mode);
	Vec(const Vec& vec);
public:
	Vec operator=(const Vec& vec);
	Vec operator=(Vec&& vec);
	Vec operator+(const Vec& vec) const;
	Vec operator-(const Vec& vec) const;
	Vec operator*(floating_type cofficient) const;

	
public:
	void set(int i, floating_type val);
	void set(int i, int n,floating_type val);
	void print() const;
	floating_type norm2d() const;
	void coeff(floating_type cofficient);
	static void add(const BaseArray& arr1, const BaseArray& arr2, BaseArray& arr3,int n);
};

//------------Mat------------
class Mat:public BaseArray
{
private:
	int rows_;
	int cols_;
public:
	Mat() = default;
	~Mat() = default;
	Mat(int rows, int cols);
	Mat(int rows, int cols, floating_type val);
	Mat(int rows, int cols, floating_type* host_data, COPY_MODE copy_mode);
	Mat(const Mat& m);
	Mat(const Vec& v,bool column);
public:
	int rows()const;
	int cols()const;
	floating_type& operator()(int i, int j);
	void set(int i, int j, floating_type val);
	floating_type at(int i, int j);
	Mat operator*(const Mat& m) const;
	Mat operator*(floating_type coffecient) const;
	Mat operator+(const Mat& m) const;
	Mat operator-(const Mat& m);
	Mat operator=(const Mat& m);
	Mat operator=(Mat&& m);
	Mat subMat(int i, int j, int rows_sub, int cols_sub);
	void setRow(int i, floating_type val);
	Vec getRow(int i) const;
	Vec getCol(int j) const;
	Vec getCol(int j, int i1, int i2) const;
	Vec getEye() const;
	Mat transpose();
	std::tuple<Mat,Mat,Mat> PL() const;
	std::tuple<Mat, Mat> cholesky() const;
	std::tuple<Mat,int> eliminate() const;
	std::tuple<int, int> rank() const;
	std::tuple<Mat, Mat> QRbyH() const;
	std::tuple<Mat, Mat> QRbyH2() const;
	floating_type det() const;
	//test
	void t_temp();
	//end test

public:
	void print() const;
	void txt(const std::string &txt_path) const;
public:
	static floating_type randVal();
	static Mat randomMN(int m, int n);
	static Mat randomNNSymmetry(int n);
	static Mat I(int n);
	static Mat zeros(int m, int n);
};

//------------util functions------------
int findMaxRow(const Mat& U, int i, int j, int rows_U, int cols_U);
floating_type dot(floating_type* a, floating_type* b, int n);
floating_type dotAtomic(floating_type* a, floating_type* b, int n);


//------------solve functions------------
Mat solverLU( const Mat& A,const Mat& y);
CU_END

//void t_data();


//void reduction1(int* answer, int* partial, const int* in, const size_t N, const int numBlocks, int numThreads)
//{
//	unsigned int sharedSize = numThreads * sizeof(int);
//
//	 kernel execution
//	reduction1_kernel << <numBlocks, numThreads, sharedSize >> > (partial, in, N);
//	reduction1_kernel << <1, numThreads, sharedSize >> > (answer, partial, numBlocks);
//}
