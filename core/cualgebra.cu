#include "cualgebra.h"
#include "basic_functions.h"
#include "common.h"
#include "utils.h"
#include "allocator.hpp"
#include <thread>


CU_BEGIN

//------------BaseArray------------

BaseArray::~BaseArray()
{
	if (flag_release_explicit_ == false)
		releaseExplicit()；
};

BaseArray::BaseArray(int size)
{
	size_ = size;
	allocateDevice<floating_type>(&data_device_, size_);
}

BaseArray::BaseArray(int size, floating_type val)
{
	size_ = size;
	allocateDevice<floating_type>(&data_device_, size_,val);
}

BaseArray::BaseArray(floating_type* src_data, int size, COPY_MODE copy_mode)
{
	size_ = size;
	allocateDevice<floating_type>(&data_device_, size_);
	switch (copy_mode)
	{
	case COPY_MODE::host_to_device:
		memcpyHostToDevice<floating_type>(data_device_, src_data, size_);
		break;
	case COPY_MODE::device_to_device:
		memcpyDeviceToDevice<floating_type>(data_device_, src_data, this->size_);
		break;
	default:
		break;
	}
}

BaseArray::BaseArray(const BaseArray& arr)
{
	this->size_ = arr.size_;
	allocateDevice<floating_type>(&(this->data_device_),size_);
	memcpyDeviceToDevice<floating_type>(this->data_device_, arr.data_device_, size_);
}

BaseArray::BaseArray(BaseArray&& arr)
{
	this->size_ = arr.size_;
	this->data_device_ = arr.data_device_;
	arr.size_ = 0;
	arr.data_device_ = nullptr;
}

void BaseArray::releaseExplicit()
{
	if (data_device_ != nullptr && size_ != 0)
	{
		cudaFree(data_device_);
		size_ = 0;
	}
	flag_release_explicit_ = true;
}
/*BaseArray operator=(const BaseArray& arr)
	{
		this->size_ = arr.size_;
		this->data_device_ = arr.data_device_;
		return *this;
	}
	BaseArray operator=(BaseArray&& arr)
	{
		this->size_ = arr.size_;
		this->data_device_ = arr.data_device_;
		arr.size_ = 0;
		arr.data_device_ = nullptr;
		return *this;
	}*/

floating_type& BaseArray::operator[](int i)
{
	return data_device_[i];
}

floating_type BaseArray::at(int i)
{
	floating_type val;
	memcpyDeviceToHost<floating_type>(&val, this->data_device_ + i, 1);
	return val;
}

void BaseArray::set(int i, floating_type val)
{
	memcpyHostToDevice<floating_type>(this->data_device_  +i, &val, 1);
}

void BaseArray::resize(int size)
{
	releaseExplicit();
	size_ = size;
	allocateDevice<floating_type>(&data_device_, size_);
}

floating_type* BaseArray::data() 
{
	return data_device_;
}

BaseArray BaseArray::clone()
{
	BaseArray result(this->size_);
	memcpyDeviceToDevice<floating_type>(result.data_device_, this->data_device_, size_);
	return result;
}

int BaseArray::size() const
{
	return this->size_;
}

void BaseArray::toHost(floating_type* data_host) const
{
	memcpyDeviceToHost<floating_type>(data_host, this->data_device_, this->size_);
}

//------------Vec------------

Vec::Vec(int size) : BaseArray(size) {}
Vec::Vec(int size, floating_type val):BaseArray(size,val){}

Vec::Vec(floating_type* host_data, int size, COPY_MODE copy_mode) :BaseArray(host_data, size, copy_mode) {}

Vec::Vec(const Vec& vec) :BaseArray(vec) {}

Vec Vec::operator=(const Vec& vec)
{
	if (this->size_ != vec.size_)
	{
		this->releaseExplicit();
		this->flag_release_explicit_ = false;
		this->size_ = vec.size_;
		allocateDevice<floating_type>(&this->data_device_, this->size_);
	}
	memcpyDeviceToDevice<floating_type>(this->data_device_, vec.data_device_, this->size_);

	return *this;
}

Vec Vec::operator=(Vec&& vec)
{
	this->releaseExplicit();
	this->flag_release_explicit_ = false;
	this->size_ = vec.size_;
	this->data_device_ = vec.data_device_;
	vec.data_device_ = nullptr;
	vec.size_ = 0;
	vec.flag_release_explicit_ = true;
	return *this;
}

Vec Vec::operator-(const Vec& vec) const
{
	if (this->size_ != vec.size_)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	Vec result(this->size_);
	int threads = THREADS_NUM;
	int blocks = BLOCK_NUM(this->size_, threads);
	kernelAddVec << <blocks, threads >> > (result.data_device_, this->data_device_, vec.data_device_, this->size_, true);
	cudaDeviceSynchronize();
	return result;
}

Vec Vec::operator+(const Vec& vec) const
{
	if (this->size_ != vec.size_)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	Vec result(this->size_);
	int threads = THREADS_NUM;
	int blocks = BLOCK_NUM(this->size_, threads);
	kernelAddVec << <blocks, threads >> > (result.data_device_, this->data_device_, vec.data_device_, this->size_, false);
	cudaDeviceSynchronize();
	return result;
}

Vec Vec::operator*(floating_type cofficient) const
{
	Vec result(*this);
	int threads_per_block = THREADS_NUM;
	int blocks = BLOCK_NUM(result.size_, threads_per_block);
	kernelScanCoefficient << <blocks, threads_per_block >> > (result.data_device_, result.size_, cofficient);
	cudaDeviceSynchronize();
	return result;
}

void Vec::set(int i, floating_type val)
{
	if (i<0||i>=this->size_-1)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	memcpyHostToDevice(this->data_device_ + i, &val, 1);
}
void Vec::set(int i, int n,floating_type val)
{
	if (i < 0 || i >= this->size_ - 1)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	memcpyHostToDevice(this->data_device_ + i, &val, 1);
}

void Vec::print() const
{
	std::vector<floating_type> host_data(this->size_);
	memcpyDeviceToHost<floating_type>(host_data.data(), this->data_device_, this->size_);
	cudaDeviceSynchronize();
	std::cout << "\n";
	std::cout << "--------" << std::endl;
	for (int i = 0; i < this->size_; i++)
		std::cout << host_data[i] << " ";
	std::cout << std::endl;
	std::cout << "--------" << std::endl;
	std::cout << "\n";
}

floating_type Vec::norm2d() const
{
	int threads_per_block = THREADS_NUM;
	int blocks = BLOCK_NUM(this->size_, threads_per_block);
	floating_type *val_host, *val_device;
	allocateHost<floating_type>(&val_host, 1);
	allocateDevice<floating_type>(&val_device, 1, 0);
	kernelNorm2dAtomic<<<blocks,threads_per_block>>>(val_device, this->data_device_, this->size_);
	cudaDeviceSynchronize();
	memcpyDeviceToHost<floating_type>(val_host, val_device, 1);
	return *val_host;
}

void Vec::coeff(floating_type cofficient)
{
	int threads_per_block = THREADS_NUM;
	int blocks = BLOCK_NUM(this->size_, threads_per_block);
	kernelScanCoefficient << <blocks, threads_per_block >> > (this->data_device_,this->size_,cofficient);
	cudaDeviceSynchronize();
}

//------------Mat------------

Mat::Mat(int rows, int cols) :
	rows_(rows), cols_(cols), BaseArray(rows* cols) {}

Mat::Mat(int rows, int cols, floating_type val):
	rows_(rows), cols_(cols), BaseArray(rows* cols,val) {}

Mat::Mat(int rows, int cols, floating_type* host_data, COPY_MODE copy_mode) :
	rows_(rows), cols_(cols), BaseArray(host_data, rows* cols, copy_mode) {}

Mat::Mat(const Mat& m) :rows_(m.rows_), cols_(m.cols_), BaseArray(m) {}
Mat::Mat(const Vec& v, bool column) :BaseArray(v) 
{
	if (column)
	{
		this->rows_ = v.size_;
		this->cols_ = 1;
	}
	else
	{
		this->cols_ = v.size_;
		this->rows_ = 1;
	}
}

int Mat::rows()const { return rows_; }
int Mat::cols()const { return cols_; }

void Mat::set(int i, int j, floating_type val)
{
	memcpyHostToDevice<floating_type>(this->data_device_ + this->cols_ * i + j, &val, 1);
}
floating_type& Mat::operator()(int i, int j)
{
	int index = rows_ * i + j;
	return data_device_[index];
}
floating_type Mat::at(int i, int j)
{
	floating_type val;
	memcpyDeviceToHost<floating_type>(&val, this->data_device_ + this->cols_ * i + j, 1);
	return val;
}

void Mat::print() const
{
	std::vector<floating_type> host_data(rows_ * cols_);
	toHost(host_data.data());
	std::cout << "\n";
	std::cout << "--------" << std::endl;
	for (int i = 0; i < this->size_; i++)
	{
		std::cout << host_data[i] << " ";
		if ((i + 1) % cols_ == 0)
			std::cout << std::endl;
	}
	std::cout << "--------" << std::endl;
	std::cout << "\n";
}
void Mat::txt(const std::string &txt_path) const
{
	std::vector<floating_type> host_data(rows_ * cols_);
	toHost(host_data.data());
	std::fstream f;
	f.open(txt_path, std::ios::out | std::ios::app);
	f << "\n";
	f << "--------------------------" << std::endl;
	for (int i = 0; i < this->size_; i++)
	{
		f << host_data[i] << " ";
		if ((i + 1) % cols_ == 0)
			f << std::endl;
	}
	f << "--------------------------" << std::endl;
	f << "\n";
	f << "\n";

	f.close();
}

Mat Mat::operator*(const Mat& m) const
{
	if (this->cols_ != m.rows_)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	Mat result(this->rows_, m.cols_);
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	dim3 block((result.cols_ - 1) / threads_per_block.x + 1, (result.rows_ - 1) / threads_per_block.y + 1);	//multiply multiplyAccelerate
	multiply <<<block, threads_per_block >>>(this->data_device_, m.data_device_, result.data_device_, this->rows_, this->cols_, m.rows_, m.cols_);
	cudaDeviceSynchronize();
	return result;
}

Mat Mat::operator*(floating_type coffecient) const
{
	Mat result(*this);
	int threads_per_block = THREADS_NUM;
	int blocks = BLOCK_NUM(this->size_, threads_per_block);
	kernelScanCoefficient << <blocks, threads_per_block >> > (result.data_device_, result.size_, coffecient);
	cudaDeviceSynchronize();
	return result;
}

Mat Mat::operator+(const Mat& m) const
{
	/*if (this->rows_ != m.rows_ || this->cols_ != m.cols_)
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
	Mat result(this->rows_, this->cols_);
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	dim3 block((result.cols_ - 1) / threads_per_block.x + 1, (result.rows_ - 1) / threads_per_block.y + 1);
	kernnelMatAdd << <block, threads_per_block >> > (result.data_device_, this->data_device_, m.data_device_, this->rows_, this->cols_);
	cudaDeviceSynchronize();
	return result;*/


	if (this->rows_ != m.rows_ || this->cols_ != m.cols_)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	Mat result(this->rows_, this->cols_);
	int threads = THREADS_NUM;
	int blocks = BLOCK_NUM(this->size_, threads);
	kernelAddVec << <blocks, threads >> > (result.data_device_, this->data_device_, m.data_device_, this->size_, false);
	cudaDeviceSynchronize();
	return result;
}
Mat Mat::operator-(const Mat& m)
{
	if (this->rows_ != m.rows_ || this->cols_ != m.cols_)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	Mat result(this->rows_, this->cols_);
	int threads = THREADS_NUM;
	int blocks = BLOCK_NUM(this->size_, threads);
	kernelAddVec << <blocks, threads >> > (result.data_device_, this->data_device_, m.data_device_, this->size_, true);
	cudaDeviceSynchronize();
	return result;
}
Mat  Mat::operator=(const Mat& m)
{
	this->releaseExplicit();
	this->flag_release_explicit_ = false;
	this->size_ = m.size_;
	this->rows_ = m.rows_;
	this->cols_ = m.cols_;
	allocateDevice<floating_type>(&this->data_device_, this->size_);
	memcpyDeviceToDevice<floating_type>(this->data_device_, m.data_device_, this->size_);
	return *this;
}
Mat  Mat::operator=(Mat&& m)
{
	this->releaseExplicit();
	this->flag_release_explicit_ = false;
	this->size_ = m.size_;
	this->rows_ = m.rows_;
	this->cols_ = m.cols_;
	this->data_device_ = m.data_device_;
	m.data_device_ = nullptr;
	m.size_ = 0;
	m.rows_ = 0;
	m.cols_ = 0;
	m.flag_release_explicit_ = true;
	return *this;
}
Mat Mat::subMat(int i, int j, int rows_sub, int cols_sub)
{
	if (i<0 || j<0 || rows_sub <= 0 || cols_sub <= 0 || (i + rows_sub)>this->rows_ || (j + cols_sub)>this->cols_)
	{
		CULMC_ERROR(ERROR_STATUS::INDEX_ERROR);
		exit(EXIT_FAILURE);
	}
		
	Mat sub_mat(rows_sub, cols_sub);
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	dim3 block((sub_mat.cols_ - 1) / threads_per_block.x + 1, (sub_mat.rows_ - 1) / threads_per_block.y + 1);
	copySub <<<block, threads_per_block >>> (sub_mat.data_device_, this->data_device_, this->rows_, this->cols_, i, j, rows_sub, cols_sub);
	cudaDeviceSynchronize();
	return sub_mat;
}

void Mat::setRow(int i, floating_type val)
{
	if (i<0 || i>this->rows_)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		MAT_RELEASE(*this);
		exit(EXIT_FAILURE);
	}
	int threads_per_block = THREADS_NUM;
	int blocks = (this->cols_ - 1) / THREADS_NUM + 1;
	kernelSetRow << <blocks, threads_per_block >> > (this->data_device_, i, this->cols_);
	cudaDeviceSynchronize();
}

Vec Mat::getRow(int i) const
{
	if (i >= this->rows_ || i < 0)
	{
		CULMC_ERROR(ERROR_STATUS::INDEX_ERROR);
		exit(EXIT_FAILURE);
	}
	Vec vec(this->cols_);
	memcpyDeviceToDevice<floating_type>(vec.data_device_, this->data_device_ + i * this->cols_, this->cols_);
	cudaDeviceSynchronize();
	return vec;
}
Vec Mat::getCol(int j) const
{
	if (j>= this->cols_||j<0)
	{
		CULMC_ERROR(ERROR_STATUS::INDEX_ERROR);
		exit(EXIT_FAILURE);
	}
	Vec vec(this->rows_);
	int threads_per_block = THREADS_NUM;
	int blocks = (this->rows_ - 1) / THREADS_NUM + 1;
	kernelGetCol << <blocks, threads_per_block >> > (vec.data_device_, this->data_device_, j, this->rows_,this->cols_);
	cudaDeviceSynchronize();
	return vec;
}
Vec Mat::getCol(int j, int i1, int i2) const
{
	if (j >= this->cols_ || j < 0||i1<0||i1>i2||i2>=this->rows_)
	{
		CULMC_ERROR(ERROR_STATUS::INDEX_ERROR);
		ERROR_LINE;
		exit(EXIT_FAILURE);
	}
	Vec vec(i2 - i1+1);
	int threads_per_block = THREADS_NUM;
	int blocks = BLOCK_NUM(this->rows_, threads_per_block);
	kernelGetCol << <blocks, threads_per_block >> > (vec.data_device_, this->data_device_ + i1 * this->cols_, j, i2+1, this->cols_);
	cudaDeviceSynchronize();
	return vec;
}
Vec Mat::getEye() const
{
	if (this->rows_ != this->cols_)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		exit(EXIT_FAILURE);
	}
		
	Vec vec(this->rows_);
	assert(this->rows_ == this->cols_, ERROR_INFO::ROWS_COLS_NO_EQUAL);
	int threads_per_block = THREADS_NUM;
	int blocks = (this->rows_ - 1) / THREADS_NUM + 1;
	kernelGetEye << <blocks, threads_per_block >> > (vec.data_device_, this->data_device_, this->rows_, this->cols_);
	cudaDeviceSynchronize();
	return vec;
}
Mat Mat::transpose()
{
	Timer timer;
	Mat result(this->cols_, this->rows_);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	dim3 blocks((this->cols_ - 1) / threads_per_block.x + 1, (this->rows_ - 1) / threads_per_block.x + 1);
	timer.start();
	kernelTranspose << <blocks, threads_per_block,0,stream >> > (result.data_device_, this->data_device_, this->rows_, this -> cols_);
	//cudaDeviceSynchronize();
	cudaStreamSynchronize(stream);
	timer.stop();
	float time = timer.timeMs();
	cudaStreamDestroy(stream);
	return result;
}
floating_type Mat::randVal()
{
	rand_engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
	return floating_type(unifIntDis(rand_engine));
}
floating_type Mat::det() const
{
	Mat U(*this);
	int rank_row = U.rows_;
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	dim3 block((U.cols_ - 1) / threads_per_block.x + 1, (U.rows_ - 1) / threads_per_block.y + 1);
	int n = U.rows_;
	int coeff_positive_negative=1;
	for (int j = 0; j < n; j++)//U.cols_
	{
		int i = j;
		int max_row = findMaxRow(U, i, j, U.rows_ - i, U.cols_ - j);
		if (max_row < 0 || max_row >= U.rows_)
			break;
		if (i != max_row)
		{
			int blocks_U = (U.cols_ - j - 1) / THREADS_NUM + 1;
			pivotU << <blocks_U, THREADS_NUM >> > (U.data_device_, max_row, i, j, U.rows_ - i, U.cols_ - j);
			cudaDeviceSynchronize();
			coeff_positive_negative *= -1;
		}
		if (U.at(i, j) == 0)
		{
			rank_row = i;
			break;
		}
		dim3 blocks_2((U.cols_ - j - 1) / threads_per_block.x + 1, (U.rows_ - i - 1 - 1) / threads_per_block.y + 1);
		kernelEliminate << <blocks_2, threads_per_block >> > (U.data_device_, i, j, U.rows_, U.cols_);
	}
	cudaDeviceSynchronize();

	floating_type det_val;
	if (rank_row == U.rows_)
	{
		Vec eye = U.getEye();
		int threads_per_block = THREADS_NUM;
		int blocks = BLOCK_NUM(eye.size_,threads_per_block); //(eye.size_ - 1) / THREADS_NUM + 1;
		Vec arr(blocks);
		kernelAccumulateMultiply << <blocks, threads_per_block >> > (arr.data_device_, eye.data_device_, eye.size_);
		cudaDeviceSynchronize();
		kernelAccumulateMultiply << <1, blocks >> > (arr.data_device_, arr.data_device_, arr.size_);
		cudaDeviceSynchronize();
		det_val = arr.at(0)*coeff_positive_negative;
	}
	else
		det_val = 0;
	cudaDeviceSynchronize();
	return det_val;
}
Mat Mat::randomMN(int m, int n)
{
	if (m <= 0 || n <= 0)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		exit(EXIT_FAILURE);
	}
	std::vector<floating_type> arr(m * n);
	for (int i = 0; i < arr.size(); i++)
		arr[i] = randVal();
	Mat result(m, n, arr.data(), COPY_MODE::host_to_device);
	return result;
}
Mat Mat::randomNNSymmetry(int n)
{
	std::vector<floating_type> arr(n * n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			floating_type val;
			while (true)
			{
				val = randVal();
				if (val != 0)
					break;
			}
			if (i > j)
			{
				arr.at(i * n + j) = val;
				arr.at(j * n + i) = val;
			}
			if (i == j)
				arr.at(i * n + j) = floating_type(std::abs(double(val)));
		}
	}
	Mat result(n, n, arr.data(), COPY_MODE::host_to_device);
	return result;
}
Mat Mat::I(int n)
{
	if (n <= 0)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		exit(EXIT_FAILURE);
	}
	std::vector<floating_type> arr(n * n);
	for (int i = 0, j = 0; i < n && j < n; i++, j++)
	{
		if (i == j)
			arr.at(i * n + j) = 1;
	}
	Mat result(n, n, arr.data(), COPY_MODE::host_to_device);
	return result;
}
Mat Mat::zeros(int m, int n)
{
	if (m <= 0 || n <= 0)
	{
		CULMC_ERROR(ERROR_STATUS::DEMENSION_ERROR);
		exit(EXIT_FAILURE);
	}
	Mat mat(m, n, floating_type(0));
	return mat;
}

std::tuple<Mat, Mat, Mat> Mat::PL() const
{
	Mat U(*this);
	Mat P = Mat::I(U.rows_);
	Mat L = Mat::I(U.rows_);
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	//dim3 block((U.cols_ - 1) / threads_per_block.x + 1, (U.rows_ - 1) / threads_per_block.y + 1);
	for (int j = 0; j < U.cols_; j++)//U.cols_
	{
		int i = j + 1;
		if (j == 1)
			int aaaaaaa = 1;
		int max_row = findMaxRow(U, i - 1, j, U.rows_ - (i - 1), U.cols_ - j);
		if (max_row < 0 || max_row >= U.rows_)
		{
			CULMC_ERROR(ERROR_STATUS::OUT_RANGE_ERROR);
			MAT_RELEASE(U, P, L);
			exit(EXIT_FAILURE);
		}	
		if (i - 1 != max_row)
		{
			int blocks_U = (U.cols_ - j - 1) / THREADS_NUM + 1;
			int blocks_L = (j - 1) / THREADS_NUM + 1;
			int blocks_P = (U.cols_ - 1) / THREADS_NUM + 1;
			pivotU << <blocks_U, THREADS_NUM >> > (U.data_device_, max_row, i - 1, j, U.rows_ - (i - 1), U.cols_ - j);
			pivotL << <blocks_L, THREADS_NUM >> > (L.data_device_, max_row, i - 1, j, L.rows_, L.cols_);
			pivotP << <blocks_P, THREADS_NUM >> > (P.data_device_, max_row, i - 1, j, P.rows_, P.cols_);
		}
		cudaDeviceSynchronize();
		if (U.at(i - 1, j) == 0)
		{
			CULMC_ERROR(ERROR_STATUS::ZERO_DIV_BOTTOM);
			MAT_RELEASE(U, P, L);
			exit(EXIT_FAILURE);
		}
		int blocks_1 = (U.rows_ - i) / THREADS_NUM + 1;
		updateL << <blocks_1, THREADS_NUM >> > (L.data_device_, U.data_device_, i - 1, j, U.rows_, U.cols_);
		dim3 blocks_2((U.cols_ - j - 1) / threads_per_block.x + 1, (U.rows_ - (i - 1)-1-1) / threads_per_block.y + 1);
		eliminateLU2 << <blocks_2, threads_per_block >> > (L.data_device_, U.data_device_, i - 1, j, U.rows_, U.cols_);
	}
	cudaDeviceSynchronize();
	return std::make_tuple(L,U,P);
}

std::tuple<Mat,int> Mat::eliminate() const
{
	Mat U(*this);
	int rank_row = U.rows_;
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	dim3 block((U.cols_ - 1) / threads_per_block.x + 1, (U.rows_ - 1) / threads_per_block.y + 1);
	int n = U.rows_;
	for (int j = 0; j < n; j++)//U.cols_
	{
		int i = j;
		int max_row = findMaxRow(U, i, j, U.rows_ - i, U.cols_ - j);
		if (max_row < 0 || max_row >= U.rows_)
			break;
		if (i != max_row)
		{
			int blocks_U = (U.cols_ - j - 1) / THREADS_NUM + 1;
			pivotU << <blocks_U, THREADS_NUM >> > (U.data_device_, max_row, i, j, U.rows_ - i, U.cols_ - j);
			cudaDeviceSynchronize();
		}
		if (U.at(i , j) == 0)
		{
			rank_row = i;
			break;
		}
		dim3 blocks_2((U.cols_ - j - 1) / threads_per_block.x + 1, (U.rows_ - i - 1 - 1) / threads_per_block.y + 1);
		kernelEliminate << <blocks_2, threads_per_block >> > (U.data_device_, i , j, U.rows_, U.cols_);
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	return std::make_tuple(U,rank_row);
}

std::tuple<int, int> Mat::rank() const
{
	int rank_row, rank_col;		
	Mat U_t;
	std::tie(U_t, rank_row) = this->eliminate();
	U_t = std::move(U_t.transpose());
	rank_col = std::get<1>(U_t.eliminate());
	return std::make_tuple(rank_row, rank_col);
}

std::tuple<Mat, Mat> Mat::cholesky() const
{
	Mat A(*this);
	Mat L=Mat::zeros(this->rows_, this->cols_);
	Mat Lt(L);
	for (int j = 0; j < A.cols_; j++)
	{
		int i = j;
		L.set(j, j, floating_type(std::sqrt(double(A.at(j,j)))));
		Lt.set(j, j, floating_type(std::sqrt(double(A.at(j, j)))));
		int threads_per_block_c = THREADS_NUM;
		int blocks_c = (A.rows_-i-1 - 1) / threads_per_block_c + 1;
		kernelUpdateLColesky << <blocks_c, threads_per_block_c >> > (L.data_device_, Lt.data_device_, A.data_device_, i, j, A.rows_, A.cols_);

		dim3 threads_per_block_sub_A(THREADS_NUM, THREADS_NUM);
		dim3 block_sub_A((A.cols_ -i-1- 1) / threads_per_block_sub_A.x + 1, (A.rows_ -i-1- 1) / threads_per_block_sub_A.y + 1);
		kernelUpdateSubAColesky<<<block_sub_A , threads_per_block_sub_A >>> (L.data_device_, Lt.data_device_, A.data_device_, i, j, A.rows_, A.cols_);
	}
	cudaDeviceSynchronize();
	return std::make_tuple(L, Lt);
}

std::tuple<Mat, Mat> Mat::QRbyH() const
{
	Mat Q(this->rows_, this->cols_);
	Mat R(*this);
	Mat A(*this);
	for (int j = 0; j < this->cols_; j++)
	{
		int i = j;
		Vec a = std::move(this->getCol(j, i, this->rows_-1));
		Vec e(a.size_, 0.f);
		e.set(0,1);
		floating_type alpha = floating_type(std::sqrt(double(a.norm2d())));
		Vec w = a - e*alpha;
		floating_type alpha_new = floating_type(std::sqrt(double(w.norm2d())));
		floating_type temp = 1 / alpha_new;
		w = std::move(w * (1 / alpha_new));
		Mat w_(w, true);
		Mat w_transpose(w, false);
		Mat H = Mat::I(w.size_);
		H = H - (w_ * w_.transpose())*2;
		Mat temp2 = H * R;
		temp2.print();
		H.print();
		cudaDeviceSynchronize();
	}

	return std::make_tuple(Q, R);
}

std::tuple<Mat, Mat> Mat::QRbyH2() const
{
	Mat Q(this->rows_, this->cols_);
	Mat R(*this);
	Mat A(*this);
	Mat H = Mat::I(A.rows_);
	Vec a(A.rows_,0);
	Vec omega(A.rows_, 0);
	Vec e(a.size_, 0.f);
	for (int j = 0; j < this->cols_; j++)
	{
		int i = j;
		a = std::move(R.getCol(j, 0, R.rows_ - 1));
		//if(j!=0)
		//	a.set(0)
		e.set(j, 1);
		floating_type alpha = floating_type(std::sqrt(double(a.norm2d())));
		omega = a - e * alpha;
		//R.print(); a.print(); omega.print();
		floating_type alpha_new = floating_type(std::sqrt(double(omega.norm2d())));
		floating_type temp = 1 / alpha_new;
		omega = std::move(omega * (1 / alpha_new));
		Mat w_(omega, true);
		Mat w_transpose(omega, false);
		H = H - (w_ * w_.transpose()) * 2;
		R = H * R;
		H.print();
		cudaDeviceSynchronize();
	}
	//Q.print(); R.print();
	return std::make_tuple(Q, R);
}

//------------util functions------------
int findMaxRow(const Mat& U, int i, int j, int rows_sub, int cols_sub)
{
	//test
	{
		floating_type* arr_1; floating_type* arr_2;
		CHECK(cudaMalloc(&arr_1, nbytes(10)));
		arr_2 = (floating_type*)malloc(nbytes(10));
		CHECK(cudaMemcpy(arr_2, arr_1, nbytes(10), DEVICE_TO_HOST));
		free(arr_2); cudaFree(arr_1);
	}
	//end test

	if (rows_sub == 1)
		return i;
	int threads_per_block = THREADS_NUM;
	const int blocks = (rows_sub - 1) / threads_per_block + 1;

	floating_type* arr_val_device; floating_type* arr_val;
	int* arr_index_device; int* arr_index;

	CHECK(cudaMalloc(&arr_val_device, nbytes(blocks)));
	CHECK(cudaMalloc(&arr_index_device, blocks * sizeof(int)));
	arr_val = (floating_type*)malloc(nbytes(blocks));
	arr_index = (int*)malloc(blocks * sizeof(int));
	kernelFindMaxCol << <blocks, threads_per_block >> > (U.data_device_, arr_val_device, arr_index_device, i, j, rows_sub, cols_sub);
	//cudaDeviceSynchronize();
	CHECK(cudaMemcpy(arr_val, arr_val_device, nbytes(blocks), DEVICE_TO_HOST));
	CHECK(cudaMemcpy(arr_index, arr_index_device, blocks * sizeof(int), DEVICE_TO_HOST));
	floating_type max_val = *arr_val;
	int max_index = *arr_index;

	std::vector<floating_type> vec_val; std::vector<int> vec_index;
	for (int i = 0; i < blocks; i++)
	{
		floating_type val = *(arr_val + i);
		vec_val.push_back(val);
		vec_index.push_back(*(arr_index + i));
		if (val > max_val)
		{
			max_val = val;
			max_index = *(arr_index + i);
		}
	}

	if (max_index<0 || max_index>U.rows())
	{
		PRINT("maxval=" + std::to_string(max_val));
		CULMC_ERROR(ERROR_STATUS::OUT_RANGE_ERROR);
	}
	cudaFree(arr_val_device);
	cudaFree(arr_index_device);
	free(arr_val);
	free(arr_index);
	return max_index;
}
floating_type dot(floating_type* a, floating_type* b, int n)
{
	floating_type sum = 0;
	if (n == 0)
		return sum;
	int threads_per_block = THREADS_NUM;
	int blocks = BLOCK_NUM(n,threads_per_block);
	floating_type *c, *c_device;
	c = (floating_type*)malloc(nbytes(blocks));
	CHECK(cudaMalloc(&c_device, nbytes(blocks)));
	kernelDot<<<blocks, threads_per_block >>>(a, b, c_device, n);
	CHECK(cudaMemcpy(c, c_device, nbytes(blocks), DEVICE_TO_HOST));
	cudaDeviceSynchronize();
	for (int i = 0; i < blocks; i++)
		sum += c[i];
	free(c);
	cudaFree(c_device);
	return sum;
}
floating_type dotAtomic(floating_type* a, floating_type* b, int n)
{
	floating_type sum = 0;
	if (n == 0)
		return sum;
	int threads_per_block = THREADS_NUM;
	int blocks = BLOCK_NUM(n,threads_per_block);
	floating_type* c, * c_device;
	allocateHost<floating_type>(&c, 1);
	allocateDevice<floating_type>(&c_device, 1, 0);
	kernelDotAtomic << <blocks, threads_per_block >> > (c_device, a, b, n);
	memcpyDeviceToHost<floating_type>(c, c_device, 1);
	cudaDeviceSynchronize();
	sum = c[0];
	deallocateHost<floating_type>(&c);
	deallocateDevice<floating_type>(&c_device);
	return sum;
}

Mat solverLU(const Mat& A, const Mat& y)
{
	Mat x(A.rows(),1);
	Mat x_temp(A.rows(), 1);
	Mat L, U, P;
	std::tie(L, U, P) = A.PL();
	Mat y_p = P * y;
	for (int i = 0; i < L.rows(); i++)
	{
		int n = i;
		floating_type accumulate_temp = dotAtomic(L.data_device_ + i * L.cols(), x_temp.data_device_, n);
		x_temp.set(i, 0, y_p.at(i, 0) - accumulate_temp);	
	}
	cudaDeviceSynchronize();
	for (int i = (U.rows() - 1); i >= 0; i--)															// 6 5
	{
		int j = i;
		int n= U.cols()-1 - j;																			// 0 1
		floating_type accumulate_temp = dotAtomic(U.data_device_ + i * U.cols()+j+1, x.data_device_+j+1, n);	// 6 5
		x.set(i, 0, (x_temp.at(i, 0) - accumulate_temp) / U.at(i, i));
	}
	cudaDeviceSynchronize();
	return x;						//Ax=y PAx=Py 
}

//test
int findMaxRow_2(const Mat& U, int i, int j, int rows_sub, int cols_sub)
{
	//test
	int max_index = 1;
	{
		floating_type* arr_1; floating_type* arr_2;
		CHECK(cudaMalloc(&arr_1, nbytes(10)));
		arr_2 = (floating_type*)malloc(nbytes(10));
		CHECK(cudaMemcpy(arr_2, arr_1, nbytes(10), DEVICE_TO_HOST));
		free(arr_2); cudaFree(arr_1);
	}
	//end test

	//if (rows_sub == 1)
	//	return i;
	//int threads_per_block = THREADS_NUM;
	//const int blocks = (rows_sub - 1) / threads_per_block + 1;

	//floating_type* arr_val_device; floating_type* arr_val;
	//int* arr_index_device; int* arr_index;

	//CHECK(cudaMalloc(&arr_val_device, nbytes(blocks)));
	//CHECK(cudaMalloc(&arr_index_device, blocks * sizeof(int)));
	//arr_val = (floating_type*)malloc(nbytes(blocks));
	//arr_index = (int*)malloc(blocks * sizeof(int));
	//kernelFindMaxCol << <blocks, threads_per_block >> > (U.data_device_, arr_val_device, arr_index_device, i, j, rows_sub, cols_sub);
	////cudaDeviceSynchronize();
	//CHECK(cudaMemcpy(arr_val, arr_val_device, nbytes(blocks), DEVICE_TO_HOST));
	//CHECK(cudaMemcpy(arr_index, arr_index_device, blocks * sizeof(int), DEVICE_TO_HOST));
	//floating_type max_val = *arr_val;
	//max_index = *arr_index;

	//std::vector<floating_type> vec_val; std::vector<int> vec_index;
	//for (int i = 0; i < blocks; i++)
	//{
	//	floating_type val = *(arr_val + i);
	//	vec_val.push_back(val);
	//	vec_index.push_back(*(arr_index + i));
	//	if (val > max_val)
	//	{
	//		max_val = val;
	//		max_index = *(arr_index + i);
	//	}
	//}

	//if (max_index<0 || max_index>U.rows())
	//{
	//	PRINT("maxval=" + std::to_string(max_val));
	//	CULMC_ERROR(ERROR_STATUS::OUT_RANGE_ERROR);
	//}
	//cudaFree(arr_val_device);
	//cudaFree(arr_index_device);
	//free(arr_val);
	//free(arr_index);
	return max_index;
}

void Mat::t_temp()
{
	Mat U(*this);
	Mat P = Mat::I(U.rows_);
	Mat L = Mat::I(U.rows_);
	dim3 threads_per_block(THREADS_NUM, THREADS_NUM);
	//dim3 block((U.cols_ - 1) / threads_per_block.x + 1, (U.rows_ - 1) / threads_per_block.y + 1);
	for (int j = 0; j < U.cols_; j++)//U.cols_
	{
		int i = j + 1;
		if (j == 1)
			int aaaaaaa = 1;
		int max_row = findMaxRow_2(U, i - 1, j, U.rows_ - (i - 1), U.cols_ - j);
		if (i - 1 != max_row)
		{
			int blocks_U = (U.cols_ - j - 1) / THREADS_NUM + 1;
			int blocks_L = (j - 1) / THREADS_NUM + 1;
			int blocks_P = (U.cols_ - 1) / THREADS_NUM + 1;
			pivotU << <blocks_U, THREADS_NUM >> > (U.data_device_, max_row, i - 1, j, U.rows_ - (i - 1), U.cols_ - j);
			pivotL << <blocks_L, THREADS_NUM >> > (L.data_device_, max_row, i - 1, j, L.rows_, L.cols_);
			pivotP << <blocks_P, THREADS_NUM >> > (P.data_device_, max_row, i - 1, j, P.rows_, P.cols_);
		}

		int blocks_1 = (U.rows_ - i) / THREADS_NUM + 1;
		updateL << <blocks_1, THREADS_NUM >> > (L.data_device_, U.data_device_, i - 1, j, U.rows_, U.cols_);
		dim3 blocks_2((U.cols_ - j - 1) / threads_per_block.x + 1, (U.rows_ - (i - 1) - 1 - 1) / threads_per_block.y + 1);
		eliminateLU2 << <blocks_2, threads_per_block >> > (L.data_device_, U.data_device_, i - 1, j, U.rows_, U.cols_);
	}
}
//end test
CU_END




