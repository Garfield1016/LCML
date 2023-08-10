
#include <iostream>
#include "basic_functions.h"
#include "cualgebra.h"
#include <cuda_runtime.h>

CU_BEGIN

int nbytes(int n)
{
	return n * sizeof(floating_type);
}

void CULMC_ERROR(ERROR_STATUS error_status)
{
	switch (error_status)
	{
		case ERROR_STATUS::DEMENSION_ERROR:
		{
			PRINT(std::string("demension error"));
			break;
		}
		case ERROR_STATUS::INDEX_ERROR:
		{
			PRINT("index error");
			break;
		}
		case ERROR_STATUS::ZERO_DIV_BOTTOM:
		{
			PRINT("zero div error");
			break;
		}
		case ERROR_STATUS::OUT_RANGE_ERROR:
		{
			PRINT("out range error");
			break;
		}
		default:
			break;
	}
}

void PRINT(const std::string& message)
{
	std::cout << message << std::endl;
}
CU_END