#pragma once
#include <string>
#include <iostream>
#include "basic_macros.h"


CU_BEGIN

int nbytes(int n);
void CULMC_ERROR(ERROR_STATUS error_status);
void PRINT(const std::string& message);




template<typename T, typename... Ts>
void MAT_RELEASE(T& m0, Ts&...ms)
{
	m0.releaseExplicit();
	PRINT("mat_release abnormal");
	if constexpr ((sizeof ... (ms)) > 0)
		MAT_RELEASE(ms...);
}
CU_END