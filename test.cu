#include "core/cualgebra.h"

void t_data()
{
	int n = 10;
	float* data_h = new float[n];// (floating_type*)malloc(n * sizeof(floating_type));
	std::fill(data_h, data_h + n, 9.9);
	memset(data_h, 9.9, n * sizeof(float));

	float aaaaaaa = *(data_h + 1)；

	int asda = 1;

	//cual::BaseArray data1;

	/*int n = 10;
	floating_type* h_data = (floating_type*)malloc(n * sizeof(floating_type));
	cual::BaseArray data2(h_data, n,cual::COPY_MODE::host_to_device);
	cual::BaseArray data3 (data2);

	floating_type a1[] = { 1,2,3,4,5,6,7,8 };
	floating_type a2[] = { 1,2,3,4,5,6,7,8 };
	floating_type* a3 = new floating_type[8];
	cual::BaseArray ad1(a1, 8, cual::COPY_MODE::host_to_device);
	cual::BaseArray ad2(a2, 8, cual::COPY_MODE::host_to_device);
	cual::BaseArray ad3(a3, 8, cual::COPY_MODE::host_to_device);
	cual::BaseArray ad4(ad1);
	floating_type* res = new floating_type[8];
	ad1.toHost(res);
	floating_type aaa = res[2];
	cual::Vec::add(ad1, ad2, ad3, 8);*/


	/*floating_type a1[] = { 1,2,3,
						   4,5,6};
	cual::Mat m1(2, 3,a1,cual::COPY_MODE::host_to_device);
	cual::Mat m2(3, 2, a1, cual::COPY_MODE::host_to_device);
	cual::Mat m3 = m1 * m2;*/
	//m3.print();
	/*floating_type* res = new floating_type[4];
	m3.toHost(res);
	floating_type aaa = res[0];
	std::cout << aaa << std::endl;*/

	//cual::Mat m5 = m4.subMat(1, 1, 2, 2);
	//cual::Mat m6(m5);
	int aaaaa = 1;
	std::string txt_path = "C:/Users/Lenovo/Desktop/temp/m_out.txt";

	cual::Mat m4 = cual::Mat::randomMN(64, 64);	//eliminate
	cual::Mat L, U, P;
	//m4.t_temp();
	std::tie(L, U, P) = m4.PL();


	//cual::Mat m9 = P * m4;
	//cual::Mat m10 = L * U;


	//cual::Mat m44 = cual::Mat::randomMN(4, 7);
	//m44.setRow(1, 0);

	//int r_row, r_col;
	//std::tie(r_row, r_col) = m44.rank();

	floating_type a3[] = {
		4,12,-16,
		12,37,-43,
		-16,-43,98
	};
	cual::Mat m1(3, 3, a3, cual::COPY_MODE::host_to_device);
	cual::Mat l, lt;
	std::tie(l, lt) = m1.cholesky();
	//cual::Mat LLt = l * lt;
	////LLt.print();

	//cual::Vec vec = m1.getEye();
	//floating_type det_val = m1.det();

	//cual::Mat A = cual::Mat::randomMN(3, 3);
	//cual::Mat y = cual::Mat::randomMN(3, 1);

	//cual::Mat x = cual::solverLU(A, y);

	floating_type a4[] = { 0,3,1,								//0 0 1	
						   0,4,-2,								//0 1 0
						   2,1,1 };								//1 0 0
	cual::Mat B(3, 3, a4, cual::COPY_MODE::host_to_device);
	cual::Mat Q, R;
	std::tie(Q, R) = B.QRbyH2();

	//std::cout << "end" << std::endl；
}