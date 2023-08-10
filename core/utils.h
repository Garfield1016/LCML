#include "basic_macros.h"
#include <chrono>
using namespace std::chrono;

class Timer
{
private:
	cudaEvent_t start_;
	cudaEvent_t stop_;
public:
	Timer()
	{
		cudaEventCreate(&start_);
		cudaEventCreate(&stop_);
	}
	~Timer()
	{
		cudaEventDestroy(start_);
		cudaEventDestroy(stop_);
	}
	void start()
	{
		cudaEventRecord(start_, 0);
	}
	void stop()
	{
		cudaEventRecord(stop_, 0);
	}
	float timeMs()
	{
		float elapsed;
		cudaEventSynchronize(stop_);
		cudaEventElapsedTime(&elapsed, start_, stop_);
		return elapsed;
	}
};

class CPUTimer
{
public:
	CPUTimer();
	~CPUTimer();
	void reset();

	void update();
	double getTimerSecond();
	double getTimerMilliSec();
	long long getTimerMicroSec();
private:
	time_point<high_resolution_clock>_start;
	time_point<high_resolution_clock>_stop;
};

CPUTimer::CPUTimer() { reset(); }
CPUTimer::~CPUTimer() {}
void CPUTimer::reset()
{
	_start = high_resolution_clock::now();
	_stop = high_resolution_clock::now();
}
void CPUTimer::update() { _stop = high_resolution_clock::now(); }
double CPUTimer::getTimerSecond() { return getTimerMicroSec() * 0.000001; }//√Î
double CPUTimer::getTimerMilliSec() { return getTimerMicroSec() * 0.001; }//∫¡√Î
long long CPUTimer::getTimerMicroSec() { return duration_cast<microseconds>(_stop - _start).count(); }//Œ¢√Î