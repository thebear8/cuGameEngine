#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <vector>

#include "cuTri.cuh"

class triBuffer
{
public:
	int64_t count, sizeBytes;
	
	std::vector<tri> cpuBuffer;
	tri* gpuBuffer;

	triBuffer(int64_t count)
	{
		this->count = count;
		this->sizeBytes = count * sizeof(tri);
		this->gpuBuffer = 0;

		cpuBuffer.reserve(count);
	}

	triBuffer() : triBuffer(0) 
	{

	}

	~triBuffer()
	{
		cudaFree(gpuBuffer);
	}

	void sync()
	{
		cudaFree(gpuBuffer);
		cudaMalloc(&gpuBuffer, sizeBytes);
		cudaMemcpy(gpuBuffer, cpuBuffer.data(), sizeBytes, cudaMemcpyDefault);
	}

	void push(tri& tri)
	{
		count++;
		sizeBytes += sizeof(tri);
		cpuBuffer.push_back(tri);
	}

	void clear()
	{
		cpuBuffer.clear();
		cudaFree(gpuBuffer);
	}
};