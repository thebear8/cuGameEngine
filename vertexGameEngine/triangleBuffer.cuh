#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "math.cuh"
#include "../cuGameEngine/cuPixel.cuh"

struct vertexTriangle
{
	cuPixel color;
	vec3 a, b, c;

	__device__ __host__ __inline__ vertexTriangle(cuPixel color, vec3 a, vec3 b, vec3 c) : color(color), a(a), b(b), c(c) { }
};

class triangleBuffer
{
	std::vector<vertexTriangle> cpuTriangles;
	vertexTriangle* gpuTriangles;
	size_t gpuTriangleCount;
	size_t gpuTriangleMaxCount;

public:
	triangleBuffer()
	{
		this->gpuTriangles = nullptr;
		this->gpuTriangleCount = 0;
		this->gpuTriangleMaxCount = 0;
	}

	~triangleBuffer()
	{
		cudaFree(gpuTriangles);
	}

	inline void cloneFrom(const triangleBuffer& buffer)
	{
		cpuTriangles.clear();
		cpuTriangles.reserve(buffer.cpuTriangles.size());
		cpuTriangles.insert(cpuTriangles.begin(), buffer.cpuTriangles.data(), buffer.cpuTriangles.data() + buffer.cpuTriangles.size());
	}

	inline void push(const vertexTriangle& tri)
	{
		cpuTriangles.push_back(tri);
	}

	inline void clear()
	{
		cpuTriangles.clear();
	}

	inline void upload()
	{
		if (cpuTriangles.size() > gpuTriangleMaxCount)
		{
			cudaFree(gpuTriangles);
			gpuTriangleMaxCount = cpuTriangles.size();
			cudaMalloc(&gpuTriangles, gpuTriangleMaxCount);
		}

		gpuTriangleCount = cpuTriangles.size();
		cudaMemcpy(gpuTriangles, cpuTriangles.data(), gpuTriangleCount * sizeof(vertexTriangle), cudaMemcpyDefault);
	}

	inline void download()
	{
		if (gpuTriangleCount != cpuTriangles.size())
		{
			cpuTriangles.clear();
			cpuTriangles.resize(gpuTriangleCount, vertexTriangle({ 0, 0, 0, 0 }, {0, 0, 0}, { 0, 0, 0 }, { 0, 0, 0 }));
		}

		cudaMemcpy(cpuTriangles.data(), gpuTriangles, gpuTriangleCount * sizeof(vertexTriangle), cudaMemcpyDefault);
	}

	inline size_t getGpuTriangleCount()
	{
		return gpuTriangleCount;
	}

	inline vertexTriangle* getGpuTrianglePtr()
	{
		return gpuTriangles;
	}
};