#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../cuGameEngine/cuPixel.cuh"
#include "../cuGameEngine/cuSurface.cuh"

static __device__ __inline__ int cuXIdx()
{
	return blockDim.x * blockIdx.x + threadIdx.x;
}

static __device__ __inline__ int cuYIdx()
{
	return blockDim.y * blockIdx.y + threadIdx.y;
}

static __device__ __inline__ int cuZIdx()
{
	return blockDim.z * blockIdx.z + threadIdx.z;
}

static __device__ __inline__ int cuXSize()
{
	return gridDim.x * blockDim.x;
}

static __device__ __inline__ int cuYSize()
{
	return gridDim.y * blockDim.y;
}
static __device__ __inline__ int cuZSize()
{
	return gridDim.z * blockDim.z;
}

static __device__ __host__ __inline__ cuPixel& tex2d(cuGpuSurface* tex, float x, float y)
{
	int cx = roundf(mapf(x, -1, 1, 0, tex->width - 1));
	int cy = roundf(mapf(y, -1, 1, 0, tex->height - 1));
	return tex->buffer[cy * tex->width + cx];
}

static __device__ __host__ __inline__ cuPixel& tex2d(cuGpuSurface* tex, int x, int y)
{
	return tex->buffer[y * tex->width + x];
}