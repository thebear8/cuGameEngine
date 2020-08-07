#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

#include "cuPixel.cuh"

__device__ __host__ __forceinline__ float map(float x, float in_min, float in_max, float out_min, float out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__device__ __host__ __forceinline__ float clamp(float x, float min, float max)
{
	return x < min ? min : (x > max ? max : x);
}

__device__ __host__ __forceinline__ float degToRad(float deg)
{
	return deg * (3.1415926535f / 180.f);
}

__device__ __host__ __forceinline__ float radToDeg(float rad)
{
	return rad * (180.f / 3.1415926535f);
}

__device__ __host__ __forceinline__ cuPixel blendColor(cuPixel c1, cuPixel c2, float c1Alpha)
{
	float r = (c1.r * c1Alpha) + (c2.r * (1 - c1Alpha));
	float g = (c1.g * c1Alpha) + (c2.g * (1 - c1Alpha));
	float b = (c1.b * c1Alpha) + (c2.b * (1 - c1Alpha));
	return cuPixel(255, r, g, b);
}

__host__ __forceinline__ cuPixel randColor()
{
	return cuPixel(255, map(rand(), 0, RAND_MAX, 0, 255), map(rand(), 0, RAND_MAX, 0, 255), map(rand(), 0, RAND_MAX, 0, 255));
}

