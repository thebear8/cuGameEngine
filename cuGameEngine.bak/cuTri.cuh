#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>

#include "cuVec.cuh"
#include "cuPixel.cuh"

struct tri
{
	vec3f p1, p2, p3;
	cuPixel color;

	__device__ __host__ __forceinline__ tri(vec3f p1, vec3f p2, vec3f p3, cuPixel color) : p1(p1), p2(p2), p3(p3), color(color)
	{
		
	}

	__device__ __host__ __forceinline__ tri() : tri({ 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0, 0})
	{

	}
};