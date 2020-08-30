#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

#include "cuPixel.cuh"

__device__ __host__ __forceinline__ float mapf(float x, float in_min, float in_max, float out_min, float out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__device__ __host__ __forceinline__ float clampf(float x, float min, float max)
{
	return x < min ? min : (x > max ? max : x);
}

__device__ __host__ __forceinline__ float smoothStepf(float x, float in_min, float in_max)
{
	return clampf((x - in_min) / (in_max - in_min), 0, 1);
}

__device__ __host__ __forceinline__ float degToRadf(float deg)
{
	return deg * (3.1415926535f / 180.f);
}

__device__ __host__ __forceinline__ float radToDegf(float rad)
{
	return rad * (180.f / 3.1415926535f);
}


__device__ __host__ __forceinline__ float byteToFloat(unsigned char x)
{
	return x / 256.0f;
}

__device__ __host__ __forceinline__ unsigned char floatToByte(float x)
{
	return x * 256.0f;
}


__device__ __host__ __forceinline__ double map(double x, double in_min, double in_max, double out_min, double out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__device__ __host__ __forceinline__ double clamp(double x, double min, double max)
{
	return x < min ? min : (x > max ? max : x);
}

__device__ __host__ __forceinline__ float smoothStep(double x, double in_min, double in_max)
{
	return clamp((x - in_min) / (in_max - in_min), 0, 1);
}

__device__ __host__ __forceinline__ double degToRad(double deg)
{
	return deg * (3.1415926535f / 180.f);
}

__device__ __host__ __forceinline__ double radToDeg(double rad)
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

__device__ __host__ __forceinline__ cuPixel blendPixel(cuPixel c1, cuPixel c2, float c1Alpha)
{
	float a = (c1.a * c1Alpha) + (c2.a * (1 - c1Alpha));
	float r = (c1.r * c1Alpha) + (c2.r * (1 - c1Alpha));
	float g = (c1.g * c1Alpha) + (c2.g * (1 - c1Alpha));
	float b = (c1.b * c1Alpha) + (c2.b * (1 - c1Alpha));
	return cuPixel(a, r, g, b);
}

__device__ __host__ __forceinline__ cuPixel interpolatePixel(cuPixel* surface, int64_t width, int64_t height, float x, float y)
{
	float x1 = clampf(floorf(x), 0, width - 1), x2 = clampf(ceilf(x), 0, width - 1), xRatio = ceilf(x) - x;
	float y1 = clampf(floorf(y), 0, height - 1), y2 = clampf(ceilf(y), 0, height - 1), yRatio = ceilf(y) - y;

	auto cX1Y1 = surface[(int64_t)(y1 * width + x1)];
	auto cX2Y1 = surface[(int64_t)(y1 * width + x2)];
	auto cX1Y2 = surface[(int64_t)(y2 * width + x1)];
	auto cX2Y2 = surface[(int64_t)(y2 * width + x2)];

	auto cX1 = blendPixel(cX1Y1, cX2Y1, xRatio), cX2 = blendPixel(cX1Y2, cX2Y2, xRatio);
	return blendPixel(cX1, cX2, yRatio);
}

__host__ __forceinline__ cuPixel randColor()
{
	return cuPixel(255, map(rand(), 0, RAND_MAX, 0, 255), map(rand(), 0, RAND_MAX, 0, 255), map(rand(), 0, RAND_MAX, 0, 255));
}

__host__ __forceinline__ int randRange(int min, int max) 
{
	return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

__host__ __forceinline__ float randRange(float min, float max)
{
	return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}