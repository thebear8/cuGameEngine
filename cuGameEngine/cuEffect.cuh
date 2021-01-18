#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <math.h>
#include <exception>

#include "cuSurface.cuh"

class cuEffect
{
public:
	virtual void apply(cuSurface* in, cuSurface* out) = 0;
	static void calcGrid(cuSurface* in, cuSurface* out, int64_t& width, int64_t& height, dim3& blocks, dim3& threads)
	{
		if (in->width != out->width || in->height != out->height)
		{
			throw std::exception("surface dimensions not equal");
		}
		else
		{
			width = in->width;
			height = in->height;
			threads = dim3(20, 20, 1);
			blocks = dim3(ceil(width / 20.0f), ceil(height / 20.0f), 1);
		}
	}

	static void calcGrid(int64_t width, int64_t height, dim3& blocks, dim3& threads)
	{
		threads = dim3(20, 20, 1);
		blocks = dim3(ceil(width / 20.0f), ceil(height / 20.0f), 1);
	}
};