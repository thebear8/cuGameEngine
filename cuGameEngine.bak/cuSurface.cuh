#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <gdiplus.h>
#pragma comment (lib,"Gdiplus.lib")

#include "cuPixel.cuh"
#include "mathUtils.cuh"

__global__ void cuSurfaceBlit(cuPixel* to, int64_t toWidth, int64_t toHeight, int64_t toXOff, int64_t toYOff, cuPixel* from, int64_t fromWidth, int64_t fromHeight, int64_t fromXOff, int64_t fromYOff, int64_t countX, int64_t countY)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= 0 && x < countX && y >= 0 && y < countY)
	{
		auto toX = toXOff + x;
		auto toY = toYOff + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			auto fromX = fromXOff + x;
			auto fromY = fromYOff + y;
			if (fromX >= 0 && fromX < toWidth && fromY >= 0 && fromY < toHeight)
			{
				to[toY * toWidth + toX] = from[fromY * fromWidth + fromX];
			}
		}
	}
}

__global__ void cuSurfaceBlitAlpha(cuPixel* to, int64_t toWidth, int64_t toHeight, int64_t toXOff, int64_t toYOff, cuPixel* from, int64_t fromWidth, int64_t fromHeight, int64_t fromXOff, int64_t fromYOff, int64_t countX, int64_t countY)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= 0 && x < countX && y >= 0 && y < countY)
	{
		auto toX = toXOff + x;
		auto toY = toYOff + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			auto fromX = fromXOff + x;
			auto fromY = fromYOff + y;
			if (fromX >= 0 && fromX < toWidth && fromY >= 0 && fromY < toHeight)
			{
				cuPixel* toPixel = &to[toY * toWidth + toX];
				cuPixel* fromPixel = &from[fromY * fromWidth + fromX];
				float alpha = map(fromPixel->a, 0, 255, 0, 1);

				toPixel->r = (fromPixel->r * alpha) + (toPixel->r * (1 - alpha));
				toPixel->g = (fromPixel->g * alpha) + (toPixel->g * (1 - alpha));
				toPixel->b = (fromPixel->b * alpha) + (toPixel->b * (1 - alpha));
			}
		}
	}
}

class cuSurface
{
public:
	cuPixel* buffer;
	int64_t width, height, pxCount, sizeBytes;

	cuSurface(int64_t width, int64_t height)
	{
		this->width = width;
		this->height = height;
		this->pxCount = width * height;
		this->sizeBytes = width * height * sizeof(cuPixel);
		cudaMalloc(&buffer, sizeBytes);
	}
	
	cuSurface(wchar_t* file, bool* success)
	{
		auto bmp = Gdiplus::Bitmap::FromFile(file);
		if (bmp->GetLastStatus() == Gdiplus::Ok)
		{
			this->width = width;
			this->height = height;
			this->pxCount = width * height;
			this->sizeBytes = width * height * sizeof(cuPixel);
			cudaMalloc(&buffer, sizeBytes);

			Gdiplus::Rect rect = Gdiplus::Rect(0, 0, bmp->GetWidth(), bmp->GetHeight());
			Gdiplus::BitmapData data;
			bmp->LockBits(&rect, 0, PixelFormat32bppARGB, &data);
			cudaMemcpy(buffer, data.Scan0, width * height * sizeof(cuPixel), cudaMemcpyDefault);
			bmp->UnlockBits(&data);

			*success = true;
		}
		else
		{
			*success = false;
		}

		delete bmp;
	}

	~cuSurface()
	{
		cudaFree(buffer);
	}

	bool copyToBuffer(void* copyTo)
	{
		return cudaMemcpy(copyTo, buffer, sizeBytes, cudaMemcpyDefault) == cudaSuccess;
	}

	bool copyFromBuffer(void* copyFrom)
	{
		return cudaMemcpy(buffer, copyFrom, sizeBytes, cudaMemcpyDefault) == cudaSuccess;
	}

	void blitTo(cuSurface* to, int64_t toX, int64_t toY, int64_t fromX, int64_t fromY, int64_t countX, int64_t countY)
	{
		dim3 threads = dim3(20, 20, 1);
		dim3 blocks = dim3(ceil(countX / 20.0f), ceil(countY / 20.0f), 1);
		cuSurfaceBlit << <blocks, threads >> > (to->buffer, to->width, to->height, toX, toY, buffer, width, height, fromX, fromY, countX, countY);
	}

	void alphaBlitTo(cuSurface* to, int64_t toX, int64_t toY, int64_t fromX, int64_t fromY, int64_t countX, int64_t countY)
	{
		dim3 threads = dim3(20, 20, 1);
		dim3 blocks = dim3(ceil(countX / 20.0f), ceil(countY / 20.0f), 1);
		cuSurfaceBlitAlpha << <blocks, threads >> > (to->buffer, to->width, to->height, toX, toY, buffer, width, height, fromX, fromY, countX, countY);
	}

	static bool loadFromFile(wchar_t* file, cuSurface** surface)
	{
		bool success;
		cuSurface* tmpSurface = new cuSurface(file, &success);

		if (success)
		{
			*surface = tmpSurface;
			return true;
		}
		else
		{
			delete tmpSurface;
			return false;
		}
	}
};