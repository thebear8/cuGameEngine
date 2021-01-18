#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <gdiplus.h>
#pragma comment (lib,"Gdiplus.lib")

#include "cuPixel.cuh"
#include "mathUtils.cuh"

__global__ void cuSurfaceBlit(cuPixel* to, float toWidth, float toHeight, float toXOff, float toYOff, cuPixel* from, float fromWidth, float fromHeight, float fromXOff, float fromYOff, float countX, float countY)
{
	float x = blockDim.x * blockIdx.x + threadIdx.x;
	float y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= 0 && x < countX && y >= 0 && y < countY)
	{
		float toX = toXOff + x;
		float toY = toYOff + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			float fromX = fromXOff + x;
			float fromY = fromYOff + y;
			if (fromX >= 0 && fromX < fromWidth && fromY >= 0 && fromY < fromHeight)
			{
				to[int(toY * toWidth + toX)] = from[int(fromY * fromWidth + fromX)];
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
		int toX = toXOff + x;
		int toY = toYOff + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			int fromX = fromXOff + x;
			int fromY = fromYOff + y;
			if (fromX >= 0 && fromX < fromWidth && fromY >= 0 && fromY < fromHeight)
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

__global__ void cuSurfaceScaleBlit(cuPixel* to, int64_t toWidth, int64_t toHeight, int64_t toXOff, int64_t toYOff, cuPixel* from, int64_t fromWidth, int64_t fromHeight, int64_t fromXOff, int64_t fromYOff, int64_t toCountX, int64_t toCountY, int64_t fromCountX, int64_t fromCountY)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= 0 && x < toCountX && y >= 0 && y < toCountY)
	{
		int toX = toXOff + x;
		int toY = toYOff + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			int fromX = fromXOff + map(x, 0, toCountX, 0, fromCountX);
			int fromY = fromYOff + map(y, 0, toCountY, 0, fromCountY);
			if (fromX >= 0 && fromX < fromWidth && fromY >= 0 && fromY < fromHeight)
			{
				to[toY * toWidth + toX] = from[fromY * fromWidth + fromX];
			}
		}
	}
}

__global__ void cuSurfaceScaleAlphaBlit(cuPixel* to, int64_t toWidth, int64_t toHeight, int64_t toXOff, int64_t toYOff, cuPixel* from, int64_t fromWidth, int64_t fromHeight, int64_t fromXOff, int64_t fromYOff, int64_t toCountX, int64_t toCountY, int64_t fromCountX, int64_t fromCountY)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= 0 && x < toCountX && y >= 0 && y < toCountY)
	{
		int toX = toXOff + x;
		int toY = toYOff + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			int fromX = fromXOff + map(x, 0, toCountX, 0, fromCountX);
			int fromY = fromYOff + map(y, 0, toCountY, 0, fromCountY);
			if (fromX >= 0 && fromX < fromWidth && fromY >= 0 && fromY < fromHeight)
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

__global__ void cuSurfaceRotateBlit(cuPixel* to, int64_t toWidth, int64_t toHeight, float toXRotCenter, float toYRotCenter, cuPixel* from, int64_t fromWidth, int64_t fromHeight, float crossDist, float sinAngle, float cosAngle)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x - crossDist;
	int y = blockDim.y * blockIdx.y + threadIdx.y - crossDist;

	if (x >= -crossDist && x <= crossDist && y >= -crossDist && y <= crossDist)
	{
		int toX = toXRotCenter + x;
		int toY = toYRotCenter + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			int fromX = x * cosAngle - y * sinAngle + (fromWidth / 2);
			int fromY = y * cosAngle + x * sinAngle + (fromHeight / 2);
			if (fromX >= 0 && fromX < fromWidth && fromY >= 0 && fromY < fromHeight)
			{
				to[toY * toWidth + toX] = from[fromY * fromWidth + fromX];
			}
		}
	}
}

__global__ void cuSurfaceRotateAlphaBlit(cuPixel* to, int64_t toWidth, int64_t toHeight, float toXRotCenter, float toYRotCenter, cuPixel* from, int64_t fromWidth, int64_t fromHeight, float crossDist, float sinAngle, float cosAngle)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x - crossDist;
	int y = blockDim.y * blockIdx.y + threadIdx.y - crossDist;

	if (x >= -crossDist && x <= crossDist && y >= -crossDist && y <= crossDist)
	{
		int toX = toXRotCenter + x;
		int toY = toYRotCenter + y;
		if (toX >= 0 && toX < toWidth && toY >= 0 && toY < toHeight)
		{
			int fromX = x * cosAngle - y * sinAngle + (fromWidth / 2);
			int fromY = y * cosAngle + x * sinAngle + (fromHeight / 2);
			if (fromX >= 0 && fromX < fromWidth && fromY >= 0 && fromY < fromHeight)
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

__global__ void cuSurfaceFillColor(cuPixel* buffer, int64_t width, int64_t height, cuPixel color)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= 0 && x < width && y >= 0 && y < height)
	{
		buffer[y * width + x] = color;
	}
}

class cuGpuSurface
{
public:
	cuPixel* buffer;
	int width, height;

	__device__ __inline__ bool isInBounds(int x, int y)
	{
		return x >= 0 && x < width && y >= 0 && y < height;
	}
};

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

	cuSurface(wchar_t const* file, bool* success)
	{
		auto bmp = Gdiplus::Bitmap::FromFile(file);
		if (bmp && bmp->GetLastStatus() == Gdiplus::Ok)
		{
			this->width = bmp->GetWidth();
			this->height = bmp->GetHeight();
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

	cuGpuSurface getGpuSurfaceInfo()
	{
		return { buffer, (int)width, (int)height };
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
		dim3 threads = dim3(32, 32, 1);
		dim3 blocks = dim3(ceil(countX / 32.0f), ceil(countY / 32.0f), 1);
		cuSurfaceBlit << <blocks, threads >> > (to->buffer, to->width, to->height, toX, toY, buffer, width, height, fromX, fromY, countX, countY);
	}

	void alphaBlitTo(cuSurface* to, int64_t toX, int64_t toY, int64_t fromX, int64_t fromY, int64_t countX, int64_t countY)
	{
		dim3 threads = dim3(32, 32, 1);
		dim3 blocks = dim3(ceil(countX / 32.0f), ceil(countY / 32.0f), 1);
		cuSurfaceBlitAlpha << <blocks, threads >> > (to->buffer, to->width, to->height, toX, toY, buffer, width, height, fromX, fromY, countX, countY);
	}

	void scaleBlitTo(cuSurface* to, int64_t toX, int64_t toY, int64_t fromX, int64_t fromY, int64_t toCountX, int64_t toCountY, int64_t fromCountX, int64_t fromCountY)
	{
		dim3 threads = dim3(32, 32, 1);
		dim3 blocks = dim3(ceil(toCountX / 32.0f), ceil(toCountY / 32.0f), 1);
		cuSurfaceScaleBlit << <blocks, threads >> > (to->buffer, to->width, to->height, toX, toY, buffer, width, height, fromX, fromY, toCountX, toCountY, fromCountX, fromCountY);
	}

	void scaleAlphaBlitTo(cuSurface* to, int64_t toX, int64_t toY, int64_t fromX, int64_t fromY, int64_t toCountX, int64_t toCountY, int64_t fromCountX, int64_t fromCountY)
	{
		dim3 threads = dim3(32, 32, 1);
		dim3 blocks = dim3(ceil(toCountX / 32.0f), ceil(toCountY / 32.0f), 1);
		cuSurfaceScaleAlphaBlit << <blocks, threads >> > (to->buffer, to->width, to->height, toX, toY, buffer, width, height, fromX, fromY, toCountX, toCountY, fromCountX, fromCountY);
	}

	void rotateBlitTo(cuSurface* to, int64_t toXCenter, int64_t toYCenter, int64_t fromX, int64_t fromY, int64_t countX, int64_t countY, float angleRad)
	{
		float crossDist = sqrtf(countX * countX + countY * countY);
		dim3 threads = dim3(32, 32, 1);
		dim3 blocks = dim3(ceil((2 * crossDist) / 32.0f), ceil((2 * crossDist) / 32.0f), 1);
		cuSurfaceRotateBlit << <blocks, threads >> > (to->buffer, to->width, to->height, toXCenter, toYCenter, buffer, width, height, crossDist, sin(angleRad), cos(angleRad));
	}

	void rotateAlphaBlitTo(cuSurface* to, int64_t toXCenter, int64_t toYCenter, int64_t fromX, int64_t fromY, int64_t countX, int64_t countY, float angleRad)
	{
		float crossDist = sqrtf(countX * countX + countY * countY);
		dim3 threads = dim3(32, 32, 1);
		dim3 blocks = dim3(ceil((2 * crossDist) / 32.0f), ceil((2 * crossDist) / 32.0f), 1);
		cuSurfaceRotateAlphaBlit << <blocks, threads >> > (to->buffer, to->width, to->height, toXCenter, toYCenter, buffer, width, height, crossDist, sin(angleRad), cos(angleRad));
	}

	void fill(cuPixel color)
	{
		dim3 threads = dim3(32, 32, 1);
		dim3 blocks = dim3(ceil(width / 32.0f), ceil(height / 32.0f), 1);
		cuSurfaceFillColor << <blocks, threads >> > (buffer, width, height, color);
	}

	static bool loadFromFile(wchar_t const* file, cuSurface** surface)
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
