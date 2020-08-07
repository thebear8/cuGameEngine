#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <gdiplus.h>
#pragma comment (lib,"Gdiplus.lib")

#include "cuPixel.cuh"

class gdiSurface
{
public:
	HDC hdc, dHdc;
	HBITMAP bmp;
	BITMAPINFO info;

	cuPixel* buffer;
	int64_t width, height, pxCount, sizeBytes;

	gdiSurface(int64_t width, int64_t height)
	{
		this->width = width;
		this->height = height;
		this->pxCount = width * height;
		this->sizeBytes = width * height * sizeof(cuPixel);
		this->dHdc = GetDC(0);
		this->hdc = CreateCompatibleDC(dHdc);

		this->info = { 0 };
		this->info.bmiHeader.biSize = sizeof(info.bmiHeader);
		GetDIBits(dHdc, (HBITMAP)GetCurrentObject(dHdc, OBJ_BITMAP), 0, 0, 0, &info, DIB_RGB_COLORS);
		this->info.bmiHeader.biCompression = 0;
		this->info.bmiHeader.biSizeImage = 0;
		this->info.bmiHeader.biWidth = width;
		this->info.bmiHeader.biHeight = height * -1;

		this->bmp = CreateDIBSection(hdc, &info, DIB_RGB_COLORS, (void**)&buffer, 0, 0);
		SelectObject(hdc, bmp);
	}

	gdiSurface(wchar_t* file, bool* success)
	{
		auto bmp = Gdiplus::Bitmap::FromFile(file);
		if (bmp->GetLastStatus() == Gdiplus::Ok)
		{
			this->width = bmp->GetWidth();
			this->height = bmp->GetHeight();
			this->pxCount = width * height;
			this->sizeBytes = width * height * sizeof(cuPixel);
			this->dHdc = GetDC(0);
			this->hdc = CreateCompatibleDC(dHdc);

			this->info = { 0 };
			this->info.bmiHeader.biSize = sizeof(info.bmiHeader);
			GetDIBits(dHdc, (HBITMAP)GetCurrentObject(dHdc, OBJ_BITMAP), 0, 0, 0, &info, DIB_RGB_COLORS);
			this->info.bmiHeader.biCompression = 0;
			this->info.bmiHeader.biSizeImage = 0;
			this->info.bmiHeader.biWidth = width;
			this->info.bmiHeader.biHeight = height;

			this->bmp = CreateDIBSection(hdc, &info, DIB_RGB_COLORS, (void**)&buffer, 0, 0);
			SelectObject(hdc, bmp);

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

	~gdiSurface()
	{
		DeleteObject(bmp);
		DeleteDC(hdc);
		ReleaseDC(0, dHdc);
	}

	bool copyToBuffer(void* copyTo)
	{
		return cudaMemcpy(copyTo, buffer, sizeBytes, cudaMemcpyDefault) == cudaSuccess;
	}

	bool copyFromBuffer(void* copyFrom)
	{
		return cudaMemcpy(buffer, copyFrom, sizeBytes, cudaMemcpyDefault) == cudaSuccess;
	}

	static bool loadFromFile(wchar_t* file, gdiSurface** surface)
	{
		bool success;
		gdiSurface* tmpSurface = new gdiSurface(file, &success);

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