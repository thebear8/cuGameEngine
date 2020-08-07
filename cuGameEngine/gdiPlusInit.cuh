#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <gdiplus.h>
#pragma comment (lib,"Gdiplus.lib")

class gdiPlusInitClass
{
	static const bool unused;
};

const bool gdiPlusInitClass::unused = []
{
	Gdiplus::GdiplusStartupInput gdiPlusStartupInput;
	ULONG_PTR gdiPlusToken;
	Gdiplus::GdiplusStartup(&gdiPlusToken, &gdiPlusStartupInput, 0);

	return true;
}();