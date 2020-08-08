#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>

#include "../cuGameEngine/cuSurface.cuh"
#include "../cuGameEngine/renderWindow.cuh"
#include "../cuGameEngine/mathUtils.cuh"

__global__ void renderMandelbrot(cuPixel* buffer, int64_t width, int64_t height, double nLeft, double nRight, double nTop, double nBottom, float maxIterations, float logMaxIterations, float log2)
{
	int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

	if (xIdx < width && yIdx < height)
	{
		float iterations = 0;
		float x = map(xIdx, 0, width, nLeft, nRight), x1 = 0, x2 = 0;
		float y = map(yIdx, 0, height, nTop, nBottom), y1 = 0, y2 = 0;

		while (iterations < maxIterations && x2 + y2 < 4.0f)
		{
			y1 = (x1 + x1) * y1 + y;
			x1 = x2 - y2 + x;

			x2 = x1 * x1;
			y2 = y1 * y1;

			++iterations;
		}

		float i = iterations - (float)(log(log(x1 * x1 + y1 * y1))) / log2;
		buffer[yIdx * width + xIdx].r = mapf(log(i), 0, logMaxIterations, 0, 255);
		buffer[yIdx * width + xIdx].g = mapf(log(i), 0, logMaxIterations, 0, 16);
		buffer[yIdx * width + xIdx].b = mapf(log(i), 0, logMaxIterations, 0, 96);
	}
}

class mandelbrotRenderer : public cuEffect
{
private:
	renderWindow wnd;
	double nLeft = -2, nRight = 2, nTop = -2, nBottom = 2;

	bool isZooming = false;
	float zoomDirection = 1;

public:
	mandelbrotRenderer() : wnd(1024, 768, true, L"Mandelbrot")
	{
		wnd.pipeLine->addEffect(this);

		wnd.inputMgr->key += createBoundHandler(&mandelbrotRenderer::onKey, this);
		wnd.inputMgr->keyDown += createBoundHandler(&mandelbrotRenderer::onKeyDown, this);
		wnd.inputMgr->keyUp += createBoundHandler(&mandelbrotRenderer::onKeyUp, this);
		wnd.inputMgr->mouseWheel += createBoundHandler(&mandelbrotRenderer::onMouseWheel, this);
	}

	void run()
	{
		bool isRunning = true;
		wnd.runLoop(true, false, isRunning);
	}

	void onKey(keyboardEventArgs* e)
	{
		if (e->key == VK_ESCAPE)
		{
			ExitProcess(0);
		}
	}

	void onKeyDown(keyboardEventArgs* e)
	{
		if (e->c == 'W')
		{
			isZooming = true;
			zoomDirection = 0.95f;
		}
		else if (e->c == 'S')
		{
			isZooming = true;
			zoomDirection = 1.05f;
		}
	}

	void onKeyUp(keyboardEventArgs* e)
	{
		if (e->c == 'W')
		{
			isZooming = false;
		}
		else if (e->c == 'S')
		{
			isZooming = false;
		}
	}

	void onMouseWheel(mouseEventArgs* e)
	{
		auto m = e->delta > 0 ? 0.9f : 1.1f;
		zoom(m);
	}

	void zoom(float m)
	{
		auto nx = (((float)wnd.inputMgr->prevCursorPos.x / wnd.width) * 2.0f) - 1.0f;
		auto ny = (((float)wnd.inputMgr->prevCursorPos.y / wnd.height) * 2.0f) - 1.0f;

		auto x = map(nx, -1, 1, nLeft, nRight);
		auto y = map(ny, -1, 1, nTop, nBottom);

		nLeft = x + (nLeft - x) * m;
		nRight = x + (nRight - x) * m;
		nTop = y + (nTop - y) * m;
		nBottom = y + (nBottom - y) * m;
	}

	void apply(cuSurface* in, cuSurface* out)
	{
		if (isZooming)
		{
			zoom(zoomDirection);
		}

		int64_t width, height;
		dim3 blocks, threads;
		calcGrid(in, out, width, height, blocks, threads);

		auto iterations = 20000;
		renderMandelbrot<<<blocks, threads>>>(out->buffer, width, height, nLeft, nRight, nTop * (height / (double)width), nBottom * (height / (double)width), iterations, log(iterations), log(2.0f));
	}
};

int main()
{
	auto renderer = new mandelbrotRenderer();
	renderer->run();
}
