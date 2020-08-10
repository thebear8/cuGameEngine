#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <sstream>

#include "../cuGameEngine/gdiPlusInit.cuh"
#include "../cuGameEngine/renderWindow.cuh"
#include "../cuGameEngine/cuSurface.cuh"
#include "../cuGameEngine/sdfTextRenderer.cuh"

__global__ void render(cuPixel* buffer, int width, int height, float xOffset, float yOffset, float maxXOffset, float maxYOffset, cuPixel c1, cuPixel c2)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= 0 && x < width && y >= 0 && y < height)
	{
		buffer[y * width + x].r = map(x + xOffset, -maxXOffset, width + maxXOffset, c1.r, c2.r);
		buffer[y * width + x].g = map(x + y + xOffset + yOffset, -xOffset - yOffset, width + height + maxXOffset + maxYOffset, c1.g, c2.g);
		buffer[y * width + x].b = map(y + yOffset, -maxYOffset, height + maxYOffset, c1.b, c2.b);
	}
}

class screenSaver : public cuEffect
{
private:
	renderWindow wnd;
	sdfTextRenderer renderer{ L"lucidaconsole.fnt", L"lucidaconsole.png" };

	bool colorSwitched = false;
	cuPixel c1 = randColor();
	cuPixel c2 = randColor();
	cuPixel lastC1 = randColor();
	cuPixel lastC2 = randColor();
	int frameCounter = 0;

public:
	screenSaver() : wnd(1024, 768, true, L"Render Test")
	{
		wnd.pipeLine->addEffect(this);
		wnd.inputMgr->key += createBoundHandler(&screenSaver::onKey, this);
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

	void apply(cuSurface* in, cuSurface* out)
	{
		int64_t width, height;
		dim3 blocks, threads;
		calcGrid(in, out, width, height, blocks, threads);

		clock_t t = clock();
		float maxXOffset = (width / 5.0f);
		float maxYOffset = (height / 5.0f);
		float xOffset = sin(t / 2500.0f) * maxXOffset;
		float yOffset = sin(t / 2500.0f + 1.0f) * maxYOffset;

		cuPixel renderC1 = blendColor(c1, lastC1, frameCounter / 150.0f);
		cuPixel renderC2 = blendColor(c2, lastC2, frameCounter / 150.0f);

		if (frameCounter++ == 150)
		{
			lastC1 = c1;
			lastC2 = c2;
			c1 = randColor();
			c2 = randColor();

			frameCounter = 0;
		}

		render<<<blocks, threads>>>(out->buffer, width, height, xOffset, yOffset, maxXOffset, maxYOffset, renderC1, renderC2);

		std::wstringstream str;
		str << "FPS:\t\t" << wnd.lastFps << "\nFrametime:\t" << wnd.lastTotalTime << "us";
		renderer.renderString(out, str.str(), 4, 4, out->width, 3, cuPixel(255, 255, 255, 255), true);
	}
};

int main()
{
	//ShowCursor(false);
	auto render = new screenSaver();
	render->run();
}