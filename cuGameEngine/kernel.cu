#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#include "gdiPlusInit.cuh"
#include "renderWindow.cuh"
#include "renderPipeline.cuh"
#include "cuSurface.cuh"

int posX;
int posY;
float angle;

void key(keyboardEventArgs* e)
{
	if (e->key == VK_ESCAPE)
	{
		ExitProcess(0);
	}
}

void mouseMove(mouseEventArgs* e)
{
	posX = e->x;
	posY = e->y;
}

int main()
{
	cuSurface* trollFace;
	cuSurface::loadFromFile(L"test.png", &trollFace);

	//cuSurface* text;
	//cuSurface::loadFromFile(L"test.png", &text);

	renderWindow wnd(1024, 768, true, L"Test");
	renderPipeline pipeline(wnd.width, wnd.height);

	wnd.inputMgr->keyUp += createHandler(key);
	wnd.inputMgr->mouseMove += createHandler(mouseMove);

	double timeNs = 0;
	int64_t counter = 0;

	for (; !wnd.disposed;)
	{
		auto start = std::chrono::high_resolution_clock::now();

		pipeline.swChain->back->fill(cuPixel(255, 0, 0, 64));
		trollFace->rotateAlphaBlitTo(pipeline.swChain->back, posX, posY, 0, 0, trollFace->width, trollFace->height, degToRad(angle += 0.3));
		cudaDeviceSynchronize();

		//auto s = pipeline.swChain->back->copyToBuffer(wnd.backBuffer->buffer);
		//wnd.repaintWindow();
		wnd.pollMessage();

		auto end = std::chrono::high_resolution_clock::now();

		timeNs += (end - start).count();
		if (counter++ == 240)
		{
			std::cout << "FPS:" << 1000.0 / (timeNs / 1000000.0 / 240) << "\n";
			counter = 0;
			timeNs = 0;
		}
	}
}