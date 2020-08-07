#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

#include "renderWindow.cuh"
#include "renderPipeline.cuh"
#include "rayTracer.cuh"
#include "cuSurface.cuh"

void key(keyboardEventArgs* e)
{
	if (e->key == VK_ESCAPE)
	{
		ExitProcess(0);
	}
}

int main()
{
	cuSurface* trollFace;
	cuSurface::loadFromFile(L"troll.png", &trollFace);

	renderWindow wnd(1024, 768, false, L"Test");
	renderPipeline pipeline(wnd.width, wnd.height);

	wnd.inputMgr->keyUp += createHandler(key);

	double timeNs = 0;
	int64_t counter = 0;

	for (; !wnd.disposed;)
	{
		auto start = std::chrono::high_resolution_clock::now();

		pipeline.render();
		auto s = pipeline.swChain->back->copyToBuffer(wnd.backBuffer->buffer);
		wnd.repaintWindow();
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