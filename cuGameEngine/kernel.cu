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
cuSurface* trollFace;

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

class simpleRenderer : public cuEffect
{
public:
	void apply(cuSurface* in, cuSurface* out)
	{
		out->fill(cuPixel(255, 0, 0, 64));
		trollFace->rotateAlphaBlitTo(out, posX, posY, 0, 0, trollFace->width, trollFace->height, degToRad(angle += 0.3));
	}
};

int main()
{
	cuSurface::loadFromFile(L"test.png", &trollFace);
	renderWindow wnd(1024, 768, true, L"Test");

	wnd.inputMgr->keyUp += createHandler(key);
	wnd.inputMgr->mouseMove += createHandler(mouseMove);

	wnd.pipeLine->addEffect(new simpleRenderer());

	bool isRunning = true;
	wnd.runLoop(false, false, isRunning);
}