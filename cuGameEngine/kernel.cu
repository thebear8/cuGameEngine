#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "gdiPlusInit.cuh"
#include "renderWindow.cuh"
#include "cuSurface.cuh"

class renderTest : public cuEffect
{
private:
	renderWindow wnd;
	cuSurface* trollFace;

	int posX;
	int posY;
	float angle;

public:
	renderTest() : wnd(1024, 768, false, L"Render Test")
	{
		cuSurface::loadFromFile(L"troll.png", &trollFace);
		wnd.pipeLine->addEffect(this);
		wnd.inputMgr->key += createBoundHandler(&renderTest::onKey, this);
		wnd.inputMgr->mouseMove += createBoundHandler(&renderTest::onMouse, this);
	}

	void run()
	{
		bool isRunning = true;
		wnd.runLoop(false, false, isRunning);
	}

	void onKey(keyboardEventArgs* e)
	{
		if (e->key == VK_ESCAPE)
		{
			ExitProcess(0);
		}
	}

	void onMouse(mouseEventArgs* e)
	{
		posX = e->x;
		posY = e->y;
	}

	void apply(cuSurface* in, cuSurface* out)
	{
		out->fill(cuPixel(255, 0, 0, 64));
		trollFace->rotateAlphaBlitTo(out, posX, posY, 0, 0, trollFace->width, trollFace->height, degToRad(angle += 0.3));
	}
};

int main()
{
	auto render = new renderTest();
	render->run();
}