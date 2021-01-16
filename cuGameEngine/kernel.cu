#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <sstream>

#include "gdiPlusInit.cuh"
#include "renderWindow.cuh"
#include "cuSurface.cuh"
#include "sdfTextRenderer.cuh"

class renderTest : public cuEffect
{
private:
	renderWindow wnd;
	sdfTextRenderer renderer{ L"lucidaconsole.fnt", L"lucidaconsole.png" };
	sdfTextBuffer buffer;

	int posX;
	int posY;
	float angle;

public:
	renderTest() : wnd(1024, 768, false, L"Render Test")
	{
		wnd.pipeLine->addEffect(this);
		wnd.inputMgr->key += createBoundHandler(&renderTest::onKey, this);
		wnd.inputMgr->mouseMove += createBoundHandler(&renderTest::onMouse, this);
	}

	void run()
	{
		bool isRunning = true;
		wnd.runLoop(false, true, isRunning);
	}

	void runOnce()
	{
		bool isRunning = false;
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

		std::wstringstream str;
		for (int i = 0; i < 200; i++)
		{
			str << L"AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789!\"§$%&/()=?`*'#+~-.:,;|<>";
		}

		//buffer.clear();
		//renderer.addStringToBuffer(&buffer, str.str(), 0, 0, out->width, 2.5, cuPixel(255, 255, 255, 255), true);

		//buffer.clipForSurface(out);
		//buffer.uploadClippedGlyphs();
		//renderer.renderTextBuffer(out, &buffer, true);

		//volatile auto r = cudaDeviceSynchronize();

		//str << L"FPS:\t\t" << wnd.lastFps << "\nFrametime:\t" << wnd.lastTotalTime << "us";
		renderer.renderString(out, str.str(), 0, 0, out->width, 0.25, cuPixel(255, 255, 255, 255), true, false);
		cudaDeviceSynchronize();
	}
};

int main()
{
	auto render = new renderTest();
	render->run();
	//render->runOnce();
	return 0;
}