#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <iostream>
#include <functional>
#include <chrono>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#include "cuPixel.cuh"
#include "cuSurface.cuh"
#include "gdiSurface.cuh"
#include "inputManager.cuh"
#include "renderPipeline.cuh"

class renderWindow
{
private:
	static ATOM wndClass;
	static HINSTANCE hInst;
	static wchar_t const* wndClassName;

public:
	HWND hwnd;
	bool fullScreen;
	int64_t width, height, x, y;
	wchar_t const* windowName;
	DWORD windowStyle;
	RECT windowRect;
	MSG currentMsg;
	bool disposed = false;

	gdiSurface* backBuffer;
	inputManager* inputMgr;
	renderPipeline* pipeLine;

	double lastMsgTime = 0;
	double lastRenderTime = 0;
	double lastPresentTime = 0;
	double lastVSyncTime = 0;
	double lastTotalTime = 0;
	double lastFps = 0;

	renderWindow(int width, int height, bool fullScreen, wchar_t const* name)
	{
		this->width = width;
		this->height = height;
		this->fullScreen = fullScreen;
		this->windowName = name;

		ensureRegisteredWndClass();
		createWindow();
		backBuffer = new gdiSurface(this->width, this->height);
		inputMgr = new inputManager();
		pipeLine = new renderPipeline(this->width, this->height);
	}

	~renderWindow()
	{
		delete backBuffer;
		delete inputMgr;
		delete pipeLine;
	}

	void pollMessage()
	{
		while (PeekMessageW(&currentMsg, hwnd, 0, 0, true))
		{
			TranslateMessage(&currentMsg);
			DispatchMessageW(&currentMsg);
		}
	}

	void repaintWindow()
	{
		InvalidateRect(hwnd, 0, true);

		PAINTSTRUCT ps;
		BeginPaint(hwnd, &ps);
		BitBlt(ps.hdc, 0, 0, width, height, backBuffer->hdc, 0, 0, SRCCOPY);
		EndPaint(hwnd, &ps);
	}

	void runLoop(bool useVSync, bool logFrameTimes, bool& isRunning)
	{
		for (; isRunning && !this->disposed; )
		{
			auto before = std::chrono::high_resolution_clock::now();

			pollMessage();

			auto afterMsg = std::chrono::high_resolution_clock::now();

			pipeLine->render();
			cudaDeviceSynchronize();

			auto afterRender = std::chrono::high_resolution_clock::now();

			pipeLine->swChain->back->copyToBuffer(backBuffer->buffer);
			repaintWindow();

			auto afterPresent = std::chrono::high_resolution_clock::now();

			if (useVSync)
			{
				DwmFlush();
			}

			auto afterVSync = std::chrono::high_resolution_clock::now();

			lastMsgTime = (afterMsg - before).count() / 1000.0;
			lastRenderTime = (afterRender - afterMsg).count() / 1000.0;
			lastPresentTime = (afterPresent - afterRender).count() / 1000.0;
			lastVSyncTime = (afterVSync - afterPresent).count() / 1000.0;
			lastTotalTime = (afterVSync - before).count() / 1000.0;
			lastFps = 1000000.0 / lastTotalTime;

			if (logFrameTimes)
			{
				std::cout << "Render Time: " << lastRenderTime << "us, Present Time: " << lastPresentTime << "us, Message Loop Time: " << lastMsgTime << "us, VSync Time: " << lastVSyncTime << "us. FPS: " << lastFps <<"\n";
			}
		}
	}

private:
	static LRESULT wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
	{
		LPARAM returnValue;
		renderWindow* rWnd = (renderWindow*)GetWindowLongPtrW(hwnd, GWLP_USERDATA);

		if (rWnd && rWnd->inputMgr->handleMsg(msg, wp, lp, returnValue))
		{
			return returnValue;
		}

		switch (msg)
		{
		case WM_PAINT:
			rWnd->repaintWindow();
			break;

		case WM_CLOSE:
			DestroyWindow(hwnd);
			break;

		case WM_DESTROY:
			PostQuitMessage(0);
			rWnd->disposed = true;
			break;

		default:
			return DefWindowProcW(hwnd, msg, wp, lp);
		}

		return 0;
	}

	void createWindow()
	{
		if (fullScreen)
		{
			HMONITOR hMon = MonitorFromWindow(0, MONITOR_DEFAULTTONEAREST);
			MONITORINFO mi = { sizeof(mi) };

			GetMonitorInfoW(hMon, &mi);
			windowRect = mi.rcMonitor;
			windowStyle = WS_POPUP | WS_VISIBLE;
			x = mi.rcMonitor.left;
			y = mi.rcMonitor.top;
			width = (int64_t)mi.rcMonitor.right - mi.rcMonitor.left;
			height = (int64_t)mi.rcMonitor.bottom - mi.rcMonitor.top;
		}
		else
		{
			windowRect = { 0, 0, (LONG)width, (LONG)height };
			windowStyle = WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE;
			x = CW_USEDEFAULT;
			y = CW_USEDEFAULT;
		}

		AdjustWindowRect(&windowRect, windowStyle, false);
		hwnd = CreateWindowExW(WS_EX_WINDOWEDGE, wndClassName, windowName, windowStyle, x, y, width, height, 0, 0, hInst, 0);
		SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)this);
	}

	void ensureRegisteredWndClass()
	{
		if (!wndClass)
		{
			WNDCLASSEXW wClass = { 0 };
			wClass.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
			wClass.cbSize = sizeof(wClass);
			wClass.lpszClassName = wndClassName;
			wClass.lpfnWndProc = wndProc;
			wClass.hInstance = hInst;
			wClass.hIcon = LoadIconA(0, IDI_APPLICATION);
			wClass.hIconSm = LoadIconA(0, IDI_APPLICATION);
			wClass.hCursor = LoadCursorA(0, IDC_ARROW);

			wndClass = RegisterClassExW(&wClass);
		}
	}
};

ATOM renderWindow::wndClass = 0;
HINSTANCE renderWindow::hInst = GetModuleHandleW(0);
wchar_t const* renderWindow::wndClassName = L"cuGameEngineWindowClass";