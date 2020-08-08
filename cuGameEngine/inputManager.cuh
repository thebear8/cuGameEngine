#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <windowsx.h>
#include <functional>

#include "inputEvent.cuh"

class inputManager
{
private:
	keyboardEventArgs keyboardArgs{};
	mouseEventArgs mouseArgs{};

public:
	POINT prevCursorPos = { 0 };
	keyboardEvent key, keyDown, keyUp;
	mouseEvent mouse, mouseDown, mouseUp, mouseWheel, mouseMove;
	bool captureMouse = false;

	inputManager()
	{
		GetCursorPos(&prevCursorPos);
	}

	bool handleMsg(UINT msg, WPARAM wp, LPARAM lp, LPARAM& returnValue)
	{
		switch (msg)
		{
		case WM_KEYDOWN:
		case WM_SYSKEYDOWN:
			returnValue = handleKeyDown(wp, lp);
			break;

		case WM_KEYUP:
		case WM_SYSKEYUP:
			returnValue = handleKeyUp(wp, lp);
			break;

		case WM_LBUTTONDOWN:
			returnValue = handleMouseDown(VK_LBUTTON, wp, lp);
			break;
		case WM_LBUTTONUP:
			returnValue = handleMouseUp(VK_LBUTTON, wp, lp);
			break;

		case WM_MBUTTONDOWN:
			returnValue = handleMouseDown(VK_MBUTTON, wp, lp);
			break;
		case WM_MBUTTONUP:
			returnValue = handleMouseUp(VK_MBUTTON, wp, lp);
			break;

		case WM_RBUTTONDOWN:
			returnValue = handleMouseDown(VK_RBUTTON, wp, lp);
			break;
		case WM_RBUTTONUP:
			returnValue = handleMouseUp(VK_RBUTTON, wp, lp);
			break;

		case WM_XBUTTONDOWN:
			returnValue = handleMouseXDown(wp, lp);
			break;
		case WM_XBUTTONUP:
			returnValue = handleMouseXDown(wp, lp);
			break;

		case WM_MOUSEMOVE:
			returnValue = handleMouseMove(wp, lp);
			break;

		case WM_MOUSEWHEEL:
			returnValue = handleMouseWheel(wp, lp);
			break;

		default:
			return false;
		}

		return true;
	}

private:
	LPARAM handleKeyDown(WPARAM wp, LPARAM lp)
	{
		if (~lp & (1<<30))
		{
			keyboardArgs.key = wp;
			keyboardArgs.c = MapVirtualKeyA(wp, MAPVK_VK_TO_CHAR);
			keyboardArgs.type = keyboardEventArgs::keyboardEventType::keyDown;

			key(&keyboardArgs);
			keyDown(&keyboardArgs);
			return 0;
		}
		else
		{
			return 1;
		}
	}

	LPARAM handleKeyUp(WPARAM wp, LPARAM lp)
	{
		keyboardArgs.key = wp;
		keyboardArgs.c = MapVirtualKeyA(wp, MAPVK_VK_TO_CHAR);
		keyboardArgs.type = keyboardEventArgs::keyboardEventType::keyUp;

		key(&keyboardArgs);
		keyUp(&keyboardArgs);
		return 0;
	}

	LPARAM handleMouseDown(int key, WPARAM wp, LPARAM lp)
	{
		mouseArgs.key = key;
		mouseArgs.delta = 0;
		mouseArgs.x = GET_X_LPARAM(lp);
		mouseArgs.y = GET_Y_LPARAM(lp);
		mouseArgs.dx = mouseArgs.x - prevCursorPos.x;
		mouseArgs.dy = mouseArgs.y - prevCursorPos.y;
		mouseArgs.type = mouseEventArgs::mouseEventType::mouseDown;

		prevCursorPos.x = mouseArgs.x;
		prevCursorPos.y = mouseArgs.y;

		mouse(&mouseArgs);
		mouseDown(&mouseArgs);
		return 0;
	}

	LPARAM handleMouseUp(int key, WPARAM wp, LPARAM lp)
	{
		mouseArgs.key = key;
		mouseArgs.delta = 0;
		mouseArgs.x = GET_X_LPARAM(lp);
		mouseArgs.y = GET_Y_LPARAM(lp);
		mouseArgs.dx = mouseArgs.x - prevCursorPos.x;
		mouseArgs.dy = mouseArgs.y - prevCursorPos.y;
		mouseArgs.type = mouseEventArgs::mouseEventType::mouseUp;

		prevCursorPos.x = mouseArgs.x;
		prevCursorPos.y = mouseArgs.y;

		mouse(&mouseArgs);
		mouseUp(&mouseArgs);
		return 0;
	}

	LPARAM handleMouseXDown(WPARAM wp, LPARAM lp)
	{
		mouseArgs.key = GET_XBUTTON_WPARAM(wp);
		mouseArgs.delta = 0;
		mouseArgs.x = GET_X_LPARAM(lp);
		mouseArgs.y = GET_Y_LPARAM(lp);
		mouseArgs.dx = mouseArgs.x - prevCursorPos.x;
		mouseArgs.dy = mouseArgs.y - prevCursorPos.y;
		mouseArgs.type = mouseEventArgs::mouseEventType::mouseDown;

		prevCursorPos.x = mouseArgs.x;
		prevCursorPos.y = mouseArgs.y;

		mouse(&mouseArgs);
		mouseDown(&mouseArgs);
		return 1;
	}

	LPARAM handleMouseXUp(WPARAM wp, LPARAM lp)
	{
		mouseArgs.key = GET_XBUTTON_WPARAM(wp);
		mouseArgs.delta = 0;
		mouseArgs.x = GET_X_LPARAM(lp);
		mouseArgs.y = GET_Y_LPARAM(lp);
		mouseArgs.dx = mouseArgs.x - prevCursorPos.x;
		mouseArgs.dy = mouseArgs.y - prevCursorPos.y;
		mouseArgs.type = mouseEventArgs::mouseEventType::mouseUp;

		prevCursorPos.x = mouseArgs.x;
		prevCursorPos.y = mouseArgs.y;

		mouse(&mouseArgs);
		mouseUp(&mouseArgs);
		return 1;
	}

	LPARAM handleMouseWheel(WPARAM wp, LPARAM lp)
	{
		mouseArgs.key = 0;
		mouseArgs.delta = GET_WHEEL_DELTA_WPARAM(wp);
		mouseArgs.x = GET_X_LPARAM(lp);
		mouseArgs.y = GET_Y_LPARAM(lp);
		mouseArgs.dx = mouseArgs.x - prevCursorPos.x;
		mouseArgs.dy = mouseArgs.y - prevCursorPos.y;
		mouseArgs.type = mouseEventArgs::mouseEventType::mouseWheel;

		prevCursorPos.x = mouseArgs.x;
		prevCursorPos.y = mouseArgs.y;

		mouse(&mouseArgs);
		mouseWheel(&mouseArgs);
		return 0;
	}

	LPARAM handleMouseMove(WPARAM wp, LPARAM lp)
	{
		mouseArgs.key = 0;
		mouseArgs.delta = 0;
		mouseArgs.x = GET_X_LPARAM(lp);
		mouseArgs.y = GET_Y_LPARAM(lp);
		mouseArgs.dx = mouseArgs.x - prevCursorPos.x;
		mouseArgs.dy = mouseArgs.y - prevCursorPos.y;
		mouseArgs.type = mouseEventArgs::mouseEventType::mouseMove;
		
		prevCursorPos.x = mouseArgs.x;
		prevCursorPos.y = mouseArgs.y;

		mouse(&mouseArgs);
		mouseMove(&mouseArgs);
		return 0;
	}
};