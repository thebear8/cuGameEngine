#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <Windows.h>
#include <list>

#include "event.cuh"

class keyboardEventArgs
{
public:
	int key;
	char c;
	enum class keyboardEventType
	{
		keyDown,
		keyUp,
	} type;
};

class mouseEventArgs
{
public:
	int key;
	int x, y, dx, dy;
	int delta;
	enum class mouseEventType
	{
		mouseDown,
		mouseUp,
		mouseMove,
		mouseWheel
	} type;
};

class mouseEvent : public eventListener<void, mouseEventArgs*>
{

};

class keyboardEvent : public eventListener<void, keyboardEventArgs*>
{

};