#pragma once

template<class T>
class swapChain
{
public:
	T front, back;

	swapChain(T front, T back)
	{
		this->front = front;
		this->back = back;
	}

	void swap()
	{
		auto oldFront = front;
		auto oldBack = back;
		front = oldBack;
		back = oldFront;
	}
};