#pragma once

#pragma pack(push, 1)

struct cuPixel
{
	unsigned char b, g, r, a;

	cuPixel(unsigned char a, unsigned char r, unsigned char g, unsigned char b)
	{
		this->a = a;
		this->r = r;
		this->g = g;
		this->b = b;
	}
};

#pragma pack(pop, 1)