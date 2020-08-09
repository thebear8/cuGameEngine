#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct fontGlyph
{
	wchar_t id;
	int x, y, width, height;
	int xOffset, yOffset;
	int xAdvance;

	fontGlyph()
	{
		this->id = 0;
		this->x = 0;
		this->y = 0;
		this->width = 0;
		this->height = 0;
		this->xOffset = 0;
		this->yOffset = 0;
		this->xAdvance = 0;
	}

	fontGlyph(const fontGlyph& glyph)
	{
		this->id = glyph.id;
		this->x = glyph.x;
		this->y = glyph.y;
		this->width = glyph.width;
		this->height = glyph.height;
		this->xOffset = glyph.xOffset;
		this->yOffset = glyph.yOffset;
		this->xAdvance = glyph.xAdvance;
	}

	fontGlyph(int id, int x, int y, int width, int height, int xOffset, int yOffset, int xAdvance)
	{
		this->id = id;
		this->x = x;
		this->y = y;
		this->width = width;
		this->height = height;
		this->xOffset = xOffset;
		this->yOffset = yOffset;
		this->xAdvance = xAdvance;
	}
};