#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fntParser.cuh"
#include "cuSurface.cuh"

class sdfGlyphRenderInfo
{
	fontGlyph glyph;
	float x, y, width, height;

	cuPixel color;
	float smoothing;

public:
	sdfGlyphRenderInfo(fontGlyph glyph, float x, float y, float width, float height, cuPixel color, float smoothing) : color(color)
	{
		this->glyph = glyph;
		this->x = x;
		this->y = y;
		this->width = width;
		this->height = height;
		this->smoothing = smoothing;
	}
};

class sdfTextBuffer
{
	std::vector<sdfGlyphRenderInfo> hostGlyphs;
	sdfGlyphRenderInfo* deviceGlyphs;

public:
	~sdfTextBuffer()
	{
		cudaFree(deviceGlyphs);
	}

	void addGlyph(fontGlyph glyph, float x, float y, float width, float height, cuPixel color, float smoothing)
	{
		hostGlyphs.push_back({ glyph, x, y, width, height, color, smoothing });
	}

	bool upload()
	{
		return cudaMalloc(&deviceGlyphs, hostGlyphs.size() * sizeof(sdfGlyphRenderInfo)) == 0
		&& cudaMemcpy(deviceGlyphs, hostGlyphs.data(), hostGlyphs.size() * sizeof(sdfGlyphRenderInfo), cudaMemcpyDefault) == 0;
	}
};