#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fntParser.cuh"
#include "cuSurface.cuh"

class sdfGlyphRenderInfo
{
public:
	fontGlyph glyph;
	float x, y, width, height;

	cuPixel color;
	float smoothing;

	__device__ __host__ __forceinline__ sdfGlyphRenderInfo(fontGlyph glyph, float x, float y, float width, float height, cuPixel color, float smoothing) : color(color)
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
	std::vector<sdfGlyphRenderInfo> clippedGlyphs;

	sdfGlyphRenderInfo* deviceGlyphs;
	size_t deviceGlyphsSize;

public:
	~sdfTextBuffer()
	{
		cudaFree(deviceGlyphs);
	}

	void addGlyph(fontGlyph glyph, float x, float y, float width, float height, cuPixel color, float smoothing)
	{
		hostGlyphs.push_back({ glyph, x, y, width, height, color, smoothing });
	}

	void clear()
	{
		hostGlyphs.clear();
	}

	void clipForSurface(cuSurface* surface)
	{
		clippedGlyphs.clear();
		for (auto& g : hostGlyphs)
		{
			if (g.x + g.width >= 0 && g.y + g.height >= 0 && g.x < surface->width && g.y < surface->height)
			{
				clippedGlyphs.push_back(g);
			}
		}
	}

	bool uploadClippedGlyphs()
	{
		if (clippedGlyphs.size() != deviceGlyphsSize || deviceGlyphs == 0)
		{
			cudaFree(deviceGlyphs);
			if (cudaMalloc(&deviceGlyphs, clippedGlyphs.size() * sizeof(sdfGlyphRenderInfo)) != 0)
			{
				return false;
			}
			else
			{
				deviceGlyphsSize = clippedGlyphs.size();
			}
		}

		return cudaMemcpy(deviceGlyphs, clippedGlyphs.data(), clippedGlyphs.size() * sizeof(sdfGlyphRenderInfo), cudaMemcpyDefault) == 0;
	}

	size_t getCount()
	{
		return hostGlyphs.size();
	}

	sdfGlyphRenderInfo* getDevicePtr()
	{
		return deviceGlyphs;
	}
};