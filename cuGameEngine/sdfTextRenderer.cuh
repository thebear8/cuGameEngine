#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <unordered_map>

#include "fntParser.cuh"
#include "cuSurface.cuh"

__global__ void sdfTextRendererRenderGlyph(cuPixel* surface, int64_t surfaceWidth, int64_t surfaceHeight,  float xPos, float yPos, float width, float height, fontGlyph glyph, cuPixel* atlas, int64_t atlasWidth, int64_t atlasHeight, cuPixel textColor, float smoothing)
{
	auto xIdx = blockDim.x * blockIdx.x + threadIdx.x;
	auto yIdx =  blockDim.y * blockIdx.y + threadIdx.y;
	int64_t x = xPos + xIdx;
	int64_t y = yPos + yIdx;

	if (xIdx < width && yIdx < height && x >= 0 && x < surfaceWidth && y >= 0 && y < surfaceHeight)
	{
		float atlasX = map(xIdx, 0, width, glyph.x, glyph.x + glyph.width);
		float atlasY = map(yIdx, 0, height, glyph.y, glyph.y + glyph.height);
		auto glyphPx = interpolatePixel(atlas, atlasWidth, atlasHeight, atlasX, atlasY);
		auto alpha = smoothStepf(byteToFloat(glyphPx.a), 0.5 - smoothing, 0.5 + smoothing);
		surface[y * surfaceWidth + x] = blendPixel(textColor, surface[y * surfaceWidth + x], alpha);
	}
}

class sdfTextRenderer
{
private:
	cuSurface* atlas;
	fntParser parser;
	std::unordered_map<wchar_t, fontGlyph> glyphs;

public:
	sdfTextRenderer(wchar_t const* fntFile, wchar_t const* atlasFile) : parser(fntFile)
	{
		cuSurface::loadFromFile(atlasFile, &atlas);
		for (auto& glyph : parser.glyphs)
		{
			glyphs.try_emplace(glyph.id, glyph);
		}
	}

	void renderGlyphInternal(cuSurface* surface, fontGlyph glyph, float x, float y, float width, float height, cuPixel color, float smoothing)
	{
		dim3 threads = dim3(20, 20, 1);
		dim3 blocks = dim3(ceil(width / 20.0f), ceil(height / 20.0f), 1);
		sdfTextRendererRenderGlyph<<<blocks, threads>>>(surface->buffer, surface->width, surface->height, x, y, width, height, glyph, atlas->buffer, atlas->width, atlas->height, color, smoothing);
	}

	void renderGlyph(cuSurface* surface, fontGlyph glyph, float x, float y, float scale, cuPixel color)
	{
		auto width = scale * glyph.width;
		auto height = scale * glyph.height;
		auto xPos = scale * glyph.xOffset + x;
		auto yPos = scale * glyph.yOffset + y;
		auto smoothing = (1.0f / 16.0f) / scale;
		renderGlyphInternal(surface, glyph, xPos, yPos, width, height,  color, smoothing);
	}

	void renderString(cuSurface* surface, std::wstring str, float xOffset, float yOffset, float width, float scale, cuPixel color, bool wrapText)
	{
		float x = xOffset, y = yOffset;
		for (auto c : str)
		{
			if (c == L'\n')
			{
				x = xOffset;
				y += parser.lineHeight * scale;
			}
			else if (c == L'\t')
			{
				auto space = glyphs[L' '];
				auto indent = 4 - ((int64_t)((x - xOffset) / (space.xAdvance * scale)) % 4);
				x += indent * space.xAdvance * scale;
			}
			else
			{
				auto glyph = glyphs[c];
				if (wrapText && x >= (xOffset + width) - glyph.width * scale)
				{
					x = xOffset;
					y += parser.lineHeight * scale;
				}

				renderGlyph(surface, glyph, x, y, scale, color);
				x += glyph.xAdvance * scale;
			}
		}
	}

	void renderAllGlyphs(cuSurface* surface, float fontSize)
	{
		float x = 0, y = 0;
		for (auto& glyphPair : glyphs)
		{
			auto& glyph = std::get<1>(glyphPair);
			if (glyph.width > 0 && glyph.height > 0)
			{
				renderGlyph(surface, glyph, x, y, 1, cuPixel(255, 255, 0, 0));

				x += glyph.xAdvance;
				if (x >= surface->width - glyph.xAdvance)
				{
					x = 0;
					y += parser.lineHeight;
				}
			}
		}

		cudaDeviceSynchronize();
	}
};