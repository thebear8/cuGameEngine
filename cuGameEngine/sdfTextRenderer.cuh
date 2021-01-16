#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <unordered_map>

#include "fntParser.cuh"
#include "cuSurface.cuh"
#include "sdfTextBuffer.cuh"
#include "cuStreamManager.cuh"

/*float glyphX = glyphs[i].x;
			float glyphY = glyphs[i].y;
			float glyphWidth = glyphs[i].width;
			float glyphHeight = glyphs[i].height;

			if (x >= glyphX && x < glyphX + glyphWidth && y >= glyphY && y < glyphY + glyphHeight)
			{
				float glyphAtlasX = glyphs[i].glyph.x;
				float glyphAtlasY = glyphs[i].glyph.y;
				float glyphAtlasWidth = glyphs[i].glyph.width;
				float glyphAtlasHeight = glyphs[i].glyph.height;
				cuPixel glyphColor = glyphs[i].color;
				float smoothing = glyphs[i].smoothing;

				float atlasX = clampf(mapf(x - glyphX, 0, glyphWidth, glyphAtlasX, glyphAtlasX + glyphAtlasWidth), 0, atlasWidth);
				float atlasY = clampf(mapf(y - glyphY, 0, glyphHeight, glyphAtlasY, glyphAtlasY + glyphAtlasHeight), 0, atlasHeight);
				cuPixel glyphPx = interpolatePixel(atlas, atlasWidth, atlasHeight, atlasX, atlasY);
				float alpha = smoothStepf(byteToFloat(glyphPx.a), 0.5 - smoothing, 0.5 + smoothing);

				px = blendPixel(glyphColor, px, alpha);
			}*/

__device__ __forceinline__ void sdfTextRendererRenderGlyphInternal(cuPixel* surface, float surfaceWidth, float surfaceHeight, float x, float y, sdfGlyphRenderInfo* glyph, cuPixel* atlas, float atlasWidth, float atlasHeight)
{
	if (x >= glyph->x && x < glyph->x + glyph->width && y >= glyph->y && y < glyph->y + glyph->height)
	{
		float atlasX = ((x - glyph->x) / glyph->width) * glyph->glyph.width + glyph->glyph.x;
		float atlasY = ((y - glyph->y) / glyph->height) * glyph->glyph.height + glyph->glyph.y;

		float xRatio = ceilf(atlasX) - atlasX;
		float yRatio = ceilf(atlasY) - atlasY;

		float a00 = atlas[(int)(floorf(atlasY) * atlasWidth + floorf(atlasX))].a;
		float a01 = atlas[(int)(floorf(atlasY) * atlasWidth + ceilf(atlasX))].a;
		float a10 = atlas[(int)(ceilf(atlasY) * atlasWidth + floorf(atlasX))].a;
		float a11 = atlas[(int)(ceilf(atlasY) * atlasWidth + ceilf(atlasX))].a;

		float glyphAlpha = ((((a00 * xRatio) + (a01 * (1.0f - xRatio))) * yRatio) + (((a10 * xRatio) + (a11 * (1.0f - xRatio))) * (1 - yRatio))) / 255.0f;
		float normGlyphAlpha = (glyphAlpha - (0.5 - glyph->smoothing)) / (2 * glyph->smoothing);
		float alpha = (normGlyphAlpha > 1.0f) + ((normGlyphAlpha > 0.0f && normGlyphAlpha < 1.0f) * normGlyphAlpha);

		if (alpha * 256 > 1.0f)
		{
			auto& surfacePx = surface[(int)(y * surfaceWidth + x)];
			surfacePx.a = (glyph->color.a * alpha) + (surfacePx.a * (1 - alpha));
			surfacePx.r = (glyph->color.r * alpha) + (surfacePx.r * (1 - alpha));
			surfacePx.g = (glyph->color.g * alpha) + (surfacePx.g * (1 - alpha));
			surfacePx.b = (glyph->color.b * alpha) + (surfacePx.b * (1 - alpha));
		}
	}
}

__global__ void sdfTextRendererRenderTextBuffer(cuPixel* surface, int surfaceWidth, int surfaceHeight, int xOffset, int yOffset, sdfGlyphRenderInfo* glyphs, int startIdx, int stopIdx, cuPixel* atlas, int atlasWidth, int atlasHeight)
{
	auto x = xOffset + blockDim.x * blockIdx.x + threadIdx.x;
	auto y = yOffset + blockDim.y * blockIdx.y + threadIdx.y;

	if (x < surfaceWidth && y < surfaceHeight)
	{
		for (int i = startIdx; i <= stopIdx; i++)
		{
			sdfTextRendererRenderGlyphInternal(surface, surfaceWidth, surfaceHeight, x, y, &glyphs[i], atlas, atlasWidth, atlasHeight);
		}
	}
}

__global__ void sdfTextRendererRenderGlyph(cuPixel* surface, int64_t surfaceWidth, int64_t surfaceHeight, float xPos, float yPos, float width, float height, fontGlyph glyph, cuPixel* atlas, int64_t atlasWidth, int64_t atlasHeight, cuPixel textColor, float smoothing)
{
	/*auto xIdx = blockDim.x * blockIdx.x + threadIdx.x;
	auto yIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t x = xPos + xIdx;
	int64_t y = yPos + yIdx;

	if (xIdx < width && yIdx < height && x >= 0 && x < surfaceWidth && y >= 0 && y < surfaceHeight)
	{
		float atlasX = mapf(xIdx, 0, width, glyph.x, glyph.x + glyph.width);
		float atlasY = mapf(yIdx, 0, height, glyph.y, glyph.y + glyph.height);
		auto glyphPx = interpolatePixel(atlas, atlasWidth, atlasHeight, atlasX, atlasY);
		auto alpha = smoothStepf(byteToFloat(glyphPx.a), 0.5 - smoothing, 0.5 + smoothing);
		surface[y * surfaceWidth + x] = blendPixel(textColor, surface[y * surfaceWidth + x], alpha);
	}*/

	auto xIdx = blockDim.x * blockIdx.x + threadIdx.x;
	auto yIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int64_t x = xPos + xIdx;
	int64_t y = yPos + yIdx;

	if (xIdx < width && yIdx < height && x >= 0 && x < surfaceWidth && y >= 0 && y < surfaceHeight)
	{
		sdfGlyphRenderInfo info = sdfGlyphRenderInfo(glyph, xPos, yPos, width, height, textColor, smoothing);
		sdfTextRendererRenderGlyphInternal(surface, surfaceWidth, surfaceHeight, x, y, &info, atlas, atlasWidth, atlasHeight);
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
		volatile auto s = cuSurface::loadFromFile(atlasFile, &atlas);
		for (auto& glyph : parser.glyphs)
		{
			glyphs.try_emplace(glyph.id, glyph);
		}
	}

	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void renderTextBuffer(cuSurface* surface, sdfTextBuffer* buffer, bool useMultipleStreams = false)
	{
		dim3 threads = dim3(20, 20, 1);
		dim3 blocks = dim3(ceil(surface->width / 20.0f), ceil(surface->height / 20.0f), 1);

		if (useMultipleStreams)
		{
			auto streamIdx = 0;
			auto stride = buffer->getCount() / cuStreamManager::getNumberOfStreams();
			for (int i = 0; i < buffer->getCount(); i += stride)
			{
				sdfTextRendererRenderTextBuffer << <blocks, threads, 0, cuStreamManager::getStream(streamIdx) >> > (surface->buffer, surface->width, surface->height, 0, 0, buffer->getDevicePtr(), i, clamp(i + stride, 0, buffer->getCount() - 1), atlas->buffer, atlas->width, atlas->height);
				streamIdx = (streamIdx + 1) % cuStreamManager::getNumberOfStreams();
			}
		}
		else
		{
			sdfTextRendererRenderTextBuffer<<<blocks, threads>>>(surface->buffer, surface->width, surface->height, 0, 0, buffer->getDevicePtr(), 0, buffer->getCount() - 1, atlas->buffer, atlas->width, atlas->height);
		}
	}

	void addGlyphToBufferInternal(sdfTextBuffer* buffer, fontGlyph glyph, float x, float y, float width, float height, cuPixel color, float smoothing)
	{
		buffer->addGlyph(glyph, x, y, width, height, color, smoothing);
	}

	void addGlyphToBuffer(sdfTextBuffer* buffer, fontGlyph glyph, float x, float y, float scale, cuPixel color)
	{
		auto width = scale * glyph.width;
		auto height = scale * glyph.height;
		auto xPos = scale * glyph.xOffset + x;
		auto yPos = scale * glyph.yOffset + y;
		auto smoothing = (1.0f / 16.0f) / scale;
		addGlyphToBufferInternal(buffer, glyph, xPos, yPos, width, height, color, smoothing);
	}

	void addStringToBuffer(sdfTextBuffer* buffer, std::wstring str, float xOffset, float yOffset, float lineWidth, float scale, cuPixel color, bool wrapText)
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
				if (wrapText && x >= (xOffset + lineWidth) - glyph.width * scale)
				{
					x = xOffset;
					y += parser.lineHeight * scale;
				}

				addGlyphToBuffer(buffer, glyph, x, y, scale, color);
				x += glyph.xAdvance * scale;
			}
		}
	}

	void addAllGlyphsToBuffer(sdfTextBuffer* buffer, float lineWidth, float fontSize)
	{
		float x = 0, y = 0;
		for (auto& glyphPair : glyphs)
		{
			auto& glyph = std::get<1>(glyphPair);
			if (glyph.width > 0 && glyph.height > 0)
			{
				addGlyphToBuffer(buffer, glyph, x, y, 1, cuPixel(255, 255, 0, 0));

				x += glyph.xAdvance;
				if (x >= lineWidth - glyph.xAdvance)
				{
					x = 0;
					y += parser.lineHeight;
				}
			}
		}
	}

	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void renderGlyphInternal(cudaStream_t stream, cuSurface* surface, fontGlyph glyph, float x, float y, float width, float height, cuPixel color, float smoothing)
	{
		if (x + width >= 0 && y + height >= 0 && x < surface->width && y < surface->height)
		{
			dim3 threads = dim3(20, 20, 1);
			dim3 blocks = dim3(ceil(width / 20.0f), ceil(height / 20.0f), 1);
			sdfTextRendererRenderGlyph << <blocks, threads, 0, stream >> > (surface->buffer, surface->width, surface->height, x, y, width, height, glyph, atlas->buffer, atlas->width, atlas->height, color, smoothing);
		}
	}

	void renderGlyph(cudaStream_t stream, cuSurface* surface, fontGlyph glyph, float x, float y, float scale, cuPixel color)
	{
		auto width = scale * glyph.width;
		auto height = scale * glyph.height;
		auto xPos = scale * glyph.xOffset + x;
		auto yPos = scale * glyph.yOffset + y;
		auto smoothing = (1.0f / 16.0f) / scale;
		renderGlyphInternal(stream, surface, glyph, xPos, yPos, width, height, color, smoothing);
	}

	void renderString(cuSurface* surface, std::wstring str, float xOffset, float yOffset, float width, float scale, cuPixel color, bool wrapText, bool useMultipleStreams = false)
	{
		int streamIdx = 0;
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

				if (useMultipleStreams)
				{
					renderGlyph(cuStreamManager::getStream(streamIdx), surface, glyph, x, y, scale, color);
					streamIdx = (streamIdx + 1) % cuStreamManager::getNumberOfStreams();
				}
				else
				{
					renderGlyph(0, surface, glyph, x, y, scale, color);
				}
				x += glyph.xAdvance * scale;
			}
		}
	}

	void renderAllGlyphs(cuSurface* surface, float fontSize, bool useMultipleStreams = false)
	{
		int streamIdx = 0;
		float x = 0, y = 0;
		for (auto& glyphPair : glyphs)
		{
			auto& glyph = std::get<1>(glyphPair);
			if (glyph.width > 0 && glyph.height > 0)
			{
				if (useMultipleStreams)
				{
					renderGlyph(cuStreamManager::getStream(streamIdx), surface, glyph, x, y, 1, cuPixel(255, 255, 0, 0));
					streamIdx = (streamIdx + 1) % cuStreamManager::getNumberOfStreams();
				}
				else
				{
					renderGlyph(0, surface, glyph, x, y, 1, cuPixel(255, 255, 0, 0));
				}

				x += glyph.xAdvance;
				if (x >= surface->width - glyph.xAdvance)
				{
					x = 0;
					y += parser.lineHeight;
				}
			}
		}
	}

	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
};