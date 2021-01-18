#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../cuGameEngine/cuEffect.cuh"
#include "../cuGameEngine/renderWindow.cuh"

#include "math.cuh"
#include "triangleBuffer.cuh"
#include "kernelHelpers.cuh"
#include "triangleRenderer.cuh"
#include "shader.cuh"
#include "cudaNew.cuh"

/*
vec3 projectVertex(vec3 pos, vec3 cameraPos)
{
    return { pos.x / pos.z, pos.y / pos.z, 1.0f };
}

__device__ cuPixel shadeTriangle(float x, float y)
{
	return cuPixel(255, x * 255, 0, y * 255);
}

__global__ void draw(cuPixel* out, int64_t width, int64_t height, vertexTriangle tri)
{
	auto x = cuXIdx();
	auto y = cuYIdx();

	if (x >= 0 && x < width && y >= 0 && y < height)
	{
		if (pointIsInTriangle2d({ x,y,0 }, tri.a, tri.b, tri.c))
		{
			out[y * width + x] = shadeTriangle(x / (float)width, y / (float)height);
		}
	}
}

static __device__ __host__ __inline__ float edgeFunction(const vec3& p, const vec3& i, const vec3& j)
{
	float dX = j.x - i.x;
	float dY = j.y - i.y;
	return (p.x - i.x) * dY - (p.y - i.y) * dX;
}

__global__ void drawUsingEdgeFunction(cuGpuSurface tex, vertexTriangle tri)
{
	auto x = cuXIdx();
	auto y = cuYIdx();

	if (tex.isInBounds(x, y))
	{
		vec3 ab = tri.b - tri.a;
		vec3 bc = tri.c - tri.b;
		vec3 ac = tri.c - tri.a;

		vec3 p(x, y, 0);
		float eAB = edgeFunction(p, tri.a, tri.b);
		float eBC = edgeFunction(p, tri.b, tri.c);
		float eCA = edgeFunction(p, tri.c, tri.a);

		if (eAB >= 0 && eBC >= 0 && eCA >= 0)
		{

			tex2d(&tex, x, y) = shadeTriangle(x / (float)tex.width, y / (float)tex.height);
		}
	}
}
*/

__device__ vec3 vertexShaderFunction(const vertexShader* shader, vec3 pos)
{
	return pos;
}

__device__ vec4 fragmentShaderFunction(const fragmentShader* shader, vec3 pos)
{
	auto s = shader->getParameter<0, float>();
	auto c = shader->getParameter<1, float>();
	return { 1, clampf(pos.x / 2 + 0.5f + s, 0, 1), 0, clampf(pos.y / 2 + 0.5f + c, 0, 1) };
}

class testRenderer
{
private:
	renderWindow wnd;

	vertexShader* vtxShader;
	fragmentShader* fragShader;
	triangleRenderer* renderer;

public:
	testRenderer() : wnd(1024, 768, false, L"Render Test")
	{
		vtxShader = cudaNewManaged<vertexShader>();
		fragShader = cudaNewManaged<fragmentShader>();
		renderer = new triangleRenderer();

		vtxShader->setShader<vertexShaderFunction>();
		fragShader->setShader<fragmentShaderFunction>();
		renderer->vtxShader = vtxShader;
		renderer->fragShader = fragShader;
		renderer->onRenderFrame += createBoundHandler(&testRenderer::onRenderFrame, this);

		triangleRenderInfo tris[] =
		{
			{ { -1, -1, 1 }, { -1, 1, 1 }, { 1, 1, 1 } },
			{ { -1, -1, 1 }, { 1, 1, 1 }, { 1, -1, 1 } },
		};

		for (auto& tri : tris)
		{
			renderer->pushTriangle(tri);
		}

		//renderer->pushTriangle({ {0, -.5, 1}, {-.5, .5, 1}, {.5, .5, 1} });
		//renderer->pushTriangle({ {0, -1, 1}, {-1, 1, 1}, {1, 1, 1} });

		wnd.pipeLine->addEffect(renderer);
		wnd.inputMgr->key += createBoundHandler(&testRenderer::onKey, this);
	}

	void run()
	{
		bool isRunning = true;
		wnd.runLoop(false, true, isRunning);
	}

	void onKey(keyboardEventArgs* e)
	{
		if (e->key == VK_ESCAPE)
		{
			ExitProcess(0);
		}
	}

	void onRenderFrame()
	{
		fragShader->setParameter<0, float>(sinf(clock() / 1000.0f));
		fragShader->setParameter<1, float>(cosf(clock() / 1000.0f));
	}
};

int main()
{
	for (int i = 0; i < 10000; i++)
	{
		auto tex = i / 10000.0f;
		int x = roundf(tex * 10000);

		if (x != i)
		{
			DebugBreak();
		}
	}

	auto shader = cudaNewManaged<vertexShader>();

	auto r = new testRenderer();
	r->run();
}