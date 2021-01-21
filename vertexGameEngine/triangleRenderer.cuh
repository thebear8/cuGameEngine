#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../cuGameEngine/cuEffect.cuh"

#include "triangleBuffer.cuh"
#include "kernelHelpers.cuh"
#include "math.cuh"
#include "shader.cuh"

struct triangleRenderInfo
{
	vec3 a, b, c;
	vec3 texA, texB, texC;

	triangleRenderInfo(vec3 a, vec3 b, vec3 c) :
		a(a), b(b), c(c), texA(0, 0, 0), texB(0, 0, 0), texC(0, 0, 0)
	{

	}

	triangleRenderInfo(vec3 a, vec3 b, vec3 c, vec3 texA, vec3 texB, vec3 texC) :
		a(a), b(b), c(c), texA(texA), texB(texB), texC(texC)
	{

	}
};

static __device__ __host__ __inline__ float edgeFunction(const vec3& p, const vec3& i, const vec3& j)
{
	float dX = j.x - i.x;
	float dY = j.y - i.y;
	return (p.x - i.x) * dY - (p.y - i.y) * dX;
}

static __device__ __host__ __inline__ vec3 projectToCameraSpace()
{

}

static __device__ __host__ __inline__ vec3 projectToScreenSpace(const vec3& pos, float nearPlane)
{
	return { (nearPlane * pos.x) / (-pos.z), (nearPlane * pos.y) / (-pos.z), -pos.z };
}

__global__ void runVertexShader(const vertexShader* shader, const triangleRenderInfo* input, int inputCount, triangleRenderInfo* output, int outputSize)
{
	auto idx = cuYIdx() * cuYSize() + cuXIdx();
	if (idx < inputCount && idx < outputSize)
	{
		auto a = shader->callShader(shader, input[idx].a);
		auto b = shader->callShader(shader, input[idx].b);
		auto c = shader->callShader(shader, input[idx].c);
		output[idx].a = a;
		output[idx].b = b;
		output[idx].c = c;
	}
}

__device__ void renderTriangle(const triangleRenderInfo& triangle, const fragmentShader* shader, cuGpuSurface& output, vec3 point, int xI, int yI)
{
	float eAB = edgeFunction(point, triangle.a, triangle.b);
	float eBC = edgeFunction(point, triangle.b, triangle.c);
	float eCA = edgeFunction(point, triangle.c, triangle.a);

	if (eAB >= 0.0f && eBC >= 0.0f && eCA >= 0.0f)
	{
		auto colorVec = shader->callShader(shader, point);
		//output.buffer[yI * output.width + xI] = cuPixel(colorVec.x * 255, colorVec.y * 255, colorVec.z * 255, colorVec.w * 255);
		tex2d(&output, point.x, point.y) = cuPixel(colorVec.x * 255, colorVec.y * 255, colorVec.z * 255, colorVec.w * 255);
	}
}

__global__ void renderTriangleBuffer(const fragmentShader* shader, const triangleRenderInfo* input, int inputCount, cuGpuSurface output)
{
	auto x = cuXIdx(), y = cuYIdx();
	if (x < output.width && y < output.height)
	{
		//vec3 screenPos(((float)x / ((float)output.width)) * 2.0f - 1.0f, ((float)y / ((float)output.height)) * 2.0f - 1.0f, 0);
		vec3 screenPos(mapf(x, 0, output.width - 1, -1, 1), mapf(y, 0, output.height - 1, -1, 1), 0);
		for (int i = 0; i < inputCount; i++)
		{
			renderTriangle(input[i], shader, output, screenPos, x, y);
		}
	}
}

class triangleRenderer : public cuEffect
{
private:
	triangleBuffer triBuffer;
	triangleBuffer workingBuffer;

	std::vector<triangleRenderInfo> triangles;
	triangleRenderInfo* gpuTriangles;
	int64_t gpuTriangleCount;

public:
	vertexShader* vtxShader = nullptr;
	fragmentShader* fragShader = nullptr;
	eventListener<void> onRenderFrame;

public:
	void pushTriangle(const triangleRenderInfo& tri)
	{
		triangles.push_back(tri);
	}

	void clearTriangles()
	{
		triangles.clear();
	}

	void uploadTriangles()
	{
		cudaFree(gpuTriangles);
		gpuTriangleCount = triangles.size();
		cudaMalloc(&gpuTriangles, triangles.size() * sizeof(triangleRenderInfo));
		cudaMemcpy(gpuTriangles, triangles.data(), triangles.size() * sizeof(triangleRenderInfo), cudaMemcpyDefault);
	}

	virtual void apply(cuSurface* in, cuSurface* out) override
	{
		onRenderFrame();

		int64_t width, height;
		dim3 blocks, threads;

		uploadTriangles();
		calcGrid(ceilf(sqrtf(gpuTriangleCount)), ceilf(sqrtf(gpuTriangleCount)), blocks, threads);
		runVertexShader<<<blocks, threads>>>(vtxShader, gpuTriangles, gpuTriangleCount, gpuTriangles, gpuTriangleCount);

		calcGrid(in, out, width, height, blocks, threads);
		renderTriangleBuffer<<<blocks, threads>>>(fragShader, gpuTriangles, gpuTriangleCount, out->getGpuSurfaceInfo());
		auto e = cudaDeviceSynchronize();
	}
};

/*static __device__ inline cuPixel shadeTriangle(float x, float y)
{
	return cuPixel(255, x * 255, 0, y * 255);
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
}*/