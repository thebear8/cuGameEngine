#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "math_constants.h"

#include "cuSurface.cuh"
#include "cuEffect.cuh"
#include "cuVec.cuh"
#include "cuTriBuffer.cuh"
#include "mathUtils.cuh"

__device__ __forceinline__ float intersect(vec3f& origin, vec3f& direction, tri& tr, vec3f& intersectOrigin, vec3f& intersectDirection)
{
	const float err = 0.0000001f;
	vec3f v0 = tr.p1, v1 = tr.p2, v2 = tr.p3;
	vec3f e1 = v1 - v0, e2 = v2 - v0;
	vec3f h = direction.cross(e2);
	float a = e1.dot(h);
	if (abs(a) < err)
	{
		return CUDART_INF_F;
	}

	float f = 1.0f / a;
	vec3f s = origin - v0;
	float u = f * s.dot(h);
	if (u > 1.0f || u < 0.0f)
	{
		return CUDART_INF_F;
	}

	vec3f q = s.cross(e1);
	float v = f * direction.dot(q);
	if (u + v > 1.0f || v < 0.0f)
	{
		return CUDART_INF_F;
	}

	float t = f * e2.dot(q);
	if (t > 1.0f)
	{
		intersectOrigin = origin + direction * vec3f(t, t, t);
		intersectDirection = (v1 - v0).cross(v2 - v0).norm();
		return origin.dist(intersectOrigin);
	}
	else
	{
		return CUDART_INF_F;
	}
}

__device__ __forceinline__ cuPixel trace(vec3f origin, vec3f direction, int64_t bouncesLeft, tri* tris, int64_t triCount)
{
	float dist = CUDART_INF_F;
	tri currentTri;
	vec3f intersectOrigin, intersectDirection;

	for (int64_t i = 0; i < triCount; i++)
	{
		float tmpDist = intersect(origin, direction, tris[i], intersectOrigin, intersectDirection);
		if (tmpDist < dist)
		{
			dist = tmpDist;
			currentTri = tris[i];
		}
	}

	/*if (bouncesLeft > 0 && dist < CUDART_INF_F)
	{
		return trace(intersectOrigin, intersectDirection, --bouncesLeft, tris, triCount);
	}*/

	return currentTri.color;
}

__global__ void rayTrace(cuPixel* buffer, int64_t width, int64_t height, tri* tris, int64_t triCount, vec3f camPos, float camYaw, float camPitch, float camFovDeg)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < width && y < height)
	{
		/*float nx = map(x, 0, width, -tan(camFovDeg / 2.f) * width / height + camYaw, tan(camFovDeg / 2.f) * width / height + camYaw);
		float ny = map(y, 0, height, tan(camFovDeg / 2.0f) + camPitch, -tan(camFovDeg / 2.0f) + camPitch);
		auto rayDir = (vec3f(nx, ny, 1)).norm();

		if (x == 0 && y == 0)
		{
			printf("PITCH: %f, YAW: %f\n", nx, ny);
		}*/

		/*float nx = map(x, 0, width, -1, 1) * tan(camFovDeg / 2.0f) * width / (float)height;
		float ny = -map(y, 0, height, -1, 1) * tan(camFovDeg / 2.0f);
		auto rayDir = (vec3f(camDir.x + nx, camDir.y + ny, 1 + camDir.z)).norm();*/

		/*float yawRange = camYaw / 2.f;// *(width / (float)height);
		float pitchRange = camPitch / 2.f;

		float yaw = map(x, 0, width, camYaw - yawRange, camYaw + yawRange);
		float pitch = map(y, 0, height, camPitch - pitchRange, camPitch + pitchRange);
		vec3f rayDir = vec3f(degToRad(yaw), degToRad(pitch));*/

		float nx = map(x, 0, width, -1, 1);
		float ny = map(y, 0, height, -1, 1);

		float yaw = map(tan(nx), tan(-1.f), tan(1.f), -(camFovDeg * width / height) / 2 + camYaw, (camFovDeg * width / height) / 2 + camYaw);
		float pitch = map(tan(ny), tan(-1.f), tan(1.f), -camFovDeg / 2 + camPitch, camFovDeg / 2 + camPitch);
		vec3f rayDir = vec3f(degToRad(yaw), degToRad(pitch)).norm();

		buffer[y * width + x] = trace(camPos, rayDir, 0, tris, triCount);
	}
}

class rayTracer : public cuEffect
{
public:
	triBuffer* buffer = new triBuffer();
	vec3f camPos, camDir = vec3f(0, 0, -1);
	//float camFov = degToRad(90), camYaw = 1.f/2.f * 3.1415926535f, camPitch = 3.f/2.f * 3.1415926535f;
	float camFov = 170, camYaw = 0, camPitch = 0;

	void apply(cuSurface* in, cuSurface* out)
	{
		//camDir = vec3f(cos(degToRad(camYaw)) * cos(degToRad(camPitch)), sin(degToRad(camYaw)) * cos(degToRad(camPitch)), sin(degToRad(camPitch))).norm();
		//camDir = vec3f(cos(camYaw) * cos(camPitch), sin(camYaw) * cos(camPitch), sin(camPitch)).norm();
		camDir = vec3f(camYaw, camPitch).norm();

		int64_t width, height;
		dim3 blocks, threads;
		calcGrid(in, out, width, height, blocks, threads);

		buffer->sync();
		rayTrace<<<blocks, threads>>>(out->buffer, width, height, buffer->gpuBuffer, buffer->count, camPos, camYaw, camPitch, camFov);
		cudaDeviceSynchronize();
	}
};
