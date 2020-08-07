#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>

struct vec3f
{
	float x, y, z;

	__device__ __host__ __forceinline__ vec3f(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}
	__device__ __host__ __forceinline__ vec3f(float yaw, float pitch)
	{
		/*this->x = cos(yaw) * cos(pitch);
		this->y = sin(yaw) * cos(pitch);
		this->z = sin(pitch);*/

		this->x = sin(yaw);
		this->y = -(sin(pitch) * cos(yaw));
		this->z = -(cos(pitch) * cos(yaw));
	}
	__device__ __host__ __forceinline__ vec3f() : vec3f(0, 0, 0)
	{

	}

	__device__ __host__ __forceinline__ vec3f operator +(vec3f& other)
	{
		return { x + other.x, y + other.y, z + other.z };
	}
	__device__ __host__ __forceinline__ vec3f operator -(vec3f& other)
	{
		return { x - other.x, y - other.y, z - other.z };
	}
	__device__ __host__ __forceinline__ vec3f operator *(vec3f& other)
	{
		return { x * other.x, y * other.y, z * other.z };
	}
	__device__ __host__ __forceinline__ vec3f operator /(vec3f& other)
	{
		return { x / other.x, y / other.y, z / other.z };
	}

	__device__ __host__ __forceinline__ vec3f& operator +=(vec3f& other)
	{
		this->x += other.x;
		this->y += other.y;
		this->z += other.z;
		return *this;
	}
	__device__ __host__ __forceinline__ vec3f& operator -=(vec3f& other)
	{
		this->x -= other.x;
		this->y -= other.y;
		this->z -= other.z;
		return *this;
	}
	__device__ __host__ __forceinline__ vec3f& operator *=(vec3f& other)
	{
		this->x *= other.x;
		this->y *= other.y;
		this->z *= other.z;
		return *this;
	}
	__device__ __host__ __forceinline__ vec3f& operator /=(vec3f& other)
	{
		this->x /= other.x;
		this->y /= other.y;
		this->z /= other.z;
		return *this;
	}

	__device__ __host__ __forceinline__ float dot(vec3f& other)
	{
		return (x * other.x) + (y * other.y) + (z * other.z );
	}
	__device__ __host__ __forceinline__ vec3f cross(vec3f& other)
	{
		return { (y * other.z) - (z * other.y), (z * other.x) - (x * other.z), (x * other.y) - (y * other.x)};
	}
	__device__ __host__ __forceinline__ float dist(vec3f& other)
	{
		return sqrtf(powf(x - other.x, 2) + powf(y - other.y, 2) + powf(z - other.z, 2));
	}
	__device__ __host__ __forceinline__ vec3f norm()
	{
		float len = sqrtf(powf(x, 2) + powf(y, 2) + powf(z, 2));
		return { x / len, y / len, z / len };
	}
};