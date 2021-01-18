#pragma once
#include <math.h>

class vec2
{
public:
	float x, y;

	__device__ __host__ __inline__ vec2(float x, float y)
	{
		this->x = x;
		this->y = y;
	}

	__device__ __host__ __inline__ vec2 add(float b) const { return { x + b, y + b}; };
	__device__ __host__ __inline__ vec2 sub(float b) const { return { x - b, y - b}; };
	__device__ __host__ __inline__ vec2 mul(float b) const { return { x * b, y * b}; };
	__device__ __host__ __inline__ vec2 div(float b) const { return { x / b, y / b}; };

	__device__ __host__ __inline__ vec2 add(const vec2& b) const { return { x + b.x, y + b.y }; };
	__device__ __host__ __inline__ vec2 sub(const vec2& b) const { return { x - b.x, y - b.y }; };
	__device__ __host__ __inline__ vec2 mul(const vec2& b) const { return { x * b.x, y * b.y }; };
	__device__ __host__ __inline__ vec2 div(const vec2& b) const { return { x / b.x, y / b.y }; };

	__device__ __host__ __inline__ vec2 norm() const
	{
		return mul(1.0f / sqrtf(x * x + y * y));
	}

	__device__ __host__ __inline__ float dot(const vec2& b)
	{
		return x * b.x + y * b.y;
	}
};

class vec3
{
public:
	float x, y, z;

	__device__ __host__ __inline__ vec3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__device__ __host__ __inline__ vec3 add(float b) const { return { x + b, y + b, z + b }; };
	__device__ __host__ __inline__ vec3 sub(float b) const { return { x - b, y - b, z - b }; };
	__device__ __host__ __inline__ vec3 mul(float b) const { return { x * b, y * b, z * b }; };
	__device__ __host__ __inline__ vec3 div(float b) const { return { x / b, y / b, z / b }; };

	__device__ __host__ __inline__ vec3 add(const vec3& b) const { return { x + b.x, y + b.y, z + b.z}; };
	__device__ __host__ __inline__ vec3 sub(const vec3& b) const { return { x - b.x, y - b.y, z - b.z }; };
	__device__ __host__ __inline__ vec3 mul(const vec3& b) const { return { x * b.x, y * b.y, z * b.z }; };
	__device__ __host__ __inline__ vec3 div(const vec3& b) const { return { x / b.x, y / b.y, z / b.z }; };

	__device__ __host__ __inline__ vec3 norm() const
	{
		return mul(1.0f / sqrtf(x * x + y * y + z * z));
	}

	__device__ __host__ __inline__ float dot(const vec3& b)
	{
		return x * b.x + y * b.y + z * b.z;
	}

	__device__ __host__ __inline__ vec3 cross(const vec3& b)
	{
		return { y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x };
	}
};

class vec4
{
public:
	float x, y, z, w;

	__device__ __host__ __inline__ vec4(float x, float y, float z, float w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	__device__ __host__ __inline__ vec4 add(float b) const { return { x + b, y + b, z + b, w + b }; };
	__device__ __host__ __inline__ vec4 sub(float b) const { return { x - b, y - b, z - b, w - b }; };
	__device__ __host__ __inline__ vec4 mul(float b) const { return { x * b, y * b, z * b, w * b }; };
	__device__ __host__ __inline__ vec4 div(float b) const { return { x / b, y / b, z / b, w / b }; };

	__device__ __host__ __inline__ vec4 add(const vec4& b) const { return { x + b.x, y + b.y, z + b.z, w + b.w }; };
	__device__ __host__ __inline__ vec4 sub(const vec4& b) const { return { x - b.x, y - b.y, z - b.z, w - b.w }; };
	__device__ __host__ __inline__ vec4 mul(const vec4& b) const { return { x * b.x, y * b.y, z * b.z, w * b.w }; };
	__device__ __host__ __inline__ vec4 div(const vec4& b) const { return { x / b.x, y / b.y, z / b.z, w / b.w }; };

	__device__ __host__ __inline__ vec4 norm() const
	{
		return mul(1.0f / sqrtf(x * x + y * y + z * z));
	}

	__device__ __host__ __inline__ float dot(const vec4& b)
	{
		return x * b.x + y * b.y + z * b.z + w * b.w;
	}
};

static __device__ __host__ __inline__ vec2 operator+(const vec2& a, float b) { return a.add(b); }
static __device__ __host__ __inline__ vec2 operator-(const vec2& a, float b) { return a.sub(b); }
static __device__ __host__ __inline__ vec2 operator*(const vec2& a, float b) { return a.mul(b); }
static __device__ __host__ __inline__ vec2 operator/(const vec2& a, float b) { return a.div(b); }
static __device__ __host__ __inline__ vec2 operator+(const vec2& a, const vec2& b) { return a.add(b); }
static __device__ __host__ __inline__ vec2 operator-(const vec2& a, const vec2& b) { return a.sub(b); }
static __device__ __host__ __inline__ vec2 operator*(const vec2& a, const vec2& b) { return a.mul(b); }
static __device__ __host__ __inline__ vec2 operator/(const vec2& a, const vec2& b) { return a.div(b); }

static __device__ __host__ __inline__ vec3 operator+(const vec3& a, float b) { return a.add(b); }
static __device__ __host__ __inline__ vec3 operator-(const vec3& a, float b) { return a.sub(b); }
static __device__ __host__ __inline__ vec3 operator*(const vec3& a, float b) { return a.mul(b); }
static __device__ __host__ __inline__ vec3 operator/(const vec3& a, float b) { return a.div(b); }
static __device__ __host__ __inline__ vec3 operator+(const vec3& a, const vec3& b) { return a.add(b); }
static __device__ __host__ __inline__ vec3 operator-(const vec3& a, const vec3& b) { return a.sub(b); }
static __device__ __host__ __inline__ vec3 operator*(const vec3& a, const vec3& b) { return a.mul(b); }
static __device__ __host__ __inline__ vec3 operator/(const vec3& a, const vec3& b) { return a.div(b); }

static __device__ __host__ __inline__ vec4 operator+(const vec4& a, float b) { return a.add(b); }
static __device__ __host__ __inline__ vec4 operator-(const vec4& a, float b) { return a.sub(b); }
static __device__ __host__ __inline__ vec4 operator*(const vec4& a, float b) { return a.mul(b); }
static __device__ __host__ __inline__ vec4 operator/(const vec4& a, float b) { return a.div(b); }
static __device__ __host__ __inline__ vec4 operator+(const vec4& a, const vec4& b) { return a.add(b); }
static __device__ __host__ __inline__ vec4 operator-(const vec4& a, const vec4& b) { return a.sub(b); }
static __device__ __host__ __inline__ vec4 operator*(const vec4& a, const vec4& b) { return a.mul(b); }
static __device__ __host__ __inline__ vec4 operator/(const vec4& a, const vec4& b) { return a.div(b); }

static __device__ __host__ __inline__ bool pointIsInTriangle2d(vec3 _p, vec3 _a, vec3 _b, vec3 _c)
{
	// from https://blackpawn.com/texts/pointinpoly/

	vec3 p(_p.x, _p.y, 0), a(_a.x, _a.y, 0), b(_b.x, _b.y, 0), c(_c.x, _c.y, 0);

	auto v0 = c.sub(a);
	auto v1 = b.sub(a);
	auto v2 = p.sub(a);

	auto dot00 = v0.dot(v0);
	auto dot01 = v0.dot(v1);
	auto dot02 = v0.dot(v2);
	auto dot11 = v1.dot(v1);
	auto dot12 = v1.dot(v2);

	auto invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
	auto u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	auto v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	return (u >= 0) && (v >= 0) && (u + v < 1);
}

static __device__ __inline__ vec3 absToNorm3d(int x, int y, int z, int maxX, int maxY, int maxZ)
{
	return { (x / ((float)maxX)) * 2 - 1, (y / ((float)maxY)) * 2 - 1, (z / ((float)maxZ)) * 2 - 1 };
}

static __device__ __inline__ void normToAbs3d(vec3 norm, int& x, int& y, int& z, int maxX, int maxY, int maxZ)
{
	x = maxX * (norm.x * 0.5 + 0.5);
	y = maxX * (norm.y * 0.5 + 0.5);
	z = maxX * (norm.z * 0.5 + 0.5);
}

static __device__ __host__ __inline__ vec3 idxToNorm(int x, int y, int z, int xWidth, int yHeight, int zDepth)
{
	return { (x / ((float)xWidth - 1)) * 2 - 1, (y / ((float)yHeight - 1)) * 2 - 1, (z / ((float)zDepth - 1)) * 2 - 1 };
}

static __device__ __host__ __inline__ void normToIdx(vec3 norm, int& x, int& y, int& z, int xWidth, int yHeight, int zDepth)
{
	x = (xWidth - 1) * (norm.x * 0.5 + 0.5);
	y = (yHeight - 1) * (norm.y * 0.5 + 0.5);
	z = (zDepth - 1) * (norm.z * 0.5 + 0.5);
}