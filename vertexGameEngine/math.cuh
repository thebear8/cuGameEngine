#include <math.h>

class vec2
{
public:
	float x, y;

	inline vec2(float x, float y)
	{
		this->x = x;
		this->y = y;
	}

	inline vec2 add(float b) const { return { x + b, y + b}; };
	inline vec2 sub(float b) const { return { x - b, y - b}; };
	inline vec2 mul(float b) const { return { x * b, y * b}; };
	inline vec2 div(float b) const { return { x / b, y / b}; };

	inline vec2 add(const vec2& b) const { return { x + b.x, y + b.y }; };
	inline vec2 sub(const vec2& b) const { return { x - b.x, y - b.y }; };
	inline vec2 mul(const vec2& b) const { return { x * b.x, y * b.y }; };
	inline vec2 div(const vec2& b) const { return { x / b.x, y / b.y }; };

	inline vec2 norm() const
	{
		return mul(1.0f / sqrtf(x * x + y * y));
	}

	inline float dot(const vec2& b)
	{
		return x * b.x + y * b.y;
	}
};

class vec3
{
public:
	float x, y, z;

	inline vec3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	inline vec3 add(float b) const { return { x + b, y + b, z + b }; };
	inline vec3 sub(float b) const { return { x - b, y - b, z - b }; };
	inline vec3 mul(float b) const { return { x * b, y * b, z * b }; };
	inline vec3 div(float b) const { return { x / b, y / b, z / b }; };

	inline vec3 add(const vec3& b) const { return { x + b.x, y + b.y, z + b.z}; };
	inline vec3 sub(const vec3& b) const { return { x - b.x, y - b.y, z - b.z }; };
	inline vec3 mul(const vec3& b) const { return { x * b.x, y * b.y, z * b.z }; };
	inline vec3 div(const vec3& b) const { return { x / b.x, y / b.y, z / b.z }; };

	inline vec3 norm() const
	{
		return mul(1.0f / sqrtf(x * x + y * y + z * z));
	}

	inline float dot(const vec3& b)
	{
		return x * b.x + y * b.y + z * b.z;
	}

	inline vec3 cross(const vec3& b)
	{
		return { y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x };
	}
};

class mat4
{
public:
	float mat[4][4];

	inline mat4(float mat[4][4])
	{
		for (int x = 0; x < 4; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				this->mat[x][y] = mat[x][y];
			}
		}
	}
};

bool pointIsInTriangle2d(vec3 _p, vec3 _a, vec3 _b, vec3 _c)
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