#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>

#include "math.cuh"
#include "cudaFnPointer.cuh"

template<unsigned int paramBufferSize, class retType, class... shaderArgs>
class baseShader
{
public:
	using shaderFnType = retType(*)(shaderArgs...);

private:
	uint64_t parameterBuffer[paramBufferSize] = { 0 };
	void* shaderFnPointer = nullptr;

public:
	template<shaderFnType shaderFn>
	void setShader()
	{
		shaderFnPointer = __cudaFnPointer<shaderFnType, shaderFn>();
	}

	template<unsigned int idx, class T>
	void setParameter(const T& param)
	{
		static_assert(sizeof(T) <= sizeof(uint64_t), "size of parameter type too big");
		static_assert(idx < paramBufferSize, "invalid parameter index");
		*((T*)(&parameterBuffer[idx])) = param;
	}

	template<unsigned int idx, class T>
	__device__ __host__ __inline__ const T& getParameter() const
	{
		static_assert(sizeof(T) <= sizeof(uint64_t), "size of parameter type too big");
		static_assert(idx < paramBufferSize, "invalid parameter index");
		return *((T*)(&parameterBuffer[idx]));
	}

	__device__ __forceinline__ retType callShader(shaderArgs... args) const
	{
		return ((retType(*)(shaderArgs...))shaderFnPointer)(args...);
	}
};

class vertexShader : public baseShader<16, vec3, const vertexShader*, vec3>
{
};

class fragmentShader : public baseShader<16, vec4, const fragmentShader*, vec3>
{
};