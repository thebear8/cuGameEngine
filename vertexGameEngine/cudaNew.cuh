#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<class T, class... argTypes>
T* cudaNewManaged(argTypes... args)
{
	T* ptr = nullptr;
	cudaMallocManaged(&ptr, sizeof(T));
	new (ptr) T(args...);
	return ptr;
}

template<class T, class... argTypes>
T* cudaNew(argTypes... args)
{
	T* ptr = nullptr;
	cudaMallocManaged(&ptr, sizeof(T));
	new (ptr) T(args...);
	return ptr;
}