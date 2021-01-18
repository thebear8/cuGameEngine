#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_FNPOINTER(fn) __cudaFnPointer<decltype(fn)*, fn>()

template<class fnType, fnType fn>
__global__ void __cudaFnPointerKernel(void** fnPointer)
{
    auto fnCopy = fn;
    *fnPointer = *((void**)&fnCopy);
}

template<class fnType, fnType fn>
fnType __cudaFnPointer()
{
    void** ptr = 0;
    cudaMallocManaged(&ptr, sizeof(*ptr));

    __cudaFnPointerKernel<fnType, fn> << <1, 1 >> > (ptr);
    cudaDeviceSynchronize();

    fnType ptrCopy = (fnType)*ptr;
    cudaFree(ptr);
    return ptrCopy;
}
