#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "math.cuh"

vec3 projectVertex(vec3 pos, vec3 cameraPos)
{
    return { pos.x / pos.z, pos.y / pos.z, 1.0f };
}

int main()
{
    
}