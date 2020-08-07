#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <vector>

#include "cuPixel.cuh"
#include "cuSurface.cuh"
#include "cuEffect.cuh"
#include "swapChain.cuh"

class renderPipeline
{
public:
	swapChain<cuSurface*>* swChain;
	std::vector<cuEffect*>* effects;

	renderPipeline(swapChain<cuSurface*>* swChain, std::vector<cuEffect*>* effects)
	{
		this->swChain = swChain;
		this->effects = effects;
	}

	renderPipeline(cuSurface* front, cuSurface* back) : renderPipeline(new swapChain<cuSurface*>(front, back), new std::vector<cuEffect*>())
	{
		
	}

	renderPipeline(int64_t width, int64_t height) : renderPipeline(new cuSurface(width, height), new cuSurface(width, height))
	{

	}

	void render()
	{
		for (auto preEffect : *effects)
		{
			preEffect->apply(swChain->front, swChain->back);
			swChain->swap();
		}

		swChain->swap();
	}

	void addEffect(cuEffect* effect)
	{
		effects->push_back(effect);
	}
};