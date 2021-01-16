#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

class cuStreamManager
{
	class __cuStreamManager
	{
	public:
		bool isInitialized = false;
		cudaDeviceProp props = { 0 };
		std::vector<cudaStream_t> streams;

	public:
		void init()
		{
			if (!isInitialized)
			{
				isInitialized = true;

				int device;
				cudaGetDevice(&device);
				cudaGetDeviceProperties(&props, device);
				for (int i = 0; i < props.multiProcessorCount; i++)
				{
					cudaStream_t stream;
					cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
					streams.push_back(stream);
				}
			}
		}

		size_t getNumberOfStreams()
		{
			init();
			return streams.size();
		}

		cudaStream_t getStream(size_t idx)
		{
			init();
			return streams[idx];
		}

		void syncAll()
		{
			for(auto s : streams)
			{
				cudaStreamSynchronize(s);
			}
		}
	};

private:
	static __cuStreamManager& getMgr()
	{
		static __cuStreamManager mgr;
		return mgr;
	}

public:
	static size_t getNumberOfStreams() { return getMgr().getNumberOfStreams(); }
	static cudaStream_t getStream(size_t idx) { return getMgr().getStream(idx); }
	static void syncAll() { return getMgr().syncAll(); }
};