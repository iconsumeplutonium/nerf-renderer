#include <vx_spawn2.h>
#include <vector>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	// auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	// auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	// auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);
	float* rayBufferPtr = reinterpret_cast<float*>(arg->rayBuffer);
	float* lookupTablePtr = reinterpret_cast<float*>(arg->lookupTable);
	
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arg->numRays) return;

	float rayDir[3] = {
		rayBufferPtr[idx * 3],    
		rayBufferPtr[idx * 3 + 1],
		rayBufferPtr[idx * 3 + 2] 
	};

	vec3 origin = v3(arg->camOrigin[0], arg->camOrigin[1], arg->camOrigin[2]);
	vec3 direction = v3(		
		rayBufferPtr[idx * 3],    
		rayBufferPtr[idx * 3 + 1],
		rayBufferPtr[idx * 3 + 2] 
	);

	const float near = 2.0f;
	const float far = 6.0f;
	const int numSamples = 192;
	const float step = (far - near) / (float) numSamples;

	std::vector<vec3> samplePoints;
	samplePoints.resize(numSamples);
	for (int i = 0; i < numSamples; i++) {
		// the middle of each interval
		// 2 (near plane)                                                                 6 (far plane)
		//        |----------|----------|----------|----------|----------|----------|----------|
		//             ^          ^          ^
		// first sample offset    all subsequent steps are 1 full step away.
		// by half a step
		// -----
		// thus, i*step gets the start of the interval, then you offset by half a step. 
		// then you add near, so that it starts at the near plane, and goes to the far plane
		float t = near + (i * step + (step / 2.0f));

		samplePoints[i] = v3add(origin, v3scale(direction, t));
	}

	// for each point, check if its in the box
	// if its not in the box, set predicted density/color to 0
	// else, do the hash table/density/color, and get predicted color and density


	

	// rayBufferPtr[idx * 3]     = arg->camOrigin[0];//rayBufferPtr[idx * 3] * -1.0f;
	// rayBufferPtr[idx * 3 + 1] = arg->camOrigin[1];//rayBufferPtr[idx * 3 + 1] * -1.0f;
	// rayBufferPtr[idx * 3 + 2] = arg->camOrigin[2];//rayBufferPtr[idx * 3 + 2] * -1.0f;
}
