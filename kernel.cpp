#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	// auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	// auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	// auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);
	auto rayBufferPtr = reinterpret_cast<float*>(arg->rayBuffer);
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arg->numRays) return;

	// dst_ptr[idx] = src0_ptr[idx] + src1_ptr[idx];
	rayBufferPtr[idx * 3]     = arg->camOrigin[0];//rayBufferPtr[idx * 3] * -1.0f;
	rayBufferPtr[idx * 3 + 1] = arg->camOrigin[1];//rayBufferPtr[idx * 3 + 1] * -1.0f;
	rayBufferPtr[idx * 3 + 2] = arg->camOrigin[2];//rayBufferPtr[idx * 3 + 2] * -1.0f;
}
