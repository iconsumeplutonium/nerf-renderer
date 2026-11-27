#include <vx_spawn2.h>
#include <vector>
#include "common.h"

std::vector<int> resolutions = {16, 22, 30, 42, 58, 80, 111, 153, 212, 294, 406, 561, 776, 1072, 1482, 2048};

inline float lerp(float a, float b, float p) {
	float t = (a - p) / (b - a);
	return (1.0f - t) * a + t * b; 
}

inline vec2 bilerp(vec2 a, vec2 b, vec2 c, vec2 d, vec2 p) {
	//     temp1
	// a +--.-+ b
	//   |  p |
	// c +--.-+ d
	//     temp2

	vec2 temp1 = v2(lerp(a.x, b.x, p.x), a.y);
	vec2 temp2 = v2(lerp(c.x, d.x, p.x), c.y);

	return v2(temp1.x, lerp(temp1.y, temp2.y, p.y));
}

enum Plane {
	YZ_PLANE,
	XZ_PLANE,
	XY_PLANE,
};

inline vec3 bilerp(vec3 a, vec3 b, vec3 c, vec3 d, vec3 p, Plane axisToIgnore) {
	switch (axisToIgnore) {
		case Plane::YZ_PLANE: // ignore x axis, treat as bilerp on y/z plane
			vec2 r = bilerp(
				v2(a.y, a.z),
				v2(b.y, b.z),
				v2(c.y, c.z),
				v2(d.y, d.z),
				v2(p.y, p.z)
			);
			return v3(a.x, r.x, r.y);
		case Plane::XZ_PLANE: // ignore y axis, treat as bilerp on x/z plane
			vec2 r1 = bilerp(
				v2(a.x, a.z),
				v2(b.x, b.z),
				v2(c.x, c.z),
				v2(d.x, d.z),
				v2(p.x, p.z)
			);
			return v3(r1.x, a.y, r1.y);
		case Plane::XY_PLANE: // ignore z axis, treat as bilerp on x/y plane
			vec2 r2 = bilerp(
				v2(a.x, a.y),
				v2(b.x, b.y),
				v2(c.x, c.y),
				v2(d.x, d.y),
				v2(p.x, p.y)
			);
			return v3(r2.x, r2.y, a.z);
	}
}

// this is fucked, i need to lerp the actual values not the coords, fuck
vec3 trilerp(std::vector<vec3>& verts, vec3 p) {
	vec3 top    = bilerp(verts[3], verts[7], verts[2], verts[6], p, Plane::XZ_PLANE);
	vec3 bottom = bilerp(verts[1], verts[5], verts[0], verts[4], p, Plane::XZ_PLANE);
	
	vec3 left  = bilerp(verts[3], verts[2], verts[1], verts[0], p, Plane::YZ_PLANE);
	vec3 right = bilerp(verts[7], verts[6], verts[5], verts[4], p, Plane::YZ_PLANE);

	vec3 front = bilerp(verts[2], verts[6], verts[0], verts[4], p, Plane::XY_PLANE);
	vec3 back  = bilerp(verts[3], verts[7], verts[1], verts[5], p, Plane::XY_PLANE);




}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	// auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	// auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	// auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);
	float* rayBufferPtr = reinterpret_cast<float*>(arg->rayBuffer);
	float* lookupTablePtr = reinterpret_cast<float*>(arg->lookupTable);
	
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arg->numRays) return;

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

	std::vector<float> densities(numSamples);
	std::vector<float> colors(numSamples);
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
		vec3 sample = v3add(origin, v3scale(direction, t));

		// check if it sin the box
		vec3 scaledPoint = v3scale(sample, 1.0f/3.0f);
		if (scaledPoint.x < -0.5 || scaledPoint.y < -0.5 || scaledPoint.z < -0.5 || scaledPoint.x > 0.5 || scaledPoint.y > 0.5 || scaledPoint.z > 0.5) {
			densities[i] = 0.0f; // todo: this should probably not be a float
			colors[i] = 0.0f;
			continue;
		}

		// add 0.5 so the point is in [0, 1] on all axes
		vec3 ndcPoint = v3translate(scaledPoint, 0.5f);

		// todo: hash table shit
		
		std::vector<float> features;
		features.reserve(32);

		for (int level = 0; level < 16; level++) {
			vec3 floored = v3floor(v3scale(ndcPoint, resolutions[level]));
			vec3 ceiled  = v3ceil( v3scale(ndcPoint, resolutions[level]));
			std::vector<vec3> vertices = {
				v3(floored.x, floored.y, floored.z),
				v3(floored.x, floored.y,  ceiled.z),
				v3(floored.x,  ceiled.y, floored.z),
				v3(floored.x,  ceiled.y,  ceiled.z),
				v3( ceiled.x, floored.y, floored.z),
				v3( ceiled.x, floored.y,  ceiled.z),
				v3( ceiled.x,  ceiled.y, floored.z),
				v3( ceiled.x,  ceiled.y,  ceiled.z),
			};

			std::vector<vec2> temp;
			temp.reserve(8);
			float* baseTablePtr = lookupTablePtr + (524288 * 2 * level);

			for (const vec3& vert: vertices) {
				long x = (uint32_t) vert.x * 1u;
				long y = (uint32_t) vert.y * 2654435761u;
				long z = (uint32_t) vert.z * 805459861u;
				uint32_t index = (x ^ y ^ z) % 524288u;
	
				// look up
				float feature1 = *(baseTablePtr + (index * 2));
				float feature2 = *(baseTablePtr + (index * 2 + 1));

				temp.push_back(vec2 {feature1, feature2});
			}
			
			// trilinear lerp
		}


		// todo: density shit
		// todo: color shit
	}
	
	// todo: tarnsmittence shit
	

	// rayBufferPtr[idx * 3]     = arg->camOrigin[0];//rayBufferPtr[idx * 3] * -1.0f;
	// rayBufferPtr[idx * 3 + 1] = arg->camOrigin[1];//rayBufferPtr[idx * 3 + 1] * -1.0f;
	// rayBufferPtr[idx * 3 + 2] = arg->camOrigin[2];//rayBufferPtr[idx * 3 + 2] * -1.0f;
}
