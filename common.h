#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

// borrowed from the rt_raycast test's common.h
typedef struct vec3 {
	float x, y, z;
} vec3;

static inline vec3 v3(float x, float y, float z) { vec3 r = {x, y, z}; return r; }
static inline vec3 v3add(vec3 a, vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3 v3sub(vec3 a, vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3 v3scale(vec3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }
static inline float v3dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3 v3cross(vec3 a, vec3 b) {
	return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
//extra i added
typedef struct vec2 {
	float x, y;
} vec2;
static inline vec2 v2(float x, float y) { vec3 r = {x, y}; return r; }
static inline vec3 v3translate(vec3 a, float s) { return v3(a.x+s, a.y+s, a.z+s); }
static inline vec3 v3floor(vec3 a) { return v3(floor(a.x), floor(a.y), floor(a.z)); }
static inline vec3 v3ceil(vec3 a) { return v3translate(v3floor(a), 1.0f) ; }

typedef struct {
	uint32_t numRays;
	uint64_t rayBuffer;
	float camOrigin[3];

	uint64_t lookupTable;

	uint64_t colorLayerWeightsAddr[3];
	// uint64_t colorLayerBiasAddr[3];
} kernel_arg_t;

#endif
