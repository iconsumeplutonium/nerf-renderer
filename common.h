#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
	uint32_t numRays;
	uint64_t rayBuffer;
	float camOrigin[3];

	uint64_t colorLayer0WeightsAddr;
	uint64_t colorLayer0BiasAddr;
} kernel_arg_t;

#endif
