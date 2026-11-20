#pragma once
#include <musa_fp16.h>
#if !defined(__MUSA_ARCH__) || __MUSA_ARCH__ < 220
__device__ __inline__ __half hsin(__half x) { return __float2half(sinf(__half2float(x))); }
__device__ __inline__ __half hlog2(__half x) { return __float2half(log2f(__half2float(x))); }
__device__ __inline__ __half hexp2(__half x) { return __float2half(exp2f(__half2float(x))); }
__device__ __inline__ __half hsqrt(__half x) { return __float2half(sqrtf(__half2float(x))); }
__device__ __inline__ __half htrunc(__half x) { return __float2half(truncf(__half2float(x))); }
__device__ __inline__ __half hrcp(__half x) { return __float2half(1.0f/__half2float(x)); }
#endif
