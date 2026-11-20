#pragma once
struct nv_bfloat16 {
  unsigned short data;
  __device__ nv_bfloat16() = default;
  __device__ nv_bfloat16(unsigned short raw) : data(raw) {}
  __device__ nv_bfloat16(float f);
  __device__ nv_bfloat16(double f);
  __device__ operator float() const;
};
__device__ nv_bfloat16 __float2bfloat16(float);
__device__ nv_bfloat16::nv_bfloat16(float f) : data(__float2bfloat16(f).data) {}
__device__ nv_bfloat16::nv_bfloat16(double f) : data(__float2bfloat16((float)f).data) {}
__device__ float __bfloat162float(nv_bfloat16 x) { return __uint_as_float(((unsigned int)x.data) << 16); }
__device__ nv_bfloat16::operator float() const { return __bfloat162float(*this); }
__device__ nv_bfloat16 __float2bfloat16(float f) {
  unsigned int u = __float_as_uint(f), rem = u << 16;
  if ((u & 0x7fffffffU) > 0x7f800000U) return nv_bfloat16((unsigned short)0x7fff);
  unsigned short x = u >> 16;
  if ((rem > 0x80000000U) || ((rem == 0x80000000U) && ((x & 1U) != 0U))) x++;
  return nv_bfloat16(x);
}
__device__ __inline__ nv_bfloat16 operator+(const nv_bfloat16& a,const nv_bfloat16& b) { return __float2bfloat16(__bfloat162float(a)+__bfloat162float(b)); }
__device__ __inline__ nv_bfloat16 operator-(const nv_bfloat16& a,const nv_bfloat16& b) { return __float2bfloat16(__bfloat162float(a)-__bfloat162float(b)); }
__device__ __inline__ nv_bfloat16 operator*(const nv_bfloat16& a,const nv_bfloat16& b) { return __float2bfloat16(__bfloat162float(a)*__bfloat162float(b)); }
__device__ __inline__ nv_bfloat16 operator/(const nv_bfloat16& a,const nv_bfloat16& b) { return __float2bfloat16(__bfloat162float(a)/__bfloat162float(b)); }
__device__ __inline__ nv_bfloat16 operator-(const nv_bfloat16& a) { return __float2bfloat16(-__bfloat162float(a)); }
__device__ __inline__ bool operator==(const nv_bfloat16& a,const nv_bfloat16& b) { return __bfloat162float(a)==__bfloat162float(b); }
__device__ __inline__ bool operator!=(const nv_bfloat16& a,const nv_bfloat16& b) { return __bfloat162float(a)!=__bfloat162float(b); }
__device__ __inline__ bool operator<(const nv_bfloat16& a,const nv_bfloat16& b) { return __bfloat162float(a)<__bfloat162float(b); }
__device__ __inline__ bool operator>(const nv_bfloat16& a,const nv_bfloat16& b) { return __bfloat162float(a)>__bfloat162float(b); }
__device__ __inline__ bool operator<=(const nv_bfloat16& a,const nv_bfloat16& b) { return __bfloat162float(a)<=__bfloat162float(b); }
__device__ __inline__ bool operator>=(const nv_bfloat16& a,const nv_bfloat16& b) { return __bfloat162float(a)>=__bfloat162float(b); }
__device__ __inline__ nv_bfloat16& operator+=(nv_bfloat16& a,const nv_bfloat16& b) { a=a+b; return a; }
__device__ __inline__ nv_bfloat16& operator-=(nv_bfloat16& a,const nv_bfloat16& b) { a=a-b; return a; }
__device__ __inline__ nv_bfloat16& operator*=(nv_bfloat16& a,const nv_bfloat16& b) { a=a*b; return a; }
__device__ __inline__ nv_bfloat16& operator/=(nv_bfloat16& a,const nv_bfloat16& b) { a=a/b; return a; }
#if !defined(__MUSA_ARCH__) || __MUSA_ARCH__ < 220
__device__ __inline__ nv_bfloat16 hexp2(nv_bfloat16 x) { return __float2bfloat16(exp2f(__bfloat162float(x))); }
__device__ __inline__ nv_bfloat16 hlog2(nv_bfloat16 x) { return __float2bfloat16(log2f(__bfloat162float(x))); }
__device__ __inline__ nv_bfloat16 hsin(nv_bfloat16 x) { return __float2bfloat16(sinf(__bfloat162float(x))); }
__device__ __inline__ nv_bfloat16 hsqrt(nv_bfloat16 x) { return __float2bfloat16(sqrtf(__bfloat162float(x))); }
__device__ __inline__ nv_bfloat16 htrunc(nv_bfloat16 x) { return __float2bfloat16(truncf(__bfloat162float(x))); }
__device__ __inline__ nv_bfloat16 hceil(nv_bfloat16 x) { return __float2bfloat16(ceilf(__bfloat162float(x))); }
__device__ __inline__ nv_bfloat16 hfloor(nv_bfloat16 x) { return __float2bfloat16(floorf(__bfloat162float(x))); }
__device__ __inline__ nv_bfloat16 hrcp(nv_bfloat16 x) { return __float2bfloat16(1.0f/__bfloat162float(x)); }
#endif
