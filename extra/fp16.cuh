#include <cuda_fp16.h>

struct __align__(8) half4 {
    half2 x;
    half2 y;
    __device__ explicit operator float4() const; 
};

__device__ half4 make_half4(half2 a, half2 b) {
    half4 result;
    result.x = a;
    result.y = b;
    return result;
}

__device__ half4::operator float4() const {
    float4 result;
    result.x = __half2float(x.x);
    result.y = __half2float(x.y);
    result.z = __half2float(y.x);
    result.w = __half2float(y.y);
    return result;
}

__device__ half4 operator+(half4 a, half4 b) {
    half4 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    return result;
}

__device__ half4 operator-(half4 a, half4 b) {
    half4 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    return result;
}

__device__ half4 operator*(half4 a, half4 b) {
    half4 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    return result;
}

__device__ half4 operator/(half4 a, half4 b) {
    half4 result;
    result.x = a.x / b.x;
    result.y = a.y / b.y;
    return result;
}

__device__ half4 float4_to_half4(float4 f) {
    half4 result;
    result.x = __float22half2_rn(make_float2(f.x, f.y));
    result.y = __float22half2_rn(make_float2(f.z, f.w));
    return result;
}

__device__ float4 half4_to_float4(half4 h) { return (float4)h; }
