#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
#include <cuda_fp8.h>
#include <stdio.h>

struct __align__(8) __nv_fp8_storage_t8 { __nv_fp8_storage_t x, y, z, w, a, b, c, d; }; __device__ __nv_fp8_storage_t8 make___nv_fp8_storage_t8(__nv_fp8_storage_t x, __nv_fp8_storage_t y, __nv_fp8_storage_t z, __nv_fp8_storage_t w, __nv_fp8_storage_t a, __nv_fp8_storage_t b, __nv_fp8_storage_t c, __nv_fp8_storage_t d) { __nv_fp8_storage_t8 r={x, y, z, w, a, b, c, d}; return r; }
struct __align__(16) __nv_fp8_storage_t16 { __nv_fp8_storage_t x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ __nv_fp8_storage_t16 make___nv_fp8_storage_t16(__nv_fp8_storage_t x, __nv_fp8_storage_t y, __nv_fp8_storage_t z, __nv_fp8_storage_t w, __nv_fp8_storage_t a, __nv_fp8_storage_t b, __nv_fp8_storage_t c, __nv_fp8_storage_t d, __nv_fp8_storage_t e, __nv_fp8_storage_t f, __nv_fp8_storage_t g, __nv_fp8_storage_t h, __nv_fp8_storage_t i, __nv_fp8_storage_t j, __nv_fp8_storage_t k, __nv_fp8_storage_t l) { __nv_fp8_storage_t16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
__device__ float4 __WMMA_8_16_32_fp8e4m3_float(__nv_fp8_storage_t16 a, __nv_fp8_storage_t8 b, float4 c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b);
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7},"
      "{%8, %9}, {%0, %1, %2, %3};"
    : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
    : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]), "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]));
  return c;
}
extern "C" __global__ void __launch_bounds__(32) r_2_2_2_2_2_2_2_2_2_2_2_2_2_4(__nv_fp8_storage_t* data0, __nv_fp8_storage_t* data1, __nv_fp8_storage_t* data2) {
  int lidx0 = threadIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 2 */
  int lidx2 = threadIdx.z; /* 2 */
  int alu0 = (lidx1<<6);
  int alu1 = (lidx2<<7);
  int alu2 = ((lidx0>>2)<<5);
  int alu3 = (alu0+alu2+alu1);
  __nv_fp8_storage_t val0 = *(data1+alu3);
  __nv_fp8_storage_t val1 = *(data1+(alu3+8));
  __nv_fp8_storage_t val2 = *(data1+(alu3+16));
  __nv_fp8_storage_t val3 = *(data1+(alu3+24));
  __nv_fp8_storage_t val4 = *(data1+(alu3+256));
  __nv_fp8_storage_t val5 = *(data1+(alu3+264));
  __nv_fp8_storage_t val6 = *(data1+(alu3+272));
  __nv_fp8_storage_t val7 = *(data1+(alu3+280));
  __nv_fp8_storage_t val8 = *(data1+(alu3+512));
  __nv_fp8_storage_t val9 = *(data1+(alu3+520));
  __nv_fp8_storage_t val10 = *(data1+(alu3+528));
  __nv_fp8_storage_t val11 = *(data1+(alu3+536));
  __nv_fp8_storage_t val12 = *(data1+(alu3+768));
  __nv_fp8_storage_t val13 = *(data1+(alu3+776));
  __nv_fp8_storage_t val14 = *(data1+(alu3+784));
  __nv_fp8_storage_t val15 = *(data1+(alu3+792));
  int alu4 = (((lidx0&1)<<1)+(((lidx0>>1)&1)<<2));
  __nv_fp8_storage_t val16 = *(data2+alu4);
  __nv_fp8_storage_t val17 = *(data2+(alu4+1));
  __nv_fp8_storage_t val18 = *(data2+(alu4+8));
  __nv_fp8_storage_t val19 = *(data2+(alu4+9));
  __nv_fp8_storage_t val20 = *(data2+(alu4+16));
  __nv_fp8_storage_t val21 = *(data2+(alu4+17));
  __nv_fp8_storage_t val22 = *(data2+(alu4+24));
  __nv_fp8_storage_t val23 = *(data2+(alu4+25));
  __nv_fp8_storage_t val24 = *(data2+(alu4+512));
  __nv_fp8_storage_t val25 = *(data2+(alu4+513));
  __nv_fp8_storage_t val26 = *(data2+(alu4+520));
  __nv_fp8_storage_t val27 = *(data2+(alu4+521));
  __nv_fp8_storage_t val28 = *(data2+(alu4+528));
  __nv_fp8_storage_t val29 = *(data2+(alu4+529));
  __nv_fp8_storage_t val30 = *(data2+(alu4+536));
  __nv_fp8_storage_t val31 = *(data2+(alu4+537));
  __nv_fp8_storage_t8 cast0 = make___nv_fp8_storage_t8(val18,val26,val19,val27,val18,val26,val19,val27);
  __nv_fp8_storage_t8 cast1 = make___nv_fp8_storage_t8(val20,val28,val21,val29,val20,val28,val21,val29);
  __nv_fp8_storage_t8 cast2 = make___nv_fp8_storage_t8(val22,val30,val23,val31,val22,val30,val23,val31);
  __nv_fp8_storage_t8 cast3 = make___nv_fp8_storage_t8(val16,val24,val17,val25,val16,val24,val17,val25);
  __nv_fp8_storage_t16 cast4 = make___nv_fp8_storage_t16(val8,val9,val10,val11,val8,val9,val10,val11,val12,val13,val14,val15,val12,val13,val14,val15);
  __nv_fp8_storage_t16 cast5 = make___nv_fp8_storage_t16(val0,val1,val2,val3,val0,val1,val2,val3,val4,val5,val6,val7,val4,val5,val6,val7);
  float4 cast6 = make_float4(0.0f,0.0f,0.0f,0.0f);
  float4 wmma0 = __WMMA_8_16_32_fp8e4m3_float(cast4, cast0, cast6);
  float4 wmma1 = __WMMA_8_16_32_fp8e4m3_float(cast4, cast1, cast6);
  float4 wmma2 = __WMMA_8_16_32_fp8e4m3_float(cast4, cast2, cast6);
  float4 wmma3 = __WMMA_8_16_32_fp8e4m3_float(cast4, cast3, cast6);
  float4 wmma4 = __WMMA_8_16_32_fp8e4m3_float(cast5, cast0, cast6);
  float4 wmma5 = __WMMA_8_16_32_fp8e4m3_float(cast5, cast1, cast6);
  float4 wmma6 = __WMMA_8_16_32_fp8e4m3_float(cast5, cast2, cast6);
  float4 wmma7 = __WMMA_8_16_32_fp8e4m3_float(cast5, cast3, cast6);
  int alu5 = (alu4+alu2+alu0+alu1);
  *(data0+alu5) = ((__nv_fp8_storage_t)(wmma7.x));
  *(data0+(alu5+1)) = ((__nv_fp8_storage_t)(wmma7.y));
  *(data0+(alu5+8)) = ((__nv_fp8_storage_t)(wmma4.x));
  *(data0+(alu5+9)) = ((__nv_fp8_storage_t)(wmma4.y));
  *(data0+(alu5+16)) = ((__nv_fp8_storage_t)(wmma5.x));
  *(data0+(alu5+17)) = ((__nv_fp8_storage_t)(wmma5.y));
  *(data0+(alu5+24)) = ((__nv_fp8_storage_t)(wmma6.x));
  *(data0+(alu5+25)) = ((__nv_fp8_storage_t)(wmma6.y));
  *(data0+(alu5+256)) = ((__nv_fp8_storage_t)(wmma7.z));
  *(data0+(alu5+257)) = ((__nv_fp8_storage_t)(wmma7.w));
  *(data0+(alu5+264)) = ((__nv_fp8_storage_t)(wmma4.z));
  *(data0+(alu5+265)) = ((__nv_fp8_storage_t)(wmma4.w));
  *(data0+(alu5+272)) = ((__nv_fp8_storage_t)(wmma5.z));
  *(data0+(alu5+273)) = ((__nv_fp8_storage_t)(wmma5.w));
  *(data0+(alu5+280)) = ((__nv_fp8_storage_t)(wmma6.z));
  *(data0+(alu5+281)) = ((__nv_fp8_storage_t)(wmma6.w));
  *(data0+(alu5+512)) = ((__nv_fp8_storage_t)(wmma3.x));
  *(data0+(alu5+513)) = ((__nv_fp8_storage_t)(wmma3.y));
  *(data0+(alu5+520)) = ((__nv_fp8_storage_t)(wmma0.x));
  *(data0+(alu5+521)) = ((__nv_fp8_storage_t)(wmma0.y));
  *(data0+(alu5+528)) = ((__nv_fp8_storage_t)(wmma1.x));
  *(data0+(alu5+529)) = ((__nv_fp8_storage_t)(wmma1.y));
  *(data0+(alu5+536)) = ((__nv_fp8_storage_t)(wmma2.x));
  *(data0+(alu5+537)) = ((__nv_fp8_storage_t)(wmma2.y));
  *(data0+(alu5+768)) = ((__nv_fp8_storage_t)(wmma3.z));
  *(data0+(alu5+769)) = ((__nv_fp8_storage_t)(wmma3.w));
  *(data0+(alu5+776)) = ((__nv_fp8_storage_t)(wmma0.z));
  *(data0+(alu5+777)) = ((__nv_fp8_storage_t)(wmma0.w));
  *(data0+(alu5+784)) = ((__nv_fp8_storage_t)(wmma1.z));
  *(data0+(alu5+785)) = ((__nv_fp8_storage_t)(wmma1.w));
  *(data0+(alu5+792)) = ((__nv_fp8_storage_t)(wmma2.z));
  *(data0+(alu5+793)) = ((__nv_fp8_storage_t)(wmma2.w));
}

void check(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

__global__ void init_matrices(__nv_fp8_storage_t* A, __nv_fp8_storage_t* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = __nv_fp8_storage_t(1.0f);
        B[idx] = __nv_fp8_storage_t(1.0f);
    }
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    __nv_fp8_storage_t *d_A, *d_B, *d_C;
    
    check(cudaMalloc(&d_A, M * K * sizeof(__nv_fp8_storage_t)), "Failed to allocate A");
    check(cudaMalloc(&d_B, K * N * sizeof(__nv_fp8_storage_t)), "Failed to allocate B");
    check(cudaMalloc(&d_C, M * N * sizeof(__nv_fp8_storage_t)), "Failed to allocate C");

    int threadsPerBlock = 256;
    int blocks = (M * K + threadsPerBlock - 1) / threadsPerBlock;
    init_matrices<<<blocks, threadsPerBlock>>>(d_A, d_B, M * K);
    
    dim3 grid(2, 2, 2);
    dim3 block(8, 2, 2);
    r_2_2_2_2_2_2_2_2_2_2_2_2_2_4<<<grid, block>>>(d_C, d_A, d_B);
    
    check(cudaGetLastError(), "Kernel launch failed");
    check(cudaDeviceSynchronize(), "Kernel execution failed");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("Test completed successfully\n");
    return 0;
}