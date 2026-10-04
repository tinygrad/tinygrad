#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#ifndef M_DIM
#define M_DIM 16384
#endif
#ifndef N_DIM
#define N_DIM 128256
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 1
#endif
#ifndef VEC_ROW_STORE
#define VEC_ROW_STORE 0
#endif
#ifndef LDS_BLOCK_PAD
#define LDS_BLOCK_PAD 1
#endif
#if NATIVE_MXFP8_CVT
typedef short short2v __attribute__((ext_vector_type(2)));
#endif
constexpr int BLK=32, TILE_N=THREADS_PER_WG, NT=N_DIM/TILE_N;
// Pad every 32-column row block as well as the complete LDS row. This lets the rowwise
// phase assign adjacent 32-value blocks to adjacent lanes (coalesced global stores)
// without making those lanes hit the same LDS bank. The exact GPT-OSS tile uses a wider
// block rotation selected by the caller; generic shapes retain the original one-slot pad.
constexpr int LDS_STRIDE=TILE_N+LDS_BLOCK_PAD*(TILE_N/BLK)+1;
constexpr float FP8_MAX=448.0f;
static_assert(M_DIM%BLK==0 && N_DIM%TILE_N==0);

__device__ __forceinline__ int lds_idx(int m, int n) { return m*LDS_STRIDE+n+LDS_BLOCK_PAD*(n/BLK); }

__device__ __forceinline__ int scale_e8(float amax) {
#if NATIVE_MXFP8_CVT
  return max(0, min(254, (int)((__builtin_bit_cast(unsigned, amax)>>23)&0xffu)));
#else
  return max(0, min(254, (int)floorf(log2f(fmaxf(amax,1e-38f)))+127));
#endif
}
__device__ __forceinline__ __hip_fp8_storage_t cvt_one(float v, int e8) {
#if NATIVE_MXFP8_CVT
  const float sf=__builtin_bit_cast(float,(unsigned)e8<<23);
  short2v acc={0,0}; acc=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc,v,0.0f,sf,false);
  return reinterpret_cast<unsigned char*>(&acc)[0];
#else
  const float qs=exp2f((float)(127-e8));
  return __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX,fminf(FP8_MAX,v*qs)),__HIP_SATFINITE,__HIP_E4M3);
#endif
}
__device__ __forceinline__ uint32_t cvt_four(float v0, float v1, float v2, float v3, int e8) {
#if NATIVE_MXFP8_CVT
  const float sf=__builtin_bit_cast(float,(unsigned)e8<<23);
  short2v a={0,0}, b={0,0};
  a=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(a,v0,v1,sf,false);
  b=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(b,v2,v3,sf,false);
  const unsigned char *ra=reinterpret_cast<const unsigned char*>(&a);
  const unsigned char *rb=reinterpret_cast<const unsigned char*>(&b);
  return (uint32_t)ra[0] | ((uint32_t)ra[1]<<8) | ((uint32_t)rb[0]<<16) | ((uint32_t)rb[1]<<24);
#else
  return (uint32_t)cvt_one(v0,e8) | ((uint32_t)cvt_one(v1,e8)<<8) |
         ((uint32_t)cvt_one(v2,e8)<<16) | ((uint32_t)cvt_one(v3,e8)<<24);
#endif
}
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void ce_dlogits_dual_mxfp8(
  __hip_fp8_storage_t *__restrict__ qrow, uint32_t *__restrict__ sirow,
  __hip_fp8_storage_t *__restrict__ qcol, uint32_t *__restrict__ sicol,
  const __hip_bfloat16 *__restrict__ logits, const float *__restrict__ lse,
  const int *__restrict__ targets, const float *__restrict__ loss_scale) {
  const int tid=threadIdx.x, tm=blockIdx.x/NT, tn=blockIdx.x%NT;
  const int n=tn*TILE_N+tid, m0=tm*BLK;
  __shared__ __hip_bfloat16 tile[BLK * LDS_STRIDE];
  float camax=0.0f;
  const float sc=loss_scale[0];
#pragma unroll
  for(int mm=0;mm<BLK;mm++) {
    const int m=m0+mm;
    const float z=(float)logits[(long long)m*N_DIM+n];
    // Match the accepted path: dLogits is rounded to BF16 before either MXFP8 quantization.
    const __hip_bfloat16 bv=(__hip_bfloat16)((__expf(z-lse[m])-(n==targets[m]?1.0f:0.0f))*sc);
    const float fv=(float)bv;
    tile[lds_idx(mm,tid)]=bv; camax=fmaxf(camax,fabsf(fv));
  }
  const int ce8=scale_e8(camax);
  reinterpret_cast<uint8_t *>(sicol)[(((long long)(tm>>2)*N_DIM+n)<<2)+(tm&3)]=(uint8_t)ce8;
  __syncthreads();

  // Eight lanes cooperatively write one transposed 32-value row as adjacent uint32s. Each wave covers
  // 64 columns while keeping every global store segment contiguous, and owns the scales for those columns,
  // so a lane shuffle avoids a second LDS array.
  const int lane=tid&63, wid=tid>>6, cm4=(lane&7)*4, cn_base=wid*64+(lane>>3);
#pragma unroll
  for(int j=0;j<8;j++) {
    const int cn=cn_base+j*8, cscale=__shfl(ce8,(lane>>3)+j*8,64);
    const uint32_t packed=cvt_four((float)tile[lds_idx(cm4+0,cn)], (float)tile[lds_idx(cm4+1,cn)],
                                   (float)tile[lds_idx(cm4+2,cn)], (float)tile[lds_idx(cm4+3,cn)], cscale);
    *reinterpret_cast<uint32_t*>(&qcol[(long long)(tn*TILE_N+cn)*M_DIM+m0+cm4])=packed;
  }

  // BLOCKS_N adjacent threads own the consecutive 1x32 blocks of one row. The per-block and per-row LDS
  // padding above keeps their serial reads bank-rotated.
  // The exact GPT-OSS shape uses TILE_N=768: twelve waves cover 768 columns while
  // preserving the same per-wave work and reducing launch/scheduling overhead.
  constexpr int BLOCKS_N=TILE_N/BLK;
  const int mm=tid/BLOCKS_N, nb=tid%BLOCKS_N, m=m0+mm, n0=tn*TILE_N+nb*BLK;
  float ramax=0.0f;
#pragma unroll
  for(int i=0;i<BLK;i++) { const float v=(float)tile[lds_idx(mm,nb*BLK+i)]; ramax=fmaxf(ramax,fabsf(v)); }
  const int re8=scale_e8(ramax);
  const long long rb=(long long)m*N_DIM+n0;
#if VEC_ROW_STORE
  uint32_t row_packed[BLK/4];
#endif
#pragma unroll
  for(int i=0;i<BLK;i+=4) {
    const uint32_t packed=cvt_four((float)tile[lds_idx(mm,nb*BLK+i+0)], (float)tile[lds_idx(mm,nb*BLK+i+1)],
                                   (float)tile[lds_idx(mm,nb*BLK+i+2)], (float)tile[lds_idx(mm,nb*BLK+i+3)], re8);
#if VEC_ROW_STORE
    row_packed[i/4]=packed;
#else
    *reinterpret_cast<uint32_t*>(&qrow[rb+i])=packed;
#endif
  }
#if VEC_ROW_STORE
  typedef unsigned uint8v __attribute__((ext_vector_type(8)));
  const uint8v p={row_packed[0],row_packed[1],row_packed[2],row_packed[3],
                  row_packed[4],row_packed[5],row_packed[6],row_packed[7]};
  *reinterpret_cast<uint8v*>(&qrow[rb])=p;
#endif
  const int pack_lane=lane-(nb&3);
  const uint32_t packed_e8=(uint32_t)__shfl(re8,pack_lane+0,64) | ((uint32_t)__shfl(re8,pack_lane+1,64)<<8) |
                           ((uint32_t)__shfl(re8,pack_lane+2,64)<<16) | ((uint32_t)__shfl(re8,pack_lane+3,64)<<24);
  if((nb&3)==0) sirow[(long long)(tn*(BLOCKS_N/4)+(nb>>2))*M_DIM+m]=packed_e8;
}
