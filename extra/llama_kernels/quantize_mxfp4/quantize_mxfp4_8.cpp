#include "quantize_mxfp4_8.h"

// Load the full 128x64 tile once; transpose packed results for coalesced column stores.
static_assert(M_DIM % 128 == 0 && N_DIM % 64 == 0);
__device__ __forceinline__ uint16_t* at(uint16_t (*tile)[72],int row,int col) {
  return &tile[row][col^(((row/8)&7)*8)];
}
__device__ __forceinline__ float val(uint16_t x) { return __uint_as_float(uint32_t(x)<<16); }
extern "C" __global__ __launch_bounds__(256,4)
void KERNEL_NAME(uint8_t* __restrict__ row_fp4,uint8_t* __restrict__ row_scale,
                uint8_t* __restrict__ fp4,uint8_t* __restrict__ scales,const uint16_t* __restrict__ input) {
  __shared__ uint16_t tile[128][72];
  __shared__ uint32_t packed[64][16];
  const int tid=threadIdx.x,base_row=blockIdx.x*128,base_col=blockIdx.y*64;
  #pragma unroll
  for (int q=0;q<4;q++) {
    const int row=q*32+tid/8,col=(tid%8)*8;
    *reinterpret_cast<uint4*>(at(tile,row,col))=*reinterpret_cast<const uint4*>(input+(base_row+row)*N_DIM+base_col+col);
  }
  __syncthreads();
#if WRITE_ROWWISE_VALUE
  #pragma unroll
  for (int q=0;q<4;q++) {
    const int row=q*32+tid/8,col=(tid%8)*8;
    const auto quant=mxfp4::quantize8(mxfp4::load_bf16x4(at(tile,row,col)),mxfp4::load_bf16x4(at(tile,row,col+4)),tid%4);
    mxfp4::store_fp4_8<SHUFFLE_ROWWISE_FP4_VALUE>(row_fp4,base_row+row,(base_col+col)/2,N_DIM/2,quant.fp4);
    if (tid%4==0) mxfp4::store_scale(row_scale,base_row+row,(base_col+col)/32,N_DIM/32,quant.scale);
  }
#endif
  #pragma unroll
  for (int q=0;q<4;q++) {
    const int row=q*32+(tid%4)*8,col=tid/4;
    const auto quant=mxfp4::quantize8(make_float4(val(*at(tile,row,col)),val(*at(tile,row+1,col)),
      val(*at(tile,row+2,col)),val(*at(tile,row+3,col))),
      make_float4(val(*at(tile,row+4,col)),val(*at(tile,row+5,col)),val(*at(tile,row+6,col)),val(*at(tile,row+7,col))),tid%4);
    packed[col][q*4+tid%4]=quant.fp4;
    if (tid%4==0) mxfp4::store_scale(scales,base_col+col,base_row/32+q,M_DIM/32,quant.scale);
  }
  __syncthreads();
  #pragma unroll
  for (int q=0;q<4;q++) {
    const int col=q*16+tid/16,rp=tid%16;
    mxfp4::store_fp4_8<SHUFFLE_COLWISE_FP4_VALUE>(fp4,base_col+col,base_row/2+rp*4,M_DIM/2,packed[col][rp]);
  }
}
