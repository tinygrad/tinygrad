#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#ifndef M_DIM
#define M_DIM 73728
#endif
#ifndef H_DIM
#define H_DIM 5888
#endif
#ifndef DY_STRIDE
#define DY_STRIDE 3072
#endif
#ifndef REAL_INTER
#define REAL_INTER 2880
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 1
#endif
#ifndef WRITE_DH
#define WRITE_DH 0
#endif
#ifndef GPTOSS_DSWIGLU_RCP_NR
#define GPTOSS_DSWIGLU_RCP_NR 0
#endif
#ifndef GPTOSS_DSWIGLU_SKIP_PAD_TAIL
#define GPTOSS_DSWIGLU_SKIP_PAD_TAIL 0
#endif
#ifndef GPTOSS_DSWIGLU_LDS_STRIDE
#define GPTOSS_DSWIGLU_LDS_STRIDE 257
#endif
#ifndef GPTOSS_DSWIGLU_SKIP_EMPTY_TAIL
#define GPTOSS_DSWIGLU_SKIP_EMPTY_TAIL 0
#endif
#ifndef GPTOSS_DSWIGLU_EXPERT_COUNTS
#define GPTOSS_DSWIGLU_EXPERT_COUNTS 0
#endif
#ifndef GPTOSS_DSWIGLU_XCD_MAP
#define GPTOSS_DSWIGLU_XCD_MAP 0
#endif
#ifndef GPTOSS_DSWIGLU_XCD_CHUNK
#define GPTOSS_DSWIGLU_XCD_CHUNK 112
#endif
#ifndef GPTOSS_DSWIGLU_NATIVE_EXP2
#define GPTOSS_DSWIGLU_NATIVE_EXP2 0
#endif
#ifndef GPTOSS_DSWIGLU_TAIL_ROW_E8
#define GPTOSS_DSWIGLU_TAIL_ROW_E8 0
#endif
#if WRITE_DH && (GPTOSS_DSWIGLU_NATIVE_EXP2 || GPTOSS_DSWIGLU_TAIL_ROW_E8)
#error "native exp2 and tail row-scale remapping are qualified only for the production no-dH ABI"
#endif
#if GPTOSS_DSWIGLU_EXPERT_COUNTS && WRITE_DH
#error "expert-count padding omission is restricted to the production no-dH ABI"
#endif

typedef short short2v __attribute__((ext_vector_type(2)));
extern "C" __device__ float __ocml_exp2_f32(float);

#if GPTOSS_DSWIGLU_RCP_NR
// The full IEEE division lowering is eleven vector instructions. One hardware reciprocal plus one Newton step is
// sufficient to preserve every BF16 dH bit (and therefore both MXFP8 layouts and the FP32 sums of those BF16 bits)
// on signed/extreme and random GPT-OSS tests. Preserve IEEE 1/inf == 0 explicitly: the Newton residual inf*0 is NaN.
static __device__ __forceinline__ float gptoss_recip_nr(float den) {
  if (isinf(den)) return 0.0f;
  float r=__builtin_amdgcn_rcpf(den);
  return __builtin_fmaf(__builtin_fmaf(-den,r,1.0f),r,r);
}
#endif

constexpr int ROWS=32, COLS=256, LDS_STRIDE=GPTOSS_DSWIGLU_LDS_STRIDE;
constexpr int COL_TILES=H_DIM/COLS;
constexpr float LIMIT=7.0f, ALPHA=1.7020000219345093f;
constexpr float NEG_ALPHA_LOG2E=-2.4554669857025146f, FP8_MAX=448.0f;
static_assert(M_DIM%ROWS==0 && H_DIM%COLS==0 && THREADS==256 && LDS_STRIDE>=COLS);

static __device__ __forceinline__ int mx_e8(float a) {
  int e=(int)((__builtin_bit_cast(unsigned,a)>>23)&0xffu);
  return max(0,min(254,e));
}

static __device__ __forceinline__ uint16_t cvt_pair(float a,float b,int e8) {
#if NATIVE_MXFP8_CVT
  const float s=__builtin_bit_cast(float,(unsigned)e8<<23);
  short2v p={0,0};
  p=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p,a,b,s,false);
  const unsigned char *x=reinterpret_cast<const unsigned char*>(&p);
  return (uint16_t)x[0]|((uint16_t)x[1]<<8);
#else
  const float s=exp2f((float)(127-e8));
  const __hip_fp8_storage_t x=__hip_cvt_float_to_fp8(fmaxf(-FP8_MAX,fminf(FP8_MAX,a*s)),__HIP_SATFINITE,__HIP_E4M3);
  const __hip_fp8_storage_t y=__hip_cvt_float_to_fp8(fmaxf(-FP8_MAX,fminf(FP8_MAX,b*s)),__HIP_SATFINITE,__HIP_E4M3);
  return (uint16_t)x|((uint16_t)y<<8);
#endif
}

// Exact GPT-OSS dSwiGLU producer for both FC1-backward GEMM layouts. A 32x256 tile computes and BF16-rounds dH
// once, stages those live values in LDS, then emits both rowwise and columnwise 1x32 MXFP8 plus 32-row dBias
// partials. The production mailbox has no BF16 consumer, so WRITE_DH=0 omits that dead 868 MiB store; WRITE_DH=1
// retains a standalone oracle mode and has byte-identical output to the former UOp dSwiGLU + transpose-quantize.
extern "C" __global__ __launch_bounds__(THREADS) void dswiglu_bwd_fp8_dual(
#if WRITE_DH
    __hip_bfloat16* __restrict__ dh,
#endif
    __hip_fp8_storage_t* __restrict__ row_q,uint8_t* __restrict__ row_e8,
    __hip_fp8_storage_t* __restrict__ col_q,uint8_t* __restrict__ col_e8,
    float* __restrict__ bias_partial,
    const __hip_bfloat16* __restrict__ h,const __hip_bfloat16* __restrict__ dy,
    const int* __restrict__ expert_off
#if GPTOSS_DSWIGLU_EXPERT_COUNTS
    ,const int* __restrict__ expert_counts
#endif
    ) {
  __shared__ __hip_bfloat16 tile[ROWS*LDS_STRIDE];
  const int tid=threadIdx.x;
  int wgid=(int)blockIdx.x;
#if GPTOSS_DSWIGLU_XCD_MAP
  // Keep the favorable row-major logical traversal while assigning 112-workgroup
  // chunks to the eight XCDs. The tail beyond the last complete superchunk is
  // left in launch order, making this an exact permutation of [0, gridDim.x).
  constexpr int num_xcds=8,chunk=GPTOSS_DSWIGLU_XCD_CHUNK,block=num_xcds*chunk;
  const int limit=(gridDim.x/block)*block;
  if(wgid<=limit) {
    const int xcd=wgid%num_xcds,local=wgid/num_xcds;
    wgid=(local/chunk)*block+xcd*chunk+local%chunk;
  }
#endif
  const int col_tile=wgid%COL_TILES,row_tile=wgid/COL_TILES;
  const int row0=row_tile*ROWS;

#if GPTOSS_DSWIGLU_EXPERT_COUNTS
  // FC1 wgrad contracts ceil(expert_count/128)*128 rows. Match that exact boundary: dH companions beyond it are
  // unobserved by wgrad, and the corresponding dgrad rows are routing padding. Do not read stale H or write those
  // large companions, but initialize the bias partial because its reducer intentionally walks padded expert_off.
  const int padded_end=__builtin_amdgcn_readfirstlane(expert_off[32]);
  if(row0<padded_end) {
    int e=0;
    e+=(__builtin_amdgcn_readfirstlane(expert_off[e+16])<=row0)?16:0;
    e+=(__builtin_amdgcn_readfirstlane(expert_off[e+ 8])<=row0)? 8:0;
    e+=(__builtin_amdgcn_readfirstlane(expert_off[e+ 4])<=row0)? 4:0;
    e+=(__builtin_amdgcn_readfirstlane(expert_off[e+ 2])<=row0)? 2:0;
    e+=(__builtin_amdgcn_readfirstlane(expert_off[e+ 1])<=row0)? 1:0;
    const int live128_end=__builtin_amdgcn_readfirstlane(expert_off[e]+((expert_counts[e]+127)&-128));
    if(row0>=live128_end) {
      const int phys_col=col_tile*COLS+tid;
      bias_partial[(long long)row_tile*H_DIM+phys_col]=0.0f;
      return;
    }
  }
#endif

#if GPTOSS_DSWIGLU_SKIP_EMPTY_TAIL
  // The paired FC1-forward and down-dgrad producers explicitly write +0 into every row at and beyond the final
  // padded expert offset. Bypass the derivative/SFU and LDS paths for these global tail tiles while preserving all
  // five production outputs byte-for-byte.
  if(row0>=__builtin_amdgcn_readfirstlane(expert_off[32])) {
    const int phys_col=col_tile*COLS+tid;
#if WRITE_DH
#pragma unroll
    for(int r=0;r<ROWS;r++) dh[(long long)(row0+r)*H_DIM+phys_col]=(__hip_bfloat16)0.0f;
#endif
#pragma unroll
    for(int r=0;r<ROWS;r++) row_q[(long long)(row0+r)*H_DIM+phys_col]=0;
#pragma unroll
    for(int r=0;r<ROWS;r++) col_q[(long long)phys_col*M_DIM+row0+r]=0;
#if GPTOSS_DSWIGLU_TAIL_ROW_E8
    // One scale per lane covers exactly the same 32 rows x 8 scales as the original eight-lane loop.
    row_e8[(long long)(row0+tid/8)*(H_DIM/32)+col_tile*8+(tid%8)]=0;
#else
    if(tid<8) {
#pragma unroll
      for(int r=0;r<ROWS;r++) row_e8[(long long)(row0+r)*(H_DIM/32)+col_tile*8+tid]=0;
    }
#endif
    col_e8[(long long)phys_col*(M_DIM/32)+row_tile]=0;
    bias_partial[(long long)row_tile*H_DIM+phys_col]=0.0f;
    return;
  }
#endif

  // Both 128-thread halves own one 16-row half. Each thread computes one interleaved gate/linear pair, so all four
  // waves remain active during the expensive derivative while global h/dy accesses and optional dH stores coalesce.
  const int pair=tid&127,r0=(tid>>7)*16,inter=col_tile*128+pair,pc=pair*2;
#pragma unroll
  for(int rr=0;rr<16;rr++) {
    const int lr=r0+rr,row=row_tile*ROWS+lr;
    __hip_bfloat16 ab=(__hip_bfloat16)0.0f,bb=(__hip_bfloat16)0.0f;
    if(inter<REAL_INTER) {
      const uint32_t hp=reinterpret_cast<const uint32_t*>(h)[(long long)row*(H_DIM/2)+inter];
      const __hip_bfloat16 bg=__builtin_bit_cast(__hip_bfloat16,(uint16_t)hp);
      const __hip_bfloat16 bl=__builtin_bit_cast(__hip_bfloat16,(uint16_t)(hp>>16));
      const float hg=(float)bg,hl=(float)bl,ga=(float)dy[(long long)row*DY_STRIDE+inter];
      // Preserve the exact signed/intermediate order rendered by _custom_swiglu_bwd_fp8_strided. Compiling this
      // source without -ffast-math is required: reassociation changes BF16 boundary values and signed zeros.
      const float nxg=fmaxf(-hg,-LIMIT);
#if GPTOSS_DSWIGLU_NATIVE_EXP2
      // nxg >= -7, including NaN hg through fmax: exponent >= -17.19, so OCML's underflow repair is unreachable.
      const float den=1.0f+__builtin_amdgcn_exp2f(nxg*(-NEG_ALPHA_LOG2E));
#else
      const float den=1.0f+__ocml_exp2_f32(nxg*(-NEG_ALPHA_LOG2E));
#endif
#if GPTOSS_DSWIGLU_RCP_NR
      const float sig=gptoss_recip_nr(den);
#else
      const float sig=1.0f/den;
#endif
      const float nxl=fmaxf(-fmaxf(hl,-LIMIT),-LIMIT);
      const float gok=hg<LIMIT?1.0f:0.0f;
      const float lok0=hl<LIMIT?1.0f:0.0f,lok1=-LIMIT<hl?1.0f:0.0f;
      ab=(__hip_bfloat16)(ga*sig*(1.0f+(nxg*(1.0f-sig)*-ALPHA))*(1.0f-nxl)*gok);
      bb=(__hip_bfloat16)(-(nxg*sig*ga*lok1*lok0));
    }
#if WRITE_DH
    const long long out=(long long)row*H_DIM+col_tile*COLS+pc;
    reinterpret_cast<uint32_t*>(dh)[out/2]=__builtin_bit_cast(uint32_t,short2v{
      __builtin_bit_cast(short,ab),__builtin_bit_cast(short,bb)});
#endif
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
    // The final 128 physical dH columns are padding. Do not feed their zeros through LDS; every externally visible
    // tail below is still initialized explicitly, preserving the full six-output ABI byte-for-byte.
    if(inter<REAL_INTER) {
#endif
      tile[lr*LDS_STRIDE+pc]=ab;
      tile[lr*LDS_STRIDE+pc+1]=bb;
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
    }
#endif
  }
  __syncthreads();

  // One thread owns the exact 32-row MX block for one physical column.
  const int phys_col=col_tile*COLS+tid;
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
  if(phys_col<2*REAL_INTER) {
#endif
  float camax=0.0f,sum=0.0f;
#pragma unroll
  for(int r=0;r<ROWS;r++) {
    const float v=(float)tile[r*LDS_STRIDE+tid];
    camax=fmaxf(camax,fabsf(v));sum+=v;
  }
  const int ce8=mx_e8(camax);
  const long long cb=(long long)phys_col*M_DIM+row_tile*ROWS;
#pragma unroll
  for(int r=0;r<ROWS;r+=2) reinterpret_cast<uint16_t*>(&col_q[cb])[r/2]=
    cvt_pair((float)tile[r*LDS_STRIDE+tid],(float)tile[(r+1)*LDS_STRIDE+tid],ce8);
  col_e8[(long long)phys_col*(M_DIM/32)+row_tile]=(uint8_t)ce8;
  bias_partial[(long long)row_tile*H_DIM+phys_col]=sum;
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
  } else {
    const long long cb=(long long)phys_col*M_DIM+row_tile*ROWS;
#pragma unroll
    for(int r=0;r<ROWS;r+=2) reinterpret_cast<uint16_t*>(&col_q[cb])[r/2]=0;
    col_e8[(long long)phys_col*(M_DIM/32)+row_tile]=0;
    bias_partial[(long long)row_tile*H_DIM+phys_col]=0.0f;
  }
#endif

  // One thread owns one exact rowwise 1x32 MX block.
  const int lr=tid/8,sub=tid%8,local_col=sub*32;
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
  if(col_tile*COLS+local_col<2*REAL_INTER) {
#endif
  float ramax=0.0f;
#pragma unroll
  for(int c=0;c<32;c++) ramax=fmaxf(ramax,fabsf((float)tile[lr*LDS_STRIDE+local_col+c]));
  const int re8=mx_e8(ramax),row=row_tile*ROWS+lr;
  const long long rb=(long long)row*H_DIM+col_tile*COLS+local_col;
#pragma unroll
  for(int c=0;c<32;c+=2) reinterpret_cast<uint16_t*>(&row_q[rb])[c/2]=
    cvt_pair((float)tile[lr*LDS_STRIDE+local_col+c],(float)tile[lr*LDS_STRIDE+local_col+c+1],re8);
  row_e8[(long long)row*(H_DIM/32)+col_tile*8+sub]=(uint8_t)re8;
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
  } else {
    const int row=row_tile*ROWS+lr;
    const long long rb=(long long)row*H_DIM+col_tile*COLS+local_col;
#pragma unroll
    for(int c=0;c<32;c+=2) reinterpret_cast<uint16_t*>(&row_q[rb])[c/2]=0;
    row_e8[(long long)row*(H_DIM/32)+col_tile*8+sub]=0;
  }
#endif
}
