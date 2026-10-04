#define WGRAD_SAFE_LDS_PTR 1
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


static __device__ __forceinline__ long long packed_row(int row,int scale) {
  return ((long long)(scale/4)*M_DIM+row)*4+scale%4;
}
static __device__ __forceinline__ long long packed_column(int col,int row_tile) {
  return ((long long)(row_tile/4)*H_DIM+col)*4+row_tile%4;
}

// All inter-wave barriers are in the caller, outside subgroup/tail predicates.
static __device__ __forceinline__ void dual_phase(uint8_t *row_q,uint8_t *row_e8,uint8_t *col_q,uint8_t *col_e8,
    float *bias_partial,const __hip_bfloat16 *h,const int *expert_off,const int *expert_counts,const __hip_bfloat16 *dy,
    __hip_bfloat16 *tile,int row_tile,int col_tile,int stripe_row,int block_n,int live128_end,int phase) {
  if(col_tile>=COL_TILES) return;
  const int tid=threadIdx.x&255,row0=row_tile*ROWS;
  if(phase==0) {
    if(row0>=live128_end) return;
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
      const float hg=(float)bg,hl=(float)bl,ga=(float)dy[(row-stripe_row)*256+inter-block_n];
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

  } else {
    if(row0<expert_off[32] && row0>=live128_end) {
      bias_partial[(long long)row_tile*H_DIM+col_tile*COLS+tid]=0.0f;
      return;
    }
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
    row_e8[packed_row(row0+tid/8,col_tile*8+tid%8)]=0;
#else
    if(tid<8) {
#pragma unroll
      for(int r=0;r<ROWS;r++) row_e8[packed_row(row0+r,col_tile*8+tid)]=0;
    }
#endif
    col_e8[packed_column(phys_col,row_tile)]=0;
    bias_partial[(long long)row_tile*H_DIM+phys_col]=0.0f;
    return;
  }
#endif


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
  col_e8[packed_column(phys_col,row_tile)]=(uint8_t)ce8;
  bias_partial[(long long)row_tile*H_DIM+phys_col]=sum;
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
  } else {
    const long long cb=(long long)phys_col*M_DIM+row_tile*ROWS;
#pragma unroll
    for(int r=0;r<ROWS;r+=2) reinterpret_cast<uint16_t*>(&col_q[cb])[r/2]=0;
    col_e8[packed_column(phys_col,row_tile)]=0;
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
  row_e8[packed_row(row,col_tile*8+sub)]=(uint8_t)re8;
#if GPTOSS_DSWIGLU_SKIP_PAD_TAIL
  } else {
    const int row=row_tile*ROWS+lr;
    const long long rb=(long long)row*H_DIM+col_tile*COLS+local_col;
#pragma unroll
    for(int c=0;c<32;c+=2) reinterpret_cast<uint16_t*>(&row_q[rb])[c/2]=0;
    row_e8[packed_row(row,col_tile*8+sub)]=0;
  }
#endif

  }
}
#include "kittens.cuh"
using namespace kittens;
using i8 = int __attribute__((ext_vector_type(8)));
using f16 = float __attribute__((ext_vector_type(16)));
using u4 = uint32_t __attribute__((ext_vector_type(4)));
#ifndef STREAM_B
#define STREAM_B 0
#endif
constexpr int M=73728, N=3072, K=3072, REAL_K=2880, REAL_N=2880, E=32;
constexpr int BM=128, BN=256, BK=64, NT=512;
using G = kittens::group<8>;
using STA = st_fp8e4m3<BM,BK,st_16x64_s>;
using STB = st_fp8e4m3<BN,BK,st_16x64_s>;

// K64's 16x64 subtiles require a different XOR than the K128 layout.
// Both global-to-LDS offsets and native32 register loads use this involution.
__device__ __forceinline__ uint32_t swizzle64(uint32_t offset) {
  return offset ^ (((offset & 1023u) >> 8) << 4);
}
template<typename ST> __device__ __forceinline__ i8 operand(const ST &tile,int row_base) {
  const int row=row_base+(laneid()&31), col=(laneid()>>5)*16;
  const uint32_t base=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile.data));
  const uint32_t a0=base+swizzle64(row*64+col), a1=base+swizzle64(row*64+col+32);
  i8 ret;
  asm volatile("ds_read_b128 %0, %1" : "=v"(reinterpret_cast<float4 *>(&ret)[0]) : "v"(a0) : "memory");
  asm volatile("ds_read_b128 %0, %1" : "=v"(reinterpret_cast<float4 *>(&ret)[1]) : "v"(a1) : "memory");
  return ret;
}
__device__ __forceinline__ void mma(f16 &c,i8 a,i8 b,uint32_t sa,uint32_t sb) {
  c=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a,b,c,0,0,0,sa,0,sb);
}


// Preserve the control GEMM's truncating BF16 boundary before derivative math.
__device__ __forceinline__ void stage_acc(bf16 *dy,const f16 &c,int cbase) {
  const int col=cbase+(laneid()&31);
  #pragma unroll
  for(int j=0;j<4;j++) {
    const int row=4*(laneid()>>5)+j*8;
    #pragma unroll
    for(int i=0;i<4;i++) dy[(row+i)*BN+col]=base_types::convertor<bf16,float>::convert(c[j*4+i]);
  }
}

extern "C" __global__ __launch_bounds__(512,2) void fused_down_k64_dual(
    uint8_t *row_q,uint8_t *row_e8,uint8_t *col_q,uint8_t *col_e8,
    float *bias_partial,const __hip_bfloat16 *h,const int *expert_off,const int *expert_counts,fp8e4m3 *Aptr,fp8e4m3 *Bptr,fp8e8m0 *SAptr,fp8e8m0 *SBptr) {
  const int *offsets=expert_off;
  __shared__ union { STA data[2]; bf16 dy[32*256]; } astorage;
  auto &As=astorage.data;
  __shared__ union { STB data[2]; __hip_bfloat16 tile[2][32*264]; } bstorage;
  auto &Bs=bstorage.data;
  __shared__ uint32_t SA[2][BM],SB[2][BN];
  const int tid=threadIdx.x, warp=__builtin_amdgcn_readfirstlane(tid>>6), wm=warp/4, wn=warp%4;
  constexpr int TM=M/BM,TN=N/BN,WGM=8;
  const int wgid=chiplet_transform_chunked(blockIdx.x,TM*TN,8,WGM*WGM);
  const int group=wgid/(WGM*TN), first=group*WGM, group_size=min(TM-first,WGM);
  const int br=first+wgid%(WGM*TN)%group_size, bc=wgid%(WGM*TN)/group_size;
  const int bm=br*BM,bn=bc*BN;
  if(bm>=__builtin_amdgcn_readfirstlane(offsets[E])) {
    #pragma unroll 1
    for(int r=0;r<4;r++) dual_phase(row_q,row_e8,col_q,col_e8,bias_partial,h,expert_off,expert_counts,nullptr,nullptr,bm/32+r,bn/128+(tid>>8),bm,bn,0,1);
    return;
  }
  int e=0;
  e+=(offsets[e+16]<=bm)?16:0; e+=(offsets[e+8]<=bm)?8:0; e+=(offsets[e+4]<=bm)?4:0;
  e+=(offsets[e+2]<=bm)?2:0; e+=(offsets[e+1]<=bm)?1:0;
  e=__builtin_amdgcn_readfirstlane(e);
  const int live128_end=__builtin_amdgcn_readfirstlane(offsets[e]+((expert_counts[e]+127)&-128));
  if(bm>=live128_end) {
    #pragma unroll 1
    for(int r=0;r<4;r++) dual_phase(row_q,row_e8,col_q,col_e8,bias_partial,h,expert_off,expert_counts,nullptr,nullptr,bm/32+r,bn/128+(tid>>8),bm,bn,live128_end,1);
    return;
  }
  const bool active=bn+wn*64<REAL_N;
  gl<fp8e4m3,1,1,M,K> A{Aptr,nullptr,nullptr,nullptr,nullptr};
  gl<fp8e4m3,1,1,E*N,K> B{Bptr,nullptr,nullptr,nullptr,nullptr};
  i32x4 ard=make_srsrc(Aptr,uint32_t(uint64_t(M)*K),K),brd=make_srsrc(Bptr,uint32_t(uint64_t(E)*N*K),K);
  i32x4 sard=make_srsrc(SAptr,uint32_t((K/128)*M*4),M*4),sbrd=make_srsrc(SBptr,uint32_t(E*(K/128)*N*4),N*4);
  uint32_t swa[1],swb[2];
  swa[0]=(swizzle64(tid*16)/BK)*K+swizzle64(tid*16)%BK;
  #pragma unroll
  for(int i=0;i<2;i++) swb[i]=(swizzle64(tid*16+i*NT*16)/BK)*K+swizzle64(tid*16+i*NT*16)%BK;
  #define LDSBASE(PTR) __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(PTR)))
  #define LOAD(P,KK) { \
    if(tid<32) llvm_amdgcn_raw_buffer_load_lds(sard,(as3_uint32_ptr)(uintptr_t)LDSBASE(SA[P]),16,tid*16,((KK)/2*M+bm)*4,0,0); \
    if(tid<64) llvm_amdgcn_raw_buffer_load_lds(sbrd,(as3_uint32_ptr)(uintptr_t)LDSBASE(SB[P]),16,tid*16,((e*(K/128)+(KK)/2)*N+bn)*4,0,0); \
    G::load(As[P],A,{0,0,br,KK},swa,ard,Aptr,LDSBASE(As[P].data)+warp*1024); \
    G::load(Bs[P],B,{0,0,e*TN+bc,KK},swb,brd,Bptr,LDSBASE(Bs[P].data)+warp*1024); \
  }
  f16 c00={},c01={},c10={},c11={};
  LOAD(0,0);
  asm volatile("s_waitcnt vmcnt(0)\ns_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  #pragma unroll 1
  for(int kk=0;kk<REAL_K/BK;kk++) {
    const int cur=kk&1,nxt=cur^1;
    if(kk+1<REAL_K/BK) { LOAD(nxt,kk+1); }
    if(active) {
      const int shift=8*(2*(kk&1)+(laneid()>>5));
      uint32_t sa0=(SA[cur][wm*64+(laneid()&31)]>>shift)&255u;
      uint32_t sa1=(SA[cur][wm*64+32+(laneid()&31)]>>shift)&255u;
      uint32_t sb0=(SB[cur][wn*64+(laneid()&31)]>>shift)&255u;
      uint32_t sb1=(SB[cur][wn*64+32+(laneid()&31)]>>shift)&255u;
      i8 a0=operand(As[cur],wm*64),a1=operand(As[cur],wm*64+32),b0=operand(Bs[cur],wn*64);
      #if !STREAM_B
      i8 b1=operand(Bs[cur],wn*64+32);
      #endif
      asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(sa0),"+v"(sa1),"+v"(sb0),"+v"(sb1) :: "memory");
      mma(c00,a0,b0,sa0,sb0); mma(c10,a1,b0,sa1,sb0);
      #if STREAM_B
      i8 b1=operand(Bs[cur],wn*64+32);
      asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(sb1) :: "memory");
      #endif
      mma(c01,a0,b1,sa0,sb1); mma(c11,a1,b1,sa1,sb1);
    }
    if(kk+1<REAL_K/BK) { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); __builtin_amdgcn_s_barrier(); }
  }
  #undef LOAD
  // Complete every LDS read before recycling the double-buffered matrix storage.
  asm volatile("s_waitcnt vmcnt(0)\ns_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  static_assert(sizeof(astorage)==32*256*2 && sizeof(bstorage)==2*32*264*2);
  bf16 *dy=astorage.dy;
  __hip_bfloat16 *tile=bstorage.tile[tid>>8];
  #define STRIPE(C0,C1,OWNER,ROW) { \
    if(wm==(OWNER) && active) { stage_acc(dy,C0,wn*64); stage_acc(dy,C1,wn*64+32); } \
    __builtin_amdgcn_s_barrier(); \
    dual_phase(row_q,row_e8,col_q,col_e8,bias_partial,h,expert_off,expert_counts,reinterpret_cast<__hip_bfloat16 *>(dy),tile,(bm+(ROW))/32,bn/128+(tid>>8),bm+(ROW),bn,live128_end,0); \
    __builtin_amdgcn_s_barrier(); \
    dual_phase(row_q,row_e8,col_q,col_e8,bias_partial,h,expert_off,expert_counts,reinterpret_cast<__hip_bfloat16 *>(dy),tile,(bm+(ROW))/32,bn/128+(tid>>8),bm+(ROW),bn,live128_end,1); \
    __builtin_amdgcn_s_barrier(); \
  }
  STRIPE(c00,c01,0,0);
  STRIPE(c10,c11,0,32);
  STRIPE(c00,c01,1,64);
  STRIPE(c10,c11,1,96);
  #undef STRIPE
  #undef LDSBASE
}
