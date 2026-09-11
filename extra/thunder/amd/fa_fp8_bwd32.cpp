#include "kittens.cuh"
// Eight waves own 256 K/V rows and replay 64 Q rows at a time. P/dS are built
// once for all five products; dQ uses packed BF16 atomics across K/V workgroups.
using namespace kittens;
using I8=int __attribute__((ext_vector_type(8)));
using F16=float __attribute__((ext_vector_type(16)));
using F2=float __attribute__((ext_vector_type(2)));
using Output=std::conditional_t<OUTPUT_BF16,bf16,float>;
template<int R,int C> struct alignas(1024) Tile {
  unsigned char data[R*C];
  __device__ static unsigned swizzle(unsigned r,unsigned c) {
    // 128-column panels let the 32x32 MFMA transpose loads share one base.
    return (c/128*(R*128)+r*128+c%128)^((r&6)<<3)^((r&16)<<2);
  }
};
__device__ int wave() {return __builtin_amdgcn_readfirstlane(warpid());}
__device__ float uniform(float x) {return __uint_as_float(__builtin_amdgcn_readfirstlane(__float_as_uint(x)));}
template<int A,int B> __device__ void dot(F16& out,I8 x,I8 y) {
  out=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(x,y,out,A,B,0,0,0,0);
}
template<typename T> __device__ void rows(I8& out,T& tile,unsigned row,int col) {
  #pragma unroll
  for(int i=0;i<2;i++) {
    unsigned addr=(unsigned)(uintptr_t)tile.data+tile.swizzle(row+laneid()%32,col+laneid()/32*16+i*32);
    asm volatile("ds_read_b128 %0, %1" : "=&v"(*(float4*)((int*)&out+i*4)) : "v"(addr) : "memory");
  }
}
template<typename T> __device__ void cols(I8& out,T& tile,int row,int col) {
  int r=row+laneid()/32*16+(laneid()%16)/2,c=col+laneid()/16%2*16+(laneid()%2)*8;
  unsigned addr=(unsigned)(uintptr_t)tile.data+tile.swizzle(r,c);
  // The supplied row has bit 3 clear, so the second eight rows are +1024.
  asm volatile("ds_read_b64_tr_b8 %0, %4\nds_read_b64_tr_b8 %1, %4 offset:4096\n"
               "ds_read_b64_tr_b8 %2, %4 offset:1024\nds_read_b64_tr_b8 %3, %4 offset:5120"
    : "=&v"(*(float2*)((int*)&out)),"=&v"(*(float2*)((int*)&out+4)),
      "=&v"(*(float2*)((int*)&out+2)),"=&v"(*(float2*)((int*)&out+6)) : "v"(addr) : "memory");
}
template<int R> __device__ void stage(Tile<R,128>& tile,const unsigned char* p,int pos,int h,int batch,int heads) {
  auto resource=make_srsrc(p+((batch*ATTN_N+pos)*heads+h)*128,heads*R*128);
  #pragma unroll
  for(int i=0;i<R*128/(512*16);i++) {
    int off=threadIdx.x*16+i*512*16,r=off/128,c=off%128;
    unsigned global_off=r*heads*128+tile.swizzle(r,c)%128;
    auto ptr=(as3_uint32_ptr)((uintptr_t)tile.data+wave()*1024+i*8192);
    llvm_amdgcn_raw_buffer_load_lds(resource,ptr,16,global_off,0,0,static_cast<int>(coherency::cache_all));
  }
}
__device__ void stage_stats(float* shared,const float* lse,const float* delta,int pos,int h,int batch) {
  if(wave()<2) {
    const float* p=(wave()==0?lse:delta)+(batch*ATTN_H+h)*ATTN_N+pos;
    auto resource=make_srsrc(p,64*sizeof(float));
    auto ptr=(as3_uint32_ptr)((uintptr_t)shared+wave()*256);
    llvm_amdgcn_raw_buffer_load_lds(resource,ptr,4,laneid()*4,0,0,static_cast<int>(coherency::cache_all));
  }
}
__device__ void wait_lds() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}
template<bool E5> __device__ unsigned encode4(float (&x)[4]) {
  if constexpr(E5) {
    unsigned a=__builtin_amdgcn_cvt_pk_bf8_f32(x[0],x[1],0,false);
    return __builtin_amdgcn_cvt_pk_bf8_f32(x[2],x[3],a,true);
  } else {
    unsigned a=__builtin_amdgcn_cvt_pk_fp8_f32(x[0],x[1],0,false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(x[2],x[3],a,true);
  }
}
extern "C" __global__ __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(2,2))) void fa_fp8_backward(
 Output* dq,Output* dk,Output* dv,float* amax,float* next_amax,const unsigned char* q,const unsigned char* k,
 const unsigned char* v,const unsigned char* dout,const float* delta,const float* lse,const float* scales) {
  // FP16_OVFL also saturates FP8/BF8 conversions, avoiding a clamp for every P/dS value.
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1" ::: "memory");
  constexpr int N=ATTN_N,H=ATTN_H,HK=ATTN_H_KV,GX=N/256,TOTAL=GX*H*ATTN_B;
  constexpr int HEAD_GROUP=H%4==0?4:H%2==0?2:1;
  int gid=(blockIdx.z*H+blockIdx.y)*GX+blockIdx.x;
  if constexpr(TOTAL%8==0) gid=(gid%8)*(TOTAL/8)+gid/8;
  int bx=(gid/HEAD_GROUP)%GX,h=((gid/(GX*HEAD_GROUP))*HEAD_GROUP+gid%HEAD_GROUP)%H,batch=gid/(GX*H);
  int base=bx*256,wr=wave()*32,kh=h/(H/HK);
  __shared__ Tile<256,128> ks,vs;
  __shared__ Tile<64,128> qs[2],dos[2];
  __shared__ Tile<64,256> ps,ds;
  __shared__ float stats[2][128],maxima[16];
  float vd=uniform(scales[0]),dod=uniform(scales[1]),pd=uniform(scales[2]),sd=uniform(scales[3]);
  float pi=uniform(1.f/pd),si=uniform(1.f/sd);
  float qscale=uniform(sd*(0.08838834764831845f/0.3570958286295132f)),vscale=uniform(pd*dod);
  auto dq_resource=make_srsrc(dq,ATTN_B*N*H*128*sizeof(Output));
  stage(ks,k,base,kh,batch,HK);stage(vs,v,base,kh,batch,HK);
  stage(qs[0],q,base,h,batch,H);stage(dos[0],dout,base,h,batch,H);
  stage_stats(stats[0],lse,delta,base,h,batch);
  F16 gk[4]={},gv[4]={};
  float pmax=0,smax=0;
  for(int pos=base,tic=0;pos<N;pos+=64,tic^=1) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __syncthreads();__builtin_amdgcn_sched_barrier(0);
    auto& qt=qs[tic];auto& dt=dos[tic];
    if(pos+64<N) {
      stage(qs[tic^1],q,pos+64,h,batch,H);stage(dos[tic^1],dout,pos+64,h,batch,H);
      stage_stats(stats[tic^1],lse,delta,pos+64,h,batch);
    }
    #pragma unroll
    for(int j=0;j<2;j++) {
      F16 score={},dp={};
      #pragma unroll
      for(int kk=0;kk<2;kk++) {
        I8 a,b;rows(a,ks,wr,kk*64);rows(b,qt,j*32,kk*64);wait_lds();dot<0,0>(score,a,b);
      }
      #pragma unroll
      for(int kk=0;kk<2;kk++) {
        I8 a,b;rows(a,vs,wr,kk*64);rows(b,dt,j*32,kk*64);wait_lds();dot<0,1>(dp,a,b);
      }
      int qi=pos+j*32+laneid()%32;
      float lv=stats[tic][j*32+laneid()%32]*1.4426950408889634f,d=stats[tic][64+j*32+laneid()%32];
      #pragma unroll
      for(int m=0;m<4;m++) {
        float pv4[4],sv4[4];
        #pragma unroll
        for(int t=0;t<4;t+=2) {
          F2 x=*(F2*)((float*)&score+m*4+t)-F2{lv,lv};
          F2 p={__builtin_amdgcn_exp2f(x[0]),__builtin_amdgcn_exp2f(x[1])};
          #pragma unroll
          for(int u=0;u<2;u++) {
            asm volatile("" : "+v"(p[u]));
            p[u]=base+wr+laneid()/32*4+m*8+t+u<=qi?p[u]:0.f;
          }
          F2 s=p*__builtin_elementwise_fma(*(F2*)((float*)&dp+m*4+t),F2{vd*dod,vd*dod},F2{-d,-d});
          // Keep each running amax in one register without scalar fmax chains.
          asm("v_max3_f32 %0, %1, %2, %3" : "=v"(pmax) : "v"(pmax),"v"(p[0]),"v"(p[1]));
          asm("v_max3_f32 %0, %1, |%2|, |%3|" : "=v"(smax) : "v"(smax),"v"(s[0]),"v"(s[1]));
          *(F2*)(pv4+t)=p*F2{pi,pi};*(F2*)(sv4+t)=s*F2{si,si};
        }
        int r=j*32+laneid()%32,c=wr+laneid()/32*4+m*8;
        *(unsigned*)(ps.data+ps.swizzle(r,c))=encode4<false>(pv4);
        *(unsigned*)(ds.data+ds.swizzle(r,c))=encode4<true>(sv4);
      }
    }
    wait_lds();__syncthreads();__builtin_amdgcn_sched_barrier(0);
    {
      I8 sr,pr;cols(sr,ds,0,wr);cols(pr,ps,0,wr);wait_lds();
      I8 qo,doo;cols(qo,qt,0,0);cols(doo,dt,0,0);
      #pragma unroll
      for(int dim=0;dim<4;dim++) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        dot<1,0>(gk[dim],sr,qo);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<3) cols(qo,qt,0,(dim+1)*32);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<3) asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        dot<0,1>(gv[dim],pr,doo);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<3) cols(doo,dt,0,(dim+1)*32);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    {
      F16 grad={};
      int qr=wave()%2*32,dim=wave()/2*32;
      #pragma unroll
      for(int kk=0;kk<4;kk++) {
        I8 sq,kr;rows(sq,ds,qr,kk*64);cols(kr,ks,kk*64,dim);wait_lds();dot<1,0>(grad,sq,kr);
      }
      #pragma unroll
      for(int pair=0;pair<8;pair++) {
        F2 xy;
        asm("v_pk_mul_f32 %0, %1, %2" : "=v"(xy) : "v"(*(F2*)((float*)&grad+pair*2)),"s"(F2{qscale,qscale}));
        int row=pos+qr+laneid()/32*4,col=dim+laneid()%32;
        unsigned idx=((batch*H+h)*(N/16)+row/16+pair/4)*2048+col/16*256+row%16/4*32+col%16*2;
        if constexpr(OUTPUT_BF16) {
          unsigned packed;
          asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(packed) : "v"(xy[0]),"v"(xy[1]));
          asm volatile("buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen offset:%3"
            :: "v"(packed),"v"(idx*2),"s"(dq_resource),"n"(pair%4/2*128+pair%2*256) : "memory");
        } else {idx+=pair%4/2*64+pair%2*128;atomicAdd((float*)dq+idx,xy[0]);atomicAdd((float*)dq+idx+1,xy[1]);}
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<4;j++) {
    #pragma unroll
    for(int t=0;t<16;t++) {
      int r=base+wr+laneid()/32*4+t/4*8+t%4,c=j*32+laneid()%32,idx=((batch*N+r)*H+h)*128+c;
      dk[idx]=gk[j][t]*qscale;dv[idx]=gv[j][t]*vscale;
    }
  }
  for(int off=32;off;off/=2) {pmax=fmaxf(pmax,__shfl_xor(pmax,off));smax=fmaxf(smax,__shfl_xor(smax,off));}
  if(laneid()==0) {maxima[wave()*2]=pmax;maxima[wave()*2+1]=smax;}
  __syncthreads();
  if(threadIdx.x<4) {
    int i=threadIdx.x*4,idx=((batch*H+h)*(N/64)+bx*4+threadIdx.x)*2;
    pmax=fmaxf(maxima[i],maxima[i+2]);smax=fmaxf(maxima[i+1],maxima[i+3]);
    amax[idx]=pmax;amax[idx+1]=smax;
    atomicMax((unsigned*)next_amax,__float_as_uint(pmax));atomicMax((unsigned*)next_amax+1,__float_as_uint(smax));
  }
}
