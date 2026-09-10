#include "kittens.cuh"
using namespace kittens;
constexpr int OWN=ATTN_N<128?64:128;
constexpr int WARPS=OWN/16;
using G=group<WARPS>;
struct alignas(1024) ST {
  unsigned char data[128*128];
  __device__ static unsigned swizzle(int2 rc) {return (rc.x*128+rc.y)^(((rc.x>>1)&7)<<4)^((rc.x&16)<<2);}
};
using SMALL=ST;
using RT=rt_fp8e4m3<16,128>;
using CT=rt<fp8e4m3,16,128,col_l,rt_16x128_s>;
using ACC=rt_fl<16,128,col_l,rt_16x16_s>;
using I8=int __attribute__((ext_vector_type(8)));
using F4=float __attribute__((ext_vector_type(4)));
using F2=float __attribute__((ext_vector_type(2)));
using Output=std::conditional_t<OUTPUT_BF16,bf16,float>;
__device__ int wave_id() {return __builtin_amdgcn_readfirstlane(warpid());}
__device__ float uniform(float x) {return __uint_as_float(__builtin_amdgcn_readfirstlane(__float_as_uint(x)));}

template<int A,int B,typename X,typename Y> __device__ void dot(float2* out,const X& x,const Y& y) {
  *(F4*)out=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(*(const I8*)x.tiles[0][0].data,
    *(const I8*)y.tiles[0][0].data,*(F4*)out,A,B,0,0,0,0);
}
template<typename T,typename S> __device__ void rows(T& x,S& s,int row) {
  #pragma unroll
  for(int i=0;i<2;i++) {
    unsigned addr=(unsigned)(uintptr_t)s.data+s.swizzle({row+laneid()%16,laneid()/16*16+i*64});
    asm volatile("ds_read_b128 %0, %1" : "=&v"(*(float4*)&x.tiles[0][0].data[i*4]) : "v"(addr) : "memory");
  }
}
__device__ void cols(CT& x,const ST& s,int col) {
  int r=laneid()/16*16+(laneid()%16)/2,c=col+(laneid()%2)*8;
  unsigned a=(unsigned)(uintptr_t)s.data+s.swizzle({r,c});
  asm volatile("ds_read_b64_tr_b8 %0, %2\nds_read_b64_tr_b8 %1, %2 offset:8192"
    : "=&v"(*(float2*)&x.tiles[0][0].data[0]),"=&v"(*(float2*)&x.tiles[0][0].data[4]) : "v"(a) : "memory");
  a^=1088;
  asm volatile("ds_read_b64_tr_b8 %0, %2\nds_read_b64_tr_b8 %1, %2 offset:8192"
    : "=&v"(*(float2*)&x.tiles[0][0].data[2]),"=&v"(*(float2*)&x.tiles[0][0].data[6]) : "v"(a) : "memory");
}
template<int R,typename S> __device__ void stage(S& s,const unsigned char* p,int pos,int head,int batch,int heads) {
  if constexpr(ATTN_N<128) {
    using V=unsigned __attribute__((ext_vector_type(4)));
    for(int i=threadIdx.x;i<R*8;i+=WARPS*64) {
      int r=i/8,c=i%8*16;
      V value={};
      if(pos+r<ATTN_N) value=*(const V*)(p+((batch*ATTN_N+pos+r)*heads+head)*128+c);
      *(V*)((unsigned char*)s.data+s.swizzle({r,c}))=value;
    }
  } else {
    auto resource=make_srsrc(p+((batch*ATTN_N+pos)*heads+head)*128,heads*128*128);
    #pragma unroll
    for(int i=0;i<128*128/(WARPS*64*16);i++) {
      int off=threadIdx.x*16+i*WARPS*64*16,r=off/128,c=off%128;
      unsigned global_off=r*heads*128+(s.swizzle({r,c})%128);
      auto ptr=(as3_uint32_ptr)((uintptr_t)s.data+wave_id()*1024+i*WARPS*1024);
      llvm_amdgcn_raw_buffer_load_lds(resource,ptr,16,global_off,0,0,static_cast<int>(coherency::cache_all));
    }
  }
}
__device__ void global_rows(RT& out,const unsigned char* ptr,int pos,int head,int batch,int heads) {
  using V=unsigned __attribute__((ext_vector_type(4)));
  #pragma unroll
  for(int i=0;i<2;i++) {
    int r=pos+laneid()%16,c=laneid()/16*16+i*64;
    *(V*)&out.tiles[0][0].data[i*4]=*(const V*)(ptr+((batch*ATTN_N+r)*heads+head)*128+c);
  }
}
__device__ void stage_stats(float* shared,const float* lse,const float* delta,int pos,int h,int batch) {
  if(wave_id()<4) {
    int row=threadIdx.x%128;
    const float* p=(wave_id()<2?lse:delta)+(batch*ATTN_H+h)*ATTN_N+pos;
    if constexpr(ATTN_N<128) shared[threadIdx.x]=p[row<ATTN_N?row:ATTN_N-1];
    else {
      auto resource=make_srsrc(p,128*sizeof(float));
      auto ptr=(as3_uint32_ptr)((uintptr_t)shared+wave_id()*256);
      llvm_amdgcn_raw_buffer_load_lds(resource,ptr,4,row*4,0,0,static_cast<int>(coherency::cache_all));
    }
  }
}
template<bool E5> __device__ unsigned encode4(float (&x)[4]) {
  #pragma unroll
  for(int t=0;t<4;t++) x[t]=__builtin_amdgcn_fmed3f(x[t],E5?-57344.f:-448.f,E5?57344.f:448.f);
  unsigned result;
  if constexpr(E5) {
    result=__builtin_amdgcn_cvt_pk_bf8_f32(x[0],x[1],0,false);
    return __builtin_amdgcn_cvt_pk_bf8_f32(x[2],x[3],result,true);
  } else {
    result=__builtin_amdgcn_cvt_pk_fp8_f32(x[0],x[1],0,false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(x[2],x[3],result,true);
  }
}
__device__ void body(Output* dq,Output* dk,Output* dv,float* amax,float* next_amax,
 const unsigned char* q,const unsigned char* k,const unsigned char* v,const unsigned char* dout,
 const float* delta,const float* lse,const float* scales,ST (&bs)[2],ST (&dbs)[2],SMALL& ps,SMALL& ds,ST& ks,float (&stats)[2][256],float* maxima,int bx,int bh,int bb) {
  constexpr int N=ATTN_N,H=ATTN_H,HK=ATTN_H_KV;
  int group=bx%(N/OWN),base=group*OWN,wr=wave_id()*16,h=bh,batch=bb,kh=h/(H/HK);
  float vs=uniform(scales[0]),dos=uniform(scales[1]),p_scale=uniform(scales[2]),s_scale=uniform(scales[3]);
  float pi=uniform(1.f/p_scale),si=uniform(1.f/s_scale);
  constexpr float grad_scale=0.08838834764831845f/0.3570958286295132f;
  RT ar,dr;
  global_rows(ar,k,base+wr,kh,batch,HK);
  global_rows(dr,v,base+wr,kh,batch,HK);
  stage<128>(ks,k,base,kh,batch,HK);
  if constexpr(N<128) {
    for(int i=threadIdx.x;i<128*128/4;i+=WARPS*64) ((unsigned*)ds.data)[i]=0;
  }
  auto dq_resource=make_srsrc(dq,ATTN_B*N*H*128*sizeof(Output));
  int first=base,end=N;
  stage<128>(bs[0],q,first,h,batch,H);
  stage<128>(dbs[0],dout,first,h,batch,H);
  stage_stats(stats[0],lse,delta,first,h,batch);
  ACC g0,g1;zero(g0);zero(g1);
  float pmax_lane[4]={},smax_lane[4]={};
  for(int pos=first,tic=0;pos<end;pos+=128,tic^=1) {
    __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __syncthreads(); __builtin_amdgcn_sched_barrier(0);
    ST& b=bs[tic];ST& db=dbs[tic];
    if(pos+128<end) {
      stage<128>(bs[tic^1],q,pos+128,h,batch,H);
      stage<128>(dbs[tic^1],dout,pos+128,h,batch,H);
      stage_stats(stats[tic^1],lse,delta,pos+128,h,batch);
    }
    {
      constexpr int CLUSTER=4;
      #pragma unroll
      for(int chunk=0;chunk<8;chunk+=CLUSTER) {
        RT br[CLUSTER],dbreg[CLUSTER];
        float2 pvreg[CLUSTER][2]={},dpreg[CLUSTER][2]={};
        #pragma unroll
        for(int z=0;z<CLUSTER;z++) {rows(br[z],b,(chunk+z)*16);rows(dbreg[z],db,(chunk+z)*16);}
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int z=0;z<CLUSTER;z++) {
          dot<0,0>(pvreg[z],ar,br[z]);
          dot<0,1>(dpreg[z],dr,dbreg[z]);
        }
        #pragma unroll
        for(int z=0;z<CLUSTER;z++) {
          int j=chunk+z;
          auto& p=pvreg[z];auto& dp=dpreg[z];
        float svalues[4],pvalues[4];
        int qi=pos+j*16+laneid()%16;
        float lv=stats[tic][j*16+laneid()%16]*1.4426950408889634f;
        float d=stats[tic][128+j*16+laneid()%16];
        #pragma unroll
        for(int t=0;t<4;t+=2) {
          F2 score=*(F2*)((float*)p+t)-F2{lv,lv};
          F2 pv={__builtin_amdgcn_exp2f(score[0]),__builtin_amdgcn_exp2f(score[1])};
          #pragma unroll
          for(int u=0;u<2;u++) {
            int ki=base+wr+laneid()/16*4+t+u;
            asm volatile("" : "+v"(pv[u]));
            pv[u]=pos!=base || ki<=qi?pv[u]:0.f;
            if constexpr(N<128) if(qi>=N || ki>=N) pv[u]=0.f;
          }
          F2 sv=pv*__builtin_elementwise_fma(*(F2*)((float*)dp+t),F2{vs*dos*si,vs*dos*si},F2{-d*si,-d*si});
          pv*=F2{pi,pi};
          #pragma unroll
          for(int u=0;u<2;u++) {
            pmax_lane[t+u]=fmaxf(pmax_lane[t+u],pv[u]);smax_lane[t+u]=fmaxf(smax_lane[t+u],fabsf(sv[u]));
          }
          *(F2*)(svalues+t)=sv;
          *(F2*)(pvalues+t)=pv;
        }
        int r=wr+(laneid()/16)*4,c=j*16+laneid()%16;
        *(unsigned*)((unsigned char*)ds.data+ds.swizzle({c,r}))=encode4<true>(svalues);
        *(unsigned*)((unsigned char*)ps.data+ps.swizzle({c,r}))=encode4<false>(pvalues);
      }
      }
    }
    __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);
    {
      CT sr,pr;cols(sr,ds,wr);cols(pr,ps,wr);
      RT sq;rows(sq,ds,wr);
      __builtin_amdgcn_sched_barrier(0);asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");__builtin_amdgcn_sched_barrier(0);
      CT kr,operand,operandv;
      cols(kr,ks,0);cols(operand,b,0);cols(operandv,db,0);
      #pragma unroll
      for(int dim=0;dim<8;dim++) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(8)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        float2 grad[2]={};dot<1,0>(grad,sq,kr);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<7) cols(kr,ks,(dim+1)*16);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<7) asm volatile("s_waitcnt lgkmcnt(8)" ::: "memory");
        else asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        dot<1,0>(g0.tiles[0][dim].data,sr,operand);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<7) cols(operand,b,(dim+1)*16);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<7) asm volatile("s_waitcnt lgkmcnt(8)" ::: "memory");
        else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        dot<0,1>(g1.tiles[0][dim].data,pr,operandv);
        __builtin_amdgcn_sched_barrier(0);
        if(dim<7) cols(operandv,db,(dim+1)*16);
        #pragma unroll
        for(int pair=0;pair<2;pair++) {
          float x=((float*)grad)[pair*2]*s_scale*grad_scale,y=((float*)grad)[pair*2+1]*s_scale*grad_scale;
          unsigned idx=((batch*H+h)*(N/16)+(pos+wr)/16)*2048+laneid()*2;
          if constexpr(OUTPUT_BF16) {
            unsigned packed;
            asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(packed) : "v"(x),"v"(y));
            asm volatile("buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen offset:%3"
              :: "v"(packed),"v"(idx*2),"s"(dq_resource),"n"(dim*512+pair*256) : "memory");
          } else {idx+=dim*256+pair*128;atomicAdd((float*)dq+idx,x);atomicAdd((float*)dq+idx+1,y);}
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<8;j++)
  #pragma unroll
  for(int t=0;t<4;t++) {
    int r=(laneid()/16)*4+t,c=j*16+laneid()%16,idx=((batch*N+base+wr+r)*H+h)*128+c;
    dk[idx]=((float*)g0.tiles[0][j].data)[t]*s_scale*grad_scale;
    dv[idx]=((float*)g1.tiles[0][j].data)[t]*p_scale*dos;
  }
  {
    float pm=fmaxf(fmaxf(pmax_lane[0],pmax_lane[1]),fmaxf(pmax_lane[2],pmax_lane[3]));
    float sm=fmaxf(fmaxf(smax_lane[0],smax_lane[1]),fmaxf(smax_lane[2],smax_lane[3]));
    pm*=p_scale;sm*=s_scale;
    for(int off=32;off;off/=2) {pm=fmaxf(pm,__shfl_xor(pm,off));sm=fmaxf(sm,__shfl_xor(sm,off));}
    if(laneid()==0) {
      maxima[wave_id()*2]=pm;
      maxima[wave_id()*2+1]=sm;
    }
    __syncthreads();
    if(threadIdx.x<OWN/64) {
      float pmax=0,smax=0;
      #pragma unroll
      for(int i=0;i<4;i++) {
        pmax=fmaxf(pmax,maxima[threadIdx.x*8+i*2]);
        smax=fmaxf(smax,maxima[threadIdx.x*8+i*2+1]);
      }
      int idx=((batch*H+h)*(N/64)+group*(OWN/64)+threadIdx.x)*2;
      amax[idx]=pmax;amax[idx+1]=smax;
      atomicMax((unsigned*)next_amax,__float_as_uint(pmax));
      atomicMax((unsigned*)next_amax+1,__float_as_uint(smax));
    }
  }
}
extern "C" __global__ __launch_bounds__(WARPS*64) __attribute__((amdgpu_waves_per_eu(2,2))) void fa_fp8_backward(Output* dq,Output* dk,Output* dv,float* amax,float* next_amax,
 const unsigned char* q,const unsigned char* k,const unsigned char* v,const unsigned char* dout,
 const float* delta,const float* lse,const float* scales) {
  __shared__ SMALL ps,ds;
  __shared__ ST bs[2],dbs[2],ks;
  __shared__ float stats[2][256];
  __shared__ float maxima[WARPS*2];
  constexpr int GX=ATTN_N/OWN,TOTAL=GX*ATTN_H*ATTN_B;
  constexpr int HEAD_GROUP=ATTN_H%4==0?4:ATTN_H%2==0?2:1;
  int gid=(blockIdx.z*ATTN_H+blockIdx.y)*GX+blockIdx.x;
  if constexpr(TOTAL%8==0) gid=(gid%8)*(TOTAL/8)+gid/8;
  int bx=(gid/HEAD_GROUP)%GX,bh=((gid/(GX*HEAD_GROUP))*HEAD_GROUP+gid%HEAD_GROUP)%ATTN_H,bb=gid/(GX*ATTN_H);
  body(dq,dk,dv,amax,next_amax,q,k,v,dout,delta,lse,scales,bs,dbs,ps,ds,ks,stats,maxima,bx,bh,bb);
}
