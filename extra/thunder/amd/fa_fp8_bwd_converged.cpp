#include "kittens.cuh"
using namespace kittens;
constexpr int OWN=ATTN_N<128?64:128;
constexpr int WARPS=OWN/16;
using G=group<WARPS>;
using ST=st_fp8e4m3<128,128,st_16x128_s>;
using SMALL=st_fp8e4m3<128,128,st_16x128_s>;
using RT=rt_fp8e4m3<16,128>;
using CT=rt<fp8e4m3,16,128,col_l,rt_16x128_s>;
using ACC=rt_fl<16,128,col_l,rt_16x16_s>;
using I8=int __attribute__((ext_vector_type(8)));
using F4=float __attribute__((ext_vector_type(4)));

template<int A,int B,typename X,typename Y> __device__ void dot(float2* out,const X& x,const Y& y) {
  *(F4*)out=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(*(const I8*)x.tiles[0][0].data,
    *(const I8*)y.tiles[0][0].data,*(F4*)out,A,B,0,0,0,0);
}
template<typename T,typename S> __device__ void rows(T& x,S& s,int row) {
  load(x,subtile_inplace<16,128>(s,{row/16,0}));

}
__device__ void cols(CT& x,const ST& s,int col) {
  load(x,s,col);

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
    gl<fp8e4m3,-1,-1,-1,-1> src((fp8e4m3*)p,batch+1,ATTN_N,heads,128);
    G::load<1,false>(s,src,{batch,pos/R,head,0});
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
template<bool KV> __device__ void body(float* dq,float* dk,float* dv,float* amax,float* next_amax,
 const unsigned char* q,const unsigned char* k,const unsigned char* v,const unsigned char* dout,
 const float* delta,const float* lse,const float* scales,ST (&bs)[2],ST (&dbs)[2],SMALL& ps,SMALL& ds,float* maxima,int bx,int bh,int bb) {
  constexpr int N=ATTN_N,H=ATTN_H,HK=ATTN_H_KV;
  int group=bx%(N/OWN),base=group*OWN,wr=warpid()*16,h=bh,batch=bb,kh=h/(H/HK);
  float vs=scales[0],dos=scales[1],p_scale=scales[2],s_scale=scales[3],pi=1.f/p_scale,si=1.f/s_scale;
  constexpr float grad_scale=0.08838834764831845f/0.3570958286295132f;
  RT ar,dr;
  global_rows(ar,KV?k:q,base+wr,KV?kh:h,batch,KV?HK:H);
  global_rows(dr,KV?v:dout,base+wr,KV?kh:h,batch,KV?HK:H);
  int first=KV?base:0,end=KV?N:base+OWN;
  stage<128>(bs[0],KV?q:k,first,KV?h:kh,batch,KV?H:HK);
  stage<128>(dbs[0],KV?dout:v,first,KV?h:kh,batch,KV?H:HK);
  ACC g0,g1;zero(g0);zero(g1);
  float pmax_lane[4]={},smax_lane[4]={};
  float q_lse[4],q_delta[4];
  if constexpr(!KV) {
    #pragma unroll
    for(int t=0;t<4;t++) {
      int qi=base+wr+laneid()/16*4+t;
      q_lse[t]=lse[(batch*H+h)*N+qi]*1.4426950408889634f;
      q_delta[t]=delta[(batch*H+h)*N+qi];
    }
  }
  for(int pos=first,tic=0;pos<end;pos+=128,tic^=1) {
    __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __syncthreads(); __builtin_amdgcn_sched_barrier(0);
    ST& b=bs[tic];ST& db=dbs[tic];
    if(pos+128<end) {
      stage<128>(bs[tic^1],KV?q:k,pos+128,KV?h:kh,batch,KV?H:HK);
      stage<128>(dbs[tic^1],KV?dout:v,pos+128,KV?h:kh,batch,KV?H:HK);
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
          if constexpr(KV) dot<0,1>(dpreg[z],dr,dbreg[z]);else dot<1,0>(dpreg[z],dr,dbreg[z]);
        }
        #pragma unroll
        for(int z=0;z<CLUSTER;z++) {
          int j=chunk+z;
          auto& p=pvreg[z];auto& dp=dpreg[z];
        float svalues[4],pvalues[4];
        #pragma unroll
        for(int t=0;t<4;t++) {
          int r=(laneid()/16)*4+t,c=j*16+laneid()%16;
          int qi=KV?pos+c:base+wr+r,ki=KV?base+wr+r:pos+c;
          int safe_qi=N<128 && qi>=N?N-1:qi;
          float lv=KV?lse[(batch*H+h)*N+safe_qi]*1.4426950408889634f:q_lse[t];
          float dv=KV?delta[(batch*H+h)*N+safe_qi]:q_delta[t];
          float pv=__builtin_amdgcn_exp2f(((float*)p)[t]-lv);
          asm volatile("" : "+v"(pv));
          pv=pos!=base || ki<=qi?pv:0.f;
          if constexpr(N<128) if(qi>=N || ki>=N) pv=0.f;
          float sv=pv*__builtin_fmaf(((float*)dp)[t],vs*dos,-dv);
          if constexpr(!KV) {pmax_lane[t]=fmaxf(pmax_lane[t],pv);smax_lane[t]=fmaxf(smax_lane[t],fabsf(sv));}
          svalues[t]=sv*si;
          if constexpr(KV) pvalues[t]=pv*pi;
        }
        int r=wr+(laneid()/16)*4,c=j*16+laneid()%16;
        *(unsigned*)((unsigned char*)ds.data+ds.swizzle({c,r}))=encode4<true>(svalues);
        if constexpr(KV) *(unsigned*)((unsigned char*)ps.data+ps.swizzle({c,r}))=encode4<false>(pvalues);
      }
      }
    }
    __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
    {
      CT sr,pr;cols(sr,ds,wr);if constexpr(KV) cols(pr,ps,wr);
      __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      #pragma unroll
      for(int dim=0;dim<8;dim+=2) {
        CT operand[2],operandv[2];
        #pragma unroll
        for(int z=0;z<2;z++) {
          cols(operand[z],b,(dim+z)*16);
          if constexpr(KV) cols(operandv[z],db,(dim+z)*16);
        }
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int z=0;z<2;z++) {
          dot<1,0>(g0.tiles[0][dim+z].data,sr,operand[z]);
          if constexpr(KV) dot<0,1>(g1.tiles[0][dim+z].data,pr,operandv[z]);
        }
      }
    }
  }
  #pragma unroll
  for(int j=0;j<8;j++)
  #pragma unroll
  for(int t=0;t<4;t++) {
    int r=(laneid()/16)*4+t,c=j*16+laneid()%16,idx=((batch*N+base+wr+r)*H+h)*128+c;
    (KV?dk:dq)[idx]=((float*)g0.tiles[0][j].data)[t]*s_scale*grad_scale;
    if constexpr(KV) dv[idx]=((float*)g1.tiles[0][j].data)[t]*p_scale*dos;
  }
  if constexpr(!KV) {
    float pm=fmaxf(fmaxf(pmax_lane[0],pmax_lane[1]),fmaxf(pmax_lane[2],pmax_lane[3]));
    float sm=fmaxf(fmaxf(smax_lane[0],smax_lane[1]),fmaxf(smax_lane[2],smax_lane[3]));
    for(int off=32;off;off/=2) {pm=fmaxf(pm,__shfl_xor(pm,off));sm=fmaxf(sm,__shfl_xor(sm,off));}
    if(laneid()==0) {
      maxima[warpid()*2]=pm;
      maxima[warpid()*2+1]=sm;
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
extern "C" __global__ __launch_bounds__(WARPS*64) __attribute__((amdgpu_waves_per_eu(2,2))) void fa_fp8_backward(float* dq,float* dk,float* dv,float* amax,float* next_amax,
 const unsigned char* q,const unsigned char* k,const unsigned char* v,const unsigned char* dout,
 const float* delta,const float* lse,const float* scales) {
  __shared__ SMALL ps,ds;
  __shared__ ST bs[2],dbs[2];
  __shared__ float maxima[WARPS*2];
  constexpr int GX=2*(ATTN_N/OWN),TOTAL=GX*ATTN_H*ATTN_B;
  constexpr int HEAD_GROUP=ATTN_H%4==0?4:ATTN_H%2==0?2:1;
  int gid=(blockIdx.z*ATTN_H+blockIdx.y)*GX+blockIdx.x;
  if constexpr(TOTAL%8==0) gid=(gid%8)*(TOTAL/8)+gid/8;
  int bx=(gid/HEAD_GROUP)%GX,bh=((gid/(GX*HEAD_GROUP))*HEAD_GROUP+gid%HEAD_GROUP)%ATTN_H,bb=gid/(GX*ATTN_H);
  if(bx<ATTN_N/OWN) body<false>(dq,dk,dv,amax,next_amax,q,k,v,dout,delta,lse,scales,bs,dbs,ps,ds,maxima,bx,bh,bb);
  else body<true>(dq,dk,dv,amax,next_amax,q,k,v,dout,delta,lse,scales,bs,dbs,ps,ds,maxima,bx,bh,bb);
}
