#include "kittens.cuh"
using namespace kittens;

// An intentionally simple fused baseline: one wave owns 64 query rows. P and
// dS live only in LDS; dQ stays in registers and dK/dV accumulate in FP32.
// E5M2 payloads use HK's byte-compatible FP8 register storage with an explicit
// MFMA format selector. No E4M3 conversion is applied to those payloads.
template<int R, int C> using bytes_tile = rt<fp8e4m3,R,C,row_l,rt_32x64_s>;
template<int R, int C> using accum_tile = rt<float,R,C,col_l,rt_32x32_s>;
using i32x8 = int __attribute__((ext_vector_type(8)));
using f32x16 = float __attribute__((ext_vector_type(16)));

template<int R,int C,typename Load>
__device__ void load_bytes(bytes_tile<R,C>& tile, Load load) {
  int lane = kittens::laneid();
  #pragma unroll
  for(int i=0;i<R/32;i++) {
    #pragma unroll
    for(int j=0;j<C/64;j++) {
      #pragma unroll
      for(int k=0;k<8;k++) {
        unsigned word=0;
        #pragma unroll
        for(int t=0;t<4;t++) {
          int row=i*32+lane%32, col=j*64+(lane/32)*16+(k/4)*32+(k%4)*4+t;
          word |= unsigned(load(row,col))<<(8*t);
        }
        tile.tiles[i][j].data[k]=std::bit_cast<fp8e4m3_4>(word);
      }
    }
  }
}

template<int AF,int BF,int R,int C,int K>
__device__ void dot(accum_tile<R,C>& out,const bytes_tile<R,K>& a,const bytes_tile<C,K>& b) {
  #pragma unroll
  for(int i=0;i<R/32;i++) {
    #pragma unroll
    for(int j=0;j<C/32;j++) {
      #pragma unroll
      for(int k=0;k<K/64;k++) {
        auto& d=out.tiles[i][j];
        *(f32x16*)d.data=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
          *(const i32x8*)a.tiles[i][k].data,*(const i32x8*)b.tiles[j][k].data,*(f32x16*)d.data,AF,BF,0,0,0,0);
      }
    }
  }
}

template<int R,int C,typename Fn>
__device__ void elements(accum_tile<R,C>& tile,Fn fn) {
  int lane=kittens::laneid();
  #pragma unroll
  for(int i=0;i<R/32;i++) {
    #pragma unroll
    for(int j=0;j<C/32;j++) {
      #pragma unroll
      for(int t=0;t<16;t++) {
        int row=i*32+4*(lane/32)+(t/4)*8+t%4,col=j*32+lane%32;
        fn(((float*)tile.tiles[i][j].data)[t],row,col);
      }
    }
  }
}

template<bool E5> __device__ unsigned char encode(float x) {
  constexpr float maxval=E5?57344.f:448.f;
  x=fminf(maxval,fmaxf(-maxval,x));
  if constexpr(E5) return __builtin_amdgcn_cvt_pk_bf8_f32(x,0.f,0,false)&255;
  else return __builtin_amdgcn_cvt_pk_fp8_f32(x,0.f,0,false)&255;
}

// Stage contiguous vectors before either MFMA orientation reads the tile.
__device__ void stage(unsigned char* dst,const unsigned char* src,int row,int head,int batch,int heads) {
  using vec = unsigned __attribute__((ext_vector_type(4)));
  for(int i=kittens::laneid();i<64*128/16;i+=64) {
    int r=i/8,c=(i%8)*16;
    *(vec*)(dst+i*16)=*(const vec*)(src+((batch*ATTN_N+row+r)*heads+head)*128+c);
  }
}

extern "C" __global__ __launch_bounds__(64) void fa_fp8_backward(
    float* dq,float* dk,float* dv,float* amax,float* next_amax,const unsigned char* q,const unsigned char* k,const unsigned char* v,
    const unsigned char* dout,const float* delta,const float* lse,const float* scales) {
  constexpr int N=ATTN_N,H=ATTN_H,HK=ATTN_H_KV,D=128;
  int qb=blockIdx.x*64,h=blockIdx.y,b=blockIdx.z,kh=h/(H/HK);
  float vs=scales[0],dos=scales[1],ps=scales[2],dss=scales[3];
  float pmax=0.f,dsmax=0.f;
  constexpr float grad_scale=0.08838834764831845f/0.3570958286295132f;
  __shared__ unsigned char pbuf[64*64],dsbuf[64*64],qbuf[64*128],kbuf[64*128],vbuf[64*128],dobuf[64*128];
  stage(qbuf,q,qb,h,b,H);stage(dobuf,dout,qb,h,b,H);
  accum_tile<64,128> dqr;zero(dqr);
  for(int kb=0;kb<=qb;kb+=64) {
    stage(kbuf,k,kb,kh,b,HK);stage(vbuf,v,kb,kh,b,HK);__syncthreads();
    {
    bytes_tile<64,128> qr,dr;
    load_bytes(qr,[&](int r,int c){return qbuf[r*D+c];});
    load_bytes(dr,[&](int r,int c){return dobuf[r*D+c];});
    bytes_tile<64,128> kr,vr;
    load_bytes(kr,[&](int r,int c){return kbuf[r*D+c];});
    load_bytes(vr,[&](int r,int c){return vbuf[r*D+c];});
    accum_tile<64,64> p,ds;zero(p);zero(ds);
    dot<0,0>(p,qr,kr);dot<1,0>(ds,dr,vr);
    elements(p,[&](float& x,int r,int c){
      x=kb+c<=qb+r ? exp2f(x-lse[(b*H+h)*N+qb+r]*1.4426950408889634f) : 0.f;
      pmax=fmaxf(pmax,fabsf(x));pbuf[r*64+c]=encode<false>(x/ps);
    });
    #pragma unroll
    for(int i=0;i<2;i++) {
      #pragma unroll
      for(int j=0;j<2;j++) {
        #pragma unroll
        for(int t=0;t<16;t++) {
          int r=i*32+4*(kittens::laneid()/32)+(t/4)*8+t%4,c=j*32+kittens::laneid()%32;
          float val=((float*)p.tiles[i][j].data)[t]*(((float*)ds.tiles[i][j].data)[t]*dos*vs-delta[(b*H+h)*N+qb+r]);
          dsmax=fmaxf(dsmax,fabsf(val));dsbuf[r*64+c]=encode<true>(val/dss);
        }
      }
    }
    }
    __syncthreads();
    bytes_tile<64,64> dsr;
    load_bytes(dsr,[&](int r,int c){return dsbuf[r*64+c];});
    #pragma unroll 1
    for(int dim=0;dim<4;dim++) {
      bytes_tile<32,64> kt;
      load_bytes(kt,[&](int r,int c){return kbuf[c*D+dim*32+r];});
      accum_tile<64,32> qgrad;
      for(int i=0;i<2;i++) qgrad.tiles[i][0]=dqr.tiles[i][dim];
      dot<1,0>(qgrad,dsr,kt);
      for(int i=0;i<2;i++) dqr.tiles[i][dim]=qgrad.tiles[i][0];
      accum_tile<32,32> grad;
      #pragma unroll 1
      for(int key=0;key<2;key++) {
        bytes_tile<32,64> st,operand;
        load_bytes(st,[&](int r,int c){return dsbuf[c*64+key*32+r];});
        load_bytes(operand,[&](int r,int c){return qbuf[c*D+dim*32+r];});
        zero(grad);dot<1,0>(grad,st,operand);
        elements(grad,[&](float& x,int r,int c){atomicAdd(dk+((b*N+kb+key*32+r)*H+h)*D+dim*32+c,x*dss*grad_scale);});
        load_bytes(st,[&](int r,int c){return pbuf[c*64+key*32+r];});
        load_bytes(operand,[&](int r,int c){return dobuf[c*D+dim*32+r];});
        zero(grad);dot<0,1>(grad,st,operand);
        elements(grad,[&](float& x,int r,int c){atomicAdd(dv+((b*N+kb+key*32+r)*H+h)*D+dim*32+c,x*ps*dos);});
      }
    }
    __syncthreads();
  }
  elements(dqr,[&](float& x,int r,int c){dq[((b*N+qb+r)*H+h)*D+c]=x*dss*grad_scale;});
  for(int offset=32;offset;offset/=2) {pmax=fmaxf(pmax,__shfl_xor(pmax,offset));dsmax=fmaxf(dsmax,__shfl_xor(dsmax,offset));}
  if(kittens::laneid()==0) {int idx=(b*H+h)*(N/64)+blockIdx.x;amax[2*idx]=pmax;amax[2*idx+1]=dsmax;
    atomicMax((unsigned*)next_amax,__float_as_uint(pmax));atomicMax((unsigned*)next_amax+1,__float_as_uint(dsmax));}
}
