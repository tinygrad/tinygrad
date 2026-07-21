#!/usr/bin/env python3
"""Random-data benchmark for cooperative image-to-local FP16 GEMM."""
import argparse, statistics

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


def source(n:int, k:int, stride:int, bk4:int, fp32_acc:bool=False) -> str:
  acc_t, zero, conv = ("float4", "(float4)(0)", "convert_float4") if fp32_acc else ("half4", "(half4)(0)", "")
  out_t = "float" if fp32_acc else "half"
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void local_image_gemm(read_only image2d_t A, read_only image2d_t B, __global {out_t} *C) {{
  __local half4 la[{32*bk4}];
  __local half4 lb[{bk4*4*32}];
  int lid=get_local_id(0), tm=lid>>5, tid=lid&31;
  int row0=get_group_id(1)*32+tm*8, col4=get_group_id(0)*32+tid;
  {acc_t} c0={zero},c1={zero},c2={zero},c3={zero};
  {acc_t} c4={zero},c5={zero},c6={zero},c7={zero};
  for(int kb=0;kb<{k//4};kb+={bk4}) {{
    for(int i=lid;i<{32*bk4};i+=128) {{
      int r=i/{bk4},q=i-r*{bk4};
      la[i]=read_imageh(A,smp,(int2)(kb+q,get_group_id(1)*32+r));
    }}
    for(int i=lid;i<{bk4*4*32};i+=128) {{
      int y=i>>5,x=i&31;
      lb[i]=read_imageh(B,smp,(int2)(get_group_id(0)*32+x,kb*4+y));
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int q=0;q<{bk4};q++) {{
      {acc_t} a0={conv}(la[(tm*8+0)*{bk4}+q]),a1={conv}(la[(tm*8+1)*{bk4}+q]);
      {acc_t} a2={conv}(la[(tm*8+2)*{bk4}+q]),a3={conv}(la[(tm*8+3)*{bk4}+q]);
      {acc_t} a4={conv}(la[(tm*8+4)*{bk4}+q]),a5={conv}(la[(tm*8+5)*{bk4}+q]);
      {acc_t} a6={conv}(la[(tm*8+6)*{bk4}+q]),a7={conv}(la[(tm*8+7)*{bk4}+q]);
      {acc_t} b0={conv}(lb[(q*4+0)*32+tid]),b1={conv}(lb[(q*4+1)*32+tid]);
      {acc_t} b2={conv}(lb[(q*4+2)*32+tid]),b3={conv}(lb[(q*4+3)*32+tid]);
      c0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
      c1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
      c2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
      c3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
      c4+=a4.xxxx*b0+a4.yyyy*b1+a4.zzzz*b2+a4.wwww*b3;
      c5+=a5.xxxx*b0+a5.yyyy*b1+a5.zzzz*b2+a5.wwww*b3;
      c6+=a6.xxxx*b0+a6.yyyy*b1+a6.zzzz*b2+a6.wwww*b3;
      c7+=a7.xxxx*b0+a7.yyyy*b1+a7.zzzz*b2+a7.wwww*b3;
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  vstore4(c0,0,C+(row0+0)*{stride}+col4*4); vstore4(c1,0,C+(row0+1)*{stride}+col4*4);
  vstore4(c2,0,C+(row0+2)*{stride}+col4*4); vstore4(c3,0,C+(row0+3)*{stride}+col4*4);
  vstore4(c4,0,C+(row0+4)*{stride}+col4*4); vstore4(c5,0,C+(row0+5)*{stride}+col4*4);
  vstore4(c6,0,C+(row0+6)*{stride}+col4*4); vstore4(c7,0,C+(row0+7)*{stride}+col4*4);
}}"""


def global_b_source(n:int, k:int, stride:int) -> str:
  rows = "\n".join(f"  half4 c{r}=(half4)(0);" for r in range(8))
  aloads = "\n".join(f"    half4 a{r}=read_imageh(A,smp,(int2)(q,row0+{r}));" for r in range(8))
  mads = "\n".join(f"    c{r}+=a{r}.xxxx*b0+a{r}.yyyy*b1+a{r}.zzzz*b2+a{r}.wwww*b3;" for r in range(8))
  stores = "\n".join(f"  vstore4(c{r},0,C+(row0+{r})*{stride}+col4*4);" for r in range(8))
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void local_image_gemm(read_only image2d_t A, __global half *B, __global half *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row0=get_group_id(1)*32+tm*8,col4=get_group_id(0)*32+tid;
{rows}
  for(int q=0;q<{k//4};q++) {{
{aloads}
    int p=q*4*{n}+col4*4;
    half4 b0=vload4(0,B+p),b1=vload4(0,B+p+{n});
    half4 b2=vload4(0,B+p+{2*n}),b3=vload4(0,B+p+{3*n});
{mads}
  }}
{stores}
}}"""


def local_b_fp32_source(n:int, k:int, stride:int, bk4:int) -> str:
  aloads = ",".join(f"a{r}=convert_float4(read_imageh(A,smp,(int2)(kb+q,row0+{r})))" for r in range(8))
  mads = "\n".join(f"      c{r}+=a{r}.xxxx*b0+a{r}.yyyy*b1+a{r}.zzzz*b2+a{r}.wwww*b3;" for r in range(8))
  stores = "\n".join(f"  vstore4(c{r},0,C+(row0+{r})*{stride}+col4*4);" for r in range(8))
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void local_image_gemm(read_only image2d_t A,read_only image2d_t B,__global float *C) {{
  __local half4 lb[{bk4*4*32}];
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31,row0=get_group_id(1)*32+tm*8,col4=get_group_id(0)*32+tid;
  float4 c0=(float4)(0),c1=(float4)(0),c2=(float4)(0),c3=(float4)(0);
  float4 c4=(float4)(0),c5=(float4)(0),c6=(float4)(0),c7=(float4)(0);
  for(int kb=0;kb<{k//4};kb+={bk4}) {{
    for(int i=lid;i<{bk4*4*32};i+=128) {{ int y=i>>5,x=i&31;lb[i]=read_imageh(B,smp,(int2)(get_group_id(0)*32+x,kb*4+y)); }}
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int q=0;q<{bk4};q++) {{
      float4 {aloads};
      float4 b0=convert_float4(lb[(q*4+0)*32+tid]),b1=convert_float4(lb[(q*4+1)*32+tid]);
      float4 b2=convert_float4(lb[(q*4+2)*32+tid]),b3=convert_float4(lb[(q*4+3)*32+tid]);
{mads}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
{stores}
}}"""


def upload(values:np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", values.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(values)).cast("B"))
  return ret


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--m", type=int, default=128); ap.add_argument("--n", type=int, default=1536)
  ap.add_argument("--k", type=int, default=384); ap.add_argument("--stride", type=int, default=2048)
  ap.add_argument("--bk4", type=int, choices=(2, 4, 8, 16), default=8)
  ap.add_argument("--global-b", action="store_true")
  ap.add_argument("--fp32-acc", action="store_true")
  ap.add_argument("--b-only", action="store_true")
  ap.add_argument("--seed", type=int, default=0); ap.add_argument("--runs", type=int, default=10)
  args = ap.parse_args()
  if args.m%32 or args.n%128 or (args.k//4)%args.bk4: raise ValueError("shape does not divide tile")
  rng=np.random.default_rng(args.seed)
  av=(rng.standard_normal((args.m,args.k))*.05).astype(np.float16)
  bv=(rng.standard_normal((args.k,args.n))*.05).astype(np.float16)
  a,b=upload(av,dtypes.half),upload(bv,dtypes.half)
  out_np, out_dtype = (np.float32, dtypes.float) if args.fp32_acc else (np.float16, dtypes.half)
  c=upload(np.zeros((args.m,args.stride),out_np),out_dtype)
  dev=Device["QCOM"]
  src=(global_b_source(args.n,args.k,args.stride) if args.global_b else local_b_fp32_source(args.n,args.k,args.stride,args.bk4)
       if args.b_only else source(args.n,args.k,args.stride,args.bk4,args.fp32_acc))
  specs=[((0,dtypes.half,(args.m,args.k//4,4)),),
         ((1,dtypes.half,None),) if args.global_b else ((1,dtypes.half,(args.k,args.n//4,4)),),((2,out_dtype,None),)]
  prg=dev.runtime("local_image_gemm",dev.compiler.compile(src),buf_dtypes=specs)
  gs,ls=(args.n//128,args.m//32,1),(128,1,1)
  for _ in range(2): prg(a._buf,b._buf,c._buf,global_size=gs,local_size=ls,wait=True)
  times=[prg(a._buf,b._buf,c._buf,global_size=gs,local_size=ls,wait=True)*1e3 for _ in range(args.runs)]
  storage=np.empty((args.m,args.stride),out_np); c.copyout(memoryview(storage).cast("B"))
  got=storage[:,:args.n].astype(np.float32); expected=av.astype(np.float32)@bv.astype(np.float32)
  delta=np.abs(got-expected); best=min(times)
  print(f"bk4={args.bk4} best_ms={best:.4f} median_ms={statistics.median(times):.4f} "
        f"gflops={2*args.m*args.n*args.k/best/1e6:.1f} max_abs={delta.max():.9g} "
        f"mean_abs={delta.mean():.9g} accumulate={'fp32' if args.fp32_acc else 'fp16'} "
        f"allclose={np.allclose(got,expected,rtol=1e-4 if args.fp32_acc else .02,atol=1e-4 if args.fp32_acc else .02)}")


if __name__ == "__main__": main()
