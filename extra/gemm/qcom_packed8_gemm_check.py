#!/usr/bin/env python3
"""Validate a zero-copy uint4 view that fetches eight packed FP16 values."""
import argparse, statistics

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import disasm, get_envelope


def source(k:int, stride:int) -> str:
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void packed8(read_only image2d_t A,read_only image2d_t B,__global half *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col8=get_group_id(0)*32+tid;
  half8 c0=(half8)(0),c1=(half8)(0),c2=(half8)(0),c3=(half8)(0);
  for(int k8=0;k8<{k//8};k8++) {{
    half8 a0=as_half8(read_imageui(A,smp,(int2)(k8,row+0)));
    half8 a1=as_half8(read_imageui(A,smp,(int2)(k8,row+1)));
    half8 a2=as_half8(read_imageui(A,smp,(int2)(k8,row+2)));
    half8 a3=as_half8(read_imageui(A,smp,(int2)(k8,row+3)));
    half8 b0=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+0)));
    half8 b1=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+1)));
    half8 b2=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+2)));
    half8 b3=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+3)));
    half8 b4=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+4)));
    half8 b5=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+5)));
    half8 b6=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+6)));
    half8 b7=as_half8(read_imageui(B,smp,(int2)(col8,k8*8+7)));
    c0+=a0.s0*b0+a0.s1*b1+a0.s2*b2+a0.s3*b3+a0.s4*b4+a0.s5*b5+a0.s6*b6+a0.s7*b7;
    c1+=a1.s0*b0+a1.s1*b1+a1.s2*b2+a1.s3*b3+a1.s4*b4+a1.s5*b5+a1.s6*b6+a1.s7*b7;
    c2+=a2.s0*b0+a2.s1*b1+a2.s2*b2+a2.s3*b3+a2.s4*b4+a2.s5*b5+a2.s6*b6+a2.s7*b7;
    c3+=a3.s0*b0+a3.s1*b1+a3.s2*b2+a3.s3*b3+a3.s4*b4+a3.s5*b5+a3.s6*b6+a3.s7*b7;
  }}
  vstore8(c0,0,C+(row+0)*{stride}+col8*8); vstore8(c1,0,C+(row+1)*{stride}+col8*8);
  vstore8(c2,0,C+(row+2)*{stride}+col8*8); vstore8(c3,0,C+(row+3)*{stride}+col8*8);
}}"""


def upload(x:np.ndarray, dtype) -> Buffer:
  ret=Buffer("QCOM",x.size,dtype).allocate(); raw=memoryview(np.ascontiguousarray(x)).cast("B")
  Device["QCOM"].allocator._copyin(ret._buf,raw); return ret


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("--m",type=int,default=128); ap.add_argument("--n",type=int,default=1536)
  ap.add_argument("--k",type=int,default=384); ap.add_argument("--stride",type=int,default=2048)
  ap.add_argument("--seed",type=int,default=0); ap.add_argument("--runs",type=int,default=10)
  ap.add_argument("--disasm",action="store_true"); args=ap.parse_args()
  if args.m%16 or args.n%256 or args.k%8: raise ValueError("shape does not divide 16x256x8 tile")
  rng=np.random.default_rng(args.seed); av=(rng.standard_normal((args.m,args.k))*.05).astype(np.float16)
  bv=(rng.standard_normal((args.k,args.n))*.05).astype(np.float16)
  a,b=upload(av,dtypes.half),upload(bv,dtypes.half); c=upload(np.zeros((args.m,args.stride),np.float16),dtypes.half)
  dev=Device["QCOM"]; lib=dev.compiler.compile(source(args.k,args.stride))
  if args.disasm:
    env,off,size,_=get_envelope(dev,source(args.k,args.stride)); print(disasm(bytes(env[off:off+size])))
  specs=[((0,dtypes.uint32,(args.m,args.k//8,4)),),((1,dtypes.uint32,(args.k,args.n//8,4)),),((2,dtypes.half,None),)]
  prg=dev.runtime("packed8",lib,buf_dtypes=specs); gs,ls=(args.n//256,args.m//16,1),(128,1,1)
  for _ in range(2): prg(a._buf,b._buf,c._buf,global_size=gs,local_size=ls,wait=True)
  times=[prg(a._buf,b._buf,c._buf,global_size=gs,local_size=ls,wait=True)*1e3 for _ in range(args.runs)]
  storage=np.empty((args.m,args.stride),np.float16); raw=memoryview(storage).cast("B")
  Device["QCOM"].allocator._copyout(raw,c._buf)
  got=storage[:,:args.n].astype(np.float32); expected=av.astype(np.float32)@bv.astype(np.float32); d=np.abs(got-expected); best=min(times)
  print(f"best_ms={best:.4f} median_ms={statistics.median(times):.4f} gflops={2*args.m*args.n*args.k/best/1e6:.1f} "
        f"max_abs={d.max():.9g} mean_abs={d.mean():.9g} allclose={np.allclose(got,expected,rtol=.02,atol=.02)}")


if __name__=="__main__": main()
