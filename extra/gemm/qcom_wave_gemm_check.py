#!/usr/bin/env python3
"""Random oracle for a wave-cooperative 8x4-half4 FP16 GEMM."""
import argparse, statistics

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.qcom_8x4_gemm import buf_copyin, buf_copyout


def source(m:int, n:int, k:int, stride:int) -> str:
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void wave_gemm(read_only image2d_t A,read_only image2d_t B,__global half *C) {{
  int lid=get_local_id(0),lane=lid&31,wave=lid>>5,row=lane>>2,col=lane&3;
  int row0=get_group_id(1)*16+(wave>>1)*8+row;
  int col4=get_group_id(0)*8+(wave&1)*4+col;
  half4 acc=(half4)(0);
  for(int q=0;q<{k//4};q++) {{
    half4 av=col==0?read_imageh(A,smp,(int2)(q,row0)):(half4)(0);
    uint2 au=as_uint2(av);
    au.x|=qcom_sub_group_shuffle_xor(au.x,1,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
    au.y|=qcom_sub_group_shuffle_xor(au.y,1,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
    au.x|=qcom_sub_group_shuffle_xor(au.x,2,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
    au.y|=qcom_sub_group_shuffle_xor(au.y,2,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
    av=as_half4(au);
    half4 own=row<4?read_imageh(B,smp,(int2)(col4,q*4+row)):(half4)(0);
    uint2 bu=as_uint2(own);
    for(int kk=0;kk<4;kk++) {{
      uint2 bk=row==kk?bu:(uint2)(0);
      bk.x|=qcom_sub_group_shuffle_xor(bk.x,4,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
      bk.y|=qcom_sub_group_shuffle_xor(bk.y,4,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
      bk.x|=qcom_sub_group_shuffle_xor(bk.x,8,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
      bk.y|=qcom_sub_group_shuffle_xor(bk.y,8,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
      bk.x|=qcom_sub_group_shuffle_xor(bk.x,16,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
      bk.y|=qcom_sub_group_shuffle_xor(bk.y,16,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,0u);
      half a=kk==0?av.x:kk==1?av.y:kk==2?av.z:av.w;
      acc+=(half4)(a)*as_half4(bk);
    }}
  }}
  vstore4(acc,0,C+row0*{stride}+col4*4);
}}"""


def upload(values:np.ndarray, dtype) -> Buffer:
  ret=Buffer("QCOM",values.size,dtype).allocate()
  buf_copyin(ret, memoryview(np.ascontiguousarray(values)).cast("B"))
  return ret


def main() -> None:
  ap=argparse.ArgumentParser()
  ap.add_argument("--m",type=int,default=1024)
  ap.add_argument("--n",type=int,default=1024)
  ap.add_argument("--k",type=int,default=1024)
  ap.add_argument("--stride",type=int,default=1024)
  ap.add_argument("--seed",type=int,default=0)
  ap.add_argument("--runs",type=int,default=10)
  args=ap.parse_args()
  if args.m%16 or args.n%32 or args.k%4: raise ValueError("shape must divide 16x32x4")
  rng=np.random.default_rng(args.seed)
  av=(rng.standard_normal((args.m,args.k))*.05).astype(np.float16)
  bv=(rng.standard_normal((args.k,args.n))*.05).astype(np.float16)
  a,b=upload(av,dtypes.half),upload(bv,dtypes.half)
  c=upload(np.zeros((args.m,args.stride),np.float16),dtypes.half)
  dev=Device["QCOM"]
  src=source(args.m,args.n,args.k,args.stride)
  lib=dev.compiler.compile(src)
  prg=dev.runtime("wave_gemm",lib,buf_dtypes=[((0,dtypes.half,(args.m,args.k//4,4)),),
    ((1,dtypes.half,(args.k,args.n//4,4)),),((2,dtypes.half,None),)])
  gs,ls=(args.n//32,args.m//16,1),(128,1,1)
  for _ in range(2): prg(a._buf,b._buf,c._buf,global_size=gs,local_size=ls,wait=True)
  times=[prg(a._buf,b._buf,c._buf,global_size=gs,local_size=ls,wait=True)*1e3 for _ in range(args.runs)]
  storage=np.empty((args.m,args.stride),np.float16)
  buf_copyout(c, memoryview(storage).cast("B"))
  got=storage[:,:args.n].astype(np.float32)
  expected=av.astype(np.float32)@bv.astype(np.float32)
  delta=np.abs(got-expected)
  best=min(times)
  print(f"best_ms={best:.4f} median_ms={statistics.median(times):.4f} gflops={2*args.m*args.n*args.k/best/1e6:.1f} "
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} allclose={np.allclose(got,expected,rtol=.02,atol=.02)}")


if __name__=="__main__": main()
