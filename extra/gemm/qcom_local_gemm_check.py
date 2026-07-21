#!/usr/bin/env python3
"""Random-data oracle and benchmark for a cooperative-local FP16 QCOM GEMM."""
import argparse, statistics

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void local_gemm(__global const half *A, __global const half *B, __global half *C) {
  __local half la[32*16];
  __local half lb[16*64];
  const int lid=get_local_id(0), lr=lid>>5, lc=lid&31;
  const int row0=get_group_id(1)*32+lr*8;
  const int col0=get_group_id(0)*64+lc*2;
  half2 c0=(half2)(0),c1=(half2)(0),c2=(half2)(0),c3=(half2)(0);
  half2 c4=(half2)(0),c5=(half2)(0),c6=(half2)(0),c7=(half2)(0);
  for (int k0=0;k0<@K@;k0+=16) {
    for (int i=lid;i<32*16;i+=128) la[i]=A[(get_group_id(1)*32+i/16)*@K@+k0+i%16];
    for (int i=lid;i<16*64;i+=128) lb[i]=B[(k0+i/64)*@N@+get_group_id(0)*64+i%64];
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int kk=0;kk<16;kk++) {
      half2 b=vload2(0,lb+kk*64+lc*2);
      c0+=la[(lr*8+0)*16+kk]*b; c1+=la[(lr*8+1)*16+kk]*b;
      c2+=la[(lr*8+2)*16+kk]*b; c3+=la[(lr*8+3)*16+kk]*b;
      c4+=la[(lr*8+4)*16+kk]*b; c5+=la[(lr*8+5)*16+kk]*b;
      c6+=la[(lr*8+6)*16+kk]*b; c7+=la[(lr*8+7)*16+kk]*b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  vstore2(c0,0,C+(row0+0)*@N@+col0); vstore2(c1,0,C+(row0+1)*@N@+col0);
  vstore2(c2,0,C+(row0+2)*@N@+col0); vstore2(c3,0,C+(row0+3)*@N@+col0);
  vstore2(c4,0,C+(row0+4)*@N@+col0); vstore2(c5,0,C+(row0+5)*@N@+col0);
  vstore2(c6,0,C+(row0+6)*@N@+col0); vstore2(c7,0,C+(row0+7)*@N@+col0);
}
"""


def upload(x:np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", x.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(x)).cast("B"))
  return ret


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--m", type=int, default=128)
  ap.add_argument("--n", type=int, default=1536)
  ap.add_argument("--k", type=int, default=384)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--runs", type=int, default=10)
  args = ap.parse_args()
  if args.m % 32 or args.n % 64 or args.k % 16: raise ValueError("M,N,K must divide the 32x64x16 tile")
  rng = np.random.default_rng(args.seed)
  a = (rng.standard_normal((args.m,args.k))*.05).astype(np.float16)
  b = (rng.standard_normal((args.k,args.n))*.05).astype(np.float16)
  ab, bb = upload(a, dtypes.half), upload(b, dtypes.half)
  cb = upload(np.zeros((args.m,args.n), np.float16), dtypes.half)
  dev = Device["QCOM"]
  src = SRC.replace("@K@", str(args.k)).replace("@N@", str(args.n))
  lib = dev.compiler.compile_cached(src)
  prg = dev.runtime("local_gemm", lib, buf_dtypes=[((0,dtypes.half,None),)]*3)
  gs, ls = (args.n//64,args.m//32,1), (128,1,1)
  for _ in range(2): prg(ab._buf,bb._buf,cb._buf,global_size=gs,local_size=ls,wait=True)
  times = [prg(ab._buf,bb._buf,cb._buf,global_size=gs,local_size=ls,wait=True)*1e3 for _ in range(args.runs)]
  got = np.empty((args.m,args.n),np.float16)
  cb.copyout(memoryview(got).cast("B"))
  expected = a.astype(np.float32) @ b.astype(np.float32)
  delta = np.abs(got.astype(np.float32)-expected)
  med, best = statistics.median(times), min(times)
  print(f"best_ms={best:.4f} median_ms={med:.4f} gflops={2*args.m*args.n*args.k/best/1e6:.1f} "
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} allclose={np.allclose(got,expected,rtol=.02,atol=.02)}")
  if not np.allclose(got,expected,rtol=.02,atol=.02): raise SystemExit(1)


if __name__ == "__main__": main()
