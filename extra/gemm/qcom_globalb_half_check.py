#!/usr/bin/env python3
"""Randomized oracle and benchmark for compiler FP16 GEMM with linear global weights."""
import os
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.qcom_8x4_gemm import buf_copyin, buf_copyout


def main() -> None:
  m, n, k = (int(os.getenv(x, d)) for x, d in (("M", 128), ("N", 384), ("K", 1536)))
  source = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_globalb(read_only image2d_t A,__global half *B,__global half *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col4=get_group_id(0)*32+tid;
  half4 r0=(half4)(0),r1=(half4)(0),r2=(half4)(0),r3=(half4)(0);
  for(int k4=0;k4<{k//4};k4++) {{
    half4 a0=read_imageh(A,smp,(int2)(k4,row)),a1=read_imageh(A,smp,(int2)(k4,row+1));
    half4 a2=read_imageh(A,smp,(int2)(k4,row+2)),a3=read_imageh(A,smp,(int2)(k4,row+3));
    int p=(k4*4)*{n}+col4*4;
    half4 b0=vload4(0,B+p),b1=vload4(0,B+p+{n}),b2=vload4(0,B+p+{2*n}),b3=vload4(0,B+p+{3*n});
    r0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
    r1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
    r2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
    r3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
  }}
  vstore4(r0,0,C+row*{n}+col4*4); vstore4(r1,0,C+(row+1)*{n}+col4*4);
  vstore4(r2,0,C+(row+2)*{n}+col4*4); vstore4(r3,0,C+(row+3)*{n}+col4*4);
}}"""
  dev = Device["QCOM"]
  lib = dev.compiler.compile(source)
  rng = np.random.default_rng(4)
  a = (rng.standard_normal((m, k))*0.05).astype(np.float16)
  b = (rng.standard_normal((k, n))*0.05).astype(np.float16)
  ab, bb, cb = (Buffer("QCOM", x.size, dtypes.half).allocate() for x in (a, b, np.empty((m, n), np.float16)))
  buf_copyin(ab, memoryview(a).cast("B")); buf_copyin(bb, memoryview(b).cast("B"))
  prg = dev.runtime("gemm_globalb", lib, buf_dtypes=[((0, dtypes.half, (m, k//4, 4)),),
                    ((0, dtypes.half, None),), ((0, dtypes.half, None),)])
  times = [prg(ab._buf, bb._buf, cb._buf, global_size=(n//128, m//16, 1), local_size=(128, 1, 1), wait=True) for _ in range(10)]
  got = np.empty((m, n), np.float16); buf_copyout(cb, memoryview(got).cast("B"))
  expected = a.astype(np.float32) @ b.astype(np.float32)
  err = np.abs(got.astype(np.float32)-expected)
  print(f"ms={min(times)*1e3:.4f} gflops={2*m*n*k/min(times)/1e9:.1f} max={err.max():.8g} mean={err.mean():.8g} "
        f"allclose={np.allclose(got, expected, rtol=.01, atol=.01)} finite={np.isfinite(got).all()}")


if __name__ == "__main__": main()
