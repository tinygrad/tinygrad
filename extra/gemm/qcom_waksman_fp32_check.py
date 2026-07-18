#!/usr/bin/env python3
"""Exact 4x4-commutative Waksman microsteps in a full FP16-input/FP32-accumulate GEMM."""
import os

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


def alloc(count: int, dtype) -> Buffer: return Buffer("QCOM", count, dtype).allocate()


def source(n: int, wg: int = 128) -> str:
  blocks = []
  swz = "xyzw"
  for block in range(2):
    b = f"b{block}"
    body = [
      f"float4 {b}q0={b}1*({b}0.xxxx+{b}0),{b}q1={b}3*({b}2.xxxx+{b}2);",
      f"float4 {b}p0=a0*({b}0.xxxx+a1),{b}p1=a2*({b}2.xxxx+a3);",
      f"float4 {b}r0=a1*({b}1.xxxx-a0),{b}r1=a3*({b}3.xxxx-a2);",
      f"c{block*4} += {b}p0+{b}r0+{b}p1+{b}r1;",
    ]
    for j in range(1, 4):
      s = swz[j]
      body += [
        f"float4 {b}m{j}0=(a0+{b}1.{s})*(a1+{b}0.x+{b}0.{s});",
        f"float4 {b}m{j}1=(a2+{b}3.{s})*(a3+{b}2.x+{b}2.{s});",
        f"c{block*4+j} += {b}m{j}0-{b}p0-{b}q0.{s}+{b}m{j}1-{b}p1-{b}q1.{s};",
      ]
    blocks.append("".join(body))
  stores = []
  for row in range(4):
    stores += [f"write_imagef(C,(int2)(col4,row+{row}),"
               f"(float4)(c0.{swz[row]},c1.{swz[row]},c2.{swz[row]},c3.{swz[row]}));",
               f"write_imagef(C,(int2)(col4+1,row+{row}),"
               f"(float4)(c4.{swz[row]},c5.{swz[row]},c6.{swz[row]},c7.{swz[row]}));"]
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void gemm(write_only image2d_t C,read_only image2d_t A,read_only image2d_t B) {{
  uint s=get_global_id(0),row=(s/{n//8})*4,col4=(s%{n//8})*2;
  float4 c0=(float4)(0),c1=(float4)(0),c2=(float4)(0),c3=(float4)(0);
  float4 c4=(float4)(0),c5=(float4)(0),c6=(float4)(0),c7=(float4)(0);
  for(uint k4=0;k4<{n//4};k4++) {{
    float4 ar0=convert_float4(read_imageh(A,smp,(int2)(k4,row)));
    float4 ar1=convert_float4(read_imageh(A,smp,(int2)(k4,row+1)));
    float4 ar2=convert_float4(read_imageh(A,smp,(int2)(k4,row+2)));
    float4 ar3=convert_float4(read_imageh(A,smp,(int2)(k4,row+3)));
    float4 a0=(float4)(ar0.x,ar1.x,ar2.x,ar3.x),a1=(float4)(ar0.y,ar1.y,ar2.y,ar3.y);
    float4 a2=(float4)(ar0.z,ar1.z,ar2.z,ar3.z),a3=(float4)(ar0.w,ar1.w,ar2.w,ar3.w);
    float4 b00=convert_float4(read_imageh(B,smp,(int2)(col4,k4*4)));
    float4 b01=convert_float4(read_imageh(B,smp,(int2)(col4,k4*4+1)));
    float4 b02=convert_float4(read_imageh(B,smp,(int2)(col4,k4*4+2)));
    float4 b03=convert_float4(read_imageh(B,smp,(int2)(col4,k4*4+3)));
    float4 b10=convert_float4(read_imageh(B,smp,(int2)(col4+1,k4*4)));
    float4 b11=convert_float4(read_imageh(B,smp,(int2)(col4+1,k4*4+1)));
    float4 b12=convert_float4(read_imageh(B,smp,(int2)(col4+1,k4*4+2)));
    float4 b13=convert_float4(read_imageh(B,smp,(int2)(col4+1,k4*4+3)));
    {''.join(blocks)}
  }}
  {''.join(stores)}
}}"""


def main() -> None:
  n, seed = int(os.getenv("N", "512")), int(os.getenv("SEED", "901"))
  if n % 256: raise ValueError("N must be divisible by 256")
  rng = np.random.default_rng(seed)
  a_np = (rng.standard_normal((n, n), dtype=np.float32)/32).astype(np.float16)
  b_np = (rng.standard_normal((n, n), dtype=np.float32)/32).astype(np.float16)
  dev = Device["QCOM"]
  a, b, c = alloc(n*n, dtypes.half), alloc(n*n, dtypes.half), alloc(n*n, dtypes.float)
  a.copyin(memoryview(a_np).cast("B"))
  b.copyin(memoryview(b_np).cast("B"))
  lib = dev.compiler.compile(source(n))
  prg = dev.runtime("gemm", lib, buf_dtypes=[
    ((0, dtypes.float, (n, n//4, 4)),), ((0, dtypes.half, (n, n//4, 4)),), ((1, dtypes.half, (n, n//4, 4)),)])
  groups = (n//4)*(n//8)//128
  times = [prg(c._buf, a._buf, b._buf, global_size=(groups, 1, 1), local_size=(128, 1, 1), wait=True) for _ in range(5)]
  got = np.empty((n, n), np.float32)
  c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected)
  bad = int(np.count_nonzero(~np.isclose(got, expected, rtol=1e-4, atol=1e-4)))
  elapsed = min(times)
  print(f"shape={n}x{n}x{n} algorithm=waksman4 inputs=fp16 accumulate=fp32 elapsed_ms={elapsed*1e3:.3f} "
        f"gflops={2*n**3/elapsed/1e9:.1f} max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} bad_count={bad} "
        f"fregs={prg.fregs} hregs={prg.hregs} max_threads={prg.max_threads}")
  if bad: raise SystemExit(1)


if __name__ == "__main__": main()
