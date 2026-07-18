#!/usr/bin/env python3
"""One-level full Strassen GEMM with operand transforms fused into FP32 leaves."""
import os

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


def alloc(count: int, dtype) -> Buffer: return Buffer("QCOM", count, dtype).allocate()


def leaf_src(p: int, n: int = 512) -> str:
  h, h4 = n//2, n//8
  ae = (("a0+a3"), ("a2+a3"), "a0", "a3", "a0+a1", "a2-a0", "a1-a3")[p]
  be = (("b0+b3"), "b0", "b1-b3", "b2-b0", "b3", "b0+b1", "b2+b3")[p]
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void leaf(write_only image2d_t C,read_only image2d_t A,read_only image2d_t B) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col=get_group_id(0)*32+tid;
  float4 r0=(float4)(0),r1=(float4)(0),r2=(float4)(0),r3=(float4)(0);
  for(int k4=0;k4<{h4};k4++) {{
    float4 a0=convert_float4(read_imageh(A,smp,(int2)(k4,row+0)));
    float4 a1=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+0)));
    float4 a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+{h}+0)));
    float4 a3=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+{h}+0))),aa0={ae};
    a0=convert_float4(read_imageh(A,smp,(int2)(k4,row+1)));a1=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+1)));
    a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+{h}+1)));a3=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+{h}+1)));float4 aa1={ae};
    a0=convert_float4(read_imageh(A,smp,(int2)(k4,row+2)));a1=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+2)));
    a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+{h}+2)));a3=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+{h}+2)));float4 aa2={ae};
    a0=convert_float4(read_imageh(A,smp,(int2)(k4,row+3)));a1=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+3)));
    a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+{h}+3)));a3=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+{h}+3)));float4 aa3={ae};
    float4 b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+0)));
    float4 b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+0)));
    float4 b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h})));
    float4 b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h})));
    float4 bb0={be};
    b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+1)));
    b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+1)));
    b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h}+1)));
    b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h}+1)));
    float4 bb1={be};
    b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+2)));
    b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+2)));
    b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h}+2)));
    b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h}+2)));
    float4 bb2={be};
    b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+3)));
    b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+3)));
    b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h}+3)));
    b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h}+3)));
    float4 bb3={be};
    r0+=aa0.x*bb0+aa0.y*bb1+aa0.z*bb2+aa0.w*bb3;
    r1+=aa1.x*bb0+aa1.y*bb1+aa1.z*bb2+aa1.w*bb3;
    r2+=aa2.x*bb0+aa2.y*bb1+aa2.z*bb2+aa2.w*bb3;
    r3+=aa3.x*bb0+aa3.y*bb1+aa3.z*bb2+aa3.w*bb3;
  }}
  write_imagef(C,(int2)(col,row+0),r0);write_imagef(C,(int2)(col,row+1),r1);
  write_imagef(C,(int2)(col,row+2),r2);write_imagef(C,(int2)(col,row+3),r3);
}}"""


def combine_src(n: int = 512) -> str:
  h = n//2
  rd = lambda p: f"read_imagef(M,smp,(int2)(x,{p*h}+r))"  # noqa: E731
  return f"""const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void combine(write_only image2d_t C,read_only image2d_t M) {{
  uint i=get_global_id(0),q=i/{h*h//4},j=i%{h*h//4},r=j/{h//4},x=j%{h//4};float4 v;
  if(q==0)v={rd(0)}+{rd(3)}-{rd(4)}+{rd(6)};
  else if(q==1)v={rd(2)}+{rd(4)};else if(q==2)v={rd(1)}+{rd(3)};
  else v={rd(0)}-{rd(1)}+{rd(2)}+{rd(5)};
  write_imagef(C,(int2)(x+(q&1)*{h//4},r+(q>>1)*{h}),v);
}}"""


def main() -> None:
  n, seed = 512, int(os.getenv("SEED", "701"))
  rng = np.random.default_rng(seed)
  a_np = rng.normal(0, 1/32, (n, n)).astype(np.float16)
  b_np = rng.normal(0, 1/32, (n, n)).astype(np.float16)
  dev = Device["QCOM"]
  a, b, mm, c = alloc(n*n, dtypes.half), alloc(n*n, dtypes.half), alloc(7*n*n//4, dtypes.float), alloc(n*n, dtypes.float)
  a.copyin(memoryview(a_np).cast("B"))
  b.copyin(memoryview(b_np).cast("B"))
  leaves = [dev.runtime("leaf", dev.compiler.compile(leaf_src(p)), buf_dtypes=[
    ((0, dtypes.float, (n//2, n//8, 4)),), ((0, dtypes.half, (n, n//4, 4)),), ((1, dtypes.half, (n, n//4, 4)),)]) for p in range(7)]
  combine = dev.runtime("combine", dev.compiler.compile(combine_src()), buf_dtypes=[
    ((0, dtypes.float, (n, n//4, 4)),), ((0, dtypes.float, (7*n//2, n//8, 4)),)])
  times = []
  for p, prg in enumerate(leaves):
    out = mm._buf.offset(p*n*n, n*n)
    times.append(prg(out, a._buf, b._buf, global_size=(2, 16, 1), local_size=(128, 1, 1), wait=True))
  times.append(combine(c._buf, mm._buf, global_size=(n*n//4//256, 1, 1), local_size=(256, 1, 1), wait=True))
  got = np.empty((n, n), np.float32)
  c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected)
  elapsed = sum(times)
  print(f"elapsed_ms={elapsed*1e3:.3f} gflops={2*n**3/elapsed/1e9:.1f} leaf_ms={sum(times[:-1])*1e3:.3f} "
        f"combine_ms={times[-1]*1e3:.3f} max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} "
        f"allclose={np.allclose(got, expected, rtol=1e-3, atol=1e-3)} leaf_parts_ms={[round(x*1e3, 3) for x in times[:-1]]}")


if __name__ == "__main__": main()
