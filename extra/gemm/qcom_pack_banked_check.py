#!/usr/bin/env python3
"""Check FP32 banked-image to row-major FP16 packing on QCOM."""
import argparse

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--m", type=int, default=192)
  ap.add_argument("--k", type=int, default=768)
  ap.add_argument("--constant", action="store_true")
  args = ap.parse_args()
  rng = np.random.default_rng(0)
  source = rng.standard_normal((args.m//4, args.k, 4)).astype(np.float32)
  expected = source.transpose(0, 2, 1).reshape(args.m, args.k).astype(np.float16)
  inp = Buffer("QCOM", source.size, dtypes.float).allocate()
  out = Buffer("QCOM", source.size, dtypes.half).allocate()
  inp.copyin(memoryview(source).cast("B"))
  out.copyin(memoryview(np.zeros(source.size, np.float16)).cast("B"))
  kernel = f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void pack_banked(__global uint *O, read_only image2d_t X) {{
  int t=get_global_id(0),row=t/{args.k//4},k4=t-row*{args.k//4},lane=row&3,y=row>>2,x=k4*4;
  float4 p0=read_imagef(X,smp,(int2)(x+0,y)),p1=read_imagef(X,smp,(int2)(x+1,y));
  float4 p2=read_imagef(X,smp,(int2)(x+2,y)),p3=read_imagef(X,smp,(int2)(x+3,y));
  float4 v=lane==0?(float4)(p0.x,p1.x,p2.x,p3.x):lane==1?(float4)(p0.y,p1.y,p2.y,p3.y):
           lane==2?(float4)(p0.z,p1.z,p2.z,p3.z):(float4)(p0.w,p1.w,p2.w,p3.w);
  vstore2(as_uint2({"(half4)(1.0h)" if args.constant else "convert_half4(v)"}),0,O+t*2);
}}"""
  specs = [((0, dtypes.half, None),), ((1, dtypes.float, (args.m//4, args.k, 4)),)]
  program = Device["QCOM"].runtime("pack_banked", Device["QCOM"].compiler.compile(kernel), buf_dtypes=specs)
  times = [program(out._buf, inp._buf, global_size=(args.m*args.k//(4*128), 1, 1),
                   local_size=(128, 1, 1), wait=True)*1e3 for _ in range(5)]
  got = np.empty_like(expected)
  out.copyout(memoryview(got).cast("B"))
  delta = np.abs(got.astype(np.float32)-expected.astype(np.float32))
  print(f"ms={min(times):.4f} max_abs={float(delta.max()):.9g} nonzero={int(np.count_nonzero(got))}/{got.size}")


if __name__ == "__main__": main()
