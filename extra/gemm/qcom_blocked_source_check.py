#!/usr/bin/env python3
"""Validate and time an OpenCL blocked-half/FP32 GEMM on QCOM."""
import argparse

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


def upload(values:np.ndarray, dtype) -> Buffer:
  return Buffer("QCOM", values.size, dtype, initial_value=np.ascontiguousarray(values).tobytes())


def source(m:int, n:int, k:int, stride:int, block4:int, linear:bool=False, ldib:bool=False) -> str:
  assert m % 16 == 0 and n % 128 == 0 and k % (block4*4) == 0
  image_type = "read_write image2d_t" if ldib else "read_only image1d_buffer_t" if linear else "read_only image2d_t"
  def coord(index:str) -> str: return f"(int2)(({index})&16383,({index})>>14)"
  def a_load(row:str) -> str: return coord(f"({row})*{k//4}+k4") if ldib else f"{row}*{k//4}+k4" if linear else f"(int2)(k4,{row})"
  def b_load(krow:str) -> str: return coord(f"({krow})*{n//4}+col4") if ldib else f"{krow}*{n//4}+col4" if linear else f"(int2)(col4,{krow})"
  image_args = "," if (linear or ldib) else ",smp,"
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void gemm_blocked({image_type} A,{image_type} B,__global float *C) {{
  int lid=get_local_id(0), row=get_group_id(1)*16+(lid>>5)*4;
  int col4=get_group_id(0)*32+(lid&31);
  float4 t0=(float4)(0),t1=(float4)(0),t2=(float4)(0),t3=(float4)(0);
  for(int kb=0;kb<{k//4};kb+={block4}) {{
    half4 h0=(half4)(0),h1=(half4)(0),h2=(half4)(0),h3=(half4)(0);
    #pragma unroll
    for(int q=0;q<{block4};q++) {{
      int k4=kb+q;
      half4 a0=read_imageh(A{image_args}{a_load('row+0')});
      half4 a1=read_imageh(A{image_args}{a_load('row+1')});
      half4 a2=read_imageh(A{image_args}{a_load('row+2')});
      half4 a3=read_imageh(A{image_args}{a_load('row+3')});
      half4 b0=read_imageh(B{image_args}{b_load('k4*4+0')});
      half4 b1=read_imageh(B{image_args}{b_load('k4*4+1')});
      half4 b2=read_imageh(B{image_args}{b_load('k4*4+2')});
      half4 b3=read_imageh(B{image_args}{b_load('k4*4+3')});
      h0+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
      h1+=a1.xxxx*b0+a1.yyyy*b1+a1.zzzz*b2+a1.wwww*b3;
      h2+=a2.xxxx*b0+a2.yyyy*b1+a2.zzzz*b2+a2.wwww*b3;
      h3+=a3.xxxx*b0+a3.yyyy*b1+a3.zzzz*b2+a3.wwww*b3;
    }}
    t0+=convert_float4(h0);t1+=convert_float4(h1);t2+=convert_float4(h2);t3+=convert_float4(h3);
  }}
  vstore4(t0,0,C+(row+0)*{stride}+col4*4);vstore4(t1,0,C+(row+1)*{stride}+col4*4);
  vstore4(t2,0,C+(row+2)*{stride}+col4*4);vstore4(t3,0,C+(row+3)*{stride}+col4*4);
}}"""


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--m", type=int, default=128)
  ap.add_argument("--n", type=int, default=1536)
  ap.add_argument("--k", type=int, default=384)
  ap.add_argument("--stride", type=int, default=2048)
  ap.add_argument("--block4", type=int, default=4)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--float-a", action="store_true", help="sample an FP32 activation image with read_imageh")
  ap.add_argument("--linear", action="store_true", help="use image1d_buffer_t with explicit flattened indices")
  ap.add_argument("--ldib", action="store_true", help="use read-write image2d_t and LDIB with flattened 2D indices")
  args = ap.parse_args()
  rng = np.random.default_rng(args.seed)
  av = (rng.standard_normal((args.m, args.k))*0.05).astype(np.float32 if args.float_a else np.float16)
  bv = (rng.standard_normal((args.k, args.n))*0.05).astype(np.float16)
  a, b = upload(av, dtypes.float if args.float_a else dtypes.half), upload(bv, dtypes.half)
  c = upload(np.zeros(args.m*args.stride, np.float32), dtypes.float)
  src = source(args.m, args.n, args.k, args.stride, args.block4, args.linear, args.ldib)
  if args.ldib:
    ashape = ((args.m*(args.k//4)+16383)//16384, 16384, 4)
    bshape = ((args.k*(args.n//4)+16383)//16384, 16384, 4)
  else:
    ashape = (1, args.m*(args.k//4), 4) if args.linear else (args.m, args.k//4, 4)
    bshape = (1, args.k*(args.n//4), 4) if args.linear else (args.k, args.n//4, 4)
  specs = [((0, dtypes.float if args.float_a else dtypes.half, ashape),),
           ((1, dtypes.half, bshape),), ((2, dtypes.float, (args.m*args.stride,)),)]
  program = Device["QCOM"].runtime("gemm_blocked", Device["QCOM"].compiler.compile(src), buf_dtypes=specs)
  times = [program(a._buf, b._buf, c._buf, global_size=(args.n//128, args.m//16, 1),
                   local_size=(128, 1, 1), wait=True)*1e3 for _ in range(8)]
  storage = c.numpy().reshape(args.m, args.stride)
  got, expected = storage[:, :args.n], av.astype(np.float32) @ bv.astype(np.float32)
  delta = np.abs(got-expected)
  print(f"block4={args.block4} ms={min(times):.4f} max_abs={float(delta.max()):.9g} "
        f"mean_abs={float(delta.mean()):.9g} allclose={np.allclose(got, expected, rtol=1e-2, atol=1e-2)}")


if __name__ == "__main__": main()
