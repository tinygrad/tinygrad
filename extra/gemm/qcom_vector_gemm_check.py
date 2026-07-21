#!/usr/bin/env python3
"""Oracle/benchmark for a 4x16 GEMM using cl_qcom_vector_image_ops."""
import argparse, os, statistics

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import disasm, get_envelope


def make_src(m:int, n:int, k:int, stride:int) -> str:
  del m
  decl = ",".join(f"c{r}{c}=(half4)(0)" for r in range(4) for c in range(4))
  mads = "\n".join(f"c{r}{c}+=a{r}.{'xyzw'[c]}*b{c};" for r in range(4) for c in range(4))
  stores = []
  for r in range(4):
    for p in range(4):
      vals = ",".join(f"c{r}{c}.{'xyzw'[p]}" for c in range(4))
      stores.append(f"vstore4((half4)({vals}),0,C+(row+{r})*{stride}+(col4+{p})*4);")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void vector_gemm(read_only image2d_t A,read_only image2d_t B,__global half *C) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*16+tm*4,col4=get_group_id(0)*128+tid*4;
  half4 {decl};
  for(int k4=0;k4<{k//4};k4++) {{
    half4 a0=read_imageh(A,smp,(int2)(k4,row+0));
    half4 a1=read_imageh(A,smp,(int2)(k4,row+1));
    half4 a2=read_imageh(A,smp,(int2)(k4,row+2));
    half4 a3=read_imageh(A,smp,(int2)(k4,row+3));
    half4 b0=qcom_read_imageh_4x1(B,smp,(float2)(col4,k4*4+0),0);
    half4 b1=qcom_read_imageh_4x1(B,smp,(float2)(col4,k4*4+1),1);
    half4 b2=qcom_read_imageh_4x1(B,smp,(float2)(col4,k4*4+2),2);
    half4 b3=qcom_read_imageh_4x1(B,smp,(float2)(col4,k4*4+3),3);
    {mads}
  }}
  {' '.join(stores)}
}}"""


def upload(x:np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", x.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(x)).cast("B"))
  return ret


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--m", type=int, default=128)
  ap.add_argument("--n", type=int, default=1536)
  ap.add_argument("--k", type=int, default=384)
  ap.add_argument("--stride", type=int, default=2048)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--runs", type=int, default=10)
  ap.add_argument("--disasm", action="store_true")
  args = ap.parse_args()
  if args.m%16 or args.n%512 or args.k%4: raise ValueError("shape must divide 16x512x4 launch tile")
  rng=np.random.default_rng(args.seed)
  a=(rng.standard_normal((args.m,args.k))*.05).astype(np.float16)
  b=(rng.standard_normal((args.k,args.n))*.05).astype(np.float16)
  if int(os.getenv("PATTERN", "0")):
    a.fill(0); b.fill(0)
    a[:, 0] = np.arange(1, args.m+1, dtype=np.float16)
    b[0] = (np.arange(args.n) % 251) + 1
  ab,bb=upload(a,dtypes.half),upload(b,dtypes.half)
  cb=upload(np.zeros((args.m,args.stride),np.float16),dtypes.half)
  dev=Device["QCOM"]
  lib=dev.compiler.compile_cached(make_src(args.m,args.n,args.k,args.stride))
  if args.disasm:
    env, off, size, _ = get_envelope(dev, make_src(args.m,args.n,args.k,args.stride))
    print(disasm(bytes(env[off:off+size])))
  specs=[((0,dtypes.half,(args.m,args.k//4,4)),),((1,dtypes.half,(args.k,args.n//4,4)),),((2,dtypes.half,None),)]
  prg=dev.runtime("vector_gemm",lib,buf_dtypes=specs)
  gs,ls=(args.n//512,args.m//16,1),(128,1,1)
  for _ in range(2): prg(ab._buf,bb._buf,cb._buf,global_size=gs,local_size=ls,wait=True)
  times=[prg(ab._buf,bb._buf,cb._buf,global_size=gs,local_size=ls,wait=True)*1e3 for _ in range(args.runs)]
  storage=np.empty((args.m,args.stride),np.float16); cb.copyout(memoryview(storage).cast("B"))
  got=storage[:,:args.n].astype(np.float32); expected=a.astype(np.float32)@b.astype(np.float32)
  delta=np.abs(got-expected); best=min(times)
  passed=bool(np.allclose(got,expected,rtol=.02,atol=.02))
  print(f"best_ms={best:.4f} median_ms={statistics.median(times):.4f} gflops={2*args.m*args.n*args.k/best/1e6:.1f} "
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} allclose={passed}")
  if int(os.getenv("DEBUG", "0")):
    print("got0", got[0, :64].tolist()); print("exp0", expected[0, :64].tolist())
  if not passed: raise SystemExit(1)


if __name__=="__main__": main()
