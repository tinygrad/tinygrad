#!/usr/bin/env python3
"""Fully checked two-level Strassen FP16 GEMM for Adreno 630."""
import os
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm import qcom_intensity_gemm as q4
from extra.gemm.ir3asm import get_envelope, inject


def add(*xs: np.ndarray) -> np.ndarray: return sum(xs, np.zeros_like(xs[0]))
def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray: return x-y


def operands(blocks: list[np.ndarray], side: str) -> list[np.ndarray]:
  x11, x12, x21, x22 = blocks
  if side == "A": return [add(x11,x22), add(x21,x22), x11, x22, add(x11,x12), sub(x21,x11), sub(x12,x22)]
  return [add(x11,x22), x11, sub(x12,x22), sub(x21,x11), x22, add(x11,x12), add(x21,x22)]


def combine(ms: list[np.ndarray]) -> list[np.ndarray]:
  m1,m2,m3,m4,m5,m6,m7 = ms
  return [m1+m4-m5+m7, m3+m5, m2+m4, m1-m2+m3+m6]


def operand_coeffs(side: str) -> list[np.ndarray]:
  eye = np.eye(16, dtype=np.int32).reshape(4,4,16)
  top = operands([eye[:2,:2], eye[:2,2:], eye[2:,:2], eye[2:,2:]], side)
  return [x for t in top for x in operands([t[0,0],t[0,1],t[1,0],t[1,1]], side)]


def output_coeffs() -> list[np.ndarray]:
  eye = np.eye(49, dtype=np.int32).reshape(7,7,49)
  inner = [np.array(combine(list(eye[p]))).reshape(2,2,49) for p in range(7)]
  outer = combine(inner)
  grid = np.empty((4,4,49),np.int32)
  grid[:2,:2],grid[:2,2:],grid[2:,:2],grid[2:,2:] = outer
  return [grid[r,c] for r in range(4) for c in range(4)]


def expr(coeff: np.ndarray, names: list[str]) -> str:
  terms: list[str] = []
  for c, name in zip(coeff.tolist(), names):
    terms += ([name]*c if c > 0 else [f"(-{name})"]*(-c))
  return "+".join(terms) if terms else "(half4)(0)"


def prep_src(side: str) -> str:
  coeffs, names = operand_coeffs(side), [f"x{i}" for i in range(16)]
  lines = ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
           "__attribute__((reqd_work_group_size(128,1,1)))",
           "__kernel void prep(__global const half *X,__global half *O){",
           "int i=get_global_id(0),r=i>>6,c=(i&63)<<2,o=r*256+c;"]
  for br in range(4):
    for bc in range(4):
      j=br*4+bc
      lines.append(f"half4 x{j}=vload4(0,X+({br}*256+r)*1024+{bc}*256+c);")
  for p, c in enumerate(coeffs): lines.append(f"vstore4({expr(c,names)},0,O+{p}*65536+o);")
  return "\n".join(lines+["}"])


def post_src() -> str:
  coeffs = output_coeffs()
  lines = ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
           "__attribute__((reqd_work_group_size(128,1,1)))",
           "__kernel void post(__global const half *M,__global half *C){",
           "int i=get_global_id(0),block=i>>14,j=i&16383,r=j>>6,c=(j&63)<<2; half4 v=(half4)(0);"]
  for block, coeff in enumerate(coeffs):
    nz = np.flatnonzero(coeff)
    loads = [f"vload4(0,M+{p}*65536+r*256+c)" for p in nz]
    e = expr(coeff[nz], loads)
    lines.append(f"{'if' if block == 0 else 'else if'}(block=={block}) v={e};")
  lines += ["int br=block>>2,bc=block&3;vstore4(v,0,C+(br*256+r)*1024+bc*256+c);", "}"]
  return "\n".join(lines)


def alloc_half(count: int) -> Buffer: return Buffer("QCOM", count, dtypes.half).allocate()


def main() -> None:
  seed, runs = int(os.getenv("SEED", "367")), int(os.getenv("BENCH_RUNS", "5"))
  rng = np.random.default_rng(seed)
  a_np=(rng.standard_normal((1024,1024))*0.05).astype(np.float16)
  b_np=(rng.standard_normal((1024,1024))*0.05).astype(np.float16)
  dev=Device["QCOM"]
  a,b,pa,pb,pm,c = alloc_half(1024**2),alloc_half(1024**2),alloc_half(49*256**2),alloc_half(49*256**2),alloc_half(49*256**2),alloc_half(1024**2)
  a.copyin(memoryview(a_np).cast("B")); b.copyin(memoryview(b_np).cast("B"))
  spec=((0,dtypes.half,None),)
  prepa=dev.runtime("prep",dev.compiler.compile(prep_src("A")),buf_dtypes=[spec,spec])
  prepb=dev.runtime("prep",dev.compiler.compile(prep_src("B")),buf_dtypes=[spec,spec])
  post=dev.runtime("post",dev.compiler.compile(post_src()),buf_dtypes=[spec,spec])

  q8.M=q8.N=q8.K=256; q8.K4=64
  env,io,sz,ro=get_envelope(dev,q4.make_direct_image_donor_src(4,128))
  shader,hregs,fregs,_=q8.build_8x8_persistent_shader(dev,128,batch_m=256,batch_n=256,batch_k=256,
    batch_fixed_b=-2,dynamic_a4_dual=True,image_store=True)
  lib=inject(env,io,sz,ro,shader,fregs=fregs,hregs=hregs,mergedregs=False)
  images=[((0,dtypes.half,(49*256,64,4)),),((0,dtypes.half,(49*256,64,4)),),((1,dtypes.half,(49*256,64,4)),)]
  gemm=dev.runtime("gemm_h",lib,buf_dtypes=images)

  def iteration() -> tuple[float,float,float,float]:
    ta=prepa(a._buf,pa._buf,global_size=(128,1,1),local_size=(128,1,1),wait=True)
    tb=prepb(b._buf,pb._buf,global_size=(128,1,1),local_size=(128,1,1),wait=True)
    tg=gemm(pm._buf,pa._buf,pb._buf,global_size=(1,8,49),local_size=(128,1,1),wait=True)
    to=post(pm._buf,c._buf,global_size=(2048,1,1),local_size=(128,1,1),wait=True)
    return ta,tb,tg,to

  for _ in range(2): iteration()
  measured=[iteration() for _ in range(runs)]
  best=min(measured,key=sum); elapsed=sum(best)
  got=np.empty((1024,1024),np.float16); c.copyout(memoryview(got).cast("B"))
  expected=a_np.astype(np.float32)@b_np.astype(np.float32)
  delta=np.abs(got.astype(np.float32)-expected); correct=np.allclose(got,expected,rtol=2e-2,atol=2e-2)
  bad=~np.isfinite(got)|(delta>.02)
  print(f"shape=1024x1024x1024 algorithm=strassen2 accumulate=fp16 elapsed_ms={elapsed*1e3:.3f} gflops={2*1024**3/elapsed/1e9:.1f} "
        f"prepA_ms={best[0]*1e3:.3f} prepB_ms={best[1]*1e3:.3f} gemm_ms={best[2]*1e3:.3f} post_ms={best[3]*1e3:.3f} "
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} allclose={correct}")
  print(f"bad_count={int(bad.sum())}")
  if not correct: raise SystemExit(1)


if __name__ == "__main__": main()
