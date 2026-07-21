#!/usr/bin/env python3
"""Full one-level Strassen GEMM with all transforms and combines fused into one kernel."""
import os

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import (ADD_F, ADD_S, ADD_S_REG, AND_B, BR, CMPS_S_EQ, END, MAD_F32, MOV_F32, MOV_S32,
                               NOP, SHR_B, STIB_F32, SUB_F, assemble, get_envelope, inject)


def alloc(count: int, dtype) -> Buffer: return Buffer("QCOM", count, dtype).allocate()


def source(n: int = 512, wg: int = 128) -> str:
  h, h4 = n//2, n//8
  ae = ("a0+a3", "a2+a3", "a0", "a3", "a0+a1", "a2-a0", "a1-a3")
  be = ("b0+b3", "b0", "b1-b3", "b2-b0", "b3", "b0+b1", "b2+b3")
  contributions = (((0, 1), (3, 1)), ((2, 1), (3, -1)), ((1, 1), (3, 1)), ((0, 1), (2, 1)),
                   ((0, -1), (1, 1)), ((3, 1),), ((0, 1),))
  phases = []
  for p in range(7):
    updates = []
    for quadrant, sign in contributions[p]:
      op = "+=" if sign > 0 else "-="
      updates.extend((f"c{quadrant}0{op}m0;", f"c{quadrant}1{op}m1;"))
    phases.append(f"""
  {{ float4 m0=(float4)(0),m1=(float4)(0);
    for(int k4=0;k4<{h4};k4++) {{
      float4 a0=convert_float4(read_imageh(A,smp,(int2)(k4,row+0)));
      float4 a1=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+0)));
      float4 a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+{h}+0)));
      float4 a3=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+{h}+0))),aa0={ae[p]};
      a0=convert_float4(read_imageh(A,smp,(int2)(k4,row+1)));a1=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+1)));
      a2=convert_float4(read_imageh(A,smp,(int2)(k4,row+{h}+1)));a3=convert_float4(read_imageh(A,smp,(int2)(k4+{h4},row+{h}+1)));
      float4 aa1={ae[p]};
      float4 b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+0)));
      float4 b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+0)));
      float4 b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h}+0)));
      float4 b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h}+0))),bb0={be[p]};
      b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+1)));b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+1)));
      b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h}+1)));b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h}+1)));
      float4 bb1={be[p]};
      b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+2)));b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+2)));
      b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h}+2)));b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h}+2)));
      float4 bb2={be[p]};
      b0=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+3)));b1=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+3)));
      b2=convert_float4(read_imageh(B,smp,(int2)(col,k4*4+{h}+3)));b3=convert_float4(read_imageh(B,smp,(int2)(col+{h4},k4*4+{h}+3)));
      float4 bb3={be[p]};
      m0+=aa0.x*bb0+aa0.y*bb1+aa0.z*bb2+aa0.w*bb3;
      m1+=aa1.x*bb0+aa1.y*bb1+aa1.z*bb2+aa1.w*bb3;
    }}
    {''.join(updates)} }}""")
  stores = []
  for quadrant in range(4):
    qr, qc = quadrant>>1, quadrant&1
    stores.extend((f"write_imagef(C,(int2)(col+{qc*h4},row+{qr*h}+0),c{quadrant}0);",
                   f"write_imagef(C,(int2)(col+{qc*h4},row+{qr*h}+1),c{quadrant}1);"))
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void gemm(write_only image2d_t C,read_only image2d_t A,read_only image2d_t B) {{
  int lid=get_local_id(0),tm=lid>>5,tid=lid&31;
  int row=get_group_id(1)*8+tm*2,col=get_group_id(0)*32+tid;
  float4 c00=(float4)(0),c01=(float4)(0),c10=(float4)(0),c11=(float4)(0);
  float4 c20=(float4)(0),c21=(float4)(0),c30=(float4)(0),c31=(float4)(0);
  {''.join(phases)}
  {''.join(stores)}
}}"""


def hand_shader(dev, n: int = 512) -> tuple[bytes, int, int]:
  h, h4 = n//2, n//8
  instrs = q.prologue_4x2(dev, 128)
  # Convert the donor's 4x8-within-full-output coordinates to a 2x4 tile
  # within one quadrant: row/=2 and group-column stride/=2, preserving tid.
  instrs += [SHR_B("r7.x", "r7.x", 1), AND_B("r6.x", "r7.y", 31),
             AND_B("r7.y", "r7.y", 0xc0), SHR_B("r7.y", "r7.y", 1),
             ADD_S_REG("r7.y", "r7.y", "r6.x"), NOP(rpt=2)]
  acc0, mi0, avec0, bvec0, tmp, state, const = 0, 8, 10, 12, 16, 17, 18
  row_base, col_base, k4, ky = (q.fvec(state, x) for x in range(4))
  hreg, h4reg = q.fvec(const, 0), q.fvec(const, 1)
  instrs += [MOV_F32(row_base, "r7.x"), MOV_F32(col_base, "r7.y"), MOV_S32(hreg, h), MOV_S32(h4reg, h4)]
  for vec in range(acc0, acc0+8): q.emit_f32_vec_imm(instrs, vec, 0)

  ta = (((0, 1), (3, 1)), ((2, 1), (3, 1)), ((0, 1),), ((3, 1),),
        ((0, 1), (1, 1)), ((2, 1), (0, -1)), ((1, 1), (3, -1)))
  tb = (((0, 1), (3, 1)), ((0, 1),), ((1, 1), (3, -1)), ((2, 1), (0, -1)),
        ((3, 1),), ((0, 1), (1, 1)), ((2, 1), (3, 1)))
  contributions = (((0, 1), (3, 1)), ((2, 1), (3, -1)), ((1, 1), (3, 1)), ((0, 1), (2, 1)),
                   ((0, -1), (1, 1)), ((3, 1),), ((0, 1),))

  def coords(dst: int, tex: int, quadrant: int, item: int) -> None:
    qr, qc = quadrant>>1, quadrant&1
    if tex == 0:
      instrs.append(MOV_F32(q.fvec(dst, 0), k4) if not qc else ADD_S_REG(q.fvec(dst, 0), k4, h4reg))
      instrs.append(MOV_F32(q.fvec(dst, 1), row_base) if not qr else ADD_S_REG(q.fvec(dst, 1), row_base, hreg))
      if item: instrs.append(ADD_S(q.fvec(dst, 1), q.fvec(dst, 1), item))
    else:
      instrs.append(MOV_F32(q.fvec(dst, 0), col_base) if not qc else ADD_S_REG(q.fvec(dst, 0), col_base, h4reg))
      instrs.append(MOV_F32(q.fvec(dst, 1), ky) if not qr else ADD_S_REG(q.fvec(dst, 1), ky, hreg))
      if item != 3: instrs.append(ADD_S(q.fvec(dst, 1), q.fvec(dst, 1), item-3))

  def load_expr(dst: int, tex: int, terms: tuple[tuple[int, int], ...], item: int) -> None:
    nonlocal instrs
    coords(dst, tex, terms[0][0], item)
    q.emit_isam_f32_vec(instrs, dst, q.fvec(dst, 0), tex, True)
    if len(terms) == 1: return
    coords(tmp, tex, terms[1][0], item)
    q.emit_isam_f32_vec(instrs, tmp, q.fvec(tmp, 0), tex, True)
    instrs += [MOV_F32(q.fvec(dst), q.fvec(dst), sy=True), NOP(rpt=5)]
    op = ADD_F if terms[1][1] > 0 else SUB_F
    for comp in range(4): instrs.append(op(dst*4+comp, dst*4+comp, tmp*4+comp))

  for p in range(7):
    for vec in range(mi0, mi0+2): q.emit_f32_vec_imm(instrs, vec, 0)
    instrs += [MOV_S32(k4, 0), MOV_S32(ky, 3)]
    loop_start = len(instrs)
    for kk in range(4): load_expr(bvec0+kk, 1, tb[p], kk)
    for row in range(2): load_expr(avec0+row, 0, ta[p], row)
    first = True
    for kk in range(4):
      for row in range(2):
        instrs.append(MAD_F32(q.fvec(mi0+row), q.fvec(avec0+row, kk), q.fvec(bvec0+kk),
                              q.fvec(mi0+row), rpt=3, sy=first, r=True))
        first = False
    instrs += [ADD_S("r6.x", k4, 1), ADD_S(ky, ky, 4), CMPS_S_EQ(k4, h4-1, nop=1),
               MOV_F32(k4, "r6.x"), NOP(rpt=3)]
    loop_end = len(instrs)
    instrs.append(BR(loop_start-loop_end))
    instrs += [MOV_F32(q.fvec(mi0), q.fvec(mi0), sy=True), NOP(rpt=3)]
    for quadrant, sign in contributions[p]:
      for row in range(2):
        dst, src = acc0+quadrant*2+row, mi0+row
        op = ADD_F if sign > 0 else SUB_F
        for comp in range(4): instrs.append(op(dst*4+comp, dst*4+comp, src*4+comp))

  instrs += [MOV_F32("r6.x", "r6.x", sy=True), NOP(rpt=8)]
  for quadrant in range(4):
    qr, qc = quadrant>>1, quadrant&1
    for row in range(2):
      instrs.append(MOV_F32("r4.x", col_base) if not qc else ADD_S_REG("r4.x", col_base, h4reg))
      instrs.append(MOV_F32("r4.y", row_base) if not qr else ADD_S_REG("r4.y", row_base, hreg))
      if row: instrs.append(ADD_S("r4.y", "r4.y", row))
      instrs += [MOV_F32(q.fvec(acc0+quadrant*2+row), q.fvec(acc0+quadrant*2+row), sy=True), NOP(rpt=5),
                 STIB_F32(q.fvec(acc0+quadrant*2+row), "r4.x"), NOP(rpt=16)]
  instrs.append(END())
  return assemble(instrs), 1, 19


def main() -> None:
  n, seed = 512, int(os.getenv("SEED", "701"))
  rng = np.random.default_rng(seed)
  a_np = rng.normal(0, 1/32, (n, n)).astype(np.float16)
  b_np = rng.normal(0, 1/32, (n, n)).astype(np.float16)
  dev = Device["QCOM"]
  a, b, c = alloc(n*n, dtypes.half), alloc(n*n, dtypes.half), alloc(n*n, dtypes.float)
  a.copyin(memoryview(a_np).cast("B"))
  b.copyin(memoryview(b_np).cast("B"))
  src = source()
  if bool(int(os.getenv("HAND", "0"))):
    q.M = q.N = q.K = n
    q.K4 = n//4
    env, io, sz, ro = get_envelope(dev, src)
    shader, hregs, fregs = hand_shader(dev, n)
    if len(shader) > sz: raise RuntimeError(f"shader {len(shader)} exceeds envelope {sz}")
    lib = inject(env, io, sz, ro, shader, hregs=hregs, fregs=fregs, mergedregs=False)
  else: lib = dev.compiler.compile(src)
  prg = dev.runtime("gemm", lib, buf_dtypes=[
    ((0, dtypes.float, (n, n//4, 4)),), ((0, dtypes.half, (n, n//4, 4)),), ((1, dtypes.half, (n, n//4, 4)),)])
  times = [prg(c._buf, a._buf, b._buf, global_size=(2, 32, 1), local_size=(128, 1, 1), wait=True) for _ in range(5)]
  got = np.empty((n, n), np.float32)
  c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected)
  elapsed = min(times)
  print(f"shape={n}x{n}x{n} algorithm=strassen1_fused inputs=fp16 accumulate=fp32 elapsed_ms={elapsed*1e3:.3f} "
        f"gflops={2*n**3/elapsed/1e9:.1f} max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} "
        f"allclose={np.allclose(got, expected, rtol=1e-3, atol=1e-3)}")


if __name__ == "__main__": main()
