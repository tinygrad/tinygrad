#!/usr/bin/env python3
"""Randomized GEMM using four planar B textures to remove coordinate churn."""
import os
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm import qcom_intensity_gemm as q4
from extra.gemm.ir3asm import *
from extra.gemm.ir3asm import _hreg


def make_envelope_src(k4: int, n: int, threads: int = 128) -> str:
  src = f'''#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({threads},1,1)))
__kernel void gemm_h(write_only image2d_t C,read_only image2d_t A,read_only image2d_t B0,
                     read_only image2d_t B1,read_only image2d_t B2,read_only image2d_t B3) {{
  int lid=get_local_id(0), row=get_group_id(1)*32+(lid>>5)*8;
  int col4=get_group_id(0)*64+(lid&31);
  half4 c[16]; for(int i=0;i<16;i++) c[i]=(half4)(0);
  for(int k4=0;k4<{k4};k4++) {{
    half4 a0=read_imageh(A,smp,(int2)(k4,row));
    half4 b0=read_imageh(B0,smp,(int2)(col4,k4));
    half4 b1=read_imageh(B1,smp,(int2)(col4,k4));
    half4 b2=read_imageh(B2,smp,(int2)(col4,k4));
    half4 b3=read_imageh(B3,smp,(int2)(col4,k4));
    c[0]+=a0.xxxx*b0+a0.yyyy*b1+a0.zzzz*b2+a0.wwww*b3;
  }}
  for(int i=0;i<16;i++) write_imageh(C,(int2)(col4+(i&1)*32,row+(i>>1)),c[i]);
}}'''
  pad = ''.join('  c[0]=mad(c[0],(half4)(1.0009765625h),(half4)(0.0009765625h));\n' for _ in range(512))
  return src.replace('  for(int i=0;i<16;i++) write_imageh', pad+'  for(int i=0;i<16;i++) write_imageh')


def build_shader(dev, threads: int, k4: int, unroll: int, no_store: bool = False, control_only: bool = False,
                 post_constant: bool = False) -> tuple[bytes, int, int]:
  instrs = q8.prologue_8x4(dev, threads)
  q8.emit_col_stride(instrs, 2)
  kz, row, col, col1 = 'r6.z', 'r7.x', 'r7.y', 'r6.w'
  acc0 = _hreg('hr12.x')
  for base in range(acc0, acc0+64, 4): q8.emit_hvec_imm(instrs, base, 0)
  instrs += [ADD_S(col1, col, 32),
             MOV_F32('r14.x', col), MOV_F32('r14.z', col1)]
  # Four independent sampler coordinate pairs avoid source WAR hazards between
  # the four outstanding A reads.  They are reused only after the first MAD group.
  a_coords = (('r15.x', 'r15.y'), ('r15.z', 'r15.w'),
              ('r16.x', 'r16.y'), ('r16.z', 'r16.w'))
  b_regs = [[_hreg(f'hr{i}.x') for i in range(4)], [_hreg(f'hr{i}.x') for i in range(4, 8)]]
  a_regs = [_hreg(f'hr{i}.x') for i in range(8, 12)]

  def mad(r: int, c: int, kk: int, sy: bool = False) -> None:
    dst = acc0+(r*2+c)*4
    instrs.append(MAD_F16(dst, a_regs[r&3]+kk, b_regs[c][kk], dst, rpt=3, sy=sy, r=True))

  loop_start = len(instrs)
  for ku in range(unroll):
    if ku: instrs += [ADD_S(kz, kz, 1), NOP(rpt=2)]
    if control_only: continue
    instrs += [MOV_F32('r14.y', kz), MOV_F32('r14.w', kz)]
    for c, cr in enumerate(('r14.x', 'r14.z')):
      for kk in range(4): instrs.append(ISAM_F16(b_regs[c][kk], cr, 1+kk))
    instrs += [MOV_F32(ac[0], kz) for ac in a_coords]
    for r in range(4):
      ac = a_coords[r]
      instrs += [MOV_F32(ac[1], row) if r == 0 else ADD_S(ac[1], row, r), NOP(rpt=int(os.getenv('ACOORD_DELAY', '5'))),
                 ISAM_F16(a_regs[r], ac[0], 0)]
    first = True
    for r in range(4):
      for c in range(2):
        for kk in range(4):
          mad(r, c, kk, first)
          first = False
    for r in range(4):
      ac = a_coords[r]
      instrs += [ADD_S(ac[1], row, r+4), NOP(rpt=int(os.getenv('ACOORD_DELAY', '5'))), ISAM_F16(a_regs[r], ac[0], 0)]
    first = True
    for r in range(4, 8):
      for c in range(2):
        for kk in range(4):
          mad(r, c, kk, first)
          first = False
  instrs += [ADD_S('r0.x', kz, 1), NOP(rpt=2), CMPS_S_EQ(kz, k4-1, nop=1), MOV_F32(kz, 'r0.x'), NOP(rpt=3)]
  loop_end = len(instrs)
  instrs.append(BR(loop_start-loop_end))
  if post_constant:
    for base in range(acc0, acc0+64, 4): q8.emit_hvec_imm(instrs, base, 0x6400)
  if not no_store:
    for r in range(8):
      for c in range(2):
        instrs += [MOV_F32('r14.x', col) if c == 0 else ADD_S('r14.x', col, 32),
                   MOV_F32('r14.y', row) if r == 0 else ADD_S('r14.y', row, r),
                   COV_F16F32('r19.x', acc0+(r*2+c)*4, sy=True, rpt=3, r=True), NOP(rpt=5),
                   STIB_F32('r19.x', 'r14.x'), NOP(rpt=16)]
  instrs.append(END())
  return assemble(instrs), 28, 17


def main() -> None:
  m, n, k = (int(os.getenv(x, '1024')) for x in ('M','N','K'))
  seed, threads, unroll = int(os.getenv('SEED','0')), 128, int(os.getenv('KUNROLL','4'))
  if m%32 or n%256 or k%4 or (k//4)%unroll: raise ValueError('unsupported tile shape')
  rng = np.random.default_rng(seed)
  a = (rng.standard_normal((m,k))*0.05).astype(np.float16)
  b = (rng.standard_normal((k,n))*0.05).astype(np.float16)
  planes = [np.ascontiguousarray(b[p::4]) for p in range(4)]
  q8.M=q4.M=m; q8.N=q4.N=n; q8.K=q4.K=k; q8.K4=q4.K4=k//4
  dev = Device['QCOM']
  env, io, sz, ro = get_envelope(dev, make_envelope_src(k//4, n, threads))
  no_store = bool(int(os.getenv('NO_STORE', '0')))
  build_k4 = int(os.getenv('BUILD_K4', str(k//4)))
  shader, hregs, fregs = build_shader(dev, threads, build_k4, unroll, no_store,
                                      bool(int(os.getenv('CONTROL_ONLY', '0'))), bool(int(os.getenv('POST_CONSTANT', '0'))))
  if len(shader)>sz: raise ValueError(f'shader {len(shader)} > envelope {sz}')
  if int(os.getenv('DUMP', '0')):
    print(f'shader_bytes={len(shader)} envelope_bytes={sz} fregs={fregs} hregs={hregs}')
    print(disasm(shader))
    return
  lib = bytes(env) if int(os.getenv('COMPILER', '0')) else \
        inject(env, io, sz, ro, shader, fregs=int(os.getenv('FREGS',str(fregs))), hregs=int(os.getenv('HREGS',str(hregs))),
               mergedregs=False if int(os.getenv('SEPARATE_REGS', '0')) else None)
  ab=Buffer('QCOM',a.size,dtypes.half).allocate(); ab.copyin(memoryview(a).cast('B'))
  pbs=[]
  for p in planes:
    pb=Buffer('QCOM',p.size,dtypes.half).allocate(); pb.copyin(memoryview(p).cast('B')); pbs.append(pb)
  cb=Buffer('QCOM',m*n,dtypes.half).allocate(); cb.copyin(memoryview(np.zeros((m,n),np.float16)).cast('B'))
  specs=[((0,dtypes.half,(m,n//4,4)),),((1,dtypes.half,(m,k//4,4)),),((2,dtypes.half,(k//4,n//4,4)),),
         ((3,dtypes.half,(k//4,n//4,4)),),((4,dtypes.half,(k//4,n//4,4)),),((5,dtypes.half,(k//4,n//4,4)),)]
  prg=dev.runtime('gemm_h',lib,buf_dtypes=specs)
  args=(cb._buf,ab._buf,pbs[0]._buf,pbs[1]._buf,pbs[2]._buf,pbs[3]._buf)
  times=[prg(*args,global_size=(n//256,m//32,1),local_size=(threads,1,1),wait=True) for _ in range(10)]
  if no_store:
    print(f'no_store elapsed_ms={min(times)*1e3:.3f}')
    return
  got=np.empty((m,n),np.float16); cb.copyout(memoryview(got).cast('B'))
  expected=(np.full((m,n),1024,np.float32) if int(os.getenv('POST_CONSTANT','0')) else
            a[:,:build_k4*4].astype(np.float32)@b[:build_k4*4].astype(np.float32)); delta=np.abs(expected-got.astype(np.float32))
  correct=np.allclose(expected,got,rtol=2e-2,atol=2e-2); best=min(times)
  print(f'shape={m}x{n}x{k} planar_b=4 accumulate=fp16 elapsed_ms={best*1e3:.3f} gflops={2*m*n*k/best/1e9:.1f} '
        f'max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} allclose={correct} bad_count={(delta>.02).sum()}')
  if not correct and int(os.getenv('DEBUG', '0')):
    print('bad_by_row', np.count_nonzero(delta > .02, axis=1).tolist())
    print('max_by_row', delta.max(axis=1).tolist())
  if not correct: raise SystemExit(1)


if __name__=='__main__': main()
