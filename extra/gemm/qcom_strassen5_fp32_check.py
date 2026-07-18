#!/usr/bin/env python3
"""Full Strassen GEMM with dense FP16 inputs, FP32 transforms/accumulation, and output validation."""
import os
from dataclasses import dataclass

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject


def transform_src(n: int, side: str, parallel: bool = False, buffer_output: bool = False,
                  fp32: bool = False, fp32_input: bool = False, wg: int = 128) -> str:
  h = n//2
  vals = (["x0+x3", "x2+x3", "x0", "x3", "x0+x1", "x2-x0", "x1-x3"] if side == "A" else
          ["x0+x3", "x0", "x1-x3", "x2-x0", "x3", "x0+x1", "x2+x3"])
  vec, read, write = ("float4", "read_imagef" if fp32_input else "convert_float4(read_imageh", "write_imagef") if fp32 else \
                     ("half4", "read_imageh", "write_imageh")
  def rd(coord: str) -> str:
    call = f"{read}(X,smp,(int2)({coord}))"
    return call+")" if fp32 and not fp32_input else call
  if parallel:
    # One output pixel per invocation avoids seven dependent image stores in a
    # single thread.  It rereads source quadrants, but substantially increases
    # the number of memory operations the GPU can keep in flight.
    cases = "\n".join(f"{'if' if p == 0 else 'else if'}(p=={p}) v={v};" for p, v in enumerate(vals))
    return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform(write_only image2d_t O,read_only image2d_t X) {{
  uint i=get_global_id(0),unit={h*h//4},b=i/(7*unit),j=i%(7*unit),pix=j%unit,p=j/unit,r=pix/{h//4},x=pix%{h//4};
  uint iy=b*{n};
  {vec} x0={rd('x,iy+r')},x1={rd('x+'+str(h//4)+',iy+r')};
  {vec} x2={rd('x,iy+r+'+str(h))},x3={rd('x+'+str(h//4)+',iy+r+'+str(h))},v;
  {cases}
  {write}(O,(int2)(x,(b*7+p)*{h}+r),v);
}}"""
  stores = "\n".join(f"write_imageh(O,(int2)(x,{p*h}+r),{v});" for p, v in enumerate(vals))
  if buffer_output:
    stores = "\n".join(f"O[((b*7+{p})*{h}+r)*{h//4}+x]={v};" for p, v in enumerate(vals))
  elif fp32:
    stores = "\n".join(f"write_imagef(O,(int2)(x,{p*h}+r),{v});" for p, v in enumerate(vals))
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform({'__global '+vec+' *O' if buffer_output else 'write_only image2d_t O'},read_only image2d_t X) {{
  uint i=get_global_id(0),unit={h*h//4},b=i/unit,pix=i%unit,r=pix/{h//4},x=pix%{h//4},iy=b*{n};
  {vec} x0={rd('x,iy+r')},x1={rd('x+'+str(h//4)+',iy+r')};
  {vec} x2={rd('x,iy+r+'+str(h))},x3={rd('x+'+str(h//4)+',iy+r+'+str(h))};
  {stores if buffer_output else stores.replace('(x,', '(x,b*7*'+str(h)+'+')}
}}"""


def combine_src(n: int, wg: int = 128) -> str:
  h = n//2
  def rd(p: int) -> str: return f"read_imagef(M,smp,(int2)(xx,(b*7+{p})*{h}+rr))"
  return f"""const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(write_only image2d_t C,read_only image2d_t M) {{
  uint i=get_global_id(0),unit={n*n//4},b=i/unit,pix=i%unit,r=pix/{n//4},x=pix%{n//4},rr=r%{h},xx=x%{h//4}; float4 v;
  if(r<{h}&&x<{h//4}) v={rd(0)}+{rd(3)}-{rd(4)}+{rd(6)};
  else if(r<{h}) v={rd(2)}+{rd(4)};
  else if(x<{h//4}) v={rd(1)}+{rd(3)};
  else v={rd(0)}-{rd(1)}+{rd(2)}+{rd(5)};
  write_imagef(C,(int2)(x,b*{n}+r),v);
}}"""


def alloc(count: int, dtype) -> Buffer:
  return Buffer("QCOM", count, dtype).allocate()


@dataclass
class Times:
  transform: float = 0.0
  gemm: float = 0.0
  combine: float = 0.0
  leaves: int = 0

  @property
  def total(self) -> float: return self.transform+self.gemm+self.combine


def main() -> None:
  size, levels = int(os.getenv("N", "8192")), int(os.getenv("LEVELS", "5"))
  seed = int(os.getenv("SEED", "701"))
  parallel_transform = bool(int(os.getenv("PARALLEL_TRANSFORM", "0")))
  buffer_transform = bool(int(os.getenv("BUFFER_TRANSFORM", "0")))
  fp32_transform = bool(int(os.getenv("FP32_TRANSFORM", "1")))
  memory_wg = int(os.getenv("MEMORY_WG", "256"))
  if parallel_transform and buffer_transform: raise ValueError("parallel buffer transform is not implemented")
  leaf = size >> levels
  if leaf != 256: raise ValueError("current FP32 leaf requires N >> LEVELS == 256")
  rng = np.random.default_rng(seed)
  data = os.getenv("DATA", "gaussian")
  if data == "exact":
    # Dense signed powers of two keep all transforms and the FP32 reference
    # exactly representable.  This is a strict indexing/instruction oracle.
    a_np = (rng.integers(0, 2, (size, size), dtype=np.int8)*2-1).astype(np.float16)*np.float16(1/256)
    b_np = (rng.integers(0, 2, (size, size), dtype=np.int8)*2-1).astype(np.float16)*np.float16(1/256)
  elif data == "gaussian":
    a_np = rng.normal(0, 1/32, (size, size)).astype(np.float16)
    b_np = rng.normal(0, 1/32, (size, size)).astype(np.float16)
  else: raise ValueError("DATA must be exact or gaussian")
  dev = Device["QCOM"]
  a, b, c = alloc(size*size, dtypes.half), alloc(size*size, dtypes.half), alloc(size*size, dtypes.float)
  a.copyin(memoryview(a_np).cast("B"))
  b.copyin(memoryview(b_np).cast("B"))

  # Rebuild with the selected workgroup size; this only affects the bandwidth
  # transforms/combine, not the 128-thread hand-written GEMM leaf.
  transform_libs = {(n, side): dev.compiler.compile(transform_src(n, side, parallel_transform, buffer_transform,
                                                                  fp32_transform, fp32_transform and n != size, memory_wg))
                    for n in (size >> x for x in range(levels)) for side in "AB"}
  combine_libs = {n: dev.compiler.compile(combine_src(n, memory_wg)) for n in (size >> x for x in range(levels))}
  transforms, combines = {}, {}

  def get_transform(n: int, side: str, count: int):
    key = (n, side, count)
    if key not in transforms:
      odt, idt = (dtypes.float if fp32_transform else dtypes.half), \
                 (dtypes.float if fp32_transform and n != size else dtypes.half)
      transforms[key] = dev.runtime("transform", transform_libs[(n, side)],
        buf_dtypes=[((0, odt, (count*7*(n//2)*(n//2),)),) if buffer_transform else
                    ((0, odt, (count*7*(n//2), n//8, 4)),),
                    ((0, idt, (count*n, n//4, 4)),)])
    return transforms[key]

  def get_combine(n: int, count: int):
    key = (n, count)
    if key not in combines:
      combines[key] = dev.runtime("combine", combine_libs[n],
        buf_dtypes=[((0, dtypes.float, (count*n, n//4, 4)),),
                    ((0, dtypes.float, (count*7*(n//2), n//8, 4)),)])
    return combines[key]

  q.M = q.N = q.K = leaf
  q.K4 = leaf//4
  env, io, sz, ro = get_envelope(dev, q.make_direct_image_donor_src(2, 128))
  shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_rotate_shader(
    dev, 128, k_count=leaf//4, batch_stride=leaf, batch_from_row=True)
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs, mergedregs=False)
  gemms = {}
  def get_gemm(count: int):
    if count not in gemms:
      out_image = ((0, dtypes.float, (count*7*leaf, leaf//4, 4)),)
      leaf_dt = dtypes.float if fp32_transform else dtypes.half
      a_image = ((0, leaf_dt, (count*7*leaf, leaf//4, 4)),)
      b_image = ((1, leaf_dt, (count*7*leaf, leaf//4, 4)),)
      gemms[count] = dev.runtime("gemm_h", lib, buf_dtypes=[out_image, a_image, b_image])
    return gemms[count]
  print(f"shape={size}x{size}x{size} levels={levels} leaf={leaf} leaf_fregs={fregs} leaf_loop={loop_instrs}", flush=True)

  keepalive: list[Buffer] = []
  times = Times()

  max_batch = int(os.getenv("BATCH_NODES", "4"))
  if not 1 <= max_batch <= 4: raise ValueError("BATCH_NODES must be 1..4 for the 8192-row image limit")

  def recurse(n: int, count: int, ax, bx, out) -> None:
    h = n//2
    transform_dt = dtypes.float if fp32_transform else dtypes.half
    ac, bc, mm = alloc(count*7*h*h, transform_dt), alloc(count*7*h*h, transform_dt), alloc(count*7*h*h, dtypes.float)
    keepalive.extend((ac, bc, mm))
    groups = (count*(7 if parallel_transform else 1)*h*h//4 + memory_wg-1)//memory_wg
    times.transform += get_transform(n, "A", count)(ac._buf, ax, global_size=(groups, 1, 1), local_size=(memory_wg, 1, 1), wait=True)
    times.transform += get_transform(n, "B", count)(bc._buf, bx, global_size=(groups, 1, 1), local_size=(memory_wg, 1, 1), wait=True)
    if n == size and count == 1 and int(os.getenv("DEBUG_STAGE", "0")):
      tnp = np.float32 if fp32_transform else np.float16
      ah = np.empty((7, h, h), tnp)
      bh = np.empty((7, h, h), tnp)
      ac.copyout(memoryview(ah).cast("B"))
      bc.copyout(memoryview(bh).cast("B"))
      aq = [a_np[:h,:h], a_np[:h,h:], a_np[h:,:h], a_np[h:,h:]]
      bq = [b_np[:h,:h], b_np[:h,h:], b_np[h:,:h], b_np[h:,h:]]
      ae = [aq[0]+aq[3], aq[2]+aq[3], aq[0], aq[3], aq[0]+aq[1], aq[2]-aq[0], aq[1]-aq[3]]
      be = [bq[0]+bq[3], bq[0], bq[1]-bq[3], bq[2]-bq[0], bq[3], bq[0]+bq[1], bq[2]+bq[3]]
      print("stage_transform", max(float(np.max(np.abs(ah[i].astype(np.float32)-ae[i]))) for i in range(7)),
            max(float(np.max(np.abs(bh[i].astype(np.float32)-be[i]))) for i in range(7)), flush=True)
      print("stage_a_parts", [(i, int(np.count_nonzero(ah[i] != ae[i])), ah[i,0,:8].tolist(), ae[i][0,:8].tolist())
                              for i in range(7)], flush=True)
    if h == leaf:
      times.gemm += get_gemm(count)(mm._buf, ac._buf, bc._buf, global_size=(1, count*7*leaf//16, 1),
                                    local_size=(128, 1, 1), wait=True)
      times.leaves += count*7
      if n == size and count == 1 and int(os.getenv("DEBUG_STAGE", "0")):
        mh = np.empty((7, h, h), np.float32)
        mm.copyout(memoryview(mh).cast("B"))
        print("stage_gemm", max(float(np.max(np.abs(mh[i]-ae[i].astype(np.float32)@be[i].astype(np.float32)))) for i in range(7)), flush=True)
    else:
      hb, fb = h*h*(4 if fp32_transform else 2), h*h*4
      children = count*7
      # The child's transform output stacks seven h/2-row matrices per input.
      # Keep every typed image at or below A630's 8192-row limit.
      child_batch = min(max_batch, 8192//(7*(h//2)))
      for first in range(0, children, child_batch):
        batch = min(child_batch, children-first)
        recurse(h, batch, ac._buf.offset(first*hb, batch*hb), bc._buf.offset(first*hb, batch*hb),
                mm._buf.offset(first*fb, batch*fb))
    groups_out = (count*n*n//4 + memory_wg-1)//memory_wg
    times.combine += get_combine(n, count)(out, mm._buf, global_size=(groups_out, 1, 1), local_size=(memory_wg, 1, 1), wait=True)
    # Recursive children have completed before their parents return. Dropping
    # these references bounds live storage to one seven-way branch per level.
    keepalive.pop()
    keepalive.pop()
    keepalive.pop()

  recurse(size, 1, a._buf, b._buf, c._buf)
  got = np.empty((size, size), np.float32)
  c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected)
  if data == "exact": wrong = np.flatnonzero(got.reshape(-1).view(np.uint32) != expected.reshape(-1).view(np.uint32))
  else:
    rtol, atol = float(os.getenv("RTOL", "1e-3")), float(os.getenv("ATOL", "1e-3"))
    wrong = np.flatnonzero(~np.isclose(got.reshape(-1), expected.reshape(-1), rtol=rtol, atol=atol))
  rel_l2 = float(np.linalg.norm((got-expected).astype(np.float64))/np.linalg.norm(expected.astype(np.float64)))
  conventional = 2*size**3
  print(f"elapsed_ms={times.total*1e3:.3f} gflops={conventional/times.total/1e9:.1f} "
        f"transform_ms={times.transform*1e3:.3f} gemm_ms={times.gemm*1e3:.3f} combine_ms={times.combine*1e3:.3f} "
        f"leaves={times.leaves} data={data} outputs={got.size} bad_count={wrong.size} "
        f"max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} rel_l2={rel_l2:.9g}", flush=True)
  if wrong.size: raise SystemExit(1)


if __name__ == "__main__": main()
