#!/usr/bin/env python3
"""Fuse two recursive Strassen output combines into one streaming kernel."""
import os

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


def alloc(count: int, dtype) -> Buffer: return Buffer("QCOM", count, dtype).allocate()


def combine_expr(names: list[str], u: int) -> str:
  ids, signs = (((0, 3, 4, 6), (1, 1, -1, 1)), ((2, 4), (1, 1)),
                ((1, 3), (1, 1)), ((0, 1, 2, 5), (1, -1, 1, 1)))[u]
  return "".join(("+" if s > 0 else "-")+names[i] for i, s in zip(ids, signs)).lstrip("+")


COMBINE = ((1, 0, 0, 1, -1, 0, 1), (0, 0, 1, 0, 1, 0, 0),
           (0, 1, 0, 1, 0, 0, 0), (1, -1, 1, 0, 0, 1, 0))


def parallel_source(n: int, wg: int, grid_x: int, fixed_block: int | None = None) -> tuple[str, int, int]:
  """One invocation per final quadrant, avoiding the serial kernel's redundant product reads."""
  leaf, paths = n//4, 49
  width, height = grid_x*leaf, ((paths+grid_x-1)//grid_x)*leaf
  cases = []
  for block_r in range(4):
    for block_c in range(4):
      u = ((block_r & 1) << 1) | (block_c & 1)
      v = ((block_r >> 1) << 1) | (block_c >> 1)
      terms = []
      for p0 in range(7):
        for p1 in range(7):
          coeff = COMBINE[v][p0] * COMBINE[u][p1]
          if not coeff: continue
          p = p0*7+p1
          value = f"M[((({p}/{grid_x})*{leaf}+r)*{width//4})+({p}%{grid_x})*{leaf//4}+x4]"
          terms.append(("+" if coeff > 0 else "-")+value)
      expr = "".join(terms).lstrip("+")
      block = block_r*4+block_c
      if fixed_block is None: cases.append(f"{'if' if block == 0 else 'else if'}(q=={block})v={expr};")
      elif block == fixed_block: cases.append(f"v={expr};")
  q_init = f"q={fixed_block},j=s" if fixed_block is not None else f"unit={leaf*leaf//4},q=s/unit,j=s%unit"
  return f"""__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(__global float4 *C,__global const float4 *M) {{
  uint s=get_global_id(0),{q_init},r=j/{leaf//4},x4=j%{leaf//4};float4 v;
  {''.join(cases)}
  C[((q/4*{leaf}+r)*{n//4})+(q%4)*{leaf//4}+x4]=v;
}}""", width, height


def reuse_source(n: int, wg: int, grid_x: int) -> tuple[str, int, int]:
  """Produce eight quadrants per pass and share each product load across four inner combines."""
  leaf, paths = n//4, 49
  width, height = grid_x*leaf, ((paths+grid_x-1)//grid_x)*leaf
  body = []
  for vs in ((0, 1), (2, 3)):
    body.append("{\n")
    for v in vs:
      for u in range(4): body.append(f"float4 o{v}_{u}=(float4)(0);\n")
    used_p0 = [p0 for p0 in range(7) if any(COMBINE[v][p0] for v in vs)]
    for p0 in used_p0:
      names = []
      for p1 in range(7):
        p = p0*7+p1
        name = f"x{p0}_{p1}"
        names.append(name)
        body.append(f"float4 {name}=M[((({p}/{grid_x})*{leaf}+r)*{width//4})+({p}%{grid_x})*{leaf//4}+x4];\n")
      for u in range(4): body.append(f"float4 d{p0}_{u}={combine_expr(names, u)};\n")
      for v in vs:
        coeff = COMBINE[v][p0]
        if coeff:
          op = "+=" if coeff > 0 else "-="
          for u in range(4): body.append(f"o{v}_{u}{op}d{p0}_{u};\n")
    for v in vs:
      for u in range(4):
        block_r, block_c = (v >> 1)*2+(u >> 1), (v & 1)*2+(u & 1)
        body.append(f"C[({block_r}*{leaf}+r)*{n//4}+{block_c}*{leaf//4}+x4]=o{v}_{u};\n")
    body.append("}\n")
  return f"""__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(__global float4 *C,__global const float4 *M) {{
  uint s=get_global_id(0),r=s/{leaf//4},x4=s%{leaf//4};
  {''.join(body)}
}}""", width, height


def vmajor_source(n: int, wg: int, grid_x: int) -> tuple[str, int, int]:
  """Accumulate four inner quadrants for one outer quadrant, sharing its product loads."""
  leaf, paths = n//4, 49
  width, height = grid_x*leaf, ((paths+grid_x-1)//grid_x)*leaf
  body = []
  for v in range(4):
    body.append("{\n")
    for u in range(4): body.append(f"float4 o{u}=(float4)(0);\n")
    for p0, coeff in enumerate(COMBINE[v]):
      if not coeff: continue
      body.append("{\n")
      names = []
      for p1 in range(7):
        p = p0*7+p1
        name = f"z{p1}"
        names.append(name)
        body.append(f"float4 {name}=M[((({p}/{grid_x})*{leaf}+r)*{width//4})+({p}%{grid_x})*{leaf//4}+x4];\n")
      op = "+=" if coeff > 0 else "-="
      for u in range(4): body.append(f"o{u}{op}{combine_expr(names, u)};\n")
      body.append("}\n")
    for u in range(4):
      block_r, block_c = (v >> 1)*2+(u >> 1), (v & 1)*2+(u & 1)
      body.append(f"C[({block_r}*{leaf}+r)*{n//4}+{block_c}*{leaf//4}+x4]=o{u};\n")
    body.append("}\n")
  source_body = "".join(body)
  return f"""__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(__global float4 *C,__global const float4 *M) {{
  uint s=get_global_id(0),r=s/{leaf//4},x4=s%{leaf//4};
  {source_body}
}}""", width, height


def source(n: int, wg: int, grid_x: int) -> tuple[str, int, int]:
  leaf, paths = n//4, 49
  width, height = grid_x*leaf, ((paths+grid_x-1)//grid_x)*leaf
  body = []
  for u in range(4):
    for p0 in range(7):
      names = [f"M[((({p0*7+p1}/{grid_x})*{leaf}+r)*{width//4})+({p0*7+p1}%{grid_x})*{leaf//4}+x4]" for p1 in range(7)]
      body.append(f"float4 d{u}_{p0}={combine_expr(names, u)};")
    ds = [f"d{u}_{x}" for x in range(7)]
    for v in range(4):
      block_r, block_c = (v>>1)*2+(u>>1), (v&1)*2+(u&1)
      body.append(f"C[({block_r}*{leaf}+r)*{n//4}+{block_c}*{leaf//4}+x4]={combine_expr(ds, v)};")
  return f"""__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(__global float4 *C,__global const float4 *M) {{
  uint s=get_global_id(0),r=s/{leaf//4},x4=s%{leaf//4};
  {''.join(body)}
}}""", width, height


def image_source(n: int, wg: int, grid_x: int) -> tuple[str, int, int]:
  leaf, paths = n//4, 49
  width, height = grid_x*leaf, ((paths+grid_x-1)//grid_x)*leaf
  body = []
  for u in range(4):
    for p0 in range(7):
      names = [f"read_imagef(M,smp,(int2)({(p0*7+p1)%grid_x}*{leaf//4}+x4,{(p0*7+p1)//grid_x}*{leaf}+r))" for p1 in range(7)]
      body.append(f"float4 d{u}_{p0}={combine_expr(names, u)};")
    ds = [f"d{u}_{x}" for x in range(7)]
    for v in range(4):
      block_r, block_c = (v>>1)*2+(u>>1), (v&1)*2+(u&1)
      body.append(f"write_imagef(C,(int2)({block_c}*{leaf//4}+x4,{block_r}*{leaf}+r),{combine_expr(ds, v)});")
  return f"""const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(write_only image2d_t C,read_only image2d_t M) {{
  uint s=get_global_id(0),r=s/{leaf//4},x4=s%{leaf//4};
  {''.join(body)}
}}""", width, height


def combine_once(m: np.ndarray) -> np.ndarray:
  h = m.shape[1]
  out = np.empty((m.shape[0]//7, h*2, h*2), np.float32)
  for b in range(out.shape[0]):
    x = m[b*7:(b+1)*7]
    out[b, :h, :h] = x[0]+x[3]-x[4]+x[6]
    out[b, :h, h:] = x[2]+x[4]
    out[b, h:, :h] = x[1]+x[3]
    out[b, h:, h:] = x[0]-x[1]+x[2]+x[5]
  return out


def main() -> None:
  n, wg, grid_x = int(os.getenv("N", "1024")), int(os.getenv("WG", "128")), int(os.getenv("GRID_X", "32"))
  leaf, paths = n//4, 49
  rng = np.random.default_rng(int(os.getenv("SEED", "701")))
  mats = rng.normal(0, 1/32, (paths, leaf, leaf)).astype(np.float32)
  image = bool(int(os.getenv("IMAGE_COMBINE", "0")))
  parallel = bool(int(os.getenv("PARALLEL", "0")))
  split = bool(int(os.getenv("PARALLEL_SPLIT", "0")))
  reuse = bool(int(os.getenv("REUSE", "0")))
  vmajor = bool(int(os.getenv("VMAJOR", "0")))
  src, width, height = (image_source(n, wg, grid_x) if image else
                        vmajor_source(n, wg, grid_x) if vmajor else
                        reuse_source(n, wg, grid_x) if reuse else
                        parallel_source(n, wg, grid_x) if parallel else source(n, wg, grid_x))
  storage = np.zeros((height, width), np.float32)
  for p in range(paths):
    storage[(p//grid_x)*leaf:(p//grid_x+1)*leaf, (p%grid_x)*leaf:(p%grid_x+1)*leaf] = mats[p]
  dev = Device["QCOM"]
  mb, cb = alloc(storage.size, dtypes.float), alloc(n*n, dtypes.float)
  mb.copyin(memoryview(storage).cast("B"))
  specs = ([((0, dtypes.float, (n, n//4, 4)),), ((0, dtypes.float, (height, width//4, 4)),)] if image else
           [((0, dtypes.float, (n*n,)),), ((0, dtypes.float, (storage.size,)),)])
  prgs = ([dev.runtime("combine", dev.compiler.compile(parallel_source(n, wg, grid_x, block)[0]), buf_dtypes=specs)
           for block in range(16)] if split else [dev.runtime("combine", dev.compiler.compile(src), buf_dtypes=specs)])
  groups = leaf*leaf//4//wg*(16 if parallel else 1)
  if split: groups //= 16
  times = [sum(prg(cb._buf, mb._buf, global_size=(groups, 1, 1), local_size=(wg, 1, 1), wait=True) for prg in prgs) for _ in range(5)]
  got = np.empty((n, n), np.float32)
  cb.copyout(memoryview(got).cast("B"))
  expected = combine_once(combine_once(mats))
  delta = np.abs(got-expected)
  print(f"n={n} best_ms={min(times)*1e3:.3f} max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} "
        f"exact={np.array_equal(got, expected)}")


if __name__ == "__main__": main()
