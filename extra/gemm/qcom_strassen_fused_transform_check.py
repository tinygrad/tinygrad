#!/usr/bin/env python3
"""Fuse every recursive Strassen operand transform into one local-memory kernel."""
import os

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


def alloc(count: int, dtype) -> Buffer: return Buffer("QCOM", count, dtype).allocate()


TA = ((1, 0, 0, 1), (0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 0, 1),
      (1, 1, 0, 0), (-1, 0, 1, 0), (0, 1, 0, -1))
TB = ((1, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, -1), (-1, 0, 1, 0),
      (0, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1))


def direct2_source(n: int, side: str, wg: int, grid_x: int) -> tuple[str, int, int]:
  leaf, paths = n//4, 49
  width, height = grid_x*leaf, ((paths+grid_x-1)//grid_x)*leaf
  t = TA if side == "A" else TB
  loads = []
  for q0 in range(4):
    for q1 in range(4):
      rb = ((q0>>1)<<1)|(q1>>1)
      cb = ((q0&1)<<1)|(q1&1)
      loads.append(f"float4 v{q0*4+q1}=convert_float4(vload4(0,I+({rb}*{leaf}+r)*{n}+{cb}*{leaf}+x4*4));")
  stores = []
  for p0 in range(7):
    for p1 in range(7):
      terms = []
      for q0 in range(4):
        for q1 in range(4):
          c = t[p0][q0]*t[p1][q1]
          if c: terms.append(("+" if c > 0 else "-")+f"v{q0*4+q1}")
      expr = "".join(terms).lstrip("+")
      p = p0*7+p1
      stores.append(f"O[((({p}/{grid_x})*{leaf}+r)*{width//4})+({p}%{grid_x})*{leaf//4}+x4]={expr};")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform(__global float4 *O,__global const half *I) {{
  uint s=get_global_id(0),r=s/{leaf//4},x4=s%{leaf//4};
  {''.join(loads)}
  {''.join(stores)}
}}""", width, height


def direct3_source(n: int, side: str, wg: int, vec: int = 2) -> tuple[str, int, int]:
  """Fuse three transforms, processing one innermost path at a time to bound registers."""
  leaf, paths = n//8, 343
  t = TA if side == "A" else TB
  body = []
  for p2 in range(7):
    body.append("{")
    vals = []
    for q0 in range(4):
      for q1 in range(4):
        terms = []
        for q2 in range(4):
          coeff = t[p2][q2]
          if not coeff: continue
          rb = ((q0 >> 1) << 2) | ((q1 >> 1) << 1) | (q2 >> 1)
          cb = ((q0 & 1) << 2) | ((q1 & 1) << 1) | (q2 & 1)
          load = f"convert_float{vec}(vload{vec}(0,I+({rb}*{leaf}+r)*{n}+{cb}*{leaf}+xv*{vec}))"
          terms.append(("+" if coeff > 0 else "-")+load)
        name = f"v{q0*4+q1}"
        vals.append(name)
        body.append(f"float{vec} {name}={''.join(terms).lstrip('+')};")
    for p0 in range(7):
      for p1 in range(7):
        terms = []
        for q0 in range(4):
          for q1 in range(4):
            coeff = t[p0][q0]*t[p1][q1]
            if coeff: terms.append(("+" if coeff > 0 else "-")+vals[q0*4+q1])
        p = (p0*7+p1)*7+p2
        body.append(f"vstore{vec}(convert_half{vec}({''.join(terms).lstrip('+')}),0,O+{p*leaf*leaf}+r*{leaf}+xv*{vec});")
    body.append("}")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform(__global half *O,__global const half *I) {{
  uint s=get_global_id(0),r=s/{leaf//vec},xv=s%{leaf//vec};
  {''.join(body)}
}}""", leaf, paths*leaf


def direct3_full_source(n: int, side: str, wg: int, vec: int = 2) -> tuple[str, int, int]:
  """Fuse three transforms while retaining all 64 source subtiles in registers."""
  leaf, paths = n//8, 343
  t = TA if side == "A" else TB
  loads, stores = [], []
  for q0 in range(4):
    for q1 in range(4):
      for q2 in range(4):
        q = (q0*4+q1)*4+q2
        rb = ((q0 >> 1) << 2) | ((q1 >> 1) << 1) | (q2 >> 1)
        cb = ((q0 & 1) << 2) | ((q1 & 1) << 1) | (q2 & 1)
        loads.append(f"float{vec} v{q}=convert_float{vec}(vload{vec}(0,I+({rb}*{leaf}+r)*{n}+{cb}*{leaf}+xv*{vec}));")
  for p0 in range(7):
    for p1 in range(7):
      for p2 in range(7):
        terms = []
        for q0 in range(4):
          for q1 in range(4):
            for q2 in range(4):
              coeff = t[p0][q0]*t[p1][q1]*t[p2][q2]
              q = (q0*4+q1)*4+q2
              if coeff: terms.append(("+" if coeff > 0 else "-")+f"v{q}")
        p = (p0*7+p1)*7+p2
        stores.append(f"vstore{vec}(convert_half{vec}({''.join(terms).lstrip('+')}),0,O+{p*leaf*leaf}+r*{leaf}+xv*{vec});")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform(__global half *O,__global const half *I) {{
  uint s=get_global_id(0),r=s/{leaf//vec},xv=s%{leaf//vec};
  {''.join(loads)}
  {''.join(stores)}
}}""", leaf, paths*leaf


def source(n: int, levels: int, side: str, wg: int, grid_x: int) -> tuple[str, int, int]:
  leaf, paths = n >> levels, 7**levels
  width, height = grid_x*leaf, ((paths+grid_x-1)//grid_x)*leaf
  width4 = width//4
  max_state = paths
  stages = []
  for d in range(levels):
    old_prefixes, rest = 7**d, 4**(levels-d-1)
    out_len = old_prefixes*7*rest
    src, dst = ("x", "y") if d % 2 == 0 else ("y", "x")
    stages.append(f"""
  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint i=lid;i<{out_len};i+={wg}) {{
    uint np=i/{rest},p=np%7,op=np/7,suf=i%{rest},base=op*{4*rest}+suf;
    float4 q0={src}[base],q1={src}[base+{rest}],q2={src}[base+{2*rest}],q3={src}[base+{3*rest}],v;
    {('if(p==0)v=q0+q3;else if(p==1)v=q2+q3;else if(p==2)v=q0;else if(p==3)v=q3;'
      'else if(p==4)v=q0+q1;else if(p==5)v=q2-q0;else v=q1-q3;' if side == 'A' else
      'if(p==0)v=q0+q3;else if(p==1)v=q0;else if(p==2)v=q1-q3;else if(p==3)v=q2-q0;'
      'else if(p==4)v=q3;else if(p==5)v=q0+q1;else v=q2+q3;')}
    {dst}[i]=v;
  }}""")
  final = "y" if levels % 2 else "x"
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform(__global float4 *O,__global const half *I) {{
  uint lid=get_local_id(0),s=get_group_id(0),ir=s/{leaf//4},ic4=s%{leaf//4};
  __local float4 x[{max_state}],y[{max_state}];
  for(uint q=lid;q<{4**levels};q+={wg}) {{
    uint rb=0,cb=0;
    for(uint d=0;d<{levels};d++) {{uint qd=(q>>(2*({levels}-1-d)))&3;rb|=(qd>>1)<<({levels}-1-d);cb|=(qd&1)<<({levels}-1-d);}}
    x[q]=convert_float4(vload4(0,I+(rb*{leaf}+ir)*{n}+cb*{leaf}+ic4*4));
  }}
  {''.join(stages)}
  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint p=lid;p<{paths};p+={wg}) {{
    uint orow=(p/{grid_x})*{leaf}+ir,ocol4=(p%{grid_x})*{leaf//4}+ic4;
    O[orow*{width4}+ocol4]={final}[p];
  }}
}}""", width, height


def cpu_transform(x: np.ndarray, levels: int, side: str) -> np.ndarray:
  mats = [x.astype(np.float32)]
  for _ in range(levels):
    out = []
    for m in mats:
      h = m.shape[0]//2
      q0, q1, q2, q3 = m[:h, :h], m[:h, h:], m[h:, :h], m[h:, h:]
      out.extend(([q0+q3, q2+q3, q0, q3, q0+q1, q2-q0, q1-q3] if side == "A" else
                  [q0+q3, q0, q1-q3, q2-q0, q3, q0+q1, q2+q3]))
    mats = out
  return np.stack(mats)


def main() -> None:
  n, levels = int(os.getenv("N", "1024")), int(os.getenv("LEVELS", "2"))
  wg, grid_x, seed = int(os.getenv("WG", "128")), int(os.getenv("GRID_X", "128")), int(os.getenv("SEED", "701"))
  leaf, paths = n >> levels, 7**levels
  rng = np.random.default_rng(seed)
  inp = rng.normal(0, 1/32, (n, n)).astype(np.float16)
  dev = Device["QCOM"]
  ib = alloc(n*n, dtypes.half)
  ib.copyin(memoryview(inp).cast("B"))
  direct2 = levels == 2 and bool(int(os.getenv("DIRECT2", "1")))
  direct3 = levels == 3 and bool(int(os.getenv("DIRECT3", "0")))
  direct3_full = direct3 and bool(int(os.getenv("DIRECT3_FULL", "0")))
  for side in "AB":
    src, width, height = (direct2_source(n, side, wg, grid_x) if direct2 else
                          direct3_full_source(n, side, wg) if direct3_full else
                          direct3_source(n, side, wg) if direct3 else source(n, levels, side, wg, grid_x))
    odt = dtypes.half if direct3 else dtypes.float
    ob = alloc(width*height, odt)
    prg = dev.runtime("transform", dev.compiler.compile(src), buf_dtypes=[
      ((0, odt, (width*height,)),), ((0, dtypes.half, (n*n,)),)])
    groups = leaf*leaf//(2 if direct3 else 4)//wg if direct2 or direct3 else leaf*leaf//4
    times = [prg(ob._buf, ib._buf, global_size=(groups, 1, 1), local_size=(wg, 1, 1), wait=True) for _ in range(3)]
    got_storage = np.empty((height, width), np.float16 if direct3 else np.float32)
    ob.copyout(memoryview(got_storage).cast("B"))
    got = (got_storage.reshape(paths, leaf, leaf) if direct3 else
           np.stack([got_storage[(p//grid_x)*leaf:(p//grid_x+1)*leaf,
                                 (p%grid_x)*leaf:(p%grid_x+1)*leaf] for p in range(paths)]))
    expected = cpu_transform(inp, levels, side).astype(np.float16 if direct3 else np.float32)
    delta = np.abs(got-expected)
    print(f"side={side} n={n} levels={levels} paths={paths} layout={height}x{width} best_ms={min(times)*1e3:.3f} "
          f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} exact={np.array_equal(got, expected)}")


if __name__ == "__main__": main()
