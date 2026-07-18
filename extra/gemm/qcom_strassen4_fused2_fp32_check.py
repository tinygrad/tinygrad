#!/usr/bin/env python3
"""Four-level full Strassen GEMM using register-fused pairs of transforms/combines."""
import ctypes, gc, hashlib, mmap, os, subprocess, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_strassen_fused_transform_check import TA, TB
from extra.gemm.qcom_strassen_fused_combine_check import COMBINE, combine_expr


def alloc(count: int, dtype) -> Buffer: return Buffer("QCOM", count, dtype).allocate()


def alloc_external(count: int, dtype) -> tuple[Buffer, mmap.mmap]:
  nbytes = count*dtype.itemsize
  backing = mmap.mmap(-1, nbytes)
  ptr = ctypes.addressof(ctypes.c_char.from_buffer(backing))
  return Buffer("QCOM", count, dtype).allocate(external_ptr=ptr), backing


def transform2_src(n: int, side: str, input_half: bool, output_half: bool = False, wg: int = 256, vec: int = 4,
                   input_image: bool = False, output_image: bool = False, compute_half: bool = False) -> str:
  if output_image and vec != 4: raise ValueError("image transform output requires vec=4")
  if compute_half and not (input_half and output_half): raise ValueError("half transform compute requires half input and output")
  leaf, t = n//4, TA if side == "A" else TB
  groups = leaf*leaf//vec//wg
  grid_x = min(groups, 1024)
  while groups % grid_x: grid_x -= 1
  sid = "get_global_id(0)" if groups <= 1024 else f"get_global_id(0)+get_group_id(1)*{grid_x*wg}"
  loads, stores = [], []
  for q0 in range(4):
    for q1 in range(4):
      rb = ((q0>>1)<<1)|(q1>>1)
      cb = ((q0&1)<<1)|(q1&1)
      offset = f"({rb}*{leaf}+r)*{n}+{cb}*{leaf}+xv*{vec}"
      if input_image:
        value = f"read_image{'h' if input_half else 'f'}(I,smp,(int2)({cb*leaf//4}+xv,{rb*leaf}+r))"
        if input_half and not compute_half: value = f"convert_float4({value})"
      else:
        value = ((f"vload{vec}(0,I+{offset})" if compute_half else f"convert_float{vec}(vload{vec}(0,I+{offset}))")
                 if input_half else f"vload{vec}(0,I+{offset})") if vec > 1 else \
                ((f"I[{offset}]" if compute_half else f"convert_float(I[{offset}])") if input_half else f"I[{offset}]")
      ctype = "half" if compute_half else "float"
      loads.append(f"{ctype if vec == 1 else f'{ctype}{vec}'} v{q0*4+q1}={value};")
  # Factor the Kronecker transform one inner path at a time.  The direct
  # 49-expression form repeats each inner sum for every outer path (95 vector
  # adds); this schedule computes four inner temporaries, consumes them into
  # seven outputs, then reuses the registers (55 vector adds total).
  scalar_type = "half" if compute_half else "float"
  vtype = scalar_type if vec == 1 else f"{scalar_type}{vec}"
  for p1 in range(7):
    stores.append("{")
    for q0 in range(4):
      terms = [("+" if t[p1][q1] > 0 else "-")+f"v{q0*4+q1}" for q1 in range(4) if t[p1][q1]]
      stores.append(f"{vtype} w{q0}={''.join(terms).lstrip('+')};")
    for p0 in range(7):
      terms = [("+" if t[p0][q0] > 0 else "-")+f"w{q0}" for q0 in range(4) if t[p0][q0]]
      p, expr = p0*7+p1, "".join(terms).lstrip("+")
      oval = ((expr if compute_half else f"convert_half{vec}({expr})") if vec > 1 else
              (expr if compute_half else f"convert_half({expr})")) if output_half else expr
      offset = f"{p*leaf*leaf}+r*{leaf}+xv*{vec}"
      if output_image: stores.append(f"write_image{'h' if output_half else 'f'}(O,(int2)(xv,{p*leaf}+r),{oval});")
      else: stores.append(f"vstore{vec}({oval},0,O+{offset});" if vec > 1 else f"O[{offset}]={oval};")
    stores.append("}")
  itype, otype = ("half" if input_half else "float"), ("half" if output_half else "float")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
{'const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;' if input_image else ''}
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform({'write_only image2d_t O' if output_image else f'__global {otype} *O'},
                        {'read_only image2d_t I' if input_image else f'__global const {itype} *I'}) {{
  uint s={sid},r=s/{leaf//vec},xv=s%{leaf//vec};
  {''.join(loads)}
  {''.join(stores)}
}}"""


def transform2_path_src(n: int, side: str, path: int, wg: int = 256, vec: int = 4, compute_half: bool = False) -> str:
  leaf = n//4
  groups = leaf*leaf//vec//wg
  grid_x = min(groups, 1024)
  while groups % grid_x: grid_x -= 1
  sid = "get_global_id(0)" if groups <= 1024 else f"get_global_id(0)+get_group_id(1)*{grid_x*wg}"
  t = TA if side == "A" else TB
  p0, p1 = divmod(path, 7)
  loads, terms = [], []
  for q0 in range(4):
    for q1 in range(4):
      coeff = t[p0][q0]*t[p1][q1]
      if not coeff: continue
      rb = ((q0>>1)<<1)|(q1>>1)
      cb = ((q0&1)<<1)|(q1&1)
      name = f"v{q0}_{q1}"
      value = f"read_imageh(I,smp,(int2)({cb*leaf//4}+xv,{rb}*{leaf}+r))"
      loads.append(f"{'half' if compute_half else 'float'}{vec} {name}="
                   f"{value if compute_half else f'convert_float{vec}({value})'}; ")
      terms.append(("+" if coeff > 0 else "-")+name)
  expr = "".join(terms).lstrip("+")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform_path(__global half *O,read_only image2d_t I) {{
  uint s={sid},r=s/{leaf//vec},xv=s%{leaf//vec};
  {''.join(loads)}
  vstore{vec}({expr if compute_half else f'convert_half{vec}({expr})'},0,O+r*{leaf}+xv*{vec});
}}"""


def transform1_batch_src(n: int, batch: int, side: str, wg: int = 256, vec: int = 2, input_half: bool = True) -> str:
  """One additional Strassen level over a dense batch of FP16 matrices."""
  leaf, t = n//2, TA if side == "A" else TB
  groups = batch*leaf*leaf//vec//wg
  grid_x = min(groups, 1024)
  while groups % grid_x: grid_x -= 1
  sid = "get_global_id(0)" if groups <= 1024 else f"get_global_id(0)+get_group_id(1)*{grid_x*wg}"
  scalar, vtype = ("half" if input_half else "float"), ("half" if input_half else "float") if vec == 1 else \
                  f"{'half' if input_half else 'float'}{vec}"
  loads, stores = [], []
  for q in range(4):
    rb, cb = q//2, q%2
    offset = f"b*{n*n}+({rb}*{leaf}+r)*{n}+{cb}*{leaf}+xv*{vec}"
    loads.append(f"{vtype} v{q}={'I['+offset+']' if vec == 1 else f'vload{vec}(0,I+{offset})'};")
  for p in range(7):
    terms = [("+" if t[p][q] > 0 else "-")+f"v{q}" for q in range(4) if t[p][q]]
    value = "".join(terms).lstrip("+")
    offset = f"(b*7+{p})*{leaf*leaf}+r*{leaf}+xv*{vec}"
    value = value if input_half else (f"convert_half({value})" if vec == 1 else f"convert_half{vec}({value})")
    stores.append(f"O[{offset}]={value};" if vec == 1 else f"vstore{vec}({value},0,O+{offset});")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void transform1(__global half *O,__global const {scalar} *I) {{
  uint s={sid},b=s/{leaf*leaf//vec},z=s%{leaf*leaf//vec},r=z/{leaf//vec},xv=z%{leaf//vec};
  {''.join(loads)}
  {''.join(stores)}
}}"""


def combine1_batch_src(n: int, batch: int, wg: int = 256, vec: int = 4) -> str:
  """Inverse of transform1_batch_src, retaining FP16 product storage semantics."""
  leaf = n//2
  groups = batch*leaf*leaf//vec//wg
  grid_x = min(groups, 1024)
  while groups % grid_x: grid_x -= 1
  sid = "get_global_id(0)" if groups <= 1024 else f"get_global_id(0)+get_group_id(1)*{grid_x*wg}"
  body = []
  names = [f"convert_float{vec}(vload{vec}(0,M+(b*7+{p})*{leaf*leaf}+r*{leaf}+xv*{vec}))" for p in range(7)]
  for u in range(4):
    value = f"convert_half{vec}({combine_expr(names, u)})"
    body.append(f"vstore{vec}({value},0,C+b*{n*n}+({u//2}*{leaf}+r)*{n}+{u%2}*{leaf}+xv*{vec});")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine1(__global half *C,__global const half *M) {{
  uint s={sid},b=s/{leaf*leaf//vec},z=s%{leaf*leaf//vec},r=z/{leaf//vec},xv=z%{leaf//vec};
  {''.join(body)}
}}"""


def combine2_src(n: int, wg: int = 128, input_half: bool = False, output_half: bool = False, vec: int = 4,
                 compute_half: bool = False) -> str:
  if compute_half and not (input_half and output_half): raise ValueError("half combine compute requires half input and output")
  leaf = n//4
  groups = leaf*leaf//vec//wg
  grid_x = min(groups, 1024)
  while groups % grid_x: grid_x -= 1
  sid = "get_global_id(0)" if groups <= 1024 else f"get_global_id(0)+get_group_id(1)*{grid_x*wg}"
  body = []
  for u in range(4):
    for p0 in range(7):
      names = [f"vload{vec}(0,M+{(p0*7+p1)*leaf*leaf}+r*{leaf}+xv*{vec})" for p1 in range(7)]
      if input_half and not compute_half: names = [f"convert_float{vec}({name})" for name in names]
      body.append(f"{'half' if compute_half else 'float'}{vec} d{u}_{p0}={combine_expr(names, u)};")
    ds = [f"d{u}_{x}" for x in range(7)]
    for v in range(4):
      block_r, block_c = (v>>1)*2+(u>>1), (v&1)*2+(u&1)
      value = combine_expr(ds, v)
      if output_half and not compute_half: value = f"convert_half{vec}({value})"
      body.append(f"vstore{vec}({value},0,C+({block_r}*{leaf}+r)*{n}+{block_c}*{leaf}+xv*{vec});")
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(__global {'half' if output_half else 'float'} *C,__global const {'half' if input_half else 'float'} *M) {{
  uint s={sid},r=s/{leaf//vec},xv=s%{leaf//vec};
  {''.join(body)}
}}"""


def combine2_block_src(n: int, block: int, wg: int = 128, input_half: bool = False,
                       output_half: bool = False, vec: int = 4) -> str:
  """One final block with only its mathematically nonzero product reads."""
  leaf, block_r, block_c = n//4, block//4, block%4
  u = ((block_r&1)<<1)|(block_c&1)
  v = ((block_r>>1)<<1)|(block_c>>1)
  terms = []
  for p0 in range(7):
    for p1 in range(7):
      coeff = COMBINE[v][p0]*COMBINE[u][p1]
      if not coeff: continue
      value = f"vload{vec}(0,M+{(p0*7+p1)*leaf*leaf}+r*{leaf}+xv*{vec})"
      if input_half: value = f"convert_float{vec}({value})"
      terms.append(("+" if coeff > 0 else "-")+value)
  value = "".join(terms).lstrip("+")
  if output_half: value = f"convert_half{vec}({value})"
  return f"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size({wg},1,1)))
__kernel void combine(__global {'half' if output_half else 'float'} *C,
                      __global const {'half' if input_half else 'float'} *M) {{
  uint s=get_global_id(0),r=s/{leaf//vec},xv=s%{leaf//vec};
  vstore{vec}({value},0,C+({block_r}*{leaf}+r)*{n}+{block_c}*{leaf}+xv*{vec});
}}"""


@dataclass
class Times:
  transform: float = 0
  gemm: float = 0
  combine: float = 0
  wall: float | None = None
  @property
  def total(self) -> float: return self.wall if self.wall is not None else self.transform+self.gemm+self.combine


def cpu_pipeline_lib(threads: int):
  src = Path(__file__).with_name("qcom_strassen_cpu_pipeline.c")
  digest = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
  so = Path(f"/tmp/qcom_strassen_cpu_pipeline_{digest}.so")
  if not so.exists():
    subprocess.run(["clang", "-O3", "-march=armv8.2-a+fp16", "-fopenmp", "-shared", "-fPIC", str(src), "-o", str(so)], check=True)
  lib = ctypes.CDLL(str(so))
  lib.set_threads.argtypes = [ctypes.c_int]
  lib.transform2_f16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
  lib.combine2_f16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
  lib.cache_clean.argtypes = [ctypes.c_void_p, ctypes.c_int64]
  lib.cache_invalidate.argtypes = [ctypes.c_void_p, ctypes.c_int64]
  lib.set_threads(threads)
  return lib


def main() -> None:
  n, seed = int(os.getenv("N", "4096")), int(os.getenv("SEED", "701"))
  if n//16 not in (64, 128, 256, 512, 768): raise ValueError("N/16 must be one of 64, 128, 256, 512, or 768")
  child, mid, wg = n//4, n//16, int(os.getenv("TRANSFORM_WG", "256"))
  strassen5 = bool(int(os.getenv("STRASSEN5", "0")))
  leaf = mid//2 if strassen5 else mid
  combine_wg = int(os.getenv("COMBINE_WG", "256"))
  leaf_threads = int(os.getenv("LEAF_THREADS", "64"))
  if leaf_threads not in (64, 128): raise ValueError("LEAF_THREADS must be 64 or 128")
  leaf_halfwave = bool(int(os.getenv("LEAF_HALFWAVE", "0")))
  if leaf_halfwave and (strassen5 or leaf != 64 or leaf_threads != 64):
    raise ValueError("LEAF_HALFWAVE requires a four-level N=1024 run with LEAF_THREADS=64")
  transform_half = bool(int(os.getenv("TRANSFORM_HALF", "1")))
  product_half = bool(int(os.getenv("PRODUCT_HALF", "1")))
  transform_vec = int(os.getenv("TRANSFORM_VEC", os.getenv("MEMORY_VEC", "2")))
  combine_vec = int(os.getenv("COMBINE_VEC", os.getenv("MEMORY_VEC", "4")))
  transform_image = bool(int(os.getenv("TRANSFORM_IMAGE", "0")))
  transform_output_image = bool(int(os.getenv("TRANSFORM_OUTPUT_IMAGE", "0")))
  transform_compute_half = bool(int(os.getenv("TRANSFORM_COMPUTE_HALF", "1")))
  combine_compute_half = bool(int(os.getenv("COMBINE_COMPUTE_HALF", "0")))
  split_top_combine = bool(int(os.getenv("SPLIT_TOP_COMBINE", "0")))
  cpu_pipeline = bool(int(os.getenv("CPU_PIPELINE", "0")))
  stream_top = bool(int(os.getenv("STREAM_TOP", "0")))
  cpu_threads = int(os.getenv("CPU_THREADS", "6"))
  if cpu_pipeline and stream_top: raise ValueError("CPU_PIPELINE and STREAM_TOP are mutually exclusive")
  if strassen5 and cpu_pipeline: raise ValueError("STRASSEN5 does not support CPU_PIPELINE")
  if strassen5 and (not transform_half or not product_half):
    raise ValueError("STRASSEN5 currently requires FP16 transform and product storage")
  if transform_image and transform_vec != 4: raise ValueError("TRANSFORM_IMAGE requires TRANSFORM_VEC=4")
  if transform_output_image and transform_vec != 4: raise ValueError("TRANSFORM_OUTPUT_IMAGE requires TRANSFORM_VEC=4")
  if cpu_pipeline and transform_output_image: raise ValueError("CPU_PIPELINE does not support image transform output")
  if transform_vec not in (1, 2, 4, 8, 16) or combine_vec not in (2, 4, 8, 16):
    raise ValueError("TRANSFORM_VEC must be 1, 2, 4, 8, or 16; COMBINE_VEC must be 2, 4, 8, or 16")
  rng = np.random.default_rng(seed)
  def random_half() -> np.ndarray:
    out = np.empty((n, n), np.float16)
    for first in range(0, n, 64):
      out[first:first+64] = rng.standard_normal((min(64, n-first), n), dtype=np.float32)*np.float32(1/32)
    return out
  a_np, b_np = random_half(), random_half()
  dev = Device["QCOM"]
  a, b = alloc(n*n, dtypes.half), alloc(n*n, dtypes.half)
  a.copyin(memoryview(a_np).cast("B"))
  b.copyin(memoryview(b_np).cast("B"))
  gpu_ref = bool(int(os.getenv("GPU_REF", "1" if n >= 8192 else "0")))
  if gpu_ref:
    del a_np, b_np
    gc.collect()

  def transform_runtime(size: int, side: str, input_half: bool):
    idt = dtypes.half if input_half else dtypes.float
    odt = dtypes.half if transform_half else dtypes.float
    input_image = transform_image or (transform_output_image and size == child and not stream_top)
    return dev.runtime("transform", dev.compiler.compile(
      transform2_src(size, side, input_half, transform_half, wg, transform_vec, input_image, transform_output_image,
                     transform_compute_half)), buf_dtypes=[
      ((0, odt, (49*size//4, size//16, 4)),) if transform_output_image else ((0, odt, (49*(size//4)**2,)),),
      ((0, idt, (size, size//4, 4)),) if input_image else ((0, idt, (size*size,)),)])

  path_vec = int(os.getenv("PATH_VEC", "4"))
  if path_vec != 4: raise ValueError("PATH_VEC must be 4 for image-backed streamed paths")
  top_paths = ({side: [dev.runtime("transform_path", dev.compiler.compile(
    transform2_path_src(n, side, p, wg, path_vec, transform_compute_half)), buf_dtypes=[
    ((0, dtypes.half, (child*child,)),), ((0, dtypes.half, (n, n//4, 4)),)]) for p in range(49)] for side in "AB"}
    if cpu_pipeline or stream_top else None)
  top_t = None if cpu_pipeline or stream_top else {side: transform_runtime(n, side, True) for side in "AB"}
  child_t = None if cpu_pipeline else {side: transform_runtime(child, side, transform_half) for side in "AB"}
  extra_t = ({side: dev.runtime("transform1", dev.compiler.compile(
    transform1_batch_src(mid, 49, side, wg, transform_vec)), buf_dtypes=[
      ((0, dtypes.half, (343*leaf*leaf,)),), ((0, dtypes.half, (49*mid*mid,)),)]) for side in "AB"}
    if strassen5 and not cpu_pipeline else None)
  top_c_specs = [((0, dtypes.float, (n*n,)),),
                 ((0, dtypes.half if product_half else dtypes.float, (49*child*child,)),)]
  top_c = ([dev.runtime("combine", dev.compiler.compile(
    combine2_block_src(n, block, combine_wg, product_half, False, combine_vec)), buf_dtypes=top_c_specs) for block in range(16)]
    if split_top_combine else dev.runtime("combine", dev.compiler.compile(
      combine2_src(n, combine_wg, product_half, False, combine_vec)), buf_dtypes=top_c_specs))
  child_c = (None if cpu_pipeline else dev.runtime("combine", dev.compiler.compile(
    combine2_src(child, combine_wg, product_half, product_half, combine_vec, combine_compute_half)), buf_dtypes=[
      ((0, dtypes.half if product_half else dtypes.float, (child*child,)),),
      ((0, dtypes.half if product_half else dtypes.float, (49*mid*mid,)),)]))
  extra_c = (dev.runtime("combine1", dev.compiler.compile(
    combine1_batch_src(mid, 49, combine_wg, combine_vec)), buf_dtypes=[
      ((0, dtypes.half, (49*mid*mid,)),), ((0, dtypes.half, (343*leaf*leaf,)),)])
    if strassen5 and not cpu_pipeline else None)

  q.M = q.N = q.K = leaf
  q.K4 = leaf//4
  leaf_unroll = int(os.getenv("LEAF_UNROLL", "3"))
  leaf_batch_from_row = leaf > 0 and not (leaf & (leaf-1))
  env, io, sz, ro = get_envelope(dev, q.make_direct_image_donor_src(2, leaf_threads))
  if leaf_halfwave:
    shader, hregs, fregs, loop_instrs = q.build_4x4_fp32_halfwave_batch_shader(dev, leaf_threads, batch_stride=leaf)
  else:
    shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_rotate_shader(
      dev, leaf_threads, k_count=leaf//4, batch_stride=leaf, batch_from_row=leaf_batch_from_row, k_unroll=leaf_unroll)
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs, mergedregs=False)
  gemms = {}
  def gemm_runtime(batch: int):
    if batch not in gemms:
      gemms[batch] = dev.runtime("gemm_h", lib, buf_dtypes=[
        ((0, dtypes.half if product_half else dtypes.float, (batch*leaf, leaf//4, 4)),),
        ((0, dtypes.half if transform_half else dtypes.float, (batch*leaf, leaf//4, 4)),),
        ((1, dtypes.half if transform_half else dtypes.float, (batch*leaf, leaf//4, 4)),)])
    return gemms[batch]

  def grid(groups: int) -> tuple[int, int, int]:
    gx = min(groups, 1024)
    while groups % gx: gx -= 1
    return gx, groups//gx, 1

  transform_dt = dtypes.half if transform_half else dtypes.float
  product_dt = dtypes.half if product_half else dtypes.float
  if cpu_pipeline:
    top_a = top_b = None
  elif stream_top:
    top_a, top_b = alloc(child*child, transform_dt), alloc(child*child, transform_dt)
  else:
    top_a, top_b = alloc(49*child*child, transform_dt), alloc(49*child*child, transform_dt)
  top_m = alloc(49*child*child, product_dt)
  times = Times()
  candidate_start = time.perf_counter()
  top_groups = (child*child//transform_vec+wg-1)//wg
  path_groups = (child*child//path_vec+wg-1)//wg
  top_combine_groups = (child*child//combine_vec+combine_wg-1)//combine_wg
  if not cpu_pipeline and not stream_top:
    times.transform += top_t["A"](top_a._buf, a._buf, global_size=grid(top_groups), local_size=(wg, 1, 1), wait=True)
    times.transform += top_t["B"](top_b._buf, b._buf, global_size=grid(top_groups), local_size=(wg, 1, 1), wait=True)
  child_input_bytes = child*child*(2 if transform_half else 4)
  child_output_bytes = child*child*(2 if product_half else 4)
  leaf_input_bytes = leaf*leaf*(2 if transform_half else 4)
  leaf_output_bytes = leaf*leaf*(2 if product_half else 4)
  child_transform_groups = (mid*mid//transform_vec+wg-1)//wg
  extra_transform_groups = (49*leaf*leaf//transform_vec+wg-1)//wg
  extra_combine_groups = (49*leaf*leaf//combine_vec+combine_wg-1)//combine_wg
  child_combine_groups = (mid*mid//combine_vec+combine_wg-1)//combine_wg
  def launch_leaf_products(lm, la, lb, wait: bool, products: int = 49) -> None:
    max_batch = 8192//leaf
    for first in range(0, products, max_batch):
      batch = min(max_batch, products-first)
      ispan, ospan = batch*leaf_input_bytes, batch*leaf_output_bytes
      elapsed = gemm_runtime(batch)(lm._buf.offset(first*leaf_output_bytes, ospan), la._buf.offset(first*leaf_input_bytes, ispan),
        lb._buf.offset(first*leaf_input_bytes, ispan),
        global_size=((1, batch*leaf//16, 1) if leaf_halfwave else
                     (max(1, leaf//256), batch*leaf//((leaf_threads//32)*4), 1) if leaf_batch_from_row else
                     (max(1, leaf//256), leaf//((leaf_threads//32)*4), batch)),
        local_size=(leaf_threads, 1, 1), wait=wait)
      if elapsed is not None: times.gemm += elapsed

  cpu_transform_work = cpu_combine_work = 0.0
  child_combine_time = top_combine_time = 0.0
  if cpu_pipeline:
    if not transform_half or not product_half or transform_image:
      raise ValueError("CPU_PIPELINE requires FP16 transform/product buffers and buffer-backed transforms")
    lib_cpu = cpu_pipeline_lib(cpu_threads)
    slots = []
    for _ in range(2):
      lm, lm_backing = alloc_external(49*leaf*leaf, product_dt)
      slots.append((alloc(49*leaf*leaf, transform_dt), alloc(49*leaf*leaf, transform_dt), lm, lm_backing))
    top_parent_slots = [(alloc_external(child*child, transform_dt), alloc_external(child*child, transform_dt)) for _ in range(2)]
    scratch_a = ctypes.create_string_buffer(49*leaf*leaf*2)
    scratch_b = ctypes.create_string_buffer(49*leaf*leaf*2)
    scratch_c = ctypes.create_string_buffer(child*child*2)
    scratch_a_addr, scratch_b_addr, scratch_c_addr = ctypes.addressof(scratch_a), ctypes.addressof(scratch_b), ctypes.addressof(scratch_c)
    def prepare_operands(a_parent, b_parent, slot) -> None:
      nonlocal cpu_transform_work
      la, lb, _, _ = slot
      cpu_start = time.perf_counter()
      lib_cpu.transform2_f16(scratch_a_addr, a_parent._buf.cpu_view().addr, child, 0)
      lib_cpu.transform2_f16(scratch_b_addr, b_parent._buf.cpu_view().addr, child, 1)
      ctypes.memmove(la._buf.cpu_view().addr, scratch_a_addr, la._buf.size)
      ctypes.memmove(lb._buf.cpu_view().addr, scratch_b_addr, lb._buf.size)
      cpu_transform_work += time.perf_counter()-cpu_start

    (a_parent, _), (b_parent, _) = top_parent_slots[0]
    times.transform += top_paths["A"][0](a_parent._buf, a._buf, global_size=grid(path_groups), local_size=(wg, 1, 1), wait=True)
    times.transform += top_paths["B"][0](b_parent._buf, b._buf, global_size=grid(path_groups), local_size=(wg, 1, 1), wait=True)
    lib_cpu.cache_invalidate(a_parent._buf.cpu_view().addr, a_parent._buf.size)
    lib_cpu.cache_invalidate(b_parent._buf.cpu_view().addr, b_parent._buf.size)
    prepare_operands(a_parent, b_parent, slots[0])
    for p in range(49):
      la, lb, lm, _ = slots[p&1]
      path_signal = None
      if p+1 < 49:
        (next_a_parent, _), (next_b_parent, _) = top_parent_slots[(p+1)&1]
        top_paths["A"][p+1](next_a_parent._buf, a._buf, global_size=grid(path_groups), local_size=(wg, 1, 1), wait=False)
        top_paths["B"][p+1](next_b_parent._buf, b._buf, global_size=grid(path_groups), local_size=(wg, 1, 1), wait=False)
        path_signal = dev.timeline_value-1
      launch_leaf_products(lm, la, lb, wait=False)
      if path_signal is not None:
        dev.timeline_signal.wait(path_signal)
        lib_cpu.cache_invalidate(next_a_parent._buf.cpu_view().addr, next_a_parent._buf.size)
        lib_cpu.cache_invalidate(next_b_parent._buf.cpu_view().addr, next_b_parent._buf.size)
        prepare_operands(next_a_parent, next_b_parent, slots[(p+1)&1])
      if p:
        cpu_start = time.perf_counter()
        prev_lm = slots[(p-1)&1][2]
        lib_cpu.cache_invalidate(prev_lm._buf.cpu_view().addr, prev_lm._buf.size)
        out = top_m._buf.offset((p-1)*child_output_bytes, child_output_bytes)
        lib_cpu.combine2_f16(scratch_c_addr, prev_lm._buf.cpu_view().addr, leaf)
        ctypes.memmove(out.cpu_view().addr, scratch_c_addr, out.size)
        cpu_combine_work += time.perf_counter()-cpu_start
      dev.synchronize()
    la, lb, lm, _ = slots[0]
    lib_cpu.cache_invalidate(lm._buf.cpu_view().addr, lm._buf.size)
    cpu_start = time.perf_counter()
    out = top_m._buf.offset(48*child_output_bytes, child_output_bytes)
    lib_cpu.combine2_f16(scratch_c_addr, lm._buf.cpu_view().addr, leaf)
    ctypes.memmove(out.cpu_view().addr, scratch_c_addr, out.size)
    cpu_combine_work += time.perf_counter()-cpu_start
  else:
    for p in range(49):
      if stream_top:
        times.transform += top_paths["A"][p](top_a._buf, a._buf, global_size=grid(path_groups), local_size=(wg, 1, 1), wait=True)
        times.transform += top_paths["B"][p](top_b._buf, b._buf, global_size=grid(path_groups), local_size=(wg, 1, 1), wait=True)
      mid_a, mid_b = alloc(49*mid*mid, transform_dt), alloc(49*mid*mid, transform_dt)
      pa = top_a._buf if stream_top else top_a._buf.offset(p*child_input_bytes, child_input_bytes)
      pb = top_b._buf if stream_top else top_b._buf.offset(p*child_input_bytes, child_input_bytes)
      times.transform += child_t["A"](mid_a._buf, pa, global_size=grid(child_transform_groups), local_size=(wg, 1, 1), wait=True)
      times.transform += child_t["B"](mid_b._buf, pb, global_size=grid(child_transform_groups), local_size=(wg, 1, 1), wait=True)
      if strassen5:
        la, lb = alloc(343*leaf*leaf, transform_dt), alloc(343*leaf*leaf, transform_dt)
        lm, mid_m = alloc(343*leaf*leaf, product_dt), alloc(49*mid*mid, product_dt)
        times.transform += extra_t["A"](la._buf, mid_a._buf, global_size=grid(extra_transform_groups), local_size=(wg, 1, 1), wait=True)
        times.transform += extra_t["B"](lb._buf, mid_b._buf, global_size=grid(extra_transform_groups), local_size=(wg, 1, 1), wait=True)
        launch_leaf_products(lm, la, lb, wait=True, products=343)
        elapsed = extra_c(mid_m._buf, lm._buf, global_size=grid(extra_combine_groups), local_size=(combine_wg, 1, 1), wait=True)
        times.combine += elapsed
        child_combine_time += elapsed
      else:
        la, lb, lm, mid_m = mid_a, mid_b, alloc(49*leaf*leaf, product_dt), None
        launch_leaf_products(lm, la, lb, wait=True)
      out = top_m._buf.offset(p*child_output_bytes, child_output_bytes)
      elapsed = child_c(out, (mid_m if strassen5 else lm)._buf, global_size=grid(child_combine_groups),
                        local_size=(combine_wg, 1, 1), wait=True)
      times.combine += elapsed
      child_combine_time += elapsed
  allocation_start = time.perf_counter()
  if cpu_pipeline: del top_parent_slots
  else: del top_a, top_b
  gc.collect()
  c = alloc(n*n, dtypes.float)
  allocation_overhead = time.perf_counter()-allocation_start
  if split_top_combine:
    split_groups = (child*child//combine_vec+combine_wg-1)//combine_wg
    top_combine_time = sum(prg(c._buf, top_m._buf, global_size=grid(split_groups), local_size=(combine_wg, 1, 1), wait=True)
                           for prg in top_c)
  else:
    top_combine_time = top_c(c._buf, top_m._buf, global_size=grid(top_combine_groups), local_size=(combine_wg, 1, 1), wait=True)
  times.combine += top_combine_time
  if cpu_pipeline: times.wall = time.perf_counter()-candidate_start-allocation_overhead

  rtol = float(os.getenv("RTOL", "1e-3"))
  atol = float(os.getenv("ATOL", "8e-3" if product_half else "5e-3" if transform_half else "2e-3"))
  ref_block = int(os.getenv("REF_BLOCK", "256"))
  bad_count, max_abs, sum_abs, err2, ref2 = 0, 0.0, 0.0, 0.0, 0.0
  reference_ms = 0.0
  if gpu_ref:
    # Free recursive intermediates, then compute an independent conventional
    # FP32-accumulating GEMM in row slabs. It is an oracle only and is excluded
    # from times.total; slab streaming avoids a second N*N float allocation.
    del top_m, la, lb, lm
    if not cpu_pipeline: del mid_a, mid_b
    if strassen5: del mid_m
    if cpu_pipeline: del slots, scratch_a, scratch_b, scratch_c
    gc.collect()
    if n % ref_block or ref_block % 16: raise ValueError("GPU reference requires REF_BLOCK to divide N and be a multiple of 16")
    ref = alloc(ref_block*n, dtypes.float)
    q.M, q.N, q.K = ref_block, n, n
    q.K4 = n//4
    renv, rio, rsz, rro = get_envelope(dev, q.make_direct_image_donor_src(2, 128))
    ref_k4 = int(os.getenv("REF_K4", "256"))
    references = []
    for kstart in range(0, n//4, ref_k4):
      kcount = min(ref_k4, n//4-kstart)
      rshader, rhregs, rfregs, _ = q.build_4x8_fp32_rotate_shader(dev, 128, k_count=kcount, k_start=kstart)
      rlib = inject(renv, rio, rsz, rro, rshader, fregs=rfregs, hregs=rhregs, mergedregs=False)
      references.append(dev.runtime("gemm_h", rlib, buf_dtypes=[
        ((0, dtypes.float, (ref_block, n//4, 4)),), ((0, dtypes.half, (ref_block, n//4, 4)),),
        ((1, dtypes.half, (n, n//4, 4)),)]))
    for first in range(0, n, ref_block):
      av = a._buf.offset(first*n*2, ref_block*n*2)
      got = np.empty((ref_block, n), np.float32)
      expected = np.zeros((ref_block, n), np.float32)
      partial = np.empty((ref_block, n), np.float32)
      for reference in references:
        reference_ms += reference(ref._buf, av, b._buf, global_size=(n//256, ref_block//16, 1),
                                  local_size=(128, 1, 1), wait=True)*1e3
        ref.copyout(memoryview(partial).cast("B"))
        expected += partial
      c.view(ref_block*n, dtypes.float, first*n*4).ensure_allocated().copyout(memoryview(got).cast("B"))
      err = got-expected
      delta = np.abs(err)
      bad_count += int(np.count_nonzero(~np.isclose(got, expected, rtol=rtol, atol=atol)))
      max_abs = max(max_abs, float(delta.max()))
      sum_abs += float(delta.astype(np.float64).sum())
      err2 += float(np.square(err.astype(np.float64)).sum())
      ref2 += float(np.square(expected.astype(np.float64)).sum())
  else:
    got = np.empty((n, n), np.float32)
    c.copyout(memoryview(got).cast("B"))
    del a, b, c, top_m, la, lb, lm
    gc.collect()
    ref_k_block = int(os.getenv("REF_K_BLOCK", "256"))
    for first in range(0, n, ref_block):
      last = min(n, first+ref_block)
      expected = np.zeros((last-first, n), np.float32)
      for kfirst in range(0, n, ref_k_block):
        klast = min(n, kfirst+ref_k_block)
        expected += a_np[first:last, kfirst:klast].astype(np.float32) @ b_np[kfirst:klast].astype(np.float32)
      err = got[first:last]-expected
      delta = np.abs(err)
      bad_count += int(np.count_nonzero(~np.isclose(got[first:last], expected, rtol=rtol, atol=atol)))
      max_abs = max(max_abs, float(delta.max()))
      sum_abs += float(delta.astype(np.float64).sum())
      err2 += float(np.square(err.astype(np.float64)).sum())
      ref2 += float(np.square(expected.astype(np.float64)).sum())
  mean_abs, rel_l2 = sum_abs/(n*n), (err2/ref2)**0.5
  print(f"shape={n}x{n}x{n} algorithm={'strassen5_fused2' if strassen5 else 'strassen4_fused2'} "
        f"inputs=fp16 accumulate=fp32 elapsed_ms={times.total*1e3:.3f} "
        f"gflops={2*n**3/times.total/1e9:.1f} transform_storage={'fp16' if transform_half else 'fp32'} "
        f"product_storage={'fp16' if product_half else 'fp32'} "
        f"transform_vec={transform_vec} combine_vec={combine_vec} leaf_unroll={leaf_unroll} "
        f"pipeline={'cpu_gpu' if cpu_pipeline else 'stream_gpu' if stream_top else 'serial_gpu'} "
        f"cpu_threads={cpu_threads if cpu_pipeline else 0} "
        f"transform_ms={times.transform*1e3:.3f} gemm_ms={times.gemm*1e3:.3f} "
        f"combine_ms={times.combine*1e3:.3f} child_combine_ms={child_combine_time*1e3:.3f} "
        f"top_combine_ms={top_combine_time*1e3:.3f} cpu_transform_work_ms={cpu_transform_work*1e3:.3f} "
        f"cpu_combine_work_ms={cpu_combine_work*1e3:.3f} outputs={n*n} bad_count={bad_count} "
        f"max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} rel_l2={rel_l2:.9g} "
        f"rtol={rtol:g} atol={atol:g} allclose={bad_count == 0} oracle={'gpu_direct' if gpu_ref else 'numpy'} "
        f"reference_ms={reference_ms:.3f} loop_instrs={loop_instrs}")
  if bad_count: raise SystemExit(1)


if __name__ == "__main__": main()
