from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import THREADS_PER_WG, alloc_like, dname_of, compile_hip

TILE_ROWS = 64

@functools.cache
def _custom_mxpack_transpose(out:UOp, inp:UOp, dname:str, rows:int, k4:int, E:int) -> UOp:
  num_wg = (rows // TILE_ROWS) * E
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  mem = E * rows * k4 * 4 * 2   # read + write uint32
  sink = UOp.sink(out.base, inp.base, threads, workgroups,
                  arg=KernelInfo(f"mxpack_transpose_{E}_{rows}_{k4}", estimates=Estimates(ops=E*rows*k4, mem=mem)))
  src = (pathlib.Path(__file__).parent/"mxpack_transpose.cpp").read_text()
  defines = [f"-DMXP_ROWS={rows}", f"-DMXP_K4={k4}", f"-DMXP_E={E}", f"-DTILE_ROWS={TILE_ROWS}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def fast_mx_pack(e8:Tensor) -> Tensor:
  # == mx_pack(e8): (rows, sk) e8 -> (sk//4, rows) uint32 (scale-major), byte-exact via a coalesced LDS transpose.
  rows, sk = e8.shape
  k4 = sk // 4
  assert sk % 4 == 0 and rows % TILE_ROWS == 0, f"fast_mx_pack: sk={sk}%4, rows={rows}%{TILE_ROWS}"
  device = e8.device
  axis = e8.uop.axis if isinstance(device, tuple) else None
  out_axis = None if axis is None else (1 if axis == 0 else 0)
  out = alloc_like((k4, rows), dtypes.uint32, device, out_axis)
  fxn = functools.partial(_custom_mxpack_transpose, dname=dname_of(device), rows=rows, k4=k4, E=1)
  return Tensor.custom_kernel(out, e8, fxn=fxn)[0]

def fast_mx_pack_3d(e8:Tensor) -> Tensor:
  # == mx_pack_3d(e8): (E, rows, sk) e8 -> (E, sk//4, rows) uint32, byte-exact via a coalesced LDS transpose.
  E, rows, sk = e8.shape
  k4 = sk // 4
  assert sk % 4 == 0 and rows % TILE_ROWS == 0, f"fast_mx_pack_3d: sk={sk}%4, rows={rows}%{TILE_ROWS}"
  device = e8.device
  axis = e8.uop.axis if isinstance(device, tuple) else None
  out_axis = None if axis is None else (0 if axis == 0 else (2 if axis == 1 else None))
  out = alloc_like((E, k4, rows), dtypes.uint32, device, out_axis)
  fxn = functools.partial(_custom_mxpack_transpose, dname=dname_of(device), rows=rows, k4=k4, E=E)
  return Tensor.custom_kernel(out, e8, fxn=fxn)[0]
