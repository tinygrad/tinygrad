import ctypes
from tinygrad.viz.serve import amd_decode
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, graph_rewrite
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size

pm_lift = PatternMatcher([])

def lift(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c,
            scratch_size: int = 0, arch: str = "rdna3", user_data: list[int]|None = None) -> int:
  # decode
  lib_bytes = ctypes.string_at(lib, lib_sz)
  insts = amd_decode(lib_bytes, arch)
  calls = []
  for inst in insts.values():
    ctx = _Ctx(inst.size(), _wave_size(arch))
    calls.append(_get_handler(inst)(inst, ctx).call())
  linear = UOp(Ops.LINEAR, src=tuple(calls))
  graph_rewrite(linear, pm_lift, name="pm_lift")
  return 0
