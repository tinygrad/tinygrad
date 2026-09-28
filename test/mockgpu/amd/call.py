import ctypes
from tinygrad.viz.serve import amd_decode
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, graph_rewrite, KernelInfo
from tinygrad.schedule.prepare import resolve_function
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.helpers import Context
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size

pm_lift = PatternMatcher([
  (UPat(Ops.CALL, name="c"), lambda c: UOp.group(*s.src) if (s:=resolve_function(c)) is not None else None),
])

def lift(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c,
            scratch_size: int = 0, arch: str = "rdna3", user_data: list[int]|None = None) -> int:
  # decode
  lib_bytes = ctypes.string_at(lib, lib_sz)
  insts = amd_decode(lib_bytes, arch)
  assigns: dict[UOp, UOp] = {}
  for inst in insts.values():
    if str(inst) == "s_code_end()": continue
    ctx = _Ctx(inst.size(), _wave_size(arch))
    sink = _get_handler(inst)(inst, ctx)
    bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
    body = sink.substitute({b:b.param_like(i) for i,b in enumerate(bufs)})
    args = [assigns.get(b, b) for b in bufs]
    call = body.call(*args)
    assigns.update((b, arg.after(call)) for b, arg in zip(bufs, args))
  sink = graph_rewrite(UOp.sink(*assigns.values()), pm_lift, name="pm_lift").replace(arg=KernelInfo(name="asm_call"))
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    to_program(sink, Device['CPU'].renderer)
  return 0
