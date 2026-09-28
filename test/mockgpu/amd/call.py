import ctypes, itertools
from tinygrad.viz.serve import amd_decode
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, graph_rewrite, KernelInfo
from tinygrad.schedule.prepare import resolve_function
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.helpers import Context
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size

def inline_call(ctx, c:UOp):
  # renumber the ranges per call body
  body = c.body.substitute({r:r.replace(arg=(next(ctx), *r.arg[1:])) for r in c.body.toposort() if r.op is Ops.RANGE})
  if (sink:=resolve_function(c.replace(src=(body, *c.src[1:])))) is not None: return UOp.group(*sink.src)
  return None

pm_lift = PatternMatcher([(UPat(Ops.CALL, name="c"), inline_call)])

def lift(lib: int, lib_sz: int, arch: str = "rdna3") -> UOp:
  # decode
  lib_bytes = ctypes.string_at(lib, lib_sz)
  insts = amd_decode(lib_bytes, arch)
  # construct CALL graph
  afters: dict[UOp, UOp] = {}
  for inst in insts.values():
    if str(inst) == "s_code_end()": continue
    ctx = _Ctx(inst.size(), _wave_size(arch))
    sink = _get_handler(inst)(inst, ctx)
    bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
    body = sink.substitute({b:b.param_like(i) for i,b in enumerate(bufs)})
    args = [afters.get(b, b) for b in bufs]
    call = body.call(*args)
    afters.update((b, arg.after(call)) for b, arg in zip(bufs, args))
  sink = graph_rewrite(UOp.sink(*afters.values()), pm_lift, ctx=itertools.count(), name="pm_lift").replace(arg=KernelInfo(name="asm_call"))
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    return to_program(sink, Device['CPU'].renderer)
