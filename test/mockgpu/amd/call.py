import ctypes, itertools
from tinygrad.viz.serve import amd_decode
from tinygrad.uop.ops import UOp, Ops, KernelInfo, PatternMatcher, UPat, graph_rewrite, rewrite_group
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.helpers import Context, getenv, TracingKey
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size, _canonical_info, PC_LO_IDX, PC_HI_IDX

asm_call_counter = itertools.count(1)

def remove_pc(ctx:int, x:UOp): pass

pm_asm_call = PatternMatcher([
  (UPat((Ops.LOAD, Ops.STORE), name="x"), remove_pc),
])

@rewrite_group(name=lambda *args,ret,**_: TracingKey(f"Lift {(k:=ret.src[0].arg).name}"))
def lift(lib: int, lib_sz: int, arch: str = "rdna3", backend: str|None = None) -> UOp:
  backend = getenv("ASM_CALL_BACKEND", "CPU") if backend is None else backend
  # decode
  lib_bytes = ctypes.string_at(lib, lib_sz)
  insts = amd_decode(lib_bytes, arch)
  # construct CALL graph
  afters: dict[UOp, UOp] = {}
  for off, inst in insts.items():
    if str(inst) == "s_code_end()": continue
    ctx = _Ctx(inst.size(), _wave_size(arch))
    sink = _get_handler(inst)(inst, ctx)
    *_, canonical_name = _canonical_info(inst, ctx, lib_bytes[off:])
    bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
    body = sink.substitute({b:b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)})
    args = [afters.get(b, b) for b in bufs]
    call = body.call(*args, name=canonical_name)
    afters.update((b, arg.after(call)) for b, arg in zip(bufs, args))
  sink = UOp.sink(*afters.values(), arg=KernelInfo(name=f"asm_call n{next(asm_call_counter)}", opts_to_apply=()))
  sink = graph_rewrite(sink, pm_asm_call, ctx=lib+off, name="pm_asm_call")
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    return to_program(sink, Device[backend].renderer)
