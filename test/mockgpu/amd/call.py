import ctypes, itertools
from tinygrad.viz.serve import amd_decode
from tinygrad.uop.ops import UOp, Ops, KernelInfo, PatternMatcher, UPat, graph_rewrite, rewrite_group
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import dtypes, Invalid
from tinygrad.helpers import Context, getenv, TracingKey
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size, _canonical_info, PC_LO_IDX, PC_HI_IDX

asm_call_counter = itertools.count(1)

def _static_pc(buf:UOp) -> int|None:
  return buf.tag if type(buf.tag) is int else None

def read_pc_lo(buf:UOp) -> UOp|None:
  return UOp.const(pc & 0xffffffff, dtypes.uint32) if (pc:=_static_pc(buf)) is not None else None

def read_pc_hi(buf:UOp) -> UOp|None:
  return UOp.const(pc >> 32, dtypes.uint32) if (pc:=_static_pc(buf)) is not None else None

def remove_pc(buf:UOp) -> UOp|None:
  return UOp(Ops.NOOP) if _static_pc(buf) is not None else None

def remove_pc_tag(buf:UOp) -> UOp|None:
  return buf.replace(tag=None) if _static_pc(buf) is not None else None

def _pc_index(idx:int) -> UPat:
  reg, null = UPat.const(idx).cast(), UPat.const(124).cast()
  return UPat.any(reg, reg.ne(null).where(reg, UPat.const(Invalid)))

pm_asm_call = PatternMatcher([
  (UPat(Ops.PARAM, name="buf").index(_pc_index(PC_LO_IDX)).load(), read_pc_lo),
  (UPat(Ops.PARAM, name="buf").index(_pc_index(PC_HI_IDX)).load(), read_pc_hi),
  (UPat(Ops.PARAM, name="buf").index(_pc_index(PC_LO_IDX)).store(UPat()), remove_pc),
  (UPat(Ops.PARAM, name="buf").index(_pc_index(PC_HI_IDX)).store(UPat()), remove_pc),
  (UPat(Ops.PARAM, name="buf"), remove_pc_tag),
])

@rewrite_group(name=lambda *args,ret,**_: TracingKey(f"Lift {(k:=ret.src[0].arg).name}", (("lift", k.function_name),)))
def lift(lib: int, lib_sz: int, arch: str = "rdna3", backend: str|None = None) -> UOp:
  backend = getenv("ASM_CALL_BACKEND", "CPU") if backend is None else backend
  # decode
  lib_bytes = ctypes.string_at(lib, lib_sz)
  insts = amd_decode(lib_bytes, arch)
  # construct CALL graph
  afters: dict[UOp, UOp] = {}
  for off, inst in insts.items():
    inst_st = str(inst)
    if inst_st.startswith("s_code_end"): continue
    if inst_st.startswith(("s_getpc", "s_setpc")): raise AssertionError("getpc and setpc are not allowed in ASM_CALL")
    ctx = _Ctx(inst.size(), _wave_size(arch))
    sink = _get_handler(inst)(inst, ctx)
    *_, canonical_name = _canonical_info(inst, ctx, lib_bytes[off:])
    bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
    # The tag makes the shared SGPR parameter instruction-local during pm_asm_call, which strips it from the result.
    body = sink.substitute({b:b.param_like(i, name=b.arg.name).replace(tag=lib+off) if b is ctx.sgpr else
                            b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)})
    args = [afters.get(b, b) for b in bufs]
    call = body.call(*args, name=canonical_name)
    afters.update((b, arg.after(call)) for b, arg in zip(bufs, args))
  sink = UOp.sink(*afters.values(), arg=KernelInfo(name=f"asm_call n{next(asm_call_counter)}", opts_to_apply=()))
  sink = graph_rewrite(sink, pm_asm_call, name="pm_asm_call", bottom_up=True, enter_calls=True)
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    return to_program(sink, Device[backend].renderer)
