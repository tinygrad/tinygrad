import ctypes, itertools
from tinygrad.viz.serve import amd_decode, get_cfg, COND_TAKEN, COND_NOT_TAKEN
from tinygrad.uop.ops import UOp, Ops, KernelInfo, PatternMatcher, UPat, graph_rewrite, rewrite_group
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import Invalid
from tinygrad.helpers import Context, getenv, TracingKey
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size, _canonical_info, PC_LO_IDX, PC_HI_IDX

asm_call_counter = itertools.count(1)

def pc_index(idx:int) -> UPat:
  reg, null = UPat.const(idx).cast(), UPat.const(124).cast()
  return UPat.any(reg, reg.ne(null).where(reg, UPat.const(Invalid)))

pm_asm_call = PatternMatcher([
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.PARAM, name="buf").index(UPat.any(pc_index(PC_LO_IDX), pc_index(PC_HI_IDX))),), allow_any_len=True),
   lambda buf: UOp(Ops.NOOP) if buf.arg.name == "sgpr" else None),
])

@rewrite_group(name=lambda *args,ret,**_: TracingKey(f"Lift {(k:=ret.src[0].arg).name}", (("lift", k.function_name),)))
def lift(lib: int, lib_sz: int, arch: str = "rdna3", backend: str|None = None) -> UOp:
  backend = getenv("ASM_CALL_BACKEND", "CPU") if backend is None else backend
  # decode
  lib_bytes = ctypes.string_at(lib, lib_sz)
  insts = amd_decode(lib_bytes, arch)
  cfg = get_cfg(insts)["data"]
  # construct CALL graph
  afters: dict[UOp, UOp] = {}
  for block_pc, block in cfg["blocks"].items():
    loop_path = cfg["paths"][block_pc].get(block_pc)
    loop = UOp.loop(block_pc) if loop_path is not None else None
    if loop is not None: afters = {b:arg.after(loop) for b,arg in afters.items()}
    branch_cond:UOp|None = None
    for off in block:
      inst = insts[off]
      inst_st = str(inst)
      if inst_st.startswith("s_code_end"): continue
      if inst_st.startswith(("s_getpc", "s_setpc")): raise AssertionError("getpc and setpc are not allowed in ASM_CALL")
      ctx = _Ctx(inst.size(), _wave_size(arch), inst_addr=lib+off)
      sink = _get_handler(inst)(inst, ctx)
      *_, canonical_name = _canonical_info(inst, ctx, lib_bytes[off:])
      bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
      args = [afters.get(b, b.after(loop) if loop is not None else b) for b in bufs]
      if ctx.branch_cond is not None and loop_path is not None:
        branch_cond = ctx.branch_cond.substitute(dict(zip(bufs, args)), walk=True)
        continue
      body = sink.substitute({b:b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)})
      call = body.call(*args, name=canonical_name)
      afters.update((b, arg.after(call)) for b, arg in zip(bufs, args))
    if loop is not None:
      assert branch_cond is not None and loop_path in (COND_TAKEN, COND_NOT_TAKEN)
      if loop_path is COND_NOT_TAKEN: branch_cond = branch_cond.logical_not()
      backedge = UOp.sink(*afters.values()).backedge(loop, branch_cond)
      afters = {b:arg.after(backedge) for b,arg in afters.items()}
  sink = UOp.sink(*afters.values(), arg=KernelInfo(name=f"asm_call n{next(asm_call_counter)}", opts_to_apply=()))
  sink = graph_rewrite(sink, pm_asm_call, name="pm_asm_call", bottom_up=True, enter_calls=True)
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    return to_program(sink, Device[backend].renderer)
