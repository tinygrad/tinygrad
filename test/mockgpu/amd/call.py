import ctypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.renderer.amd import InstDecodeError, decode_inst
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from test.mockgpu.amd.emu import _Ctx, _INST_HANDLERS, _op_name, _wave_size

def run_asm(lib:int, lib_sz:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int, args_ptr:int, rsrc2:int=0x19c,
            scratch_size:int=0, arch:str="rdna3", user_data:list[int]|None=None) -> int:
  code = ctypes.string_at(lib, lib_sz)
  offset = 0
  calls:list[UOp] = []
  while offset < len(code):
    try: inst = decode_inst(code[offset:], arch)
    except InstDecodeError: break
    if offset + inst.size() > len(code): raise RuntimeError(f"truncated instruction at {offset:#x}")
    if _op_name(inst) == "S_CODE_END":
      offset += inst.size()
      continue
    handler = next((_INST_HANDLERS[cls] for cls in type(inst).__mro__ if cls in _INST_HANDLERS), None)
    if handler is None: raise RuntimeError(f"unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")
    ctx = _Ctx(inst.size(), _wave_size(arch))
    body = handler(inst, ctx)
    calls.append(body.call(ctx.sgpr, ctx.vgpr, ctx.vmem, ctx.lds, ctx.scratch, ctx.accvgpr, name=f"{_op_name(inst).lower()}_{offset:x}"))
    offset += inst.size()
  sink = UOp.sink(UOp(Ops.LINEAR, src=tuple(calls)), arg=KernelInfo(name="asm_call")).rtag(1)
  prg = to_program(sink, Device["CPU"].renderer)
  return 0
