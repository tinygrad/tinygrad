import ctypes, itertools
from tinygrad.codegen import to_program
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import get_runtime
from tinygrad.renderer.amd import InstDecodeError, decode_inst
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from test.mockgpu.amd.emu import _Ctx, _INST_HANDLERS, _MXCSRContext, _init_wave, _op_name, _wave_size

def run_asm(lib:int, lib_sz:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int, args_ptr:int, rsrc2:int=0x19c,
            scratch_size:int=0, arch:str="rdna3", user_data:list[int]|None=None) -> int:
  code = ctypes.string_at(lib, lib_sz)
  offset = 0
  calls:list[UOp] = []
  while offset < len(code):
    try: inst = decode_inst(code[offset:], arch)
    except InstDecodeError: break
    if offset + inst.size() > len(code): raise RuntimeError(f"truncated instruction at {offset:#x}")
    if _op_name(inst) == "S_CODE_END": break
    handler = next((_INST_HANDLERS[cls] for cls in type(inst).__mro__ if cls in _INST_HANDLERS), None)
    if handler is None: raise RuntimeError(f"unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")
    ctx = _Ctx(inst.size(), _wave_size(arch))
    body = handler(inst, ctx)
    calls.append(body.call(ctx.sgpr, ctx.vgpr, ctx.vmem, ctx.lds, ctx.scratch, ctx.accvgpr, name=f"{_op_name(inst).lower()}_{offset:x}"))
    offset += inst.size()
  sink = UOp.sink(UOp(Ops.LINEAR, src=tuple(calls)), arg=KernelInfo(name="asm_call")).rtag(1)
  prg = to_program(sink, Device["CPU"].renderer)
  runtime = get_runtime("CPU", prg)
  wave_size, total_threads = _wave_size(arch), lx * ly * lz
  lds_size = ((rsrc2 >> 15) & 0x1ff) * 512
  lds = Buffer("CPU", max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()
  scratch = Buffer("CPU", max(scratch_size * wave_size, 1), dtypes.uint8).ensure_allocated()
  with _MXCSRContext():
    for gidz, gidy, gidx in itertools.product(range(gz), range(gy), range(gx)):
      ctypes.memset(lds._buf, 0, max(lds_size, 4))
      for wave_start in range(0, total_threads, wave_size):
        st = _init_wave(lib, wave_start, total_threads, lx, ly, lz, args_ptr, rsrc2, scratch_size, arch, gidx, gidy, gidz, user_data, wave_size)
        bufs = [st.sgpr_buf._buf, st.vgpr_buf._buf, 0, lds._buf, scratch._buf, st.accvgpr_buf._buf]
        runtime(*[bufs[g] for g in prg.arg.globals])
  return 0
