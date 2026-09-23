import ctypes, itertools
from tinygrad.codegen import pm_add_loads, to_program
from tinygrad.device import Buffer, Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import get_runtime
from tinygrad.renderer.amd import InstDecodeError, decode_inst
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp, UPat, PatternMatcher, graph_rewrite
from test.mockgpu.amd.emu import _Ctx, _INST_HANDLERS, _MXCSRContext, _init_wave, _op_name, _wave_size
from test.mockgpu.amd.emu import PC_LO_IDX, PC_HI_IDX

class _CallCtx(_Ctx):
  def __init__(self, code:bytes, pc:int, wave_size:int):
    super().__init__(len(code), wave_size)
    self.code, self.pc = code, pc

  def inst_word(self, dword_idx:int) -> UOp:
    return UOp.const(int.from_bytes(self.code[dword_idx*4:(dword_idx+1)*4], "little"), dtypes.uint32)

  def rpc(self) -> UOp: return UOp.const(self.pc, dtypes.uint64)
  def inc_pc(self) -> list[UOp]: return []

  def wsgpr_dyn(self, reg:UOp, val:UOp) -> UOp:
    if reg.vmin == reg.vmax and reg.vmin in (PC_LO_IDX, PC_HI_IDX):
      # The old emulator terminates by storing an all-ones PC. No runtime PC is needed here.
      if val.vmin == val.vmax == 0xffffffff: return UOp.sink()
      raise NotImplementedError("ASM_CALL requires control-flow lifting for PC writes")
    return super().wsgpr_dyn(reg, val)

class _CallGraph:
  def __init__(self, ctx:_CallCtx):
    self.banks = {ctx.sgpr:("s", 1), ctx.vgpr:("v", ctx.wave_size)}
    if ctx.accvgpr is not ctx.vgpr: self.banks[ctx.accvgpr] = ("a", ctx.wave_size)
    self.params:dict[tuple[UOp, int], UOp] = {}
    self.operands:set[UOp] = set()
    self.values:dict[UOp, UOp] = {}
    self.readers:dict[UOp, list[UOp]] = {}
    self.calls:list[UOp] = []

  def param(self, bank:UOp, reg:int=0) -> UOp:
    if (key:=(bank, reg)) not in self.params:
      name, width = self.banks.get(bank, (bank.arg.name, bank.arg.size))
      is_reg = bank in self.banks
      p = self.params[key] = UOp.param(reg if is_reg else bank.arg.slot, bank.dtype, (width,), name=name,
                                      addrspace=AddrSpace.REG if is_reg else bank.addrspace)
      self.operands.add(p)
      self.values[p] = p
      self.readers[p] = []
    return self.params[key]

  def index(self, idx:UOp) -> UOp|None:
    if (bank:=idx.src[0].without_after) not in self.banks: return None
    width = self.banks[bank][1]
    offset = idx.src[1].get_idx()
    reg = (offset // width).simplify()
    if reg.vmin != reg.vmax: raise NotImplementedError(f"dynamic register index: {reg.render()}")
    buf = self.param(bank, int(reg.vmin))
    if idx.src[0].op is Ops.AFTER: buf = buf.after(*idx.src[0].src[1:])
    return buf.index((offset % width).valid(idx.src[1].get_valid()))

  def append(self, body:UOp, name:str):
    body = graph_rewrite(body, pm_add_loads, name="explicit instruction loads")
    body = graph_rewrite(body, pm_register_operands, ctx=self, name="bind instruction registers")
    used = {p for p in body.toposort() if p.op is Ops.PARAM}
    if not used <= self.operands: raise NotImplementedError("unbound register bank")
    writes = {u.src[0].buf_uop for u in body.toposort() if u.op is Ops.STORE}
    reads = {u.src[0].buf_uop for u in body.toposort() if u.op is Ops.LOAD}
    # Consecutive registers with the same access mode form an operand STACK.
    groups:list[list[UOp]] = []
    last = None
    for (bank, reg), p in sorted(self.params.items(), key=lambda item: (item[0][0].arg.slot, item[0][1])):
      if p not in used: continue
      mode = (p in reads, p in writes)
      if bank in self.banks and last == (bank, reg-1, mode): groups[-1].append(p)
      else: groups.append([p])
      last = (bank, reg, mode)
    args = []
    for group in groups:
      vals = [self.values[p].after(*self.readers[p]) if p in writes else self.values[p] for p in group]
      arg = UOp.stack(*vals) if len(vals) > 1 else vals[0]
      args.append(arg)
    call = body.call(*args, name=name)
    self.calls.append(call)
    for p in used:
      if p in writes: self.values[p], self.readers[p] = p.after(call), []
      elif p in reads: self.readers[p].append(call)

pm_register_operands = PatternMatcher([
  (UPat(Ops.INDEX, name="idx"), lambda ctx,idx: ctx.index(idx)),
  (UPat(Ops.PARAM, name="p"), lambda ctx,p: ctx.param(p) if p not in ctx.banks and p not in ctx.operands else None),
])

def lower_register(p:UOp) -> UOp|None:
  if p.addrspace is not AddrSpace.REG: return None
  if p.arg.name == "s": bank = _Ctx.sgpr
  elif p.arg.name in ("v", "a"):
    bank = UOp.param(1 if p.arg.name == "v" else 5, dtypes.uint32, 256 * p.arg.size,
                     name="vgpr" if p.arg.name == "v" else "accvgpr")
  else: raise ValueError(f"unknown register bank: {p.arg.name}")
  start = p.arg.slot * p.arg.size
  return bank.shrink(((start, start + p.arg.size),))

class _CallRenderer(ClangRenderer):
  pre_matcher = PatternMatcher([(UPat(Ops.PARAM, name="p"), lower_register)])

def run_asm(lib:int, lib_sz:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int, args_ptr:int, rsrc2:int=0x19c,
            scratch_size:int=0, arch:str="rdna3", user_data:list[int]|None=None) -> int:
  code = ctypes.string_at(lib, lib_sz)
  offset = 0
  graph = _CallGraph(_CallCtx(b"", lib, _wave_size(arch)))
  while offset < len(code):
    try: inst = decode_inst(code[offset:], arch)
    except InstDecodeError: break
    if offset + inst.size() > len(code): raise RuntimeError(f"truncated instruction at {offset:#x}")
    if _op_name(inst) == "S_CODE_END": break
    handler = next((_INST_HANDLERS[cls] for cls in type(inst).__mro__ if cls in _INST_HANDLERS), None)
    if handler is None: raise RuntimeError(f"unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")
    ctx = _CallCtx(code[offset:offset+inst.size()], lib + offset, _wave_size(arch))
    body = handler(inst, ctx).simplify(tracked=True)
    graph.append(body, name=f"{_op_name(inst).lower()}_{offset:x}")
    offset += inst.size()
  sink = UOp.sink(*graph.calls, arg=KernelInfo(name="asm_call")).rtag(1)
  prg = to_program(sink, _CallRenderer(Device["CPU"].renderer.target))
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
