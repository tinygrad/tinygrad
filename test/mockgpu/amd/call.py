import ctypes
import re
from tinygrad.codegen import pm_add_loads, to_program
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.amd import InstDecodeError, decode_inst
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp, UPat, PatternMatcher, graph_rewrite
from test.mockgpu.amd.emu import _Ctx, _INST_HANDLERS, _op_name, _wave_size, get_pcode, _get_pcode_dict
from test.mockgpu.amd.emu import PC_LO_IDX, PC_HI_IDX, SGPR_COUNT, SCRATCH_STRIDE_IDX, F32_INLINE, EXEC_LO, ttmp, hsa

class _CallCtx(_Ctx):
  def __init__(self, code:bytes, pc:int, wave_size:int):
    super().__init__(len(code), wave_size)
    self.code, self.pc = code, pc
    self.targets:dict[int, UOp] = {}

  def inst_word(self, dword_idx:int) -> UOp:
    return UOp.const(int.from_bytes(self.code[dword_idx*4:(dword_idx+1)*4], "little"), dtypes.uint32)

  def rpc(self) -> UOp: return UOp.const(self.pc, dtypes.uint64)
  def inc_pc(self) -> list[UOp]: return []

  def wsgpr_dyn(self, reg:UOp, val:UOp) -> UOp:
    reg = reg.simplify()
    if reg.vmin == reg.vmax and reg.vmin in (PC_LO_IDX, PC_HI_IDX):
      self.targets[int(reg.vmin)] = val
      return UOp.sink()
    return super().wsgpr_dyn(reg, val)

  def rsgpr_dyn(self, reg:UOp, valid:UOp|None=None) -> UOp:
    reg = reg.simplify()
    if reg.vmin == reg.vmax:
      idx = int(reg.vmin)
      value = idx-128 if 128 <= idx <= 192 else (192-idx)&0xffffffff if 193 <= idx <= 208 else None
      # Operand reads are gated. Ungated reads in this range also hold the emulator's HW registers.
      if idx in F32_INLINE and valid is not None: value = F32_INLINE[idx]
      if value is not None:
        ret = UOp.const(value, dtypes.uint32)
        return valid.where(ret, UOp.const(0, dtypes.uint32)) if valid is not None else ret
    return super().rsgpr_dyn(reg, valid)

class _CallGraph:
  def __init__(self, ctx:_CallCtx):
    self.banks = {ctx.sgpr:("s", 1), ctx.vgpr:("v", ctx.wave_size)}
    if ctx.accvgpr is not ctx.vgpr: self.banks[ctx.accvgpr] = ("a", ctx.wave_size)
    self.params:dict[tuple[UOp, int], UOp] = {}
    self.operands:set[UOp] = set()
    self.values:dict[UOp, UOp] = {}
    self.readers:dict[UOp, list[UOp]] = {}
    self.calls:list[UOp] = []
    self.deps:tuple[UOp, ...] = ()
    self.guards:list[UOp] = []
    self.storage:dict[str, UOp] = {}

  def condition(self, cond:UOp) -> UOp:
    cond = graph_rewrite(cond, pm_register_operands, ctx=self)
    return cond.substitute({p:p.after(*self.deps) for p in self.operands}, walk=True)

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
    if not any(u.op is Ops.STORE for u in body.toposort()): return
    if self.guards:
      gate = UOp.const(True)
      for p in self.guards: gate = gate & p.index(0).load()
      body = graph_rewrite(body, pm_gate_instruction, ctx=gate, walk=True, name="gate instruction memory")
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
    deps = self.deps
    args = [UOp.stack(*(v.after(*deps) for v in a.src)) if a.op is Ops.STACK else a.after(*deps) for a in args]
    call = body.call(*args, name=name)
    self.calls.append(call)
    self.deps = (call,)
    for p in used:
      if p in writes: self.values[p], self.readers[p] = p.after(call), []
      elif p in reads: self.readers[p].append(call)

pm_register_operands = PatternMatcher([
  (UPat(Ops.INDEX, name="idx"), lambda ctx,idx: ctx.index(idx)),
  (UPat(Ops.PARAM, name="p"), lambda ctx,p: ctx.param(p) if p not in ctx.banks and p not in ctx.operands else None),
])

pm_gate_instruction = PatternMatcher([
  (UPat(Ops.INDEX, name="idx"), lambda ctx,idx: idx.replace(src=(idx.src[0], idx.src[1].valid(ctx), *idx.src[2:]))),
])

def lower_register(ctx, p:UOp) -> UOp|None:
  if p.addrspace is not AddrSpace.REG: return ctx.banks.get(p.arg.name)
  start = p.arg.slot * p.arg.size
  return ctx.banks[p.arg.name].shrink(((start, start + p.arg.size),))

class _CallRenderer(ClangRenderer):
  pre_matcher = PatternMatcher([(UPat(Ops.PARAM, name="p"), lower_register)])

  def __init__(self, target, banks:dict[str, UOp]):
    super().__init__(target)
    self.banks = banks

  def __reduce__(self): return self.__class__, (self.target, self.banks)

def state_call(body:UOp, name:str, inputs:list[UOp], *deps:UOp) -> UOp:
  params = {u:UOp.param(i, u.dtype, u.shape, addrspace=AddrSpace.ALU if u.shape == () else AddrSpace.GLOBAL)
            for i, u in enumerate(inputs)}
  return body.substitute(params, walk=True).call(*(u.after(*deps) if u.shape else u for u in inputs), name=name)

def render_call(lib:int, lib_sz:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int,
                rsrc2:int, scratch_size:int, arch:str, user_words:int) -> UOp:
  code = ctypes.string_at(lib, lib_sz)
  offset = 0
  graph = _CallGraph(_CallCtx(b"", lib, _wave_size(arch)))
  instructions:dict[int, tuple[int, str, UOp, int|None, UOp]] = {}
  loops:dict[int, int] = {}
  while offset < lib_sz:
    try: inst = decode_inst(code[offset:], arch)
    except InstDecodeError: break
    if offset + inst.size() > lib_sz: raise RuntimeError(f"truncated instruction at {offset:#x}")
    if _op_name(inst) == "S_CODE_END": break
    name = _op_name(inst)
    branch = name == "S_BRANCH" or name.startswith("S_CBRANCH_")
    if not branch and name != "S_ENDPGM" and hasattr(inst, "op") and inst.op in _get_pcode_dict(inst.op):
      assert not re.search(r'\bPC\b', get_pcode(inst.op)), f"explicit PC access is not supported: {name}"
    assert not any(x in name for x in ('GETPC', 'SETPC', 'SWAPPC', 'RFE', 'CALL_B64')), f"explicit PC access is not supported: {name}"
    handler = next((_INST_HANDLERS[cls] for cls in type(inst).__mro__ if cls in _INST_HANDLERS), None)
    if handler is None: raise RuntimeError(f"unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")
    ctx = _CallCtx(code[offset:offset+inst.size()], offset, _wave_size(arch))
    body = handler(inst, ctx).simplify(tracked=True)
    target, cond = None, UOp.const(False)
    if branch:
      assert ctx.targets, f"missing branch semantics: {name}"
      displacement = int(getattr(inst, "simm16")) & 0xffff
      if displacement & 0x8000: displacement -= 0x10000
      target = offset + 4 + displacement*4
      dest = ctx.targets[PC_LO_IDX].cast(dtypes.uint64) | (ctx.targets[PC_HI_IDX].cast(dtypes.uint64) << 32)
      cond = dest.eq(target).simplify()
      if target <= offset: loops[target] = max(loops.get(target, offset), offset)
    instructions[offset] = (inst.size(), name, body, target, cond)
    offset += inst.size()
  for _, _, _, target, _ in instructions.values():
    assert target is None or target in instructions or target == offset, f"branch target is not an instruction boundary: {target}"
  axis = 7

  def emit(start:int, end:int, active_loop:int|None=None):
    nonlocal axis
    pos = start
    forward:list[int] = []
    while pos < end:
      while forward and forward[-1] == pos:
        forward.pop()
        graph.guards.pop()
      if pos in loops and pos != active_loop:
        latch = loops[pos]
        assert latch < end, "overlapping branch regions"
        loop = UOp.loop(axis)
        axis += 1
        graph.deps += (loop,)
        emit(pos, latch, pos)
        cond = graph.condition(instructions[latch][4])
        for guard in graph.guards: cond = cond & guard.after(*graph.deps).index(0).load()
        effect = UOp.group(*graph.deps).backedge(loop, cond)
        graph.calls.append(effect)
        graph.deps = (effect,)
        for p in graph.operands: graph.values[p], graph.readers[p] = p.after(effect), []
        pos = latch + instructions[latch][0]
        continue
      size, name, body, target, cond = instructions[pos]
      if name == "S_ENDPGM":
        assert not graph.guards and active_loop is None, "conditional termination is not supported"
        break
      if target is not None:
        assert pos < target <= end, f"branch crosses region boundary: {pos:#x} -> {target:#x}"
        assert not forward or target <= forward[-1], "overlapping forward branch regions are not supported"
        flag_name = f"branch_{pos:x}"
        bank = UOp.param(7+len(graph.storage), dtypes.bool, 1, name=flag_name)
        graph.storage[flag_name] = UOp.placeholder((1,), dtypes.bool, addrspace=AddrSpace.REG, tag=flag_name)
        flag = graph.param(bank)
        graph.append(UOp.sink(flag.index(0).store(cond.logical_not())), name=f"branch_{pos:x}")
        graph.guards.append(flag)
        forward.append(target)
      else: graph.append(body, name=f"{name.lower()}_{pos:x}")
      pos += size
    for _ in forward: graph.guards.pop()

  emit(0, offset)
  wave_size, total_threads = _wave_size(arch), lx * ly * lz
  sizes = {"s":SGPR_COUNT, "v":256*wave_size, "lds":max(((rsrc2 >> 15) & 0x1ff)*128, 1),
           "scratch":max(scratch_size*wave_size, 1)}
  if wave_size == 64: sizes["a"] = 256*wave_size
  banks = {name:UOp.placeholder((size,), dtypes.uint8 if name == "scratch" else dtypes.uint32, slot=i, addrspace=AddrSpace.REG,
                                tag={"s":"sgpr", "v":"vgpr", "a":"accvgpr"}.get(name, name)) for i, (name, size) in enumerate(sizes.items())}
  banks.update(graph.storage)
  group = UOp.range(gx*gy*gz, 0, dtype=dtypes.int)
  clear_lds = UOp.range(sizes["lds"], 1)
  lds_init = state_call(UOp.sink(banks["lds"].index(clear_lds).store(0).end(clear_lds)), "init_workgroup", [banks["lds"]], group)
  wave = UOp.range((total_threads+wave_size-1)//wave_size, 2, dtype=dtypes.int)
  clears = []
  for i, name in enumerate(("s", "v", "a") if wave_size == 64 else ("s", "v")):
    idx = UOp.range(sizes[name], 3+i)
    clears.append(banks[name].index(idx).store(0).end(idx))
  sgpr = banks["s"].after(*clears)
  words = UOp.param(6, dtypes.uint32, user_words, name="user_data")
  stores = [sgpr.index(i).store(words.index(i)) for i in range(user_words)]
  n_lanes = (total_threads-wave*wave_size).minimum(wave_size)
  for i in range(wave_size//32):
    bits = (n_lanes-i*32).maximum(0).minimum(32).cast(dtypes.uint64)
    stores.append(sgpr.index(EXEC_LO.offset+i).store(((UOp.const(1, dtypes.uint64) << bits)-1).cast(dtypes.uint32)))
  gidx, gidy, gidz = group%gx, (group//gx)%gy, group//(gx*gy)
  if arch == "rdna4":
    stores += [sgpr.index(ttmp[7].offset).store((gidy & 0xffff) | ((gidz & 0xffff) << 16)), sgpr.index(ttmp[9].offset).store(gidx)]
  else:
    slot = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
    for flag, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                      (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                      (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
      if rsrc2 & flag:
        stores.append(sgpr.index(slot).store(gid))
        slot += 1
  stores += [sgpr.index(SCRATCH_STRIDE_IDX).store(scratch_size), sgpr.index(SGPR_COUNT-12).store((wave & 15) | ((wave & 3) << 4))]
  lane = UOp.range(n_lanes, 6)
  tid = wave*wave_size+lane
  stores.append(banks["v"].after(*clears).index(lane).store(((tid//(lx*ly)) << 20) | (((tid//lx)%ly) << 10) | (tid%lx)).end(lane))
  init = state_call(UOp.sink(*stores), "init_wave", [banks[n] for n in ("s", "v", "a") if n in banks]+[words, group, wave], lds_init, wave)
  body = graph.calls[-1] if graph.calls else init
  body = body.substitute({p:p.after(init) for p in graph.operands}, walk=True)
  sink = UOp.sink(body.end(wave, group), arg=KernelInfo(name="asm_call")).rtag(1)
  return to_program(sink, _CallRenderer(Device["CPU"].renderer.target, banks))
