import ctypes
import itertools
import re
from tinygrad.codegen import pm_add_loads, to_program
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.amd import InstDecodeError, decode_inst
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp, UPat, PatternMatcher, graph_rewrite
from test.mockgpu.amd.emu import _Ctx, _INST_HANDLERS, _op_name, _wave_size, get_pcode, _get_pcode_dict
from test.mockgpu.amd.emu import PC_LO_IDX, PC_HI_IDX, SGPR_COUNT, SCRATCH_STRIDE_IDX, F32_INLINE, EXEC_LO, VCC_LO, SCC, ttmp, hsa

_call_ids = itertools.count()

class _CallCtx(_Ctx):
  def __init__(self, code:bytes, pc:int, wave_size:int):
    super().__init__(len(code), wave_size)
    self.code, self.pc = code, pc
    self.targets:dict[int, UOp] = {}
    self.immediates:dict[UOp, UOp] = {}
    self.lift_immediates = False

  def immediate(self, value:UOp, slot:int, name:str) -> UOp:
    if not self.lift_immediates: return value
    p = UOp.param(slot, value.dtype, (), name=name, addrspace=AddrSpace.ALU)
    self.immediates[p] = value.simplify()
    return p

  def inst_field(self, field) -> UOp:
    value = super().inst_field(field)
    if field.name in ("offset", "ioffset", "offset0", "offset1", "literal", "simm16"):
      return self.immediate(value, -field.lo-1, field.name)
    return value

  def wmask(self, reg:UOp, val:UOp) -> list[UOp]:
    val = val.simplify()
    reductions = [u for u in val.toposort() if u.op is Ops.REDUCE and u.tag == "lane_mask"]
    if not reductions: return super().wmask(reg, val)
    assert len(reductions) == 1, "multiple lane masks in one destination"
    reduction = reductions[0]
    lane, = reduction.src[1:]
    bit = val.substitute({reduction:reduction.src[0]})
    # Capture scalar inputs (including old VCC/EXEC) before clearing the destination.
    reads = tuple(u for u in val.toposort() if u.op is Ops.LOAD and not u.ranges)
    stores = []
    for word in range(self.wave_size//32):
      idx = (reg+word).simplify()
      valid = UOp.const(True) if self.wave_size == 64 else idx.ne(124)
      dst = self.sgpr.after(*reads).index(idx.valid(valid))
      init = dst.store(0)
      dst = self.sgpr.after(init, lane).index(idx.valid(valid))
      stores.append(dst.store(dst.load() | (bit >> (word*32)).cast(dtypes.uint32)))
    return [UOp.group(*stores).end(lane)]

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
        ret = self.immediate(UOp.const(value, dtypes.uint32), -1024-idx, "imm")
        return valid.where(ret, UOp.const(0, dtypes.uint32)) if valid is not None else ret
    return super().rsgpr_dyn(reg, valid)

class _CallGraph:
  def __init__(self, ctx:_CallCtx):
    self.banks = {ctx.sgpr:("s", 1), ctx.vgpr:("v", ctx.wave_size)}
    if ctx.accvgpr is not ctx.vgpr: self.banks[ctx.accvgpr] = ("a", ctx.wave_size)
    self.buffers:dict[tuple[UOp, int], UOp] = {}
    self.registers:dict[UOp, tuple[str, int]] = {}
    self.operands:set[UOp] = set()
    self.values:dict[UOp, UOp] = {}
    self.readers:dict[UOp, list[UOp]] = {}
    self.calls:list[UOp] = []
    self.deps:tuple[UOp, ...] = ()
    self.guards:list[UOp] = []
    self.storage:dict[str, UOp] = {}
    self.constants:dict[UOp, UOp] = {}

  def condition(self, cond:UOp) -> UOp:
    cond = graph_rewrite(cond, pm_register_operands, ctx=self)
    return cond.substitute({p:p.after(*self.deps) for p in self.operands}, walk=True)

  def operand(self, bank:UOp, reg:int=0) -> UOp:
    if (key:=(bank, reg)) not in self.buffers:
      name, width = self.banks.get(bank, (bank.arg.name, bank.arg.size))
      if bank in self.banks:
        label = {EXEC_LO.offset:"exec", EXEC_LO.offset+1:"exec_hi", VCC_LO.offset:"vcc",
                 VCC_LO.offset+1:"vcc_hi", SCC.offset:"scc"}.get(reg, f"s{reg}") if name == "s" else f"{name}{reg}"
        p = UOp.placeholder((width,), bank.dtype, addrspace=AddrSpace.REG, tag=label)
        self.registers[p] = (name, reg)
      else: p = self.storage.get(name, bank)
      self.buffers[key] = p
      self.operands.add(p)
      self.values[p] = p
      self.readers[p] = []
    return self.buffers[key]

  def index(self, idx:UOp) -> UOp|None:
    view = idx.src[0].without_after
    bank = view.src[0].without_after if view.op is Ops.RESHAPE else view
    if bank not in self.banks: return None
    width = self.banks[bank][1]
    if view.op is Ops.RESHAPE and len(idx.src) == 3:
      reg = idx.src[1].get_idx().simplify()
      offset = idx.src[2].get_idx()
      valid = idx.src[1].get_valid() & idx.src[2].get_valid()
    else:
      offset = idx.src[1].get_idx()
      reg = (offset // width).simplify()
      offset = offset - reg*width
      valid = idx.src[1].get_valid()
    if reg.vmin != reg.vmax: raise NotImplementedError(f"dynamic register index: {reg.render()}")
    buf = self.operand(bank, int(reg.vmin))
    if idx.src[0].op is Ops.AFTER: buf = buf.after(*idx.src[0].src[1:])
    return buf.index(offset.simplify().valid(valid))

  def append(self, body:UOp, name:str, immediates:dict[UOp, UOp]|None=None):
    immediates = immediates or {}
    body = graph_rewrite(body, pm_add_loads, name="explicit instruction loads")
    body = graph_rewrite(body, pm_register_operands, ctx=self, name="bind instruction registers")
    if not any(u.op is Ops.STORE for u in body.toposort()): return
    if self.guards:
      gate = UOp.const(True)
      for p in self.guards: gate = gate & p.index(0).load()
      body = graph_rewrite(body, pm_gate_instruction, ctx=gate, walk=True, name="gate instruction memory")
    used = {p for p in body.toposort() if p in self.operands or p.op is Ops.PARAM}
    if not used <= self.operands | immediates.keys(): raise NotImplementedError("unbound register bank")
    writes = {u.src[0].buf_uop for u in body.toposort() if u.op is Ops.STORE}
    reads = {u.src[0].buf_uop for u in body.toposort() if u.op is Ops.LOAD}
    # Input and output roles are separate formals even when the caller binds them to the same register.
    inputs = {p:UOp.param(p.arg.slot, p.dtype, p.shape, name=f"{p.arg.name}_input", addrspace=p.addrspace)
              for p in reads & writes if p in self.registers}
    loads = {}
    for u in body.toposort():
      if u.op is not Ops.LOAD or (p:=u.src[0].buf_uop) not in inputs: continue
      idx = u.src[0]
      buf = inputs[p].after(*idx.src[0].src[1:]) if idx.src[0].op is Ops.AFTER else inputs[p]
      loads[u] = u.replace(src=(idx.replace(src=(buf, *idx.src[1:])), *u.src[1:]))
    body = body.substitute(loads, walk=True)
    aliases = {v:k for k,v in inputs.items()}
    # Formal operands follow their use in the body, independent of hardware register numbering.
    params = [p for p in body.toposort() if p in used or p in aliases]
    formal, args = {}, []
    counts:dict[str, int] = {}
    for i, p in enumerate(params):
      if p in immediates:
        n = counts.get("s_src", 0)
        counts["s_src"] = n+1
        formal[p] = UOp.param(i, p.dtype, (1,), name=f"s_src{n}", addrspace=AddrSpace.GLOBAL).index(UOp.const(0, dtypes.uint32)).load()
        value = immediates[p]
        if value not in self.constants:
          buf = UOp.placeholder((1,), p.dtype, slot=1024+len(self.constants), addrspace=AddrSpace.REG, tag=f"imm_{int(value.vmin)}")
          self.constants[value] = buf.after(buf.index(0).store(value))
        args.append(self.constants[value])
        continue
      actual = aliases.get(p, p)
      role = "dst" if p in writes else "src"
      bank_name, reg = self.registers.get(actual, (actual.arg.name, 0))
      prefix = f"{bank_name}_{role}"
      operand_name = prefix
      if actual not in self.registers: operand_name = actual.arg.name or operand_name
      elif bank_name == "s":
        special = {EXEC_LO.offset:"exec_lo", EXEC_LO.offset+1:"exec_hi", VCC_LO.offset:"vcc_lo",
                   VCC_LO.offset+1:"vcc_hi", SCC.offset:"scc"}
        if reg in special: operand_name = f"{special[reg]}_{role}"
      if operand_name == prefix:
        n = counts.get(prefix, 0)
        counts[prefix] = n+1
        operand_name += str(n)
      formal[p] = UOp.param(i, p.dtype, p.shape, name=operand_name, addrspace=AddrSpace.GLOBAL)
      val = self.values[actual].after(*self.readers[actual]) if actual in writes else self.values[actual]
      args.append(val.after(*self.deps))
    body = body.substitute(formal, walk=True).simplify(tracked=True)
    # Adjacent registers with the same operand role form one STACK argument.
    groups:list[list[int]] = []
    remaining = set(range(len(params)))
    for i, p in enumerate(params):
      if i not in remaining: continue
      group = [i]
      actual = aliases.get(p, p)
      if actual in self.registers:
        bank, reg = self.registers[actual]
        if bank != "s" or reg < 106:
          candidates = {self.registers[a][1]:j for j,q in enumerate(params) if j in remaining and
                        (a:=aliases.get(q, q)) in self.registers and self.registers[a][0] == bank and (q in writes) == (p in writes)}
          while reg+1 in candidates and (bank != "s" or reg+1 < 106):
            reg += 1
            group.append(candidates[reg])
      remaining.difference_update(group)
      groups.append(group)
    replacements, grouped_args = {}, []
    for slot, group in enumerate(groups):
      originals = [next(u for u in formal[params[i]].toposort() if u.op is Ops.PARAM) for i in group]
      first = originals[0]
      shape = (len(group), *first.shape) if len(group) > 1 else first.shape
      param = UOp.param(slot, first.dtype, shape, name=first.arg.name, addrspace=AddrSpace.GLOBAL)
      for j, p in enumerate(originals): replacements[p] = param.index(j) if len(group) > 1 else param
      grouped_args.append(UOp.stack(*(args[i] for i in group)) if len(group) > 1 else args[group[0]])
    call = body.substitute(replacements, walk=True).call(*grouped_args, name=name)
    self.calls.append(call)
    self.deps = (call,)
    for p in used - immediates.keys():
      if p in writes: self.values[p], self.readers[p] = p.after(call), []
      elif p in reads: self.readers[p].append(call)

pm_register_operands = PatternMatcher([
  (UPat(Ops.INDEX, name="idx"), lambda ctx,idx: ctx.index(idx)),
  (UPat(Ops.PARAM, name="p"), lambda ctx,p: ctx.operand(p) if p.shape and p not in ctx.banks and p not in ctx.operands else None),
])

pm_gate_instruction = PatternMatcher([
  (UPat(Ops.INDEX, name="idx"), lambda ctx,idx: idx.replace(src=(idx.src[0], idx.src[1].valid(ctx), *idx.src[2:]))),
])

def state_call(body:UOp, name:str, inputs:list[UOp], *deps:UOp) -> UOp:
  params = {u:UOp.param(i, u.dtype, u.shape, addrspace=AddrSpace.ALU if u.shape == () else AddrSpace.GLOBAL)
            for i, u in enumerate(inputs)}
  return body.substitute(params, walk=True).call(*(u.after(*deps) if u.shape else u for u in inputs), name=name)

def render_call(lib:int, lib_sz:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int,
                rsrc2:int, scratch_size:int, arch:str, user_words:int) -> UOp:
  code = ctypes.string_at(lib, lib_sz)
  offset = 0
  decode_error = None
  graph = _CallGraph(_CallCtx(b"", lib, _wave_size(arch)))
  sizes = {"lds":max(((rsrc2 >> 15) & 0x1ff)*128, 1), "scratch":max(scratch_size*_wave_size(arch), 1)}
  graph.storage.update({name:UOp.placeholder((size,), dtypes.uint8 if name == "scratch" else dtypes.uint32,
                                             addrspace=AddrSpace.REG, tag=name) for name, size in sizes.items()})
  instructions:dict[int, tuple[int, str, UOp, int|None, UOp]] = {}
  immediates:dict[int, dict[UOp, UOp]] = {}
  loops:dict[int, int] = {}
  barriers:list[int] = []
  while offset < lib_sz:
    try: inst = decode_inst(code[offset:], arch)
    except InstDecodeError as e:
      decode_error = e
      break
    if offset + inst.size() > lib_sz: raise RuntimeError(f"truncated instruction at {offset:#x}")
    if _op_name(inst) == "S_CODE_END": break
    name = _op_name(inst)
    if "BARRIER" in name: barriers.append(offset + inst.size())
    branch = name == "S_BRANCH" or name.startswith("S_CBRANCH_")
    if not branch and name != "S_ENDPGM" and hasattr(inst, "op") and inst.op in _get_pcode_dict(inst.op):
      assert not re.search(r'\bPC\b', get_pcode(inst.op)), f"explicit PC access is not supported: {name}"
    assert not any(x in name for x in ('GETPC', 'SETPC', 'SWAPPC', 'RFE', 'CALL_B64')), f"explicit PC access is not supported: {name}"
    handler = next((_INST_HANDLERS[cls] for cls in type(inst).__mro__ if cls in _INST_HANDLERS), None)
    if handler is None: raise RuntimeError(f"unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")
    ctx = _CallCtx(code[offset:offset+inst.size()], offset, _wave_size(arch))
    ctx.lift_immediates = not branch
    body = handler(inst, ctx).simplify(tracked=True)
    immediates[offset] = ctx.immediates
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
  # Rotate loops whose initial entry skips the increment/check block. The check becomes the loop latch.
  order = list(instructions)
  for head, latch in list(loops.items()):
    hi, li = order.index(head), order.index(latch)
    if not hi: continue
    entry = instructions[order[hi-1]]
    if entry[1] != "S_BRANCH" or entry[3] not in order[hi+1:li]: continue
    split = order.index(entry[3])
    check = instructions[order[split-1]]
    if not check[1].startswith("S_CBRANCH_") or check[3] != latch+instructions[latch][0]: continue
    size, _, body, _, _ = instructions[latch]
    instructions[latch] = (size, check[1], body, entry[3], check[4].logical_not())
    order[hi:li+1] = order[split:li] + order[hi:split-1] + [latch]
  positions, end = {}, 0
  for pos in order:
    positions[pos], end = end, end+instructions[pos][0]
  positions[offset] = end
  instructions = {positions[p]:(size, name, body, positions[target] if target is not None else None, cond)
                  for p in order for size,name,body,target,cond in [instructions[p]]}
  immediates = {positions[p]:immediates[p] for p in order}
  offset = end
  barriers = [p+size for p,(size,name,_,_,_) in instructions.items() if "BARRIER" in name]
  loops = {}
  for p, (_, _, _, target, _) in instructions.items():
    if target is not None and target <= p: loops[target] = max(loops.get(target, p), p)
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
        flag = graph.operand(bank)
        graph.append(UOp.sink(flag.index(0).store(cond.logical_not())), name="branch")
        graph.guards.append(flag)
        forward.append(target)
      else: graph.append(body, name=name.lower(), immediates=immediates[pos])
      pos += size
    if pos == offset and decode_error is not None: raise decode_error
    for _ in forward: graph.guards.pop()
    return pos < end and instructions[pos][1] == "S_ENDPGM"

  def reset_dependencies():
    graph.deps = ()
    for p in graph.operands: graph.values[p], graph.readers[p] = p, []

  def phase(start:int, end:int, active_loop:int|None=None) -> tuple[UOp, bool]:
    before = len(graph.calls)
    terminated = emit(start, end, active_loop)
    result = graph.calls[-1] if len(graph.calls) > before else UOp(Ops.NOOP)
    reset_dependencies()
    return UOp(Ops.LINEAR, src=(result,)), terminated

  def phases_for(start:int, end:int, active_loop:int|None=None) -> list[UOp]:
    nonlocal axis
    result = []
    while start < end:
      nested = next((h for h,l in sorted(loops.items()) if start <= h < end and h != active_loop and any(h < b <= l for b in barriers)), None)
      stop = min([b for b in barriers if start < b <= end]+[end, nested if nested is not None else end])
      if stop == start:
        latch = loops[start]
        flag_bank = UOp.param(7+len(graph.storage), dtypes.bool, 1, name=f"loop_{start:x}")
        graph.storage[flag_bank.arg.name] = UOp.placeholder((1,), dtypes.bool, addrspace=AddrSpace.REG, tag=flag_bank.arg.name)
        flag = graph.operand(flag_bank)
        graph.append(UOp.sink(flag.index(0).store(True)), "loop_enter")
        result.append(UOp(Ops.LINEAR, src=(graph.calls[-1],)))
        reset_dependencies()
        graph.guards.append(flag)
        children = phases_for(start, latch, start)
        graph.append(UOp.sink(flag.index(0).store(instructions[latch][4])), "loop_condition")
        last = children.pop().src[0]
        condition = graph.calls[-1].substitute({p:p.after(last) for p in graph.operands}, walk=True)
        children.append(UOp(Ops.LINEAR, src=(condition,)))
        graph.guards.pop()
        reset_dependencies()
        result.append(UOp(Ops.LINEAR, src=tuple(children)).backedge(UOp.loop(axis), flag))
        axis += 1
        start = latch+instructions[latch][0]
      else:
        item, terminated = phase(start, stop, active_loop)
        result.append(item)
        if terminated: break
        start = stop
    return result

  phases = phases_for(0, offset)
  wave_size, total_threads = _wave_size(arch), lx * ly * lz
  group = UOp.range(gx*gy*gz, 0, dtype=dtypes.int, tag="workgroup")
  clear_lds = UOp.range(sizes["lds"], 1)
  lds = graph.storage["lds"]
  lds_init = state_call(UOp.sink(lds.index(clear_lds).store(0).end(clear_lds)), "init_workgroup", [lds], group)
  wave = UOp.range((total_threads+wave_size-1)//wave_size, 2, dtype=dtypes.int, tag="wave")
  words = UOp.param(6, dtypes.uint32, user_words, name="user_data")
  initial = {i:words.index(i).load() for i in range(user_words)}
  n_lanes = (total_threads-wave*wave_size).minimum(wave_size)
  for i in range(wave_size//32):
    bits = (n_lanes-i*32).maximum(0).minimum(32).cast(dtypes.uint64)
    initial[EXEC_LO.offset+i] = ((UOp.const(1, dtypes.uint64) << bits)-1).cast(dtypes.uint32)
  gidx, gidy, gidz = group%gx, (group//gx)%gy, group//(gx*gy)
  if arch == "rdna4":
    initial.update({ttmp[7].offset:(gidy & 0xffff) | ((gidz & 0xffff) << 16), ttmp[9].offset:gidx})
  else:
    slot = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
    for flag, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                      (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                      (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
      if rsrc2 & flag:
        initial[slot] = gid
        slot += 1
  initial.update({SCRATCH_STRIDE_IDX:UOp.const(scratch_size), SGPR_COUNT-12:(wave & 15) | ((wave & 3) << 4)})
  lane = UOp.range(wave_size, 6, tag="lane")
  tid = wave*wave_size+lane
  tid_value = (lane < n_lanes).where(((tid//(lx*ly)) << 20) | (((tid//lx)%ly) << 10) | (tid%lx), 0)
  stores, vector_stores = [], []
  for buf, (name, reg) in graph.registers.items():
    if name == "s": stores.append(buf.index(0).store(initial.get(reg, UOp.const(0)).cast(dtypes.uint32)))
    else: vector_stores.append(buf.index(lane).store((tid_value if name == "v" and reg == 0 else UOp.const(0)).cast(dtypes.uint32)))
  if vector_stores: stores.append(UOp.group(*vector_stores).end(lane))
  init = state_call(UOp.sink(*stores), "init_wave", [*graph.registers, words, group, wave], lds_init, wave)
  # Each phase visits every wave before the next phase starts. Register and scratch storage survives the barrier.
  n_waves = (total_threads+wave_size-1)//wave_size
  persistent = {p:UOp.placeholder((n_waves*p.max_numel(),), p.dtype, addrspace=AddrSpace.REG, tag=p.tag)
                for p in [*graph.registers, *(v for k,v in graph.storage.items() if k != "lds")]} if barriers else {}
  def lower_phases(items:list[UOp], previous:UOp, first:bool=False) -> UOp:
    nonlocal axis
    for item in items:
      if item.op is Ops.BACKEDGE:
        plan, loop, flag = item.src
        effect = lower_phases(list(plan.src), UOp.group(previous, loop))
        check_wave = UOp.range(n_waves, axis, tag="active_wave")
        axis += 1
        cond = persistent[flag].after(effect).index(check_wave).load().cast(dtypes.uint32).reduce(check_wave, arg=Ops.ADD).ne(0)
        previous = effect.backedge(loop, cond)
      else:
        phase_wave = wave if first else UOp.range(n_waves, axis, dtype=dtypes.int, tag="wave_phase")
        axis += 1
        views = {p:buf.shrink(((phase_wave*p.max_numel(), phase_wave*p.max_numel()+p.max_numel()),)).simplify() for p,buf in persistent.items()}
        begin = init.substitute({wave:phase_wave, **views}, walk=True) if first else previous
        body = item.src[0].substitute({p:views.get(p, p).after(begin) for p in graph.operands}, walk=True)
        previous = UOp.group(begin, body).end(phase_wave)
      first = False
    return previous
  body = lower_phases(phases, lds_init, True)
  sink = UOp.sink(body.end(group), arg=KernelInfo(name=f"asm_call_{next(_call_ids)}")).rtag(1)
  return to_program(sink, ClangRenderer(Device["CPU"].renderer.target))
