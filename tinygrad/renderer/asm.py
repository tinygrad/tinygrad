from typing import cast
from tinygrad.uop.ops import UOp, Ops, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType, truncate
from functools import cached_property
import struct, copy, sys

def to_hex(x, dt:DType) -> str:
  if dt is dtypes.float64: return hex(struct.unpack('<Q', struct.pack('<d', x))[0])
  if dt is dtypes.float32: return hex(struct.unpack('<I', struct.pack('<f', x))[0])
  if dt is dtypes.float16: return hex(struct.unpack('<H', struct.pack('<e', x))[0])
  return cast(str, int(truncate[dt](x)))

base_rewrite = PatternMatcher([
  # local/global load
  (UPat(Ops.LOAD, name="ld"), lambda ctx,ld: f"{ctx.ops[ld.dtype][ld.op]} {ctx[ld]}, [{ctx[ld.src[0]]}]"),
  # register move
  (UPat(Ops.COPY, src=(UPat.var("x"),), name="cp"), lambda ctx,x,cp: f"{ctx.ops[x.dtype][Ops.ASSIGN]} {ctx[cp]}, {ctx[x]}"),
  # accumulator
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: f"{ctx.ops[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[0]]}"),
  # a bit hacky, if an assign moves to memory it's a store
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: ctx.string_rewrite.rewrite(x.src[0].store(x.src[1]), ctx) if not ctx.is_reg(ctx.r[x.src[0]]) else None),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}" if ctx[x] != ctx[x.src[1]] else None),
  # binary ops
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
  # unary ops
  (UPat(Ops.SQRT, name="x"), lambda ctx,x: f"{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}"),
  # branches
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: f"{ctx.ops[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.const_like(0)]}\n.LOOP_{x.arg}:"),
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"{ctx.ops[x.src[0].dtype][Ops.CMPNE]} {ctx[x.src[0]]}, {ctx[x.src[0].const_like(0)]}\n"
   f"{ctx.ops[x.dtype][x.op]} .L{ctx.uops.index(x)}"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx,x: f".L{ctx.uops.index(x.src[0])}:"),
])

asm_matcher = PatternMatcher([
  # load/store use pointer arithmetic and we get rid of the buf pointer, size in arg for define local
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))),
   lambda buf,idx: buf.replace(dtype=dtypes.uint64, arg=(buf.dtype.size*buf.dtype.itemsize) if buf.dtype.local else buf.arg) +
    idx.cast(dtypes.uint64)*buf.dtype.itemsize),
  # move mask from INDEX to the load/store
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"), UPat.var("gate"))), UPat.var("alt"))),
   lambda buf,idx,gate,alt: UOp(Ops.LOAD, alt.dtype, (buf.index(idx), alt, gate))),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"), UPat())), UPat.var("val"), UPat.var("gate"))),
   lambda buf,idx,val,gate: UOp.store(buf.index(idx), val, gate)),
  # cast to pointer is a noop
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None),
  # can't cast from float16 to ints/float64 directly and vice versa
  (UPat.var("y", dtypes.float16).cast((dtypes.float64,)+dtypes.ints, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  (UPat.var("y", (dtypes.float64,)+dtypes.ints).cast(dtypes.float16, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
  # can't cast from float to int8/16 directly and vice versa
  (UPat.var("y", dtypes.floats).cast((dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16), name="x"),
   lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  (UPat.var("y", (dtypes.bool, dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16)).cast(dtypes.floats, name="x"),
   lambda y,x: y.cast(dtypes.int32).cast(x.dtype)),
  # scalar cast to <= int, or zero extending int32 is a noop
  (UPat.var("y", dtypes.ints+(dtypes.bool,)).cast(dtypes.ints+(dtypes.bool,), name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if (x.dtype.itemsize <= y.dtype.itemsize or y.dtype is dtypes.uint32) and y.dtype.count == 1 else None),
  # vector cast between signed and unsigned is a noop
  (UPat.var("y", dtypes.ints).cast(dtypes.ints, name="x"),
   lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize == y.dtype.itemsize and y.dtype.count > 1 else None),
  # bitcast between signed and unsigned is a noop
  (UPat.var("y", dtypes.ints).bitcast(dtypes.ints).named("x"), lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.itemsize == y.dtype.itemsize else None),
  # bitcast between vectors is a noop
  (UPat.var("y").bitcast().named("x"), lambda y,x: x.replace(op=Ops.NOOP) if x.dtype.count > 1 and y.dtype.count > 1 else None),
  # TODO: fold gep in store with vpextrd and arg is imm
  # a gep in a vectorize is folded and its arg is the imm of the instruction
  #(UPat(Ops.VECTORIZE, name="x"),
  # lambda x: x.replace(src=nsrc) if (nsrc:=tuple(s.replace(op=Ops.NOOP) if s.op is Ops.GEP else s for s in x.src)) != x.src else None),
  # rewrite cast to bool to CMPNE 0
  (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
  # rewrite RECIP to FDIV
  (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
  # rewrite MAX to CMPLT + WHERE
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

# reg alloc based on https://dash.harvard.edu/server/api/core/bitstreams/7312037d-c641-6bd4-e053-0100007fdf3b/content
# TODO: right now a spilled var loaded into a register can't be coalesced at its last use but it should, maybe don't extend live ranges
class AsmRenderer(Renderer):
  regs: dict[DType, list[str]] = {}
  ops: dict[DType, dict[Ops, str]] = {}
  callee_saved: list[str] = []
  string_rewrite = base_rewrite
  @cached_property
  def all_regs(self) -> set[str]: return {reg for regs in self.regs.values() for reg in regs}
  def constraints(self, u:UOp, s:UOp|None=None) -> list[str]:
    # because of spill hoisting need to ensure acc is in memory before assign if it was spilled
    if u.op is Ops.ASSIGN and s is self.srcs(u)[0] and s in self.mem: return [self.mem[s]]
    return []
  def render_imm(self, imm:str): raise NotImplementedError("arch specific")
  def render_mem(self, sz:int): raise NotImplementedError("arch specific")
  def render_reg(self, reg:str, dt:DType, alias:bool=False): raise NotImplementedError("arch specific")
  #def two_address(self, x:UOp, y:UOp) -> str: return cast(str, l)+"\n" if (l:=self.string_rewrite.rewrite(x.assign(y), self)) is not None else ""
  def two_address(self, x:UOp, y:UOp) -> str: return f"{self.ops[x.dtype][Ops.ASSIGN]} {self[x]}, {self[y]}\n" if self[x] != self[y] else ""
  def reg_class(self, x:UOp) -> list[str]: return self.regs[x.dtype]
  def bypass(self, x:UOp) -> UOp: return self.bypass(x.src[0]) if x.op in (Ops.ASSIGN, Ops.NOOP) else x
  def srcs(self, x:UOp) -> tuple[UOp, ...]: return tuple(self.bypass(s) for s in x.src)
  def is_reg(self, r:str) -> bool: return r in self.all_regs
  def __getitem__(self, x:UOp) -> str: # hacky helper
    if x.op is Ops.CONST: return self.render_imm(to_hex(x.arg, x.dtype))
    r, dt = self.r[self.bypass(x)], x.dtype
    return self.render_reg(r, dt) if self.is_reg(r) else r
  def _render(self, uops:list[UOp]):
    self.uops = uops
    regs = copy.deepcopy(self.regs)
    callee_saved: list[str] = []
    kernel: list[str] = []
    live: dict[UOp, str] = {}
    loc: dict[UOp, str] = {}
    self.r = loc
    mem: dict[UOp, str] = {}
    self.mem = mem
    live_at_range: dict[UOp, dict[UOp, str]] = {}
    spill_place: dict[UOp, UOp] = {}
    self.stack_size: int = 0
    inst_map: dict[UOp, list[str]] = {}
    # live ranges, first pass updates start of range
    live_range: dict[UOp, list[int]] = {}
    for i,u in enumerate(uops):
      # void dtypes don't hold values, consts aren't instructions, noop and assign use location of src[0]
      u = self.bypass(u)
      if u.dtype != dtypes.void and u.op != Ops.CONST: live_range[u] = [i, i]
    # second pass updates end of range, a var defined before a range and used inside it is needed for the whole range
    ranges: list[UOp] = []
    for i,u in enumerate(reversed(uops)):
      for s in (s for s in self.srcs(u) if s in live_range):
        end = next((live_range[rng][1] for rng in ranges if live_range[s][0] < live_range[rng][0]), len(uops)-i-1)
        live_range[s][1] = max(live_range[s][1], end)
      if u.op is Ops.ENDRANGE: ranges.append(u.src[0])
      if u.op is Ops.RANGE: ranges.pop()

    def reg_class(x:UOp): return regs[x.dtype]
    def _alloc(x:UOp, cons:list[str]):
      # assign free register, otherwise spill one
      if (free:=next((r for r in reg_class(x) if r in cons), None)) is not None: return reg_class(x).pop(reg_class(x).index(free))
      # we choose the var whose next use is the latest, in case no next use we use the uop(endrange) that kills the var
      # this prioritizes vars defined outside loops
      spilled = max([k for k,v in live.items() if v in cons],
                    key=lambda k: next((j for j,u in enumerate(uops[i:live_range[k][1]]) if k in self.srcs(u)), live_range[k][1]-i))
      if spilled not in mem:
        offset = self.stack_size + (spilled.dtype.itemsize - self.stack_size % spilled.dtype.itemsize) % spilled.dtype.itemsize
        self.stack_size = offset + spilled.dtype.itemsize
        mem[spilled] = self.render_mem(offset)
        inst = spill_place.get(spilled, spilled)
        inst_map[inst].append(self.string_rewrite.rewrite(spilled.store(), ctx=self))
      loc[spilled] = mem[spilled]
      return live.pop(spilled)

    def alloc(x:UOp, cons:list[str]):
      # if x already had a reg we free it
      if x in live: reg_class(x).insert(0, live.pop(x))
      # memory constraint, should only be one
      if not self.is_reg(cons[0]):
        assert len(cons) == 1
        mem[x] = cons[0]
        return mem[x]
      live[x] = ret = _alloc(x, cons)
      if x in loc:
        # if x already in reg and we move to another reg the spill place changes
        if self.is_reg(loc[x]): spill_place[x] = u
        move = UOp(Ops.COPY, x.dtype, (x,)) if self.is_reg(self.r[x]) else x.load()
        loc[move] = ret
        inst_map[u].append(self.string_rewrite.rewrite(move, ctx=self))
      if ret in self.callee_saved and ret not in callee_saved: callee_saved.append(ret)
      return ret

    name = "test"
    for i,u in enumerate(uops):
      inst_map[u] = []
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      # free dead registers
      for v in [v for v in live if live_range[v][1] < i]: reg_class(v).insert(0, live.pop(v))
      # reload necessary vars
      if u.op is Ops.ENDRANGE:
        for k,v in live_at_range.pop(u.src[0], {}).items():
          if loc[k] != v: loc[k] = alloc(k, [v])
      # assign srcs, ignore srcs without live ranges
      src = tuple(s for s in self.srcs(u) if s in live_range)
      for s in src:
        cons = self.constraints(u, s)
        if s in loc and loc[s] not in cons:
          # if var first use in range is a load we don't need to reload
          if live_at_range:
            rng, rng_live = list(live_at_range.items())[-1]
            if s in rng_live and not (self.is_reg(loc[s]) or any(s in self.srcs(x) for x in uops[live_range[rng][0]:i-1])): rng_live.pop(s)
          loc[s] = alloc(s, cons)
      # free srcs before assigning destination to coalesce when valid
      # TODO: need to ignore noop and assign here
      for s in src:
        if u.op is Ops.VECTORIZE: continue
        if u.op is Ops.LOAD and len(u.src) == 3 and s is not u.src[1]: continue
        if isinstance(self, X86Renderer):
          if u.op is Ops.WHERE and s is not u.src[1]: continue
          if u.op is Ops.MULACC and s is not u.src[0]: continue
          if u.op in GroupOp.Binary - {Ops.CMPLT, Ops.CMPNE} and u.dtype in dtypes.ints + (dtypes.bool,) and s is not u.src[0]: continue
        if s in live and live_range[s][1] == i: reg_class(s).insert(0, live.pop(s))
      # assign destination
      if u in live_range: loc[u] = alloc(u, self.constraints(u))
      if u.op not in (Ops.NOOP, Ops.CONST, Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR, Ops.BARRIER):
        # render assembly
        if (l:=self.string_rewrite.rewrite(u, ctx=self)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        inst_map[u].append(cast(str, l))
      # define local allocates stack space
      if u.op is Ops.DEFINE_LOCAL: self.stack_size += u.arg
      # save live vars at loop entry
      if u.op is Ops.RANGE: live_at_range[u] = live.copy()
      # making sure nothing got lost
      assert len(set(live.values()) | set(regs[dtypes.int]) | set(regs[dtypes.float])) == len(self.regs[dtypes.int] + self.regs[dtypes.float])

    for u in uops: kernel.extend(inst_map[u])
    # stack must be aligned to 16 bytes
    return (name, kernel, self.stack_size + (16 - (self.stack_size + len(callee_saved)*8) % 16) % 16, callee_saved)

  def render_kernel(self, name:str, kernel:list[str], stack_size:int, called_saved:list[str]): raise NotImplementedError("arch specific")
  def render(self, uops:list[UOp]): return self.render_kernel(*self._render(uops))

# x18 clobbered on macos/windows, x29 is frame pointer, kept for stack arg access
arm64_gen_regs = ['x' + str(i) for i in range(31) if i != 29 and not (i == 18 and sys.platform in ("darwin", "win32"))]
arm64_float_regs = ['v' + str(i) for i in range(32)]
arm64_regs = {**{x:arm64_gen_regs for x in (dtypes.bool,)+dtypes.ints}, **{x:arm64_float_regs for x in dtypes.floats}}
arm64_reg_map = {**{f"x{i}": {4: f"w{i}"} for i in range(31)}, **{f"v{i}": {16: f"q{i}", 8: f"d{i}", 4: f"s{i}", 2: f"h{i}"} for i in range(32)}}

arm64_mov_ops = {Ops.STORE: "str", Ops.LOAD: "ldr", Ops.ASSIGN: "mov"}
arm64_branch_ops = {Ops.ENDRANGE: "b.lt", Ops.IF: "b.eq"}
arm64_unsigned_ops = {**arm64_mov_ops, Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "mul", Ops.MULACC: "madd", Ops.IDIV: "udiv",
                Ops.CMPNE: "cmp", Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "orr", Ops.XOR: "eor", Ops.SHL: "lsl", Ops.SHR: "lsr", Ops.WHERE: "csel"}
arm64_signed_ops = {**arm64_unsigned_ops, Ops.IDIV: "sdiv", Ops.SHR: "asr"}
# NOTE: int16/int8 alus are casted to int32
arm64_16bit_ops = {Ops.STORE: "strh", Ops.LOAD: "ldrh", Ops.ASSIGN: "mov"}
arm64_8bit_ops = {Ops.STORE: "strb", Ops.LOAD: "ldrb", Ops.ASSIGN: "mov"}
arm64_bool_ops = {**arm64_unsigned_ops, **arm64_8bit_ops}
arm64_float_ops = {Ops.ADD: "fadd", Ops.SUB: "fsub", Ops.MUL: "fmul", Ops.FDIV: "fdiv", Ops.CMPLT: "fcmp", Ops.CMPNE: "fcmp",
                  Ops.SQRT: "fsqrt", Ops.MULACC: "fmadd", Ops.WHERE: "fcsel", Ops.STORE: "str", Ops.LOAD: "ldr", Ops.ASSIGN: "fmov"}
arm64_vec_ops = arm64_mov_ops
arm64_ops = {**{x:arm64_unsigned_ops for x in dtypes.uints}, **{x:arm64_signed_ops for x in dtypes.sints},
             **{x:arm64_float_ops for x in dtypes.floats}, **{x:arm64_16bit_ops for x in (dtypes.int16, dtypes.uint16)},
             **{x:arm64_8bit_ops for x in (dtypes.int8, dtypes.uint8)}, dtypes.bool:arm64_bool_ops, dtypes.void:arm64_branch_ops,
             **{x:arm64_vec_ops for x in (dtypes.float16.vec(2), dtypes.float16.vec(4), dtypes.float32.vec(2), dtypes.float32.vec(4))}}
arm64_vec = {1: "b", 2: "h", 4: "s", 8: "d"}
arm64_cast_suffix = {1: "b", 2: "h", 4: "w"}
def arm64_cflag(x:UOp) -> str: return "ne" if x.op is Ops.CMPNE else "lo" if x.src[0].dtype in dtypes.uints else "lt"

arm64_rewrite = PatternMatcher([
  # const load
  (UPat(Ops.LOAD, src=(UPat.cvar('idx'),), name="x"), lambda ctx,x,idx: f"ldr {ctx[x]}, ={ctx[idx][1:]}"),
  # gated load
  (UPat(Ops.LOAD, src=(UPat.var('idx'), UPat.var('alt'), UPat.var('mask')), name="x"), lambda ctx,x,idx,alt,mask:
   f"{ctx.two_address(x, alt)}tst {ctx[mask]}, #1\n"
   f"b.eq .L{ctx.uops.index(x)}\n{ctx.ops[x.dtype][x.op]} {ctx.render_reg(ctx.r[x], x.dtype, True)}, [{ctx[idx]}]\n.L{ctx.uops.index(x)}:"),
  # local/global load
  (UPat(Ops.LOAD, name="x"), lambda ctx,x: f"{ctx.ops[x.dtype][x.op]} {ctx.render_reg(ctx.r[x], x.dtype, True)}, [{ctx[x.src[0]]}]"),
  # local/global store
  (UPat(Ops.STORE, src=(UPat.var("x"),), name="st"), lambda ctx,x,st:
   f"{ctx.ops[x.dtype][st.op]} {ctx.render_reg(ctx.r[ctx.bypass(x)], x.dtype, True)}, [{ctx.mem[ctx.bypass(x)]}]"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x:
   f"{ctx.ops[x.src[1].dtype][x.op]} {ctx.render_reg(ctx.r[ctx.bypass(x.src[1])], x.src[1].dtype, True)}, [{ctx[x.src[0]]}]"),
  # devectorize/vectorize
  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"mov {ctx[x]}, {ctx.r[x.src[0]]}.{arm64_vec[x.dtype.itemsize]}[{x.arg[0]}]"),
  (UPat(Ops.VECTORIZE, name="x"),
   lambda ctx,x: "\n".join(f"mov {ctx.r[x]}.{arm64_vec[s.dtype.itemsize]}[{i}], {ctx.r[ctx.bypass(s)]}.{arm64_vec[s.dtype.itemsize]}[{0}]"
     for i,s in enumerate(x.src))),
  # casts
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=(dtypes.bool,) + dtypes.uints),), name="x"),
   lambda ctx,x: f"uxt{arm64_cast_suffix.get(x.src[0].dtype.itemsize, '')} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=dtypes.sints),), name="x"),
   lambda ctx,x: f"sxt{arm64_cast_suffix.get(x.src[0].dtype.itemsize, '')} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.floats),), name="x"), lambda ctx,x: f"fcvt {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.sints),), name="x"), lambda ctx,x: f"scvtf {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.uints),), name="x"), lambda ctx,x: f"ucvtf {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.sints, src=(UPat(dtype=dtypes.floats),), name="x"), lambda ctx,x: f"fcvtzs {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.uints, src=(UPat(dtype=dtypes.floats),), name="x"), lambda ctx,x: f"fcvtzu {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"{ctx.ops[dtypes.float32][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[0]]}"),
   # ternary ops
  (UPat(Ops.WHERE, name="x"), lambda ctx,x: f"tst {ctx[x.src[0]]}, #1\n{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}, {ctx[x.src[2]]}, ne"),
  (UPat(Ops.MULACC, name="x"), lambda ctx,x: f"{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]}"),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda ctx,x: f"{ctx.ops[x.src[0].dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}\ncset {ctx[x]}, {arm64_cflag(x)}"),
  # endrange #TODO: should be in base rewrite
  (UPat(Ops.ENDRANGE, src=(UPat.var("rng")), name="x"),
   lambda ctx,x,rng: f"{ctx.ops[rng.dtype][Ops.ADD]} {ctx[rng]}, {ctx[rng]}, {ctx[rng.const_like(1)]}\n"
   f"{ctx.ops[rng.dtype][Ops.CMPLT]} {ctx[rng]}, {ctx[rng.src[0]]}\n{ctx.ops[x.dtype][x.op]} .LOOP_{rng.arg}"),
]) + base_rewrite

def arm64_load_consts(x:UOp) -> UOp|None:
  if x.op is Ops.LOAD and x.src[0].op is Ops.CONST: return None
  nsrc = []
  for s in x.src:
    if s.op is Ops.CONST:
      # NOTE: apparently just loading the consts works cause the assembler generates the correct instructions
      if s.dtype is dtypes.float16: s = s.load(dtype=dtypes.int16).bitcast(dtypes.float16)
      elif s.dtype is dtypes.float32: s = s.load(dtype=dtypes.int32).bitcast(dtypes.float32)
      elif s.dtype is dtypes.float64: s = s.load(dtype=dtypes.int64).bitcast(dtypes.float64)
      elif abs(int(truncate[s.dtype](s.arg))) > (2 ** 12) - 1: s = s.load(dtype=s.dtype)
    nsrc.append(s)
  return x.replace(src=tuple(nsrc)) if tuple(nsrc) != x.src else None

# TODO: technically loading to w reg doesn't zero extend upper 32 bits so not a NOOP, use 64 regs when loading?
arm64_matcher = asm_matcher + PatternMatcher([
  (UPat(GroupOp.All, name="x"), arm64_load_consts),
  # some ops can't take imm in srcs
  (UPat((Ops.CMPNE, Ops.CMPLT, Ops.XOR, Ops.IDIV, Ops.MUL, Ops.MULACC, Ops.WHERE, Ops.STORE), name="x"),
   lambda x: x.replace(src=nsrc) if (nsrc:=tuple(s.load(dtype=s.dtype) if s.op is Ops.CONST else s for s in x.src)) != x.src else None),
  # no modulo in arm64
  (UPat(Ops.MOD, src=(UPat.var("a"), UPat.var("b"))), lambda a,b: a - (a // b) * b),
  # int8/int16 alus perform instruction in int32
  (UPat(GroupOp.ALU, dtype=(dtypes.int8, dtypes.int16), name="x"),
   lambda x: UOp(x.op, dtypes.int32, tuple(s.cast(dtypes.int32) if s.dtype != dtypes.bool else s for s in x.src)).cast(x.dtype)),
  (UPat(GroupOp.ALU, dtype=(dtypes.uint8, dtypes.uint16), name="x"),
   lambda x: UOp(x.op, dtypes.uint32, tuple(s.cast(dtypes.uint32) if s.dtype != dtypes.bool else s for s in x.src)).cast(x.dtype)),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda x: UOp(x.op, x.dtype, tuple(s.cast(dtypes.int) for s in x.src)) if any(s.dtype in (dtypes.int8, dtypes.int16) for s in x.src) else None),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"), lambda x: UOp(x.op, x.dtype, tuple(s.cast(dtypes.uint) for s in x.src)) \
   if any(s.dtype in (dtypes.uint8, dtypes.uint16) for s in x.src) else None),
])

class Arm64Renderer(AsmRenderer):
  device = "ARM64"
  has_local = False
  global_max = None
  extra_matcher = arm64_matcher
  string_rewrite = arm64_rewrite
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.AND, Ops.SHL, Ops.SHR, Ops.MULACC)}
  ops = arm64_ops
  regs = arm64_regs

  def constraints(self, u:UOp, s:UOp|None=None) -> list[str]:
    if (base:=super().constraints(u, s)): return base
    if s is not None: return self.reg_class(s)
    # stack args are offset by 8
    if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
      return [("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7")[i]] if (i:=self.uops.index(u)) < 8 else [f"x29, #{(i-7)*8+8}"]
    return self.reg_class(u)
  def render_imm(self, imm:str) -> str: return f"#{imm}"
  def render_mem(self, sz:int) -> str: return f"sp, #{sz}"
  # arm64 vec load/store use q/d instead of v.suffix
  def render_reg(self, reg:str, dt:DType, alias:bool=False) -> str:
    if dt.count > 1 and not alias: return f"{reg}.{dt.count}{arm64_vec[dt.scalar().itemsize]}"
    if dtypes.is_float(dt): return arm64_reg_map[reg][dt.itemsize]
    return reg if dt.itemsize == 8 else arm64_reg_map[reg][max(dt.itemsize, dtypes.int32.itemsize)]
  def render_kernel(self, name:str, kernel:list[str], stack_size:int, callee_saved:list[str]) -> str:
    return "\n".join([".text", f".global {name}", f"{name}:", "stp x29, x30, [sp, #-16]!", "mov x29, sp", f"sub sp, sp, #{stack_size}"] + \
                      kernel + [f"add sp, sp, #{stack_size}", "ldp x29, x30, [sp], #16", "ret", "\n"])

# *** x86 registers ***
# rbp is frame pointer, kept for stack arg access
x86_gen_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rax", "rbx", "r10", "r11", "r12", "r13", "r14", "r15"]
x86_float_regs = ["ymm" + str(i) for i in range(0,16)]
x86_regs = {**{x:x86_gen_regs for x in (dtypes.bool,)+dtypes.ints}, **{x:x86_float_regs for x in dtypes.floats},
            **{x.vec(sz):x86_float_regs for x in dtypes.all for sz in [2,4,8,16,32]}}
x86_reg_map = {**{"rdi": {4: "edi", 2: "di", 1: "dil"}, "rsi": {4: "esi", 2: "si", 1: "sil"}, "rdx": {4: "edx", 2: "dx", 1: "dl"},
                "rcx": {4: "ecx", 2: "cx", 1: "cl"}, "rax": {4: "eax", 2: "ax", 1: "al"}, "rbx": {4: "ebx", 2: "bx", 1: "bl"},
               **{f"r{i}": {4: f"r{i}d", 2: f"r{i}w", 1: f"r{i}b"} for i in range(8,16)}},
               **{f"ymm{i}": {sz: f"xmm{i}" for sz in [16,8,4,2]} for i in range(0,16)}}
# NOTE: because of inconsistent support avx512 instructions aren't used
# *** x86 scalar ops ***
x86_mov_ops = {Ops.STORE: "mov", Ops.LOAD: "mov", Ops.ASSIGN: "mov"}
x86_branch_ops = {Ops.ENDRANGE: "jl", Ops.IF: "je"}
x86_unsigned_ops = {**x86_mov_ops, Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "imul", Ops.IDIV: "div", Ops.MOD: "div", Ops.CMPNE: "cmp",
                    Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor", Ops.SHL: "shl", Ops.SHR: "shr", Ops.WHERE: "cmove"}
x86_signed_ops = {**x86_unsigned_ops, Ops.IDIV: "idiv", Ops.MOD: "idiv", Ops.SHR: "sar"}
x86_float32_ops = {Ops.ADD: "vaddss", Ops.SUB: "vsubss", Ops.MUL: "vmulss", Ops.FDIV: "vdivss", Ops.CMPLT: "vucomiss", Ops.CMPNE: "vucomiss",
                 Ops.SQRT: "sqrtss", Ops.MULACC: "vfmadd213ss", **{k:v+"ss" for k,v in x86_mov_ops.items()}}
x86_float64_ops = {**{k:v[:-1]+"d" for k,v in x86_float32_ops.items()}}
# NOTE: these are just for local moves
x86_float16_ops = {Ops.STORE: "movss", Ops.LOAD: "movss", Ops.ASSIGN: "movss"}
# *** x86 vector ops ***
x86_vec_mov_sz = {4: {Ops.STORE: "vmovd", Ops.LOAD: "vmovd", Ops.ASSIGN: "movss"},
                  8: {Ops.STORE: "vmovq", Ops.LOAD: "vmovq", Ops.ASSIGN: "vmovq"},
                  16: {Ops.STORE: "vmovdqa", Ops.LOAD: "vmovdqa", Ops.ASSIGN: "vmovdqa"},
                  32: {Ops.STORE: "vmovdqa", Ops.LOAD: "vmovdqa", Ops.ASSIGN: "vmovdqa"}}
x86_int_suf = {1: "b", 2: "w", 4: "d", 8: "q"}
x86_uint_sz = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}
# NOTE: no cmpne and cmplt for int vec cmp
# NOTE: vec cmove is done at the byte level, x86 doesn't support 2 byte mask granularity
x86_vec_int_shared = {Ops.WHERE: "vpblendvb", Ops.AND: "vpand", Ops.OR: "vpor"}
x86_vec_sint_base = {Ops.ADD: "vpadd", Ops.SUB: "vpsub", Ops.MUL: "vpmull", Ops.SHL: "vpsllv", Ops.SHR: "vpsrav", Ops.CMPLT: "vpcmpgt",
                     Ops.VECTORIZE: "vpbroadcast"}
x86_vec_uint_base = {**x86_vec_sint_base, Ops.SHR: "vpsrlv"}
# a vec bool is a mask, always uses the full reg
x86_vec_bool_ops = {x.vec(l):{**{k:v+"b" for k,v in x86_vec_uint_base.items()}, **x86_vec_int_shared, **x86_vec_mov_sz[16]}
                    for x in (dtypes.bool,) for l in [4,8,16,32]}
x86_vec_sint_ops = {dt.vec(l):{**{k:v+x86_int_suf[dt.itemsize] for k,v in x86_vec_sint_base.items()}, **x86_vec_int_shared,
                               **x86_vec_mov_sz[dt.vec(l).itemsize]} for dt in dtypes.sints for l in [2,4,8,16,32] if 4 <= dt.vec(l).itemsize <= 32}
x86_vec_uint_ops = {dt.vec(l):{**{k:v+x86_int_suf[dt.itemsize] for k,v in x86_vec_uint_base.items()}, **x86_vec_int_shared,
                               **x86_vec_mov_sz[dt.vec(l).itemsize]} for dt in dtypes.uints for l in [2,4,8,16,32] if 4 <= dt.vec(l).itemsize <= 32}
x86_vec_float16_ops = {x.vec(l):{**x86_vec_mov_sz[x.vec(l).itemsize]} for x in (dtypes.float16,) for l in [2,4,8,16]}
x86_vec_float32_ops = {x.vec(l):{**{k:v[:-2]+"ps" for k,v in x86_float32_ops.items()}, **x86_vec_mov_sz[x.vec(l).itemsize], Ops.CMPLT: "vcmpltps",
                          Ops.CMPNE: "vcmpneqps", Ops.WHERE: "vblendvps", Ops.VECTORIZE: "vbroadcastss"} for x in (dtypes.float32,) for l in [2,4,8]}
x86_vec_float64_ops = {x.vec(l):{**{k:v[:-2]+"pd" for k,v in x86_float32_ops.items()}, **x86_vec_mov_sz[x.vec(l).itemsize], Ops.CMPLT: "vcmpltpd",
                          Ops.CMPNE: "vcmpneqpd", Ops.WHERE: "vblendvpd", Ops.VECTORIZE: "vbroadcastsd"} for x in (dtypes.float64,) for l in [2,4]}
# final mapping
x86_ops = {**{x:x86_unsigned_ops for x in (dtypes.bool,)+dtypes.uints}, **{x:x86_signed_ops for x in dtypes.sints},
          dtypes.float16:x86_float16_ops, dtypes.float32:x86_float32_ops, dtypes.float64:x86_float64_ops,
          **x86_vec_bool_ops, **x86_vec_sint_ops, **x86_vec_uint_ops, **x86_vec_float16_ops, **x86_vec_float32_ops, **x86_vec_float64_ops,
          dtypes.void:x86_branch_ops}

gep_imm = {0: "0x00", 1: "0x40", 2: "0x80", 3: "0xC0"}
#bcast_imm = {0: "0x00", 1: "0x55", 2: "0xAA", 3: "0xFF"}
vec_imm = {0: "0x00", 1: "0x10", 2: "0x20", 3: "0x30"}
vec_imm = {0: "0x00", 1: "0x10", 2: "0x20", 3: "0x30"}
#size_prefix = {1: "byte ptr", 2: "word ptr", 4: "dword ptr", 8: "qword ptr", 16: "xmmword ptr"}
idiv_signex = {1: "cbw", 2: "cwd", 4: "cdq", 8: "cqo"}
def x86_cflag(x:UOp) -> str: return "setne" if x.op is Ops.CMPNE else "setl" if x.src[0].dtype in dtypes.sints else "setb"

# TODO: switch this to avx
def float_cast(x:DType, s:DType) -> str:
  if s is dtypes.float16: return "vcvtph2ps"
  cfrom = "si" if not dtypes.is_float(s) else "sd" if s.itemsize == 8 else "ss"
  cto = "si" if not dtypes.is_float(x) else "sd" if x.itemsize == 8 else "ss"
  if cto == "si": cfrom = "t" + cfrom
  return f"cvt{cfrom}2{cto}"

x86_rewrite = PatternMatcher([
  # define local points to the start of the next stack slot, NOTE: unaligned, could cause issues
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"mov {ctx[x]}, rsp\nadd {ctx[x]}, {ctx.stack_size}"),
  # const load
  (UPat(Ops.LOAD, src=(UPat.cvar('idx'),), name="x"), lambda ctx,x,idx: f"{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[idx]}"),
  # gated load
  (UPat(Ops.LOAD, src=(UPat.var('idx'), UPat.var('alt'), UPat.var('mask')), name="x"), lambda ctx,x,idx,alt,mask:
   f"{ctx.two_address(x, alt)}test {ctx[mask]}, 1\n"
   f"je .L{ctx.uops.index(x)}\n{ctx.ops[x.dtype][x.op]} {ctx[x]}, [{ctx[idx]}]\n.L{ctx.uops.index(x)}:"),
  # local/global store
  (UPat(Ops.STORE, src=(UPat.var("x"),), name="st"), lambda ctx,x,st: f"{ctx.ops[x.dtype][st.op]} [{ctx.mem[x]}], {ctx[x]}"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x: f"{ctx.ops[x.src[1].dtype][x.op]} [{ctx[x.src[0]]}], {ctx[x.src[1]]}"),
  # devectorize
  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"insertps {ctx[x]}, {ctx[x.src[0]]}, {gep_imm[x.arg[0]]}"),
  # broadcast
  #(UPat(Ops.VECTORIZE, src=UPat(Ops.NOOP, name="g"), name="x"), lambda ctx,g,x: f"vpshufd {ctx[x]}, {ctx[g]}, {bcast_imm[g.arg[0]]}"),
  #(UPat(Ops.VECTORIZE, src=UPat(Ops.LOAD, name="g"), name="x"), lambda ctx,g,x: f"vpshufd {ctx[x]}, {ctx[g]}, {bcast_imm[0]}"),
  # vectorize
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join(f"insertps {ctx[x]}, {ctx[s]}, {vec_imm[i]}" for i,s in enumerate(x.src))),
  # casting
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=(dtypes.bool,) + dtypes.uints),), name="x"), lambda ctx,x: f"movzx {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=dtypes.sints),), name="x"),
   lambda ctx,x: f"movs{'x' if x.src[0].dtype.itemsize < 4 else 'xd'} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.float16, name="x"), lambda ctx,x: f"vcvtps2ph {ctx[x]}, {ctx[x.src[0]]}, 0x4"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"{float_cast(x.dtype, x.src[0].dtype)} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"vmov{'q' if x.dtype.itemsize == 8 else 'd'} {ctx[x]}, {ctx[x.src[0]]}"),
  # ternary ops (no cmov for floats)
  (UPat(Ops.WHERE, dtype=dtypes.floats, name="x"),
   lambda ctx,x: f"{ctx.two_address(x, x.src[1])}test {ctx[x.src[0]]}, 1\n"
   f"jne .L{ctx.uops.index(x)}\n{ctx.ops[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[2]]}\n.L{ctx.uops.index(x)}:"),
  (UPat(Ops.WHERE, name="x"),
   lambda ctx,x: f"{ctx.two_address(x, x.src[1])}test {ctx[x.src[0]]}, 1\n{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[2]]}"),
  (UPat(Ops.MULACC, name="x"),
   lambda ctx,x: f"{ctx.two_address(x, x.src[0])}{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}, {ctx[x.src[2]]}"),
  # binary ops, instructions that allow 3 operands (avx) use the base rewrite
  # float cmp requires nan check
  (UPat((Ops.CMPLT, Ops.CMPNE), src=(UPat(dtype=dtypes.floats), UPat()), name="x"), lambda ctx,x:
   f"{ctx.ops[x.src[0].dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n"
   f"{x86_cflag(x)} {ctx[x]}\nsetp {ctx[x][:-1]+'h'}\nxor {ctx[x]}, {ctx[x][:-1]+'h'}"),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda ctx,x: f"{ctx.ops[x.src[0].dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n{x86_cflag(x)} {ctx[x]}"),
   # TODO: get rid of push/pop somehow, some new constraint maybe
  (UPat(Ops.IDIV, name="x"), lambda ctx,x:
   f"{ctx.two_address(x, x.src[0])}push rdx\n"
   f"{('xor rdx, rdx' if x.dtype.itemsize > 1 else 'xor ah, ah') if x.dtype in dtypes.uints else idiv_signex[x.dtype.itemsize]}\n"
   f"{ctx.ops[x.dtype][x.op]} {ctx[x.src[1]]}\npop rdx"),
  (UPat(Ops.MOD, name="x"), lambda ctx,x:
   f"push rax\nmov rax, {ctx.r[x.src[0]]}\n"
   f"{('xor rdx, rdx' if x.dtype.itemsize > 1 else 'xor ah, ah') if x.dtype in dtypes.uints else idiv_signex[x.dtype.itemsize]}\n"
   f"{ctx.ops[x.dtype][x.op]} {ctx[x.src[1]]}\n{'mov dl, ah\n' if x.dtype.itemsize == 1 else ''}pop rax"),
  (UPat(GroupOp.Binary, dtypes.ints + (dtypes.bool,), name="x"),
   lambda ctx,x: f"{ctx.two_address(x, x.src[0])}{ctx.ops[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}"),
  # endrange
  (UPat(Ops.ENDRANGE, src=(UPat.var("rng")), name="x"), lambda ctx,x,rng: f"{ctx.ops[rng.dtype][Ops.ADD]} {ctx[rng]}, {ctx[rng.const_like(1)]}\n"
   f"{ctx.ops[rng.dtype][Ops.CMPLT]} {ctx[rng]}, {ctx[rng.src[0]]}\n{ctx.ops[x.dtype][x.op]} .LOOP_{rng.arg}"),
]) + base_rewrite

def x86_load_consts(x:UOp) -> UOp|None:
  if x.op is Ops.LOAD and x.src[0].op is Ops.CONST: return None
  nsrc = []
  for s in x.src:
    if s.op is Ops.CONST:
      if s.dtype is dtypes.float16: s = s.load(dtype=dtypes.int16).bitcast(dtypes.float16)
      elif s.dtype is dtypes.float32: s = s.load(dtype=dtypes.int32).bitcast(dtypes.float32)
      elif s.dtype is dtypes.float64: s = s.load(dtype=dtypes.int64).bitcast(dtypes.float64)
      elif x.dtype.count > 1 or abs(s.arg) > dtypes.max(dtypes.int32): s = s.load()
    nsrc.append(s)
  return x.replace(src=tuple(nsrc)) if tuple(nsrc) != x.src else None

x86_matcher = asm_matcher + PatternMatcher([
  # some consts can't be immediates
  (UPat(GroupOp.All, name="x"), x86_load_consts),
  # some ops can't take imm in srcs
  (UPat((Ops.WHERE, Ops.IDIV, Ops.MOD, Ops.STORE), name="x"),
   lambda x: x.replace(src=nsrc) if (nsrc:=tuple(s.load(dtype=s.dtype) if s.op is Ops.CONST else s for s in x.src)) != x.src else None),
  (UPat((Ops.CMPLT, Ops.CMPNE), src=(UPat.cvar("c"), UPat()), name="x"), lambda x,c: x.replace(src=(c.load(dtype=c.dtype), x.src[1]))),
  # we use general registers to load/store the 2 bytes of float16
  (UPat(Ops.LOAD, dtypes.float16, src=(UPat.var('idx'), UPat.var('alt'), UPat.var('mask')), name="x"),
   lambda x,idx,alt,mask: idx.load(alt.bitcast(dtypes.int16), mask, dtype=dtypes.int16).bitcast(x.dtype)),
  (UPat(Ops.LOAD, dtypes.float16, name="x"), lambda x: x.replace(dtype=dtypes.int16).bitcast(x.dtype)),
  (UPat(Ops.STORE, src=(UPat.var("idx"), UPat.var("x", dtypes.float16))), lambda idx,x: idx.store(x.bitcast(dtypes.int16))),
  # float16 alus are done in float32
  (UPat(GroupOp.ALU, dtypes.float16, name="x"),
   lambda x: UOp(x.op, dtypes.float32, tuple(s.cast(dtypes.float32) if s.dtype != dtypes.bool else s for s in x.src)).cast(dtypes.float16)),
  # float16 accumulator are done in float32 as there's no register move for float16
  (UPat(Ops.ASSIGN, dtypes.float16, src=(UPat.var("a"), UPat.var("b")), name="x"), lambda a,b,x:
   x.replace(dtype=dtypes.float32, src=(a.replace(dtype=dtypes.float32,
    src=(a.src[0].cast(dtypes.float32),) + a.src[1:]), b.cast(dtypes.float32))).cast(dtypes.float16)),
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda x: UOp(x.op, x.dtype, tuple(s.cast(dtypes.float32) for s in x.src)) if any(s.dtype is dtypes.float16 for s in x.src) else None),
  # can't bitcast from uint16/int16 to float16 directly and vice versa
  (UPat(Ops.BITCAST, (dtypes.uint16, dtypes.int16), (UPat(dtype=dtypes.float16),), name="c"), lambda c: c.src[0].bitcast(dtypes.uint).cast(c.dtype)),
  (UPat(Ops.BITCAST, dtypes.float16, (UPat(dtype=(dtypes.uint16, dtypes.int16)),), name="c"), lambda c: c.src[0].cast(dtypes.uint).bitcast(c.dtype)),
  # casting uint32 to float requires 64 bit register (float cast op assumes signed integers)
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.uint32),), name="c"), lambda c: c.src[0].cast(dtypes.uint64).cast(c.dtype)),
  # casting uint64 to float requires special handling if msb is 1
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=dtypes.uint64),), name="c"),
   lambda c: ((c.src[0] >> 63) != 0).where((c.src[0] & 0x7FFFFFFFFFFFFFFF).cast(dtypes.int64).cast(c.dtype) * 2, \
                                               c.src[0].cast(dtypes.int64).cast(c.dtype))),
  # 2 operand imul and cmov don't work with 8bit registers
  (UPat(Ops.MUL, dtype=(dtypes.uint8, dtypes.int8), name="x"),
    lambda x: UOp(Ops.MUL, dtype=dtypes.int16, src=(x.src[0].cast(dtypes.int16), x.src[1].cast(dtypes.int16))).cast(x.dtype)),
  (UPat(Ops.WHERE, dtype=(dtypes.bool, dtypes.uint8, dtypes.int8), name="x"),
    lambda x: UOp(Ops.WHERE, dtype=dtypes.int16, src=(x.src[0], x.src[1].cast(dtypes.int16), x.src[2].cast(dtypes.int16))).cast(x.dtype)),
  # mulacc only available for floats
  (UPat.var('a', dtypes.floats)*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c)),
])

# TODO: add x86 support for folding loads into instruction if last use of uop
class X86Renderer(AsmRenderer):
  device = "X86"
  has_local = False
  global_max = None
  extra_matcher = x86_matcher
  string_rewrite = x86_rewrite
  # TODO: fix this
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.AND, Ops.SHL, Ops.SHR)}
  ops = x86_ops
  regs = x86_regs
  callee_saved = ["rbx", "rsi", "rdi", "r12", "r13", "r14", "r15"] if sys.platform == "win32" else []

  def constraints(self, u:UOp, s:UOp|None=None) -> list[str]:
    if (base:=super().constraints(u, s)): return base
    # constraints for srcs
    if u.dtype.count == 1:
      if u.op in (Ops.IDIV, Ops.MOD) and s in self.srcs(u): return [r for r in self.reg_class(s) if r not in ("rdx", "rax")]
      if u.op in (Ops.SHL, Ops.SHR) and s is self.srcs(u)[1] and s.op != Ops.CONST: return ["rcx"]
    if s is not None: return self.reg_class(s)
    # constraints for destination
    # abi constraints, stack args are offset by 8
    if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
      # on windows, caller reserves 32 bytes for arg registers
      if sys.platform == "win32": return [("rcx", "rdx", "r8", "r9")[i]] if (i:=self.uops.index(u)) < 4 else [f"rbp + {(i-3)*8+40}"]
      return [("rdi", "rsi", "rdx", "rcx", "r8", "r9")[i]] if (i:=self.uops.index(u)) < 6 else [f"rbp + {(i-5)*8+8}"]
    if u.dtype.count == 1:
      if u.op is Ops.IDIV: return ["rax"]
      if u.op is Ops.MOD: return ["rdx"]
      # float cmp requires nan check, to avoid reserving temp reg we constrain dest to regs that have a high 8 bit portion
      if u.op in (Ops.CMPLT, Ops.CMPNE) and self.srcs(u)[0].dtype in dtypes.floats: return ["rax", "rbx", "rcx", "rdx"]
    return self.reg_class(u)
  def render_imm(self, imm:str) -> str: return imm
  def render_mem(self, sz:int) -> str: return f"rsp + {sz}"
  def render_reg(self, reg:str, dt:DType, alias:bool=False) -> str:
    if dtypes.is_float(dt) or dt.count > 1: return reg if dt.itemsize == 32 else x86_reg_map[reg][dt.itemsize]
    return reg if dt.itemsize == 8 else x86_reg_map[reg][dt.itemsize]
  def render_kernel(self, name:str, kernel:list[str], stack_size:int, callee_saved:list[str]) -> str:
    return "\n".join([".text", f".global {name}", f"{name}:"] + ["push rbp", "mov rbp, rsp"] + [f"push {r}" for r in reversed(callee_saved)] +
                    [f"sub rsp, {stack_size}"] + kernel + [f"add rsp, {stack_size}"] + [f"pop {r}" for r in callee_saved] + ["pop rbp", "ret", "\n"])
