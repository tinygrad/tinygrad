from typing import List, Dict
from tinygrad.ops import UOp, Ops, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
import struct
from collections import Counter

x86_mov_ops = {Ops.STORE: "mov", Ops.LOAD: "mov", Ops.ASSIGN: "mov", Ops.DEFINE_ACC: "mov"}
x86_unsigned_ops = {**x86_mov_ops, Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "imul", Ops.IDIV: "div", Ops.MOD: "div", Ops.CMPNE: "cmp",
                    Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor"}
x86_signed_ops = {**x86_unsigned_ops, Ops.IDIV: "idiv", Ops.MOD: "idiv"}
x86_float_ops = {Ops.ADD: "addss", Ops.SUB: "subss", Ops.MUL: "mulss", Ops.FDIV: "divss", Ops.CMPLT: "ucomiss", Ops.CMPNE: "ucomiss", Ops.SQRT: "sqrtss",
                 **{k:v+"ss" for k,v in x86_mov_ops.items()}}
x86_double_ops = {**{k:v[:-1]+'d' for k,v in x86_float_ops.items()}}
# NOTE: are doubles vectorized? 2 doubles is "ups" not "lps", use a instead of u
x86_vec2_ops = {**{k:v+"lps" for k,v in x86_mov_ops.items()}}
x86_vec4_ops = {**{k:v+"ups" for k,v in x86_mov_ops.items()}}

x86op = {**{x:x86_unsigned_ops for x in (dtypes.bool,)+dtypes.uints}, **{x:x86_signed_ops for x in dtypes.sints},
         **{x:x86_float_ops for x in dtypes.floats}, dtypes.float64:x86_double_ops, dtypes.float32.vec(2):x86_vec2_ops, dtypes.float32.vec(4):x86_vec4_ops}

gep_imm = {0: "0x00", 1: "0x40", 2:"0x80", 3:"0xC0"}
vec_imm = {0: "0x00", 1: "0x10", 2:"0x20", 3:"0x30"}

size_prefix = {1: " byte ptr", 2: " word ptr", 4: " dword ptr", 8: " qword ptr"}
mov_sufix = {4: "d", 8: "q"}

def to_hex(x: int | float, dt:DType) -> str:
  if not dtypes.is_float(dt): return hex(x)
  if dt is dtypes.double: return struct.unpack('<Q', struct.pack('<d', x))[0]
  return struct.unpack('<I', struct.pack('<f', x))[0]

def cflag(x:UOp) -> str:
  if x.op is Ops.CMPLT: return "setl" if x.src[0].dtype in dtypes.sints else "setb"
  if x.op is Ops.CMPNE: return "setne"

def float_cast(x:DType, s:DType):
  cfrom = "si" if not dtypes.is_float(s) else "sd" if s.itemsize == 8 else "ss"
  cto = "si" if not dtypes.is_float(x) else "sd" if x.itemsize == 8 else "ss"
  if cto == "si": cfrom = "t" + cfrom
  return f"cvt{cfrom}2{cto}"

x86_rewrite = PatternMatcher([
  (UPat(Ops.INDEX, name="x"), lambda ctx,x: f"lea {ctx[x]}, [{ctx[x.src[0]]} + {ctx.r[x.src[1]]}*{x.src[0].dtype.itemsize}]"),
  (UPat(Ops.LOAD, src=(UPat.var('idx'), UPat.var('alt'), UPat.var('mask')), name="x"), lambda ctx,x,idx,alt,mask:
   f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[alt]}\ntest {ctx[mask]}, 1\njz .L{ctx.uop_i[x]}\n{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, [{ctx[idx]}]\n.L{ctx.uop_i[x]}:"),
  (UPat(Ops.LOAD, src=(UPat.var("idx"),), name="x"), lambda ctx,x,idx: f"{x86op[x.dtype][x.op]} {ctx[x]}, [{ctx[idx]}]"),
  (UPat(Ops.STORE, name="x"),
   lambda ctx,x: f"{x86op[x.src[1].dtype][x.op]}{size_prefix[x.src[1].dtype.itemsize] if x.src[1].op is Ops.CONST else ''} [{ctx[x.src[0]]}], {ctx[x.src[1]]}"),
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}" if ctx[x.src[0]] != ctx[x.src[1]] else None),

  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"insertps {ctx[x]}, {ctx[x.src[0]]}, {gep_imm[x.arg[0]]}"),
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join(f"insertps {ctx[x]}, {ctx[s]}, {vec_imm[i]}" for i,s in enumerate(x.src))),

  (UPat(Ops.RANGE, name="x"), lambda ctx,x: f"mov {ctx[x]}, {ctx[x.src[0]]}\n.LOOP_{x.arg[0]}:"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x: f"inc {ctx[x.src[0]]}\ncmp {ctx[x.src[0]]}, {x.src[0].src[1].arg}\njl .LOOP_{x.src[0].arg[0]}"),

  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"mov{mov_sufix[x.dtype.itemsize] if ctx.r[x.src[0]] in ctx.all_regs else ""} {ctx[x]}, {ctx[x.src[0]]}"),
  # TODO: remove casts that are just movs?
  # casting to <= int or if src is uint32(already zero extended) we just mov, to bigger uint we zero extend, to bigger sint we sign extend
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=dtypes.ints)), name="x"),
   lambda ctx,x: f"mov {ctx[x]}, {ctx.regt(ctx.r[x.src[0]], x.dtype)}" if x.dtype.itemsize <= x.src[0].dtype.itemsize or x.src[0].dtype is dtypes.uint32 else None),
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=(dtypes.bool, *dtypes.uints))), name="x"), lambda ctx,x: f"movzx {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, dtype=dtypes.ints, src=(UPat(dtype=dtypes.sints)), name="x"), lambda ctx,x: f"movs{'x' if x.src[0].dtype.itemsize < 4 else 'xd'} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"{float_cast(x.dtype, x.src[0].dtype)} {ctx[x]}, {ctx[x.src[0]]}"),
  # no cmov for floats
  (UPat(Ops.WHERE, dtype=dtypes.floats, name="x"),
   lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[1]]}\ntest {ctx[x.src[0]]}, 1\njnz .L{ctx.uop_i[x]}\n{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[2]]}\n.L{ctx.uop_i[x]}:"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[1]]}\ntest {ctx[x.src[0]]}, 1\ncmovz {ctx[x]}, {ctx[x.src[2]]}"),

  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda ctx,x: f"{x86op[x.src[0].dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n{cflag(x)} {ctx[x]}{f'\nsetp r15b\nxor {ctx[x]}, r15b' if dtypes.is_float(x.src[0].dtype) else ''}"),
  # idiv requires rax/rdx
  (UPat((Ops.IDIV, Ops.MOD), name="x"), lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[0]]}\n{ctx.idiv(x, x.src[1])}"),
  # rest of binary ops
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[0]]}\n{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}"),

  (UPat(Ops.SQRT, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}"),

  (UPat(Ops.IF, name="x"), lambda ctx,x: f"test {ctx[x.src[0]]}, 1\njz .L{ctx.uop_i[x]}"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx,x: f".L{ctx.uop_i[x.src[0]]}:"),
])

class X86Renderer(Renderer):
  device = "X86"
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None

  extra_matcher = PatternMatcher([
    # can't cast from float to int8/16 directly and vice versa
    (UPat(Ops.CAST, dtype=(dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16), src=(UPat(dtype=dtypes.floats),), name="c"),
     lambda c: c.src[0].cast(dtypes.int32).cast(c.dtype)),
    (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(dtype=(dtypes.bool, dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16)),), name="c"),
     lambda c: c.src[0].cast(dtypes.int32).cast(c.dtype)),
    # 2 operand imul and cmov don't work with 8bit registers
    (UPat(Ops.MUL, dtype=(dtypes.uint8, dtypes.int8), name="x"),
     lambda x: UOp(Ops.MUL, dtype=dtypes.int16, src=(x.src[0].cast(dtypes.int16), x.src[1].cast(dtypes.int16))).cast(x.dtype)),
    (UPat(Ops.WHERE, dtype=(dtypes.bool, dtypes.uint8, dtypes.int8), name="x"),
     lambda x: UOp(Ops.WHERE, dtype=dtypes.int16, src=(x.src[0], x.src[1].cast(dtypes.int16), x.src[2].cast(dtypes.int16))).cast(x.dtype)),
    #TODO: get rid of ptrdtype
    #(UPat(Ops.DEFINE_GLOBAL, name="x"), lambda x: UOp(Ops.DEFINE_GLOBAL, dtype=dtypes.int64, src=x.src, arg=x.arg) if isinstance(x.dtype, PtrDType) else None),
    # *** also in ptx ***
    #(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))), lambda buf,idx: buf.cast(dtypes.int64) + idx.cast(dtypes.int64)*buf.dtype.itemsize),
    # cast between pointers does nothing
    (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) else None),
    # *** also in llvmir ***
    # rewrite cast to bool to CMPNE 0
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
    # rewrite RECIP to FDIV
    (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
    # *** also in cstyle ***
    # gate any stores that aren't gated with ifs
    (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat(), UPat(), UPat(dtype=dtypes.bool)), name="store"),
      lambda store: UOp(Ops.STORE, src=store.src[:2]+(UOp(Ops.IF, src=(store.src[2],)),))),
    # rewrite MAX to CMPLT + WHERE
    (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  ])

  def idiv(self, x:UOp, s:UOp) -> str:
    remainder_signex = {1:"cbw", 2: "cwd", 4: "cdq", 8: "cqo"}
    l = ""
    # if dividend is rax/rdx we don't push because pop would overwrite result
    if self.r[x] != "rax" and "rax" in self.r.values(): l += "push rax\n"
    if self.r[x] != "rdx" and "rdx" in self.r.values(): l += "push rdx\n"
    # divisor can't be rax or rdx
    if self.r[s] in ("rax", "rdx"): l += f"mov r15, {self.r[s]}\n"
    divisor = "r15" if self.r[s] in ("rax", "rdx") else self.r[s]
    l += f"mov {self.regt("rax", x.dtype)}, {self[x]}\n"
    if dtypes.is_unsigned(x.dtype): l += f"xor rdx, rdx\n" if x.dtype.itemsize > 1 else f"xor ah, ah\n"
    else: l += f"{remainder_signex[x.dtype.itemsize]}\n"
    l += f"{x86op[x.dtype][x.op]} {self.regt(divisor, s.dtype)}\n"
    l += f"mov {self[x]}, {self.regt("rax" if x.op is Ops.IDIV else "rdx", x.dtype)}"
    if self.r[x] != "rdx" and "rdx" in self.r.values(): l += "\npop rdx"
    if self.r[x] != "rax" and "rax" in self.r.values(): l += "\npop rax"
    return l

  # 64 bit int reg to lower bit reg
  def regt(self, reg:str, dt:DType) -> str:
    if dt.itemsize == 8 or dtypes.is_float(dt) or isinstance(dt, PtrDType): return reg
    if dt.itemsize == 4: return reg+'d' if reg[-1].isdigit() else 'e'+reg[1:]
    if dt.itemsize == 2: return reg+'w' if reg[-1].isdigit() else reg[1:]
    if dt.itemsize == 1: return reg+'b' if reg[-1].isdigit() else reg[1:]+'l' if reg[-1] == 'i' else reg[1:-1]+'l'

  def __getitem__(self, key:UOp): return self.regt(self.r[key], key.dtype) if self.r[key] in self.all_regs else self.r[key]  # hacky helper
  def render(self, name:str, uops:List[UOp]) -> str:
    # 64 bit general registers, rsp/rbp not included, r15 temp register for now
    gen_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rax", "rbx"] + ['r'+str(i) for i in range(10,15)]
    float_regs = ["xmm" + str(i) for i in range(0,16)]
    all_regs = gen_regs + float_regs
    self.all_regs = all_regs
    # can be a register, memory location or immediate value
    r: Dict[UOp, str] = {}
    self.r = r

    mem: Dict[UOp, str] = {}
    stack_size: int = 8
    kernel: List[str] = []

    child_count = Counter(v for ru in uops for v in ru.src)
    uop_i = {u:i for i,u in enumerate(uops)}
    self.uop_i = uop_i
    # move everything to stack before loop so that spilling works correctly, the alternative is more complicated
    uops_to_spill: List[UOp] = []

    def is_imm(u:UOp) -> bool: return u.op is Ops.CONST and not dtypes.is_float(u.dtype) and abs(u.arg) <= dtypes.max(dtypes.int32)
    def is_reg(u:UOp) -> bool: return u in r and r[u] in all_regs
    def is_mem(u:UOp) -> bool: return u in r and u in mem and r[u] == mem[u]
    def free_reg(reg:str): float_regs.append(reg) if reg.startswith("xmm") else gen_regs.append(reg)

    def mov_to_reg(u:UOp, temp:bool=False) -> str:
      dt = dtypes.int64 if isinstance(u.dtype, PtrDType) or temp else u.dtype
      reg = "r15" if temp else assign_reg(i, u.dtype)
      kernel.append(f"{x86op[dt][Ops.LOAD]} {reg}, {r[u]}")
      r[u] = reg

    def mov_to_stack(u:UOp) -> str:
      nonlocal stack_size
      if u not in mem:
        mem[u] = f"[rbp - {stack_size}]"
        stack_size += 8
      dt = dtypes.int64 if isinstance(u.dtype, PtrDType) or r[u] == "r15" else u.dtype
      kernel.append(f"{x86op[dt][Ops.STORE]} {mem[u]}, {r[u]}")
      r[u] = mem[u]

    def assign_reg(i:int, dt:DType) -> str:
      type_regs = float_regs if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else gen_regs
      if type_regs: return type_regs.pop(0)
      # no available regs, spill one TODO: remove live_range check once RANGE is added to live range
      t = 'x' if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else 'r'
      candidates = [u for u in r if u in live_range and live_range[u][-1] > i and r[u][0] == t]
      nonlocal stack_size
      # we choose who to spill by looking for the reg whose next instruction is the latest
      chosen = max(candidates, key=lambda u: min(v for v in live_range[u] if v >= i))
      reg = r[chosen]
      mov_to_stack(chosen)
      return reg

    # do a pass over ops to assign ranges, ranges allow us to get rid of dead regs and pick the best reg to spill
    live_range: Dict[UOp, List[int]] = {}
    for i,u in enumerate(uops):
      for s in (u,) + u.src:
        if s.op not in (Ops.RANGE,) and s.dtype != dtypes.void:
          if s not in live_range: live_range[s] = []
          live_range[s].append(i)

    for i,u in enumerate(uops):
      if u.op is Ops.CONST:
        r[u] = to_hex(u.arg, u.dtype)
        if not is_imm(u):
          mov_to_reg(u, True)
          mov_to_stack(u)
        continue

      if u.op is Ops.DEFINE_GLOBAL and u.arg > 5:
        # address is in stack instead of register
        r[u] = mem[u] = f"[rbp + {stack_size}]"
        stack_size += 8
        continue

      for s in u.src:
        # these can't take imm values
        if is_imm(s) and u.op in (Ops.WHERE, Ops.IDIV, Ops.MOD): mov_to_reg(s)
        elif is_mem(s): mov_to_reg(s)
        else: assert is_reg(s) or is_imm(s) or s.op is Ops.RANGE or u.dtype is dtypes.void

      if u.dtype != dtypes.void:
        r[u] = r[u.src[0]] if u.op in (Ops.ASSIGN,) else assign_reg(i, u.dtype)
      # TODO: can only take src reg if in same loop or no loop
      #elif u.op in GroupOp.ALU and u.op not in (Ops.CMPLT, Ops.CMPNE) and child_count[ss:=u.src[0 if u.op != Ops.WHERE else 1]] == 1 and is_reg(ss): r[u] = r[ss]

      # we spill everything before a loop, the altenative is to spill inside the loop and load before the next iteration and also not allow regs from outside to die inside a loop if used in loop
      if any(uu.op is Ops.RANGE for uu in uops[i+1:]) and is_reg(u) and u.op not in (Ops.RANGE, Ops.ENDRANGE):
        uops_to_spill.append(u)
      if u.op is Ops.RANGE:
        for uu in uops_to_spill:
          if is_reg(uu):
            free_reg(r[uu])
            mov_to_stack(uu)
        uops_to_spill.clear()

      l = x86_rewrite.rewrite(u, ctx=self)
      if l: kernel.append(l)
      else: assert u.op in (Ops.ASSIGN, Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR), f"{u.op}, {i}"

      # this should be just for uops in uops_to_spill
      for s in u.src:
        if s in mem and is_reg(s):
          reg = r[s]
          mov_to_stack(s)
          if reg not in r.values(): free_reg(reg)

      # this is fucking stupid
      for s in u.src:
        if is_imm(s) and is_reg(s):
          free_reg(r[s])
          r[s] = to_hex(s.arg, s.dtype)

      if u.op is Ops.ASSIGN:
        free_reg(r[u])
        mov_to_stack(u)

      if u.op is Ops.GEP: assert u.dtype == dtypes.float32 and u.dtype.count == 1 and len(u.arg) == 1
      if u.op is Ops.VECTORIZE: assert u.dtype.scalar() == dtypes.float32
      if u.op is Ops.BITCAST: assert dtypes.is_int(u.dtype) != dtypes.is_int(u.src[0].dtype)

      # free dead regs
      for s in (u,) + u.src:
        if s in live_range and live_range[s][-1] == i and is_reg(s):
          reg = r.pop(s)
          if reg not in r.values(): free_reg(reg)

    return "\n".join([".text", f".global {name}", f"{name}:", "push rbp", "mov rbp, rsp", f"sub rsp, {stack_size}"] + kernel + [f"add rsp, {stack_size}", "pop rbp", "ret", "\n"])

