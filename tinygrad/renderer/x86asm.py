from typing import List, Dict
from tinygrad.ops import UOp, Ops, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
import struct
from collections import Counter
from tinygrad.renderer.cstyle import CStyleLanguage

size_prefix = {1: " byte ptr", 2: " word ptr", 4: " dword ptr", 8: " qword ptr"}
mov_sufix = {4: "d", 8: "q"}

x86_mov_ops = {Ops.STORE: "mov", Ops.LOAD: "mov", Ops.ASSIGN: "mov", Ops.DEFINE_ACC: "mov"}
x86_unsigned_ops = {**x86_mov_ops, Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "imul", Ops.IDIV: "div", Ops.CMPNE: "cmp",
                    Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor"}
x86_signed_ops = {**x86_unsigned_ops, Ops.IDIV: "idiv"}
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

def to_hex(x: int | float) -> str:
  if isinstance(x, int): return hex(x)
  return "0x" + "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

def cflag(x:UOp) -> str:
  if x.op is Ops.CMPLT: return "setl" if x.src[0].dtype in dtypes.sints else "setb"
  if x.op is Ops.CMPNE: return "setne"
  assert False

x86_rewrite = PatternMatcher([
  (UPat(Ops.INDEX, name="x"), lambda ctx,x: f"lea {ctx[x]}, [{ctx[x.src[0]]} + {ctx[x.src[1]]}*{x.src[0].dtype.itemsize}]"),
  (UPat(Ops.LOAD, src=(UPat.var("idx"),), name="x"), lambda ctx,x,idx: f"{x86op[x.dtype][x.op]} {ctx[x]}, [{ctx[idx]}]"),
  (UPat(Ops.STORE, name="x"),
   lambda ctx,x: f"{x86op[x.src[1].dtype][x.op]}{size_prefix[x.src[1].dtype.itemsize] if x.src[1].op is Ops.CONST else ""} [{ctx[x.src[0]]}], {ctx[x.src[1]]}"),
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}"),
  # only assign if location isn't the same
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}" if ctx[x.src[0]] != ctx[x.src[1]] else None),

  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"insertps {ctx[x]}, {ctx[x.src[0]]}, {gep_imm[x.arg[0]]}"),
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join(f"insertps {ctx[x]}, {ctx[s]}, {vec_imm[i]}" for i,s in enumerate(x.src))),
  
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: f"mov {ctx[x]}, {ctx[x.src[0]]}\n.LOOP_{x.arg[0]}:"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x: f"inc {ctx[x.src[0]]}\ncmp {ctx[x.src[0]]}, {x.src[0].src[1].arg}\njl .LOOP_{x.src[0].arg[0]}"),
  
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"mov{mov_sufix[x.dtype.itemsize] if ctx.r[x.src[0]] in ctx.all_regs else ""} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: ctx.x86_cast(x, x.src[0])),
  # no cmov for floats
  (UPat(Ops.WHERE, dtype=dtypes.floats, name="x"),
   lambda ctx,x: f"test {ctx[x.src[0]]}, 1\njnz .L{ctx.uop_i[x]}\n{x86op[x.dtype][Ops.ASSIGN]} {ctx[x]}, {ctx[x.src[2]]}\n.L{ctx.uop_i[x]}:"),
  (UPat(Ops.WHERE, dtype=dtypes.ints, name="x"), lambda ctx,x: f"test {ctx[x.src[0]]}, 1\ncmovz {ctx[x]}, {ctx[x.src[2]]}"),
  
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda ctx,x: f"{x86op[x.src[0].dtype][x.op]} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n{cflag(x)} {ctx[x]}{f"\nsetp r15b\nxor {ctx[x]}, r15b" if dtypes.is_float(x.src[0].dtype) else ""}"),
  # idiv requires rax/rdx
  (UPat(Ops.IDIV, name="x"), lambda ctx,x: ctx.idiv(x, x.src[1])),
  # rest of binary ops
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"{x86op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[1]]}"),
])

class X86Renderer(Renderer):
  device = "X86"
  has_local = False
  global_max = None

  extra_matcher = PatternMatcher([
    # index register must be 64bit
    (UPat(Ops.INDEX, name="x"), lambda x: x.src[0].index(x.src[1].cast(dtypes.int64)) if x.src[1].dtype != dtypes.int64 else None),
    # rewrite RECIP to FDIV
    (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
    # rewrite cast to bool to CMPNE 0
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
    # gate any stores that aren't gated with ifs
    (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat(), UPat(), UPat(dtype=dtypes.bool)), name="store"),
      lambda store: UOp(Ops.STORE, src=store.src[:2]+(UOp(Ops.IF, src=(store.src[2],)),))),
    # rewrite MAX to CMPLT + WHERE
    (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  ])

  def idiv(self, x:UOp, s:UOp) -> str:
    l = ""
    # if dividend is rax/rdx we don't push because pop would overwrite result
    if self.r[x] != "rax": l += "push rax\n"
    if self.r[x] != "rdx": l += "push rdx\n"
    # divisor can't be rax or rdx
    if self.r[s] in ("rax", "rdx"): l += f"mov r15, {self.r[s]}\n"
    divisor = self.r[s] if self.r[s] not in ("rax", "rdx") else "r15"
    l += f"mov {self.regt("rax", x.dtype)}, {self[x]}\n"
    l += "xor rdx, rdx\n" if dtypes.is_unsigned(x.dtype) else "cqo\n" if x.dtype.itemsize == 8 else "cdq\n"
    l += f"{x86op[x.dtype][x.op]} {self.regt(divisor, s.dtype)}\n"
    l += f"mov {self[x]}, {self.regt("rax", x.dtype)}"
    if self.r[x] != "rdx": l += "\npop rdx"
    if self.r[x] != "rax": l += "\npop rax"
    return l
  
  def x86_cast(self, x:UOp, s:UOp) -> str:
    # NOTE: cast from uint64 to floats is complicated, might not want to allow that
    # TODO: cast from uint32 to float requires use of 64bit reg (already zero extended)
    if isinstance(x.dtype.scalar(), PtrDType):
      assert isinstance(s.dtype.scalar(), PtrDType)
      return f"mov {self[x]}, {self[s]}"

    if (dtypes.is_int(s.dtype) or s.dtype is dtypes.bool) and dtypes.is_int(x.dtype):
      if s.dtype.itemsize < x.dtype.itemsize:
        if s.dtype in dtypes.sints: return f"movs{'x' if s.dtype.itemsize < 4 else 'xd'} {self[x]}, {self[s]}"
        elif s.dtype.itemsize < 4: return f"movzx {self[x]}, {self[s]}"
      # cast to smaller int or uint32 to uint64 is just a mov
      return f"mov {self[x]}, {self.regt(self.r[s], x.dtype)}"

    # here float is present
    cfrom = "si" if not dtypes.is_float(s.dtype) else "tsd" if s.dtype.itemsize == 8 else "tss"
    cto = "si" if not dtypes.is_float(x.dtype) else "sd" if x.dtype.itemsize == 8 else "ss"
    if (dtypes.is_int(s.dtype) or s.dtype is dtypes.bool) and s.dtype.itemsize < 4:
      # need to zero/sign extend to 32bit temp reg before cast
      return f"mov{'zx' if dtypes.is_unsigned(s.dtype) else 'sx'} r15d, {self[s]}\ncvt{cfrom}2{cto} {self[x]}, r15d"

    return f"cvt{cfrom}2{cto} {self[x]}, {self[s]}"


  # 64 bit int reg to lower bit reg
  def regt(self, reg:str, dt:DType) -> str:
    if dtypes.is_float(dt) or isinstance(dt, PtrDType): return reg
    if dt.itemsize == 8: return reg
    if dt.itemsize == 4: return reg+'d' if reg[-1].isdigit() else 'e'+reg[1:]
    if dt.itemsize == 2: return reg+'w' if reg[-1].isdigit() else reg[1:]
    if dt.itemsize == 1: return reg+'b' if reg[-1].isdigit() else reg[1:]+'l' if reg[-1] == 'i' else reg[1:-1]+'l'

  def __getitem__(self, key:UOp): return self.regt(self.r[key], key.dtype) if self.r[key] in self.all_regs else self.r[key]  # hacky helper
  def render(self, name:str, ops:List[UOp]) -> str:
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

    child_count = Counter(v for ru in ops for v in ru.src)
    uop_i = {u:i for i,u in enumerate(ops)}
    self.uop_i = uop_i

    def is_imm(u:UOp) -> bool: return u.op is Ops.CONST and not dtypes.is_float(u.dtype) and abs(u.arg) <= dtypes.max(dtypes.int32)
    def is_reg(u:UOp) -> bool: return r[u] in all_regs
      
    def mov_to_reg(u:UOp) -> str:
      reg = assign_reg(i, u.dtype)
      dt = dtypes.int64 if isinstance(u.dtype, PtrDType) else u.dtype
      kernel.append(f"{x86op[dt][Ops.LOAD]} {reg}, {r[u]}")
      r[u] = reg

    def mov_to_stack(u:UOp, reg:str) -> str:
      nonlocal stack_size
      if u not in mem:
        mem[u] = f"[rbp - {stack_size}]"
        stack_size += 8
      r[u] = mem[u]
      dt = dtypes.int64 if reg == "r15" or isinstance(u.dtype, PtrDType) else u.dtype
      kernel.append(f"{x86op[dt][Ops.STORE]} {r[u]}, {reg}")
    
    def free_reg(reg:str): float_regs.append(reg) if reg.startswith("xmm") else gen_regs.append(reg)

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
      mov_to_stack(chosen, reg)
      return reg

    # do a pass over ops to assign ranges, ranges allow us to get rid of dead regs and pick the best reg to spill
    live_range: Dict[UOp, List[int]] = {}
    for i,u in enumerate(ops):
      for s in (u,) + u.src:
        if s.op not in (Ops.RANGE,):
          if s not in live_range: live_range[s] = []
          live_range[s].append(i)

    for i,u in enumerate(ops):
      if u.op is Ops.CONST:
        if not is_imm(u):
          kernel.append(f"mov r15, {to_hex(u.arg)}")
          mov_to_stack(u, "r15")
        else: r[u] = to_hex(u.arg)
        continue

      for s in u.src:
        # these can't take imms
        if u.op in (Ops.WHERE, Ops.IDIV) and not is_reg(s): mov_to_reg(s)
        # uop_i is greater in define_acc
        elif uop_i[s] < i and not is_reg(s) and not is_imm(s): mov_to_reg(s)

      if u.op is Ops.ASSIGN: r[u] = r[u.src[0]]
      elif u.op in GroupOp.ALU and u.op not in (Ops.CMPLT, Ops.CMPNE) and child_count[ss:=u.src[0 if u.op != Ops.WHERE else 1]] == 1 and is_reg(ss): r[u] = r[ss]
      else: r[u] = assign_reg(i, u.dtype)

      if u.op in GroupOp.ALU and u.op not in (Ops.CMPLT, Ops.CMPNE):
        # for cmp nothing to mov as reg depends on flag
        if r[u] != r[ss]:
          kernel.append(f"{x86op[u.dtype][Ops.ASSIGN]} {self[u]}, {self[ss]}")
      
      l = x86_rewrite.rewrite(u, ctx=self)
      if l: kernel.append(l)

      # this is fucking stupid
      for s in u.src:
        if is_imm(s) and is_reg(s):
          free_reg(r[s])
          r[s] = to_hex(s.arg)

      #if range_i and i < range_i and is_reg(u) and u.op is not Ops.DEFINE_ACC:
      if next((True for uu in ops[i+1:] if uu.op is Ops.RANGE), False) and is_reg(u) and u.op is not Ops.DEFINE_ACC:
        free_reg(r[u])
        mov_to_stack(u, r[u])
          
      if u.op is Ops.GEP: assert u.dtype == dtypes.float32 and u.dtype.count == 1 and len(u.arg) == 1
      if u.op is Ops.VECTORIZE: assert u.dtype.scalar() == dtypes.float32
      if u.op is Ops.BITCAST: assert dtypes.is_int(u.dtype) != dtypes.is_int(u.src[0].dtype)

      # free dead regs
      for s in u.src:
        if s in live_range and live_range[s][-1] == i and is_reg(s):
            reg = r.pop(s)
            if reg not in r.values(): free_reg(reg)
    
    return "\n".join([".text", f".global {name}", f"{name}:", "push rbp", "mov rbp, rsp", f"sub rsp, {stack_size}"] + kernel + [f"add rsp, {stack_size}", "pop rbp", "ret", "\n"])

# .intel_syntax noprefix <-- add this if using gas
# TODO: free loop counter for spilling
# NOTE: for now we mov all operands to regs
# TODO: handle func args in stack
# TODO: figure out spilling with loops