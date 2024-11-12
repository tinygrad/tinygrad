from typing import List, Dict, Tuple, Union
from tinygrad.ops import UOp, Ops, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
import struct
from collections import Counter
from tinygrad.renderer.cstyle import CStyleLanguage

size_prefix = {1: "byte", 2: "word", 4: "dword", 8: "qword"}
mov_sufix = {4: "d", 8: "q"}
asm_ops = {Ops.STORE: "mov", Ops.LOAD: "mov", Ops.DEFINE_ACC: "mov", Ops.ASSIGN: "mov", Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "imul", Ops.IDIV: "idiv",
            Ops.FDIV: "div", Ops.SHL: "shl", Ops.SHR: "shr", Ops.CMPNE: "cmp", Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor",
            Ops.RECIP: "rcp", Ops.SQRT: "sqrt", Ops.WHERE: "cmovz"}

gep_imm = {0: "0x00", 1: "0x40", 2:"0x80", 3:"0xC0"}
vec_imm = {0: "0x00", 1: "0x10", 2:"0x20", 3:"0x30"}

def to_hex(x: int | float) -> str:
  if isinstance(x, int): return hex(x)
  return "0x" + "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

def cflag(x:UOp) -> str:
  if x.op is Ops.CMPLT: return "setl" if x.src[0].dtype in dtypes.sints else "setb"
  if x.op is Ops.CMPNE: return "setne"
  assert False

def opc(u:UOp) -> str:
  op = asm_ops[u.op]
  # store and cmp op type is based on srcs
  return optype(op, u.src[-1].dtype) if u.op is Ops.STORE or op == "cmp" else optype(op, u.dtype)

def optype(op:str, dt:DType) -> str:
  if dtypes.is_float(dt) and not isinstance(dt, PtrDType):
    s1 = 'p' if dt.count > 1 else 's'
    s2 = 'd' if dt.scalar().itemsize == 8 else 's'
    if op == "cmp": return "ucomi" + s1 + s2
    if op == "mov":
      s0 = 'l' if dt.count == 2 and dt.scalar() is dtypes.float32 else 'u' if dt.count > 1 else ''
      return op + s0 + s1 + s2 # packed mov is unaligned
    return (op if op[0] != 'i' else op[1:]) + s1 + s2
  if dtypes.is_unsigned(dt) and op == 'idiv': return op[1:]
  return op

x86_rewrite = PatternMatcher([
  (UPat(Ops.INDEX, name="x"), lambda ctx,x: f"lea {ctx[x]}, [{ctx[x.src[0]]} + {ctx[x.src[1]]}*{x.src[0].dtype.itemsize}]"),
  (UPat(Ops.LOAD, src=(UPat.var("idx"),), name="x"), lambda ctx,x,idx: f"{opc(x)} {ctx[x]}, [{ctx[idx]}]"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x: f"{opc(x)} {size_prefix[x.src[0].dtype.itemsize]} ptr [{ctx[x.src[0]]}], {ctx[x.src[1]]}"),
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: f"{opc(x)} {ctx[x]}, {ctx[x.src[0]]}"),
  # only assign if location isn't the same
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{opc(x)} {ctx[x.src[0]]}, {ctx[x.src[1]]}" if ctx[x.src[0]] != ctx[x.src[1]] else None),

  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"insertps {ctx[x]}, {ctx[x.src[0]]}, {vec_imm[x.arg[0]]}"),
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join(f"insertps {ctx[x]}, {ctx[s]}, {vec_imm[i]}" for i,s in enumerate(x.src))),
  
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: f"mov {ctx[x]}, {ctx[x.src[0]]}\n.LOOP_{x.arg[0]}:"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x: f"inc {ctx[x.src[0]]}\ncmp {ctx[x.src[0]]}, {ctx[x.src[0].src[1]]}\njl .LOOP_{x.src[0].arg[0]}"),
  
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"mov{mov_sufix[x.dtype.itemsize] if ctx.r[x.src[0]] in ctx.all_regs else ""} {ctx[x]}, {ctx[x.src[0]]}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: ctx.x86_cast(x, x.src[0])),
  # no cmov for floats
  (UPat(Ops.WHERE, dtype=dtypes.floats, name="x"), lambda ctx,x: f"test {ctx[x.src[0]]}, 1\njnz .l{ctx.uop_i[x]}\n{optype("mov", x.dtype)} {ctx[x]}, {ctx[x.src[2]]}\n.l{ctx.uop_i[x]}:"),
  (UPat(Ops.WHERE, dtype=dtypes.ints, name="x"), lambda ctx,x: f"test {ctx[x.src[0]]}, 1\n{opc(x)} {ctx[x]}, {ctx[x.src[2]]}"),
  
  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda ctx,x: f"{opc(x)} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n{cflag(x)} {ctx[x]}{f"\nsetp r15b\nxor {ctx[x]}, r15b" if dtypes.is_float(x.src[0].dtype) else ""}"),
  # idiv requires rax/rdx
  (UPat(Ops.IDIV, name="x"), lambda ctx,x: ctx.idiv(x, x.src[1])),
  # rest of binary ops
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"{opc(x)} {ctx[x]}, {ctx[x.src[1]]}"),
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
  
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.NEG, Ops.EXP2, Ops.SIN, Ops.LOG2]})}

  def idiv(self, x:UOp, s:UOp) -> str:
    l = ""
    if self.r[x] != "rax": l += "push rax\n"
    l += "push rdx\n"
    assert self.r[s] != "rdx", "divisor can't be rdx i think\n"
    l += f"mov {self.regt("rax", x.dtype)}, {self[x]}\n"
    l += "xor rdx, rdx\n" if dtypes.is_unsigned(x.dtype) else "cqo\n" if x.dtype.itemsize == 8 else "cdq\n"
    l += f"{opc(x)} {self[s]}\n"
    l += f"mov {self[x]}, {self.regt("rax", x.dtype)}\n"
    l += "pop rdx\n"
    if self.r[x] != "rax": l += "pop rax"
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
    self.all_regs = gen_regs + float_regs
    # can be a register, memory location or immediate value
    r: Dict[UOp, str] = {}
    self.r = r

    regs: Dict[UOp, str] = {}
    mem: Dict[UOp, str] = {}
    stack_size: int = 8
    kernel: List[str] = []

    child_count = Counter(v for ru in ops for v in ru.src)
    uop_i = {u:i for i,u in enumerate(ops)}
    self.uop_i = uop_i

    def is_imm(u:UOp) -> bool:
      if u.op is Ops.CONST and not dtypes.is_float(u.dtype) and abs(u.arg) <= dtypes.max(dtypes.int32): return True
      return False

    def mov_to_reg(u:UOp) -> str:
      regs[u] = assign_reg(i, u.dtype)
      kernel.append(f"{optype("mov", u.dtype)} {regs[u]}, {mem[u] if u in mem else to_hex(u.arg)}")

    def mov_to_stack(u:UOp, reg:str) -> str:
      nonlocal stack_size
      if u not in mem:
        mem[u] = f"[rbp - {stack_size}]"
        stack_size += 8
      kernel.append(f"{optype("mov", u.dtype if u.op != Ops.CONST else dtypes.int32)} {mem[u]}, {reg}")
    
    def free_reg(reg:str): float_regs.append(reg) if reg.startswith("xmm") else gen_regs.append(reg)

    def assign_reg(i:int, dt:DType) -> str:
      type_regs = float_regs if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else gen_regs
      if type_regs: return type_regs.pop(0)
      # no available regs, spill one
      t = 'x' if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else 'r'
      candidates = [u for u in regs if u in live_range and live_range[u][-1] > i and regs[u][0] == t]
      nonlocal stack_size
      # we choose who to spill by looking for the reg whose next instruction is the latest
      chosen = max(candidates, key=lambda u: min(v for v in live_range[u] if v >= i))
      mov_to_stack(chosen, regs[chosen])
      return regs.pop(chosen)

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
        continue
      regs[u] = regs[u.src[0]] if u.op in (Ops.ASSIGN,) else assign_reg(i, u.dtype)

      if u.op is Ops.DEFINE_GLOBAL:
        mov_to_stack(u, regs[u])
        free_reg(regs.pop(u))
      if u.op in (Ops.STORE, Ops.LOAD, Ops.INDEX):
        if u.src[0] not in regs: mov_to_reg(u.src[0])
      if u.op is Ops.INDEX:
        # src[1] can't be mem
        if not is_imm(u.src[1]) and u.src[1] not in regs: mov_to_reg(u.src[1])
      if u.op in (Ops.WHERE, Ops.IDIV):
        # TODO: src[0] in WHERE doesn't need to be reg but requires size prefix
        for s in u.src:
          if s not in regs: mov_to_reg(s)
      if u.op in (Ops.CMPLT, Ops.CMPNE):
        if is_imm(u.src[0]): mov_to_reg(u.src[0])
        # only 1 operand can be memory TODO: remove from mem when moved to reg, makes this cleaner by just checking if in mem
        elif u.src[0] not in regs and u.src[1] not in regs and not is_imm(u.src[1]): mov_to_reg(u.src[0])

      for s in (u,) + u.src: r[s] = regs[s] if s in regs else mem[s] if s in mem else to_hex(s.arg)

      if u.op in GroupOp.ALU and u.op not in (Ops.CMPLT, Ops.CMPNE):
        # for cmp nothing to mov as reg depends on flag
        kernel.append(f"{optype("mov", u.dtype)} {self[u]}, {self[u.src[0]] if u.op is not Ops.WHERE else self[u.src[1]]}")
      
      l = x86_rewrite.rewrite(u, ctx=self)
      if l: kernel.append(l)

      if u.op is Ops.INDEX:
        # when this happens INDEX is before range, everything before range that's not define_acc should be moved to the stack
        if is_imm(u.src[1]):
          mov_to_stack(u, regs[u])
          free_reg(regs.pop(u))
          
      if u.op is Ops.GEP: assert u.dtype == dtypes.float32 and u.dtype.count == 1 and len(u.arg) == 1
      if u.op is Ops.VECTORIZE: assert u.dtype.scalar() == dtypes.float32
      if u.op is Ops.BITCAST: assert dtypes.is_int(u.dtype) != dtypes.is_int(u.src[0].dtype)

      # free dead regs
      for s in u.src:
        if s in live_range and live_range[s][-1] == i and s in regs:
            reg = regs.pop(s)
            if reg not in regs.values(): free_reg(reg)
    
    return "\n".join([".text", f".global {name}", f"{name}:", "push rbp", "mov rbp, rsp", f"sub rsp, {stack_size}"] + kernel + [f"add rsp, {stack_size}", "pop rbp", "ret", "\n"])

# .intel_syntax noprefix <-- add this if using gas
# TODO: free loop counter for spilling
# NOTE: for now we mov all operands to regs
# TODO: handle func args in stack
# TODO: avoid unnacessary registers using child_count
# TODO: logsumexp and softmax with NOOPT are incorrect, something about range
# TODO: everything before a range should just mov to the stack
