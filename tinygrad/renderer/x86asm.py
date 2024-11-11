from typing import List, Dict
from tinygrad.ops import UOp, Ops, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
import struct
from collections import Counter
from tinygrad.renderer.cstyle import CStyleLanguage

def to_hex(x: int | float) -> str:
  if isinstance(x, int): return hex(x)
  return "0x" + "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

def x86cast(i_dt:DType, o_dt:DType) -> str:
  pass

x86_pm = PatternMatcher([
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

# 64 bit general registers, rsp/rbp not included, r15 temp register for now
gen_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rax", "rbx"] + ['r'+str(i) for i in range(10,15)]
float_regs = ["xmm" + str(i) for i in range(0,16)]
all_regs = gen_regs + float_regs
size_prefix = {1: "byte", 2: "word", 4: "dword", 8: "qword"}
mov_sufix = {4: "d", 8: "q"}
asm_ops = {Ops.STORE: "mov", Ops.LOAD: "mov", Ops.DEFINE_ACC: "mov", Ops.ASSIGN: "mov", Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "imul", Ops.IDIV: "idiv",
            Ops.FDIV: "div", Ops.SHL: "shl", Ops.SHR: "shr", Ops.CMPNE: "cmp", Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor",
            Ops.RECIP: "rcp", Ops.SQRT: "sqrt", Ops.WHERE: "cmovz"}

gep_imm = {0: "0x00", 1: "0x40", 2:"0x80", 3:"0xC0"}
vec_imm = {0: "0x00", 1: "0x10", 2:"0x20", 3:"0x30"}

def cflag(x:UOp) -> str:
  if x.op is Ops.CMPLT: return "setl" if x.src[0].dtype in dtypes.sints else "setb"
  if x.op is Ops.CMPNE: return "setne"
  assert False

# need this to handle nans, maybe rewrite in pm cmplt/cmpne to handle nans? It would be unreadable though
def float_cmp(reg:str) -> str: return "push r15\n" + "setp r15b\n" + f"xor {reg}, r15b\n" + "pop r15"

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
  # TODO: instead of in all_regs do is_reg or in regs (pm needs access to regs)
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"mov{mov_sufix[x.dtype.itemsize] if ctx[x.src[0]] in all_regs else ""} {ctx[x]}, {ctx[x.src[0]]}"),
  #(UPat(Ops.CAST, name="x"), lambda ctx,x: ),
  # no cmov for floats
  (UPat(Ops.WHERE, dtype=dtypes.floats, name="x"), lambda ctx,x: f"test {ctx[x]}, 1\njnz .l{i}\n{optype("mov", x.dtype)} {ctx[x]}, {ctx[x.src[2]]}\n.l{i}:"),
  (UPat(Ops.WHERE, dtype=dtypes.ints, name="x"), lambda ctx,x: f"test {ctx[x]}, 1\n{opc(x)} {ctx[x]}, {ctx[x.src[2]]}"),

  (UPat((Ops.CMPLT, Ops.CMPNE), name="x"),
   lambda ctx,x: f"{opc(x)} {ctx[x.src[0]]}, {ctx[x.src[1]]}\n{cflag(x)} {ctx[x]}{"\n"+float_cmp(x) if dtypes.is_float(x.src[0].dtype) else ""}"),
  # idiv requires rax/rdx
  (UPat(Ops.IDIV, name="x"), lambda ctx,x: f""),
  # rest of binary ops
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"{opc(x)} {ctx[x]}, {ctx[x.src[1]]}"),
])

class X86Renderer(Renderer):
  device = "X86"
  has_local = False
  global_max = None

  extra_matcher = x86_pm
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.NEG, Ops.EXP2, Ops.SIN, Ops.LOG2]})}

  def render(self, name:str, ops:List[UOp]) -> str:

    regs: Dict[UOp, str] = {}
    mem: Dict[UOp, int] = {}
    # can be a register, memory location or immediate value
    r: Dict[UOp, str] = {}
    stack_size: int = 8
    ins = ""
    kernel: List[str] = []

    child_count = Counter(v for ru in ops for v in ru.src)
    uop_i = {u:i for i,u in enumerate(ops)}
    # these ops require all operands to be regs
    srcs_all_regs = (Ops.WHERE, Ops.IDIV)

    def line(op:str, outr:str=None, inr:str=None, imm:str=None) -> str:
      nonlocal ins
      if outr is None: ins += f"{op}\n"
      elif inr is None: ins += f"{op} {outr}\n"
      elif imm is None: ins += f"{op} {outr}, {inr}\n"
      else: ins += f"{op} {outr}, {inr}, {imm}\n"

    # 64 bit int reg to lower bit reg
    def regsz(reg:str, sz:int) -> str:
      if reg.startswith(("xmm")): return reg
      if sz == 8: return reg
      if sz == 4: return reg+'d' if reg[-1].isdigit() else 'e'+reg[1:]
      if sz == 2: return reg+'w' if reg[-1].isdigit() else reg[1:]
      if sz == 1: return reg+'b' if reg[-1].isdigit() else reg[1:]+'l' if reg[-1] == 'i' else reg[1:-1]+'l'

    # location is either an immediate value, register or stack offset
    def loc(u:UOp, sz=None) -> str:
      if u not in regs:
        if u.op is Ops.CONST and u not in mem: return to_hex(u.arg)
        assert u in mem
        return f"[rbp - {mem[u]}]"
      sz = sz if sz else u.dtype.itemsize if not isinstance(u.dtype, PtrDType) else 8
      return regsz(regs[u], sz)
    
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
      if chosen not in mem:
        mem[chosen] = stack_size
        stack_size += 8
      line(optype("mov", dt), f"[rbp - {mem[chosen]}]", loc(chosen))
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
        # consts to stack if they can't be immediate values
        # pattern match 64bit int consts to 32bit?
        if dtypes.is_float(u.dtype) or abs(u.arg) > (2**31-1):
          mem[u] = stack_size
          stack_size += 8
          line("mov", "r15", to_hex(u.arg))
          line("mov", loc(u), "r15")
          #r[u] = f"[rbp - {mem[u]}]"
        #else:
          #r[u] = to_hex(u.arg)
      if u.op is Ops.DEFINE_GLOBAL:
        #define globals to stack, this frees them for spilling
        reg = assign_reg(i, u.dtype)
        mem[u] = stack_size
        stack_size += 8
        free_reg(reg)
        line("mov", loc(u), reg)
      if u.op in (Ops.CONST, Ops.DEFINE_GLOBAL): continue

      # for now only non const srcs must be in registers, unless op requires all registers
      for s in u.src:
        if (s.op is not Ops.CONST or u.op in srcs_all_regs) and s not in regs:
          if uop_i[s] > i: continue # this happens in define_acc
          if s.op is not Ops.CONST: assert s in mem
          reg = assign_reg(i, s.dtype)
          line(optype("mov", s.dtype), reg, loc(s))
          regs[s] = reg

      regs[u] = regs[u.src[0]] if u.op in (Ops.ASSIGN,) else assign_reg(i, u.dtype)

      #for s in u.src: r[s] = regs[s] if s in regs else f"[rbp - {mem[s]}]" if s in mem else to_hex(s.arg)
      #r[u] = regs[u]
      #l = x86_rewrite.rewrite(u, ctx=r)

          
      if u.op is Ops.GEP: assert u.dtype == dtypes.float32 and u.dtype.count == 1 and len(u.arg) == 1
      if u.op is Ops.VECTORIZE: assert u.dtype.scalar() == dtypes.float32
      if u.op is Ops.BITCAST: assert dtypes.is_int(u.dtype) != dtypes.is_int(u.src[0].dtype)
      
      elif u.op is Ops.CAST:
        if dtypes.is_int(u.dtype) and (dtypes.is_int(u.src[0].dtype) or u.src[0].dtype is dtypes.bool):
          # sign extend if casting to larger int
          if u.dtype.itemsize > u.src[0].dtype.itemsize:
            if dtypes.is_unsigned(u.src[0].dtype):
              line("mov", loc(u), regsz(regs[u.src[0]], u.dtype.itemsize)) if u.src[0].dtype.itemsize == 4 else line("movzx", loc(u), loc(u.src[0]))
            else:
              line("movsxd", loc(u), loc(u.src[0])) if u.src[0].dtype.itemsize == 4 else line("movsx", loc(u), loc(u.src[0]))
          # casting to smaller int is just a mov
          else: line("mov", loc(u), regsz(regs[u.src[0]], u.dtype.itemsize))

        elif not isinstance(u.dtype, PtrDType):
          cfrom = "si" if not dtypes.is_float(u.src[0].dtype) else "tsd" if u.src[0].dtype.itemsize == 8 else "tss"
          cto = "si" if not dtypes.is_float(u.dtype) else "sd" if u.dtype.itemsize == 8 else "ss"
          # zero extend boolean
          if u.src[0].dtype == dtypes.bool and u.src[0] in regs: line("and", loc(u.src[0], 8), "1")
          line(f"cvt{cfrom}2{cto}", loc(u), loc(u.src[0], None if u.src[0].dtype != dtypes.bool else 4))

        else:
          # cast between pointers don't do anything, we just mov
          assert isinstance(u.src[0].dtype.scalar(), PtrDType)
          line("mov", loc(u), loc(u.src[0]))

      # alu ops
      elif u.op in GroupOp.ALU:
        # for cmp nothing to mov as reg depends on flag
        if u.op not in (Ops.CMPLT, Ops.CMPNE):
          # if cmov copy first src, mov happens if condition is false
          line(optype("mov", u.dtype), loc(u), loc(u.src[0] if u.op is not Ops.WHERE else u.src[1]))
          
        elif u.op in GroupOp.Binary:
          # for int div need to clear rax/rdx
          # NOTE: for % result is in rdx
          if u.op is Ops.IDIV:
            assert dtypes.is_int(u.dtype)
            if "rax" in regs.values() and regs[u] != "rax": line("push", "rax")
            if "rdx" in regs.values(): line("push", "rdx")
            assert regs[u.src[1]] != "rdx", "divisor can't be rdx i think"
            line("mov", regsz("rax", u.src[0].dtype.itemsize), loc(u.src[0]))
            line("xor", "rdx", "rdx") if dtypes.is_unsigned(u.dtype) else line("cqo") if u.dtype.itemsize == 8 else line("cdq")
            line(opc(u), loc(u.src[1]))
            line("mov", loc(u), regsz("rax", u.dtype.itemsize))
            if "rax" in regs.values() and regs[u] != "rax": line("pop", "rax")
            if "rdx" in regs.values(): line("pop", "rdx")

      # free dead regs
      for s in u.src:
        if s in live_range and live_range[s][-1] == i and s in regs:
            reg = regs.pop(s)
            if reg not in regs.values(): free_reg(reg)
    
    return "\n".join([".text", f".global {name}", f"{name}:", "push rbp", "mov rbp, rsp", f"sub rsp, {stack_size}"] + [ins.rstrip("\n")] + [f"add rsp, {stack_size}", "pop rbp", "ret", "\n"])

# .intel_syntax noprefix <-- add this if using gas
# TODO: free loop counter for spilling
# NOTE: for now we mov all operands to regs
# TODO: handle func args in stack
# TODO: avoid unnacessary registers using child_count
# TODO: logsumexp and softmax with NOOPT are incorrect, something about range
# TODO: everything before a range should just mov to the stack
