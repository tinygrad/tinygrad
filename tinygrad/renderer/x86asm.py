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

x86_pm = PatternMatcher([
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

class X86Renderer(Renderer):
  device = "X86"
  has_local = False
  global_max = None

  extra_matcher = x86_pm
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.NEG, Ops.EXP2, Ops.SIN, Ops.LOG2]})}

  def render(self, name:str, ops:List[UOp]) -> str:
    # 64 bit general registers, rsp/rbp not included
    gen_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rax", "rbx"] + ['r'+str(i) for i in range(10,16)]
    float_regs = ["xmm" + str(i) for i in range(0,16)]
    size_prefix = {1: "byte", 2: "word", 4: "dword", 8: "qword"}
    asm_ops = {Ops.STORE: "mov", Ops.LOAD: "mov", Ops.DEFINE_ACC: "mov", Ops.ASSIGN: "mov", Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "imul", Ops.IDIV: "idiv",
               Ops.SHL: "shl", Ops.SHR: "shr", Ops.CMPNE: "cmp", Ops.CMPLT: "cmp", Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor",
               Ops.RECIP: "rcp", Ops.SQRT: "sqrt", Ops.WHERE: "cmovz"}

    regs: Dict[UOp, str] = {}
    mem: Dict[UOp, int] = {}
    stack_size: int = 8
    ins = ""

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
    
    def op2(op:str, dt:DType) -> str:
      if dtypes.is_float(dt) and not isinstance(dt, PtrDType):
        s1 = 'p' if dt.count > 1 else 's'
        s2 = 'd' if dt.itemsize == 8 else 's'
        if op == "cmp": return "ucomi" + s1 + s2
        if op == "mov": return op + ('u' if s1 == 'p' else '') + s1 + s2 # packed mov is unaligned
        return (op if op[0] != 'i' else op[1:]) + s1 + s2
      if dtypes.is_unsigned(u.dtype) and op == 'idiv': return op[1:]
      return op

    def opc(u:UOp) -> str:
      op = asm_ops[u.op]
      # store and cmp op type is based on srcs
      return op2(op, u.src[-1].dtype) if u.op is Ops.STORE or op == "cmp" else op2(op, u.dtype)
    
    def free_reg(reg:str): float_regs.append(reg) if reg.startswith("xmm") else gen_regs.append(reg)

    def assign_reg(i:int, dt:DType) -> str:
      if dtypes.is_float(dt) and not isinstance(dt, PtrDType) and float_regs: return float_regs.pop(0)
      if (not dtypes.is_float(dt) or isinstance(dt, PtrDType)) and gen_regs: return gen_regs.pop(0)
      # no available regs, spill one
      t = 'x' if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else 'r'
      candidates = [u for u in regs if u in live_range and live_range[u][-1] > i and regs[u][0] == t]
      nonlocal stack_size
      # we choose who to spill by looking for the reg whose next instruction is the latest
      chosen = max(candidates, key=lambda u: min(v for v in live_range[u] if v >= i))
      if chosen not in mem:
        mem[chosen] = stack_size
        stack_size += 8
      line(op2("mov", dt), f"[rbp - {mem[chosen]}]", loc(chosen))
      return regs.pop(chosen)

    # need this to handle nans
    def float_cmp(u:UOp):
      temp_reg = assign_reg(i, u.dtype)
      line("setp", regsz(temp_reg, u.dtype.itemsize))
      line("xor", loc(u), regsz(temp_reg, u.dtype.itemsize))
      free_reg(temp_reg)

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
        if u.dtype.itemsize == 8 or dtypes.is_float(u.dtype):
          mem[u] = stack_size
          stack_size += 8
          line("mov", "r15", to_hex(u.arg))
          line("mov", loc(u), "r15")
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
          line(op2("mov", s.dtype), reg, loc(s))
          regs[s] = reg

      regs[u] = regs[u.src[0]] if u.op in (Ops.ASSIGN,) else assign_reg(i, u.dtype)
      
      if u.op is Ops.INDEX:
        line("lea", loc(u), f"[{loc(u.src[0])} + {loc(u.src[1], 8)}*{u.src[0].dtype.itemsize}]")
        if u.src[1].op is Ops.CONST:
          mem[u] = stack_size
          stack_size += 8
          line("mov", f"[rbp - {mem[u]}]", loc(u))
          free_reg(regs.pop(u))
      elif u.op is Ops.LOAD: line(opc(u), loc(u), f"[{loc(u.src[0])}]")
      elif u.op is Ops.STORE: line(opc(u), f"[{loc(u.src[0])}]", loc(u.src[1]))
      elif u.op is Ops.DEFINE_ACC: line(opc(u), loc(u), loc(u.src[0]))
      elif u.op is Ops.ASSIGN:
        if regs[u] != regs[u.src[1]]: line(opc(u), loc(u), loc(u.src[1]))
      elif u.op is Ops.RANGE:
        line("mov", loc(u), loc(u.src[0]))
        line(f".l{i}:")
      elif u.op is Ops.ENDRANGE:
        line("inc", loc(u.src[0]))
        line("cmp", loc(u.src[0]), loc(u.src[0].src[1]))
        line(f"jl .l{uop_i[u.src[0]]}")
      elif u.op is Ops.GEP:
        assert len(u.arg) == 1
        assert dtypes.is_float(u.dtype)
        idx_imm = {0: "0x00", 1: "0x55", 2:"0xAA", 3:"0xFF"}
        line("shufps", loc(u), loc(u.src[0]), idx_imm[u.arg[0]])
      elif u.op is Ops.VECTORIZE:
        assert dtypes.is_float(u.dtype.scalar())
        line(op2("mov", u.dtype.scalar()), loc(u), loc(u.src[0]))
        line("unpcklps", loc(u), loc(u.src[1]))
      elif u.op is Ops.BITCAST:
        # bitcast just movs to register of the type
        assert dtypes.is_int(u.dtype) != dtypes.is_int(u.src[0].dtype), "what do?"
        if u.src[0] in regs: line("movq", loc(u), loc(u.src[0])) if u.dtype.itemsize == 8 else line("movd", loc(u), loc(u.src[0]))
        else: line(op2("mov", u.dtype), loc(u), loc(u.src[0]))
      elif u.op is Ops.CAST:
        if dtypes.is_int(u.dtype) and dtypes.is_int(u.src[0].dtype):
          # sign extend if casting to larger int
          if u.dtype.itemsize > u.src[0].dtype.itemsize:
            line("movsxd", loc(u), loc(u.src[0])) if u.src[0].dtype.itemsize == 4 else line("movsx", loc(u), loc(u.src[0]))
          # casting to smaller int is just a mov
          else: line("mov", loc(u), regsz(regs[u.src[0]], u.dtype.itemsize))

        elif u.dtype is dtypes.bool:
          if dtypes.is_int(u.src[0].dtype):
            line("test", loc(u.src[0]), loc(u.src[0]))
            line("setne", loc(u))
          else: # casting float to boolean is this annoying yes
            temp_reg = assign_reg(i, u.src[0].dtype)
            free_reg(temp_reg)
            line("xorps", temp_reg, temp_reg)
            line(op2("ucomi", u.src[0].dtype), loc(u.src[0]), temp_reg)
            line("setne", loc(u))
            float_cmp(u)
        elif not isinstance(u.dtype, PtrDType):
          # cast between pointers don't do anything
          cfrom = "si" if not dtypes.is_float(u.src[0].dtype) else "tsd" if u.src[0].dtype.itemsize == 8 else "tss"
          cto = "si" if not dtypes.is_float(u.dtype) else "sd" if u.dtype.itemsize == 8 else "ss"
          # zero extend boolean
          if u.src[0].dtype == dtypes.bool and u.src[0] in regs: line("and", loc(u.src[0], 8), "1")
          line(f"cvt{cfrom}2{cto}", loc(u), loc(u.src[0], None if u.src[0].dtype != dtypes.bool else 4))

      # alu ops
      elif u.op in GroupOp.ALU:
        # for cmp nothing to mov as reg depends on flag
        if u.op not in (Ops.CMPLT, Ops.CMPNE):
          # if cmov copy first src, mov happens if condition is false
          line(op2("mov", u.dtype), loc(u), loc(u.src[0] if u.op is not Ops.WHERE else u.src[1]))
          
        if u.op is Ops.WHERE:
          line("test", loc(u.src[0]), "1")
          # cmovs don't work on floats need jump
          if dtypes.is_float(u.dtype):
            line(f"jnz .l{i}")
            line(op2("mov", u.dtype), loc(u), loc(u.src[2]))
            line(f".l{i}:")
          else:
            line(opc(u), loc(u), loc(u.src[2]))
          
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

          elif u.op is Ops.CMPLT:
            line(opc(u), loc(u.src[0]), loc(u.src[1]))
            if dtypes.is_float(u.src[0].dtype):
              line("setb", loc(u))
              float_cmp(u)
            else: line("setl", loc(u))
          
          elif u.op is Ops.CMPNE:
            line(opc(u), loc(u.src[0]), loc(u.src[1]))
            line("setne", loc(u))
            if dtypes.is_float(u.src[0].dtype): float_cmp(u)
              
          else:
            line(opc(u), loc(u), loc(u.src[1]))

        elif u.op in GroupOp.Unary:
          # NOTE: using recip loses precision so we just div
          if u.op is Ops.RECIP:
            assert u.dtype == dtypes.float32
            # load 1 into gen reg, mov to float reg and div
            temp_reg = assign_reg(i, dtypes.int32)
            line("mov", temp_reg, to_hex(1.))
            line("movd", loc(u), temp_reg)
            line("divss", loc(u), loc(u.src[0]))
            free_reg(temp_reg)
          #line(opc(u), loc(u), loc(u))

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