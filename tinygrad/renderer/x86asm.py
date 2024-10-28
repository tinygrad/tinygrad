from typing import List, Dict, cast
from tinygrad.ops import UOp, UOps, BinaryOps, UnaryOps, TernaryOps, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType
import math

base_rewrite = PatternMatcher([
  (UPat(UOps.CONST, arg=math.inf), lambda r: r.infinity),
  (UPat(UOps.CONST, arg=-math.inf), lambda r: r.neg_infinity),
  (UPat(UOps.CONST, arg=math.nan), lambda r: r.nan),
  (UPat(UOps.CONST, dtype=dtypes.bool, name="x"), lambda r,x: "1" if x.arg else "0"),
  (UPat(UOps.CONST, name="x"), lambda r,x: str(x.arg))
])

x86_pm = PatternMatcher([
  # consts are rendered to larger type and casted
  #(UPat(UOps.CONST, (dtypes.bfloat16, dtypes.half), name="c"), lambda c: UOp.const(dtypes.float, c.arg).cast(c.dtype)),
  #(UPat(UOps.CONST, (dtypes.uint8, dtypes.uint16), name="c"), lambda c: UOp.const(dtypes.uint32, c.arg).cast(c.dtype)),
  #(UPat(UOps.CONST, (dtypes.int8, dtypes.int16), name="c"), lambda c: UOp.const(dtypes.int32, c.arg).cast(c.dtype)),
  # insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
  #(UPat(UOps.BITCAST, name="x"),
  # lambda x: UOp(UOps.BITCAST, x.dtype, (UOp(UOps.NOOP, x.src[0].dtype, x.src),)) if x.src[0].op is not UOps.NOOP else None),
  # gate any stores that aren't gated with ifs
  (UPat(UOps.STORE, dtype=dtypes.void, src=(UPat(), UPat(), UPat(), UPat(dtype=dtypes.bool)), name="store"),
    lambda store: UOp(UOps.STORE, src=store.src[:3]+(UOp(UOps.IF, src=(store.src[3],)),))),
  # rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  (UPat(UOps.ALU, name="m", arg=BinaryOps.MAX), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

class X86Renderer(Renderer):
  device = "X86"
  has_local = False
  global_max = None
  infinity = "0x7F800000"
  neg_infinity = "0xFF800000"
  nan = "0x7FC00000"

  extra_matcher = x86_pm
  string_rewrite = base_rewrite
  code_for_op: Dict = {UnaryOps.NEG: None}

  def render(self, name:str, uops:List[UOp]) -> str:
    # 64 bit general registers, rsp/rbp not included
    regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rax", "rbx"] + ['r'+str(i) for i in range(10,16)]
    float_regs = ["xmm" + str(i) for i in range(0,16)]

    size_to_suffix = {1: "byte", 2: "word", 4: "dword", 8: "qword"}
    opcodes = {UOps.STORE: "mov", UOps.LOAD: "mov", UOps.DEFINE_ACC: "mov", UOps.ASSIGN: "mov", BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "imul", BinaryOps.IDIV: "idiv",
               BinaryOps.SHL: "shl", BinaryOps.SHR: "shr", BinaryOps.CMPNE: "cmp", BinaryOps.CMPLT: "cmp", BinaryOps.AND: "and", BinaryOps.OR: "or", BinaryOps.XOR: "xor",
               UnaryOps.RECIP: "rcp", UnaryOps.NEG: "neg", UnaryOps.SQRT: "sqrt", TernaryOps.WHERE: "cmov"}
    uop_reg: Dict[UOp, str] = {}
    uop_mem: Dict[UOp, int] = {}
    ins = ""

    # 64 bit int reg to lower bit reg
    def reg(reg:str, bytes:int) -> str:
      # immediate values are treated as registers
      if reg.isdigit() or reg.startswith(("-", "xmm")): return reg
      if bytes == 8: return reg
      if bytes == 4: return reg+'d' if reg[-1].isdigit() else 'e'+reg[1:]
      if bytes == 2: return reg+'w' if reg[-1].isdigit() else reg[1:]
      if bytes == 1: return reg+'b' if reg[-1].isdigit() else reg[1:]+'l' if reg[-1] == 'i' else reg[1:-1]+'l'
    
    def new_reg(dt:DType): return float_regs.pop(0) if dtypes.is_float(dt) else regs.pop(0)
    
    def opcode(op, dt:DType) -> str:
      oc = opcodes[op]
      if dtypes.is_float(dt): return (oc if oc[0] != 'i' else oc[1:]) + "ss"
      return oc
    
    def opcode2(oc:str, dt:DType) -> str:
      if dtypes.is_float(dt): return (oc if oc[0] != 'i' else oc[1:]) + "ss"
      return oc
    
    # I think we want to allocate regs before the asm rendering which should prob be patterns
    
    # use stack when no more registers available
    # 1 int reg and 1 float reg reserved to load and store to mem, load before use and stoer atthe end of loop

    # do a pass over uops to assign regs
    live_range:Dict[UOp, List[int]] = {}
    for i,u in enumerate(uops):
      # live ranges for regs so we can add non used regs back to available regs
      if u.op not in (UOps.CONST, UOps.DEFINE_GLOBAL):
        live_range[u] = [i, i] if u not in live_range else [live_range[u][0], i]
      for s in u.src:
        if s in live_range:
          live_range[s] = [live_range[s][0], i]
    
    #for k,v in live_range.items():
    #  print("UOP")
    #  print(k)
    #  print("RANGE")
    #  print(v)

    def set_reg(u:UOp):
      if u.op in (UOps.DEFINE_GLOBAL, UOps.LOAD, UOps.DEFINE_ACC, UOps.RANGE, UOps.BITCAST, UOps.CAST):
      
      if u.op is UOps.ALU:

      # if mem mov to r14/5 from mem before 

    
    for i,u in enumerate(uops):
      # if no more regs available spill r14/r15
      if u.op in (UOps.DEFINE_GLOBAL, UOps.LOAD, UOps.DEFINE_ACC, UOps.RANGE, UOps.BITCAST, UOps.CAST, )


      if u.op is UOps.DEFINE_GLOBAL: uop_reg[u] = regs.pop(0)
      
      if u.op is UOps.CONST: uop_reg[u] = self.string_rewrite.rewrite(u, ctx=self)
      
      if u.op is UOps.LOAD:
        uop_reg[u] = new_reg(u.dtype)
        ins += f"{opcode(u.op, u.dtype)} {reg(uop_reg[u], u.dtype.itemsize)}, [{uop_reg[u.src[0]]} + {uop_reg[u.src[1]]}*{u.src[0].dtype.itemsize}]" + "\n"
      
      if u.op is UOps.DEFINE_ACC:
        uop_reg[u] = new_reg(u.dtype)
        ins += f"{opcode(u.op, u.dtype)} {reg(uop_reg[u], u.dtype.itemsize)}, {uop_reg[u.src[0]]}" + "\n"
      
      if u.op is UOps.ASSIGN:
        uop_reg[u] = uop_reg[u.src[0]]
        if uop_reg[u] != uop_reg[u.src[1]]:
          ins += f"{opcode(u.op, u.dtype)} {reg(uop_reg[u], u.dtype.itemsize)}, {reg(uop_reg[u.src[1]], u.src[1].dtype.itemsize)}" + "\n"
      
      if u.op is UOps.STORE:
        if u.src[2].op is UOps.CONST:
          # if storing const val specify operand size
          ins += f"{opcode(u.op, u.src[0].dtype)} {size_to_suffix[u.src[0].dtype.itemsize]} ptr [{uop_reg[u.src[0]]} + {uop_reg[u.src[1]]}*{u.src[0].dtype.itemsize}], {uop_reg[u.src[2]]}" + "\n"
        else:  
          ins += f"{opcode(u.op, u.src[0].dtype)} [{uop_reg[u.src[0]]} + {uop_reg[u.src[1]]}*{u.src[0].dtype.itemsize}], {reg(uop_reg[u.src[2]], u.src[2].dtype.itemsize)}" + "\n"
      
      if u.op is UOps.RANGE:
        uop_reg[u] = new_reg(u.dtype)
        ins += f"xor {reg(uop_reg[u], u.dtype.itemsize)}, {reg(uop_reg[u], u.dtype.itemsize)}" + "\n" + ".loop:" + "\n"
      
      if u.op is UOps.BITCAST:
        # bitcast just movs to register of the type
        uop_reg[u] = new_reg(u.dtype)
        ins += f"mov {reg(uop_reg[u], u.dtype.itemsize)}, {reg(uop_reg[u.src[0]], u.dtype.itemsize)}" + "\n"

      if u.op is UOps.CAST:
        # for casts to > between signed ints it's movsx for 8/16 to 32/64 and movsxd for 32 to 64
        # other int casts don't need anything
        uop_reg[u] = new_reg(u.dtype)
        if u.dtype is dtypes.bool:
          ins += f"test {uop_reg[u]}, {uop_reg[u]}" + "\n" + f"setne {uop_reg[u]}" + "\n"
        else:
          cfrom = {dtypes.int32: "si", dtypes.uint32: "si", dtypes.ulong: "si", dtypes.float32: "tss", dtypes.float64: "tsd"}
          cto = {dtypes.int32: "si", dtypes.uint32: "si", dtypes.ulong: "si", dtypes.float32: "ss", dtypes.float64: "sd"}
          ins += f"cvt{cfrom[u.src[0].dtype]}2{cto[u.dtype]} {reg(uop_reg[u], u.dtype.itemsize)}, {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}" + "\n"

      if u.op is UOps.ALU:
        if u.src[0].op in (UOps.CONST, UOps.RANGE):
          # if out reg is a const or range we need a new register, technically we don't need a new reg if op is cummutative and other operand is alu
          uop_reg[u] = new_reg(u.dtype)
          ins += f"{opcode2("mov", u.dtype)} {reg(uop_reg[u], u.src[0].dtype.itemsize)}, {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}" + "\n"
        else:
          uop_reg[u] = uop_reg[u.src[0]] if not isinstance(u.arg, TernaryOps) else uop_reg[u.src[2]]

        if u.arg is TernaryOps.WHERE:
          if u.src[0].arg is BinaryOps.CMPLT:
            suffix = "l"
          elif u.src[0].arg is BinaryOps.CMPNE:
            suffix = "ne"
          elif u.src[0].op is UOps.LOAD and u.src[0].dtype is dtypes.bool:
            # if first operand is not CMP need to set flag
            suffix = "nz"
            ins += f"test {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}, {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}" + "\n"
          ins += f"{opcode(u.arg, u.dtype)}{suffix} {reg(uop_reg[u], u.dtype.itemsize)}, {reg(uop_reg[u.src[1]], u.src[1].dtype.itemsize)}" + "\n"
          continue

        if isinstance(u.arg, BinaryOps):
          # for int div need to clear rax/rdx, very cool very nice
          # NOTE: for % result is in rdx
          if u.arg is BinaryOps.IDIV and dtypes.is_int(u.dtype):
            if "rax" in uop_reg.values() and uop_reg[u] != "rax":
              ins += "push rax" + "\n"
              ins += f"mov {reg("rax", u.dtype.itemsize)}, {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}" + "\n"
            if "rdx" in uop_reg.values(): ins += "push rdx" + "\n"
            ins += "cdq" + "\n" # xor rdx, rdx for unsigned
            ins += f"idiv {reg(uop_reg[u.src[1]], u.src[1].dtype.itemsize)}" + "\n"
            if "rax" in uop_reg.values() and uop_reg[u] != "rax":
              ins += f"mov {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}, {reg("rax", u.dtype.itemsize)}" + "\n"
              ins += "pop rax" + "\n"
            if "rdx" in uop_reg.values(): ins += "pop rdx" + "\n"
            continue

          #                                                  only src[0] casue of cmp
          ins += f"{opcode(u.arg, u.dtype)} {reg(uop_reg[u], u.src[0].dtype.itemsize)}, {reg(uop_reg[u.src[1]], u.src[1].dtype.itemsize)}" + "\n"
        
          # only set reg based on flag val if the next uop is not a conditional move
          if uops[i+1].arg is not TernaryOps.WHERE:
            if u.arg is BinaryOps.CMPNE: ins += f"sete {reg(uop_reg[u], u.dtype.itemsize)}" + "\n" + f"xor {reg(uop_reg[u], u.dtype.itemsize)}, 1" + "\n"
            if u.arg is BinaryOps.CMPLT: ins += f"setl {reg(uop_reg[u], u.dtype.itemsize)}" + "\n"

        if isinstance(u.arg, UnaryOps):
          # NOTE: only for float regs, needs second operand
          if u.arg in (UnaryOps.RECIP, UnaryOps.SQRT): ins += f"{opcode(u.arg, u.dtype)} {uop_reg[u]}, {uop_reg[u]}" + "\n"
          else: ins += f"{opcode(u.arg, u.dtype)} {reg(uop_reg[u], u.dtype.itemsize)}" + "\n"

      if u.op is UOps.ENDRANGE:
        ins += f"inc {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}" + "\n" + f"cmp {reg(uop_reg[u.src[0]], u.src[0].dtype.itemsize)}, {uop_reg[u.src[0].src[1]]}" + "\n" + "jl .loop" + "\n"

      # if src regs out of scope clear from regs and add as available if no other uop uses that reg
      # this let's the current uop adopt the reg of its src, consts and define global are excluded
      for s in u.src:
        if s in live_range:
          if live_range[s][1] == i:
            ureg = uop_reg.pop(s)
            if ureg not in uop_reg.values(): float_regs.insert(0, ureg) if ureg.startswith("xmm") else regs.insert(0, ureg)

    return "\n".join([".text", f".global {name}", f"{name}:", "push rbp", "mov rbp, rsp"] + [ins] + ["pop rbp", "ret", "\n"])


# add unsigned mul/div needs rax register
# recip has too little precision, maybe add pattern matcher to change recip to idiv
# way too many movs, it's a mess, do I only create registers based on the loads and stores?
# do reg spilling


# .intel_syntax noprefix <-- add this if using gas

# only create new regs if uop used > 1 times or alu requires it
# I was on bitcast when I left off

# TODO: GET FUNCTIONS LIKE sin,log WORKING


# with chains of alus if it ends in a store we need a new reg but with an assign we don't