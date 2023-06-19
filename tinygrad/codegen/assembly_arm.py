import struct
from itertools import count
from tinygrad.codegen.assembly import AssemblyCodegen, Register, AssemblyInstruction
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

class ARMCodegen(AssemblyCodegen):
  # NOTE: But why?
  supports_load3 = False
  def specialize(self, asm):
    # TODO: fix global name to use real name
    ins =[] 
    pre_global = []
    post_global = []
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "fsub", BinaryOps.MUL: "fmul", BinaryOps.DIV: "fdiv", BinaryOps.MAX: "max",
           BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", BinaryOps.CMPEQ: "cmp",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fma.rn"}

    # TODO: hack to parse registers
    countf = 0  
    counti = 0  
    regs = {}
    parse_reg = lambda reg: regs[reg] 
    pend_reg = [] 
    #parse_reg = lambda reg: f"{'x' if reg[1].lower() in ['b', 'i'] else 's'}{reg[2:]}"
    for i, (uop, out, vin, arg) in enumerate(asm):
      print(asm[i])
      if uop == UOps.DEFINE_REGISTER:
        # TODO: Look at how new_reg handles counts and regs names 
        for x in range(arg[2]):
          if arg[0][0] == dtypes.float: 
            regs['%' + arg[1] + str(x)] = 's' + str(countf)
            countf +=1
          else:
            regs['%' + arg[1] + str(x)] = 'x' + str(counti)
            counti+=1 
      elif uop == UOps.CONST:
        ins.append(f"mov {parse_reg(out[0])}, {arg}")
      elif uop == UOps.CAST:
        # TODO: this is not how casting should be
        if out[1] != dtypes.float:
          ins.append(f"mov {parse_reg(out[0])}, {parse_reg(vin[0][0])}")
      elif uop == UOps.ALU:
        if arg != UnaryOps.NOOP:
          # TODO: better way to check for mul by a const 
          if arg == BinaryOps.MUL and len(vin) == 2 and isinstance(vin[1], int):
            # NOTE arm64 can't mul by a const, so we need to add instead
            # NOTE Mul by 4 i0 is to select index of address 
            pre_global.append(f"mul4:")
            pre_global.append(f"mov x4, xzr")
            post_global.append(f"add x4, x4, #8")
            post_global.append(f"cmp x4, #8")
            post_global.append(f"b.ne _test")
          elif arg == BinaryOps.CMPEQ:
            # NOTE: vin is set to scalar rn.
            ins.append(f"{alu[arg]} {parse_reg(out[0])}, #{vin[0]}")
          else:
            ins.append(f"{'f' if out[1] == dtypes.float else ''}{alu[arg]} {parse_reg(out[0])}, {', '.join(str(x) if x.__class__ is int else str(parse_reg(x[0])) for x in vin)};")
          
      elif uop == UOps.CONST:
        pass
      elif uop == UOps.LOAD:
        ins.append(f"ldr {parse_reg(out[0])},[{','.join(str(parse_reg(x[0])) for x in vin)}]")
      elif uop == UOps.CAST:
        pass
      elif uop == UOps.STORE:
        ins.append(f"str {parse_reg(vin[1][0])}, [{','.join(str(parse_reg(x[0])) for x in vin if vin[1] != x)}] ")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"b.ne {arg[0]}")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".balign 4"] + pre_global + ["_test:"] + ins + post_global + ["ret;"])
