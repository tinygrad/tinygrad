import struct
from itertools import count
from tinygrad.codegen.assembly import AssemblyCodegen, Register, AssemblyInstruction
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes


class ARMCodegen(AssemblyCodegen):
  def specialize(self, asm):
    ins =[] 
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "rem", BinaryOps.CMPLT: "cmp", BinaryOps.CMPEQ: "cmp",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fma.rn"}

    # TODO: hack to parse registers
    countf = countx = countw = 0 
    reg_map = {}
    reg_type = {}
    for i, (uop, out, vin, arg) in enumerate(asm):
      print(asm[i])
      if uop == UOps.DEFINE_REGISTER:
      #TODO: Using only register might break at some point. find out when and how to fix. (think using stack)
       for i in range(arg[2]):
        reg_map[f"%{arg[1]}{i}"] = f"{'s'+str(countf) if dtypes.is_float(arg[0][0]) else 'x'+str(countx)}"
        if dtypes.is_float(arg[0][0]): countf+=1
        else: countx+=1 #TODO Support w registers
      elif uop == UOps.CONST:
        ins.append(f"mov {reg_map[out.nm]}, #{arg}")
      elif uop == UOps.CAST:
        ins.append(f"sxtw {reg_map[out.nm]}, {reg_map[vin[0].nm]}")
      elif uop == UOps.ALU:
        if dtypes.is_float(out.dtype):
          ins.append(f"f{alu[arg]} {reg_map[out.nm]}, {reg_map[vin[0].nm]}, {reg_map[vin[1].nm]}")
        else:
          #TODO: vin[1] check if not a constant
          if arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
            ins.append(f"{alu[arg]} {reg_map[vin[0].nm]}, {vin[1]}")
          elif isinstance(vin[1], int):
            ins.append(f"mov x10, #{vin[1]}")
            ins.append(f"{alu[arg]} {reg_map[out.nm]}, {reg_map[vin[0].nm]}, x10")
          else: 
            ins.append(f"{alu[arg]} {reg_map[out.nm]}, {reg_map[vin[0].nm]}, {reg_map[vin[1].nm]}")
      elif uop == UOps.LOAD:
        ins.append(f"ldr {reg_map[out.nm]}, [{reg_map[vin[0].nm]}, {arg[0]}]")
      elif uop == UOps.STORE:
        ins.append(f"str {reg_map[vin[1].nm]}, [{reg_map[vin[0].nm]}, {arg[0]}]")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"b.ne {arg[0][1:]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg[1:]}:")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".balign 4"] + ["_test:"] + ins  + ["ret;"])
