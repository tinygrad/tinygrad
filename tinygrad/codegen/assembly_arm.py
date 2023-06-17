import struct
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

class ARMCodegen(AssemblyCodegen):
  # NOTE: But why?
  supports_load3 = True
  def specialize(self, asm):
    # TODO: fix global name to use real name
    ins = [".arch armv8-a",".text", ".global _test",".balign 4", "_test:"]
    alu = {BinaryOps.ADD: "fadd", BinaryOps.SUB: "fsub", BinaryOps.MUL: "fmul", BinaryOps.DIV: "fdiv", BinaryOps.MAX: "max",
           BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", BinaryOps.CMPEQ: "setp.eq",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fma.rn"}

    # TODO: hack to parse registers
    parse_reg = lambda reg: f"{'x' if reg[1].lower() in ['b', 'i'] else 's'}{reg[2:]}"

    for i, (uop, out, vin, arg) in enumerate(asm):
      print(asm[i])
      if uop == UOps.SPECIAL:
        pass
      elif uop == UOps.ALU:
        if arg != UnaryOps.NOOP:
          ins.append(f"{alu[arg]} {parse_reg(out[0])}, {', '.join(str(parse_reg(x[0])) for x in vin)};")
      elif uop == UOps.CONST:
        pass
      elif uop == UOps.LOAD:
        ins.append(f"ldr {parse_reg(out[0])}, [{parse_reg(vin[0][0])}]")
      elif uop == UOps.CAST:
        pass
      elif uop == UOps.STORE:
        ins.append(f"str {parse_reg(vin[1][0])}, [{parse_reg(vin[0][0])}] ")

    ins += ["ret;"]
    return "test", '\n'.join(ins)
