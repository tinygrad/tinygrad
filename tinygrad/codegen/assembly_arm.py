import struct
from itertools import count
from tinygrad.codegen.assembly import AssemblyCodegen, Register, AssemblyInstruction
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
class ARMCodegen(AssemblyCodegen):
  def specialize(self, asm):
    ins =[] 
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "sdiv", BinaryOps.MAX: "max",
           BinaryOps.MOD: "", BinaryOps.CMPLT: "cmp", BinaryOps.CMPEQ: "cmp",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fmadd"}

    # TODO: hack to parse registers
    countf = countx = countw = 0 
    reg_map = {}
    reg_type = {}
    stack_map = {}
    var_size = 0
    for i, (uop, out, vin, arg) in enumerate(asm):
#      print(asm[i])
      if uop == UOps.DEFINE_REGISTER and dtypes.bool != arg[0][0]: 
      #https://developer.arm.com/documentation/den0024/a/The-ABI-for-ARM-64-bit-Architecture/Register-use-in-the-AArch64-Procedure-Call-Standard/Parameters-in-general-purpose-registers
       for i in range(arg[2]):
        stack_map[f"%{arg[1]}{i}"] = f"[sp, #{var_size}]"  
        var_size += 16
        reg_map[f"%{arg[1]}{i}"] = f"{'s'+str(countf) if dtypes.is_float(arg[0][0]) else 'x'+str(countx)}"
        if dtypes.is_float(arg[0][0]): countf+=1
        else:
          countx+=1 #TODO Support w registers
      elif uop == UOps.SPECIAL:
        if arg.startswith('buf'):
          ins.append(f"str x{arg[3:]}, {stack_map[out.nm]}")
      elif uop == UOps.CONST:
        if isinstance(arg, float):
          ins.append(f"ldr x0, #{float_to_hex(arg)}")
        else:
          ins.append(f"mov x0, #{arg}")
        ins.append(f"str x0, {stack_map[out.nm]}")
      elif uop == UOps.CAST:
        ins.append(f"ldr x0, {stack_map[vin[0].nm]}")
        ins.append(f"sxtw x1, x0")
        ins.append(f"str x1, {stack_map[out.nm]}")
      elif uop == UOps.ALU:
        if arg == FusedOps.MULACC and out == vin[2]:
          ins.append(f"{alu[arg]} {reg_map[out.nm]}, {reg_map[vin[0].nm]}, {reg_map[vin[1].nm]}, {reg_map[vin[2].nm]}")
        elif dtypes.is_float(out.dtype):
          ins.append(f"ldr s0, {stack_map[vin[0].nm]}")
          ins.append(f"ldr s1, {stack_map[vin[1].nm]}")
          ins.append(f"f{alu[arg]} s1, s0, s1")
          ins.append(f"str s1, {stack_map[out.nm]}")
        else:
          #TODO: vin[1] check if not a constant
#          if arg == BinaryOps.MOD:
            #TODO: LOAD constant into memory and not into reg
#            ins.append(f"sub sp, sp, #16")
            # ins.append(f"str x25, [sp]")
            # ins.append(f"mov x25, #{vin[1]}")
            # ins.append(f"udiv {reg_map[out.nm]}, {reg_map[vin[0].nm]}, x25")
            # ins.append(f"msub {reg_map[out.nm]}, {reg_map[out.nm]}, {reg_map[vin[0].nm]}, x25")
            # ins.append(f"ldr x25, [sp]")
#            ins.append(f"add sp, sp, #16")
          if arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
            ins.append(f"ldr x0, {stack_map[vin[0].nm]}")
            ins.append(f"{alu[arg]} x0, #{vin[1]}")
          else: 
            isint = isinstance(vin[1], int)
            ins.append(f"ldr x1, {stack_map[vin[0].nm]}")
            if arg in [BinaryOps.MUL, BinaryOps.DIV] and isint:
              ins.append(f"mov x2, #{vin[1]}")
              ins.append(f"{alu[arg]} x3, x1, x2")
            # elif arg == BinaryOps.MOD:
            #   ins.append(f"str x5, [sp, #-16]!")
            #   ins.append(f"mov x5, #{vin[1]}")
            #   ins.append(f"sdiv {reg_map[out.nm]}, {reg_map[vin[0].nm]}, {'x5' if isint else reg_map[vin[1].nm]}")
            #   ins.append(f"msub {reg_map[out.nm]}, {reg_map[out.nm]}, {'x5' if isint else reg_map[vin[1].nm]}, {reg_map[vin[0].nm]}")
            #   ins.append(f"ldr x5 [sp], #16")
            else:
              ins.append(f"{'mov x2, #' + str(vin[1]) if isint else 'ldr x2, ' + stack_map[vin[1].nm]}")
              ins.append(f"{alu[arg]} x3, x1, x2")
            ins.append(f"str x3, {stack_map[out.nm]}")
      elif uop == UOps.LOAD:
        reg = 's0' if dtypes.is_float(out[1]) else 'x0'
        ins.append(f"ldr x1, {stack_map[vin[0].nm]}")
        ins.append(f"ldr {reg}, [x1, {arg[0]}]")
        ins.append(f"str {reg}, {stack_map[out.nm]}")
      elif uop == UOps.STORE:
        reg = 's0' if dtypes.is_float(vin[1][1]) else 'x0'
        ins.append(f"ldr {reg}, {stack_map[vin[1].nm]}")
        ins.append(f"ldr x1, {stack_map[vin[0].nm]}")
        ins.append(f"str {reg}, [x1, {arg[0]}]")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"{'b.ne' if arg[1] else 'beq'} {arg[0][1:]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg[1:]}:")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".balign 4"] + ["_test:"] + [f"sub sp, sp, #{var_size}"] + ins  + [f"add sp, sp, #{var_size}"]+ ["ret;"])
