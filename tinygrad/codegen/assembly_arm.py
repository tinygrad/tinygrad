import struct
from itertools import count
from tinygrad.codegen.assembly import AssemblyCodegen, Register, AssemblyInstruction
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

class ARMCodegen(AssemblyCodegen):
  # NOTE: I don't need this, however it build to better assembly when i do.  why?
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
    print("-"*5)
    reg_map = {}
    reg_type = {}
    local_var_size = 0
    for uop, out, vin, arg in filter(lambda op: op.op == UOps.DEFINE_REGISTER, asm):
      for i in range(arg[2]):
        local_var_size += arg[0][0].itemsize
        reg_map[f"%{arg[1]}{i}"] = f"[FP, #-{local_var_size}]"
        reg_type[f"%{arg[1]}{i}"] = arg[0]

    print(reg_map)
    print(reg_type)
    print(local_var_size)
    print("-"*5)
    #ins.append("stp x29, x30, [SP, #-16]!   ")
    for i, (uop, out, vin, arg) in enumerate(asm):
      if uop == UOps.SPECIAL:
        unix_call_conv = {'buf0': 'x0', 'buf1': 'x1', 'buf2': 'x2' }
        if arg.startswith('buf'):
          ins.append(f"str {unix_call_conv[arg]}, {reg_map[out.nm]}")
          # TODO pop remaining args from stack
      elif uop == UOps.CONST:
        ins.append(f"mov x1, #{arg}")
        ins.append(f"str x1, {reg_map[out[0]]}")
      elif uop == UOps.CAST:
        # TODO: this is not how casting should be
        ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
        ins.append(f"sxtw x2, x1")
        ins.append(f"str x2, {reg_map[out.nm]}")
      elif uop == UOps.ALU:
        if dtypes.is_float(out.dtype):
          ins.append(f"ldr s0, {reg_map[vin[0].nm]}")
          ins.append(f"ldr s1, {reg_map[vin[1].nm]}")
          ins.append(f"f{alu[arg]} s0, s0, s1")
          ins.append(f"str s0, {reg_map[out.nm]}")
        else:
          ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
          if arg == BinaryOps.MUL and len(vin) == 2 and isinstance(vin[1], int):
        #   # NOTE arm64 can't mul by a const, so we need to add instead
            ins.append(f"mov x2, #{vin[1]}")
            ins.append(f"mul x1, x1, x2")
          else:
            ins.append(f"ldr x2, {reg_map[vin[1].nm]}")
            ins.append(f"{alu[arg]} x1, x1, x2")
          ins.append(f"str x1, {reg_map[out.nm]}") 
          
      elif uop == UOps.LOAD:
        # acc_reg_a = "%rax" if out.dtype.itemsize == 8 else "%eax"        
        # ins.append(f"movq {reg_map[vin[0].nm]}, %rbx")
        # ins.append(f"{inst('mov', out)} {arg[0]}(%rbx), {acc_reg_a}")
        # ins.append(f"{inst('mov', out)} {acc_reg_a}, {reg_map[out.nm]}")
        ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
        ins.append(f"ldr s0, [x1]")
        ins.append(f"str s0, {reg_map[out[0]]}")
#        ins.append(f"ldr {parse_reg(out[0])},[{','.join(str(parse_reg(x[0])) for x in vin)}, {arg[0]}]")
      elif uop == UOps.STORE:
#        ins.append(f"str {parse_reg(vin[1][0])}, [{','.join(str(parse_reg(x[0])) for x in vin if vin[1] != x)}] ")
        print(f"l {vin[1][1]}")
        print(f"l {vin[0][1]}")
        ins.append(f"ldr s0, {reg_map[vin[1].nm]}")
        ins.append(f"ldr x0, {reg_map[vin[0].nm]}")
        ins.append(f"str s0, [x0]")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"b.ne {arg[0]}")
#    ins.append("ldp x29, x30, [SP], #16")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".balign 4"] + pre_global + ["_test:"] + ins + post_global + ["ret;"])
