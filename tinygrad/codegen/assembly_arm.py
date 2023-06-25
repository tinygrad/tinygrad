# from tinygrad.codegen.assembly import AssemblyCodegen, AssemblyInstruction 
# from tinygrad.ops import BinaryOps, UnaryOps
# from tinygrad.codegen.linearizer import UOps
# from tinygrad.helpers import DEBUG, dtypes

# class ARMCodegen(AssemblyCodegen):

#   def specialize(self, asm):
#     type_to_op_suffix = {dtypes.float32: '', dtypes.bool: '', dtypes.int32: '', dtypes.int64: '', dtypes.uint32: '', dtypes.uint64: ''}
#     def inst(instr, reg, fp=False, simd=False): return f"{instr}{type_to_op_suffix[reg_type[reg.nm]]}"
#     alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.CMPLT: "cmp",  BinaryOps.CMPEQ: "cmp", BinaryOps.MAX: "max",
#          UnaryOps.NOOP: "mov", UnaryOps.SIN: "bl sinf@PLT", UnaryOps.LOG2: "bl log2f@PLT", UnaryOps.EXP2: "bl exp2f@PLT"}

#     ins = []
#     reg_map = {}
#     reg_type = {}
#     local_var_size = 0
#     for uop, out, vin, arg in filter(lambda op: op.op == UOps.DEFINE_REGISTER, asm):
#       for i in range(arg[2]):
#         local_var_size += arg[0][0].itemsize
#         reg_map[f"%{arg[1]}{i}"] = f"x{i}"
#         reg_type[f"%{arg[1]}{i}"] = arg[0]

#     print(reg_map)
#     print(reg_type)
#     print(local_var_size)

#     for uop, out, vin, arg in asm:
#       if DEBUG >= 5: ins.append(f"# {AssemblyInstruction(uop, out, vin, arg)}")
#       if uop == UOps.DEFINE_REGISTER: pass
#       elif uop == UOps.DEFINE_LOCAL: pass
#       elif uop == UOps.SPECIAL:
#         ins.append(f"{inst('mov', out)} {reg_map[arg]}, {reg_map[out.nm]}")
#       elif uop == UOps.ALU:
#         if dtypes.is_float(out.dtype):
#           ins.append(f"ldr s0, {reg_map[vin[0].nm]}")
#           if arg not in [UnaryOps.SIN, UnaryOps.LOG2, UnaryOps.EXP2]: 
#             ins.append(f"{alu[arg]} s0, s0, {reg_map[vin[1].nm]}")
#           else:
#             ins.append(f"{alu[arg]}")
#           ins.append(f"str s0, {reg_map[out.nm]}")
#         else:
#           ins.append(f"{inst('ldr', vin[0])} {reg_map[vin[0].nm]}, x0")
#           ins.append(f"{inst('ldr', vin[0])} {reg_map[vin[1].nm] if not isinstance(vin[1], int) else f'#{vin[1]}'}, x1")
#           if arg == BinaryOps.MUL:
#             ins.append(f"{inst(alu[arg], vin[0])} x0, x1")
#           else:
#             ins.append(f"{inst(alu[arg], vin[0])} x1, x0")
#             if arg in [BinaryOps.CMPLT, BinaryOps.CMPEQ]: 
#               cmp_map = {BinaryOps.CMPLT: "b.ge", BinaryOps.CMPEQ: "b.eq"}
#               ins.append(f"{cmp_map[arg]} 1f")
#           ins.append(f"{inst('str', vin[0])} x0, {reg_map[out.nm]}")
#       elif uop == UOps.LOAD:
#         ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
#         ins.append(f"{inst('ldr', out)} [{arg[0]}], x0")
#         ins.append(f"{inst('str', out)} x0, {reg_map[out.nm]}")
#       elif uop == UOps.STORE:
#         ins.append(f"{inst('ldr', vin[1])} {reg_map[vin[1].nm]}, x0")
#         ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
#         ins.append(f"{inst('str', vin[1])} x0, [{arg[0]}]")
#       elif uop == UOps.CAST:     
#         ins.append(f"sxtw x0, {reg_map[vin[0].nm]}")
#         ins.append(f"str x0, {reg_map[out.nm]}")
#       elif uop == UOps.CONST:
#         ins.append(f"{inst('mov', out)} #{arg}, {reg_map[out.nm]}")
#       elif uop == UOps.LABEL:
#         ins.append(f"{arg.replace('$', '_')}:")
#       elif uop == UOps.COND_BRANCH:
#         ins.append(f"ldrb w0, {reg_map[vin[0].nm]}")
#         ins.append(f"tst w0, w0")
#         ins.append(f"{arg[1]}eq {arg[0].replace('$', '_')}")

#     return "_kernel", '\n'.join([".global _kernel", "_kernel:", "stp x29, x30, [sp, #-16]!", "mov x29, sp", f"sub sp, sp, #{local_var_size}"] + ins + ["mov sp, x29", "ldp x29, x30, [sp], #16", "ret"])
import struct
from itertools import count
from tinygrad.codegen.assembly import AssemblyCodegen, Register, AssemblyInstruction
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

class ARMCodegen(AssemblyCodegen):
  def specialize(self, asm):
    # TODO: fix global name to use real name
    ins =[] 
    pre_global = []
    post_global = []
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "fsub", BinaryOps.MUL: "mul", BinaryOps.DIV: "fdiv", BinaryOps.MAX: "max",
           BinaryOps.MOD: "rem", BinaryOps.CMPLT: "cmp", BinaryOps.CMPEQ: "cmp",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fma.rn"}

    # TODO: hack to parse registers
    countf = countx = countw = 0 
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
        reg_map[f"%{arg[1]}{i}"] = f"{'s'+str(countf) if dtypes.is_float(arg[0][0]) else 'x'+str(countx)}"
        reg_type[f"%{arg[1]}{i}"] = arg[0]
        if dtypes.is_float(arg[0][0]):
          countf+=1
        else:
          countx+=1

    print(reg_map)
    print(reg_type)
    print(local_var_size)
    print("-"*5)
    #ins.append("stp x29, x30, [SP, #-16]!   ")
    #ins.append("mov FP, SP")
    #ins.append(f"sub SP, SP, #{local_var_size}")
    for i, (uop, out, vin, arg) in enumerate(asm):
      print(asm[i])
      if uop == UOps.SPECIAL:
        unix_call_conv = {'buf0': 'x0', 'buf1': 'x1', 'buf2': 'x2'}
        if arg.startswith('buf'):
          # ins.append(f"str {unix_call_conv[arg]}, {reg_map[out.nm]}")
          ins.append(f"mov {unix_call_conv[arg]}, {reg_map[out.nm]}")
          # TODO pop remaining args from stack
      elif uop == UOps.CONST:
        ins.append(f"mov {reg_map[out.nm]}, #{arg}")
      elif uop == UOps.CAST:
        ins.append(f"sxtw {reg_map[out.nm]}, {reg_map[vin[0].nm]}")
      elif uop == UOps.ALU:
        if dtypes.is_float(out.dtype):
          ins.append(f"f{alu[arg]} {reg_map[out.nm]}, {reg_map[vin[0].nm]}, {reg_map[vin[1].nm]}")
        else:
          if isinstance(vin[1], int):
            ins.append(f"mov x10, #{vin[1]}")
            if arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
              ins.append(f"{alu[arg]} {reg_map[vin[0].nm]}, x10")
            else:
              ins.append(f"{alu[arg]} {reg_map[out.nm]}, {reg_map[vin[0].nm]}, x10")

          else: 
            if arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
              ins.append(f"{alu[arg]} {reg_map[vin[0].nm]}, {reg_map[vin[1].nm]}")
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
#    ins.append("ldp x29, x30, [SP], #16")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".balign 4"] + pre_global + ["_test:"] + ins + post_global + ["ret;"])
