import struct
from platform import system
from extra.assembly.assembly import AssemblyCodegen, Register
from typing import Tuple, Set, Dict
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.codegen.linearizer import UOps, ConstOp
from tinygrad.helpers import dtypes

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
rtor:Dict[Register, str] = {}
pend_regs:Set[Register] = set()
x_regs = ['x' + str(i) for i in reversed(range(29)) if i not in (9,10,11,12,13,14,15,16,17,18,19,20,21)]
s_regs = ['s' + str(i) for i in reversed(range(2,30))]
def alloc_reg(x):
  global x_regs, s_regs
  available_regs = s_regs if dtypes.is_float(x[1]) else x_regs
  if len(available_regs) == 0:
    var_name = max(key = lambda k: rtor[k])
    available_regs.append(rtor[var_name])
    del rtor[var_name]
  reg = available_regs.pop()
  rtor[x.nm] = reg
  return reg 

def free_reg(var_name, available_regs):
  reg = rtor.pop(var_name)
  available_regs.append(reg)

#NOTE: Darwin needs lm functions to start with a "_" 
def get_op(op): return f"bl {'_' if system() == 'Darwin' else ''}{op}"
type_to_reg = {dtypes.half: 'h', dtypes.float32: 's', dtypes.bool: 'x',dtypes.int8:'w', dtypes.int32: 'w', dtypes.int64: 'x', dtypes.uint8:'w', dtypes.uint32: 'w', dtypes.uint64: 'x'}
prev_uop = None
class ARM64Codegen(AssemblyCodegen):
  def specialize(self, asm):
    ins = [] 
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "", BinaryOps.CMPLT: "subs", BinaryOps.CMPEQ: "subs",
           UnaryOps.SIN: get_op('sinf'), UnaryOps.LOG2: get_op("log2f"), UnaryOps.EXP2: get_op("exp2f"), UnaryOps.SQRT: get_op("sqrtf"),
           TernaryOps.MULACC: "fmadd", TernaryOps.WHERE: "fcmp"}
    reg_map = {}
    var_size = 0
    def mov_imm(value, to):
        # Manually move value into reg if vin[1] can't fit
        if value > 65535:
          ins.append(f"mov w15, #{value & 0xffff}")
          ins.append(f"movk w15, #{(value >> 16) & 0xffff}, lsl #16")
          ins.append(f"sxtw {to}, w15")
        else:
          #ins.append(f"{'mov' if to[0] == 'x' else 'fmov'} {to}, {'#' + str(value) if to[0] == 'x' else '0x' + float_to_hex(arg)}")
          if to[0] == 's': 
            ins.append(f"mov x15, {'0x' + float_to_hex(value)}")
            ins.append(f"scvtf {to}, x15")
          else:
            ins.append(f"{'mov'} {to}, {'#' + str(value) if value.__class__ is int else '0x' + float_to_hex(value)}")
          #ins.append(f"{'mov' if to[0] == 'x' else 'fmov'} {to}, {'#' + str(value) if to[0] == 'x' else '0x' +float_to_hex(arg)}")

    for i, (uop, out, vin, arg) in enumerate(asm):
      if out is not None and out.nm not in rtor:
        alloc_reg(out)
        if vin.__class__ is not int:
          for v in vin:
            if v.__class__ is not int and v.nm not in rtor:
              alloc_reg(v)
      #print(rtor)
      if uop == UOps.CAST:
        if arg == BinaryOps.CMPEQ:
          ins.append("mov x15, xzr")
          ins.append("cset w15, eq")
          ins.append(f"scvtf {rtor[out.nm]}, w15")
        else:
          ins.append(f"sxtw {rtor[out.nm]}, w{rtor[vin[0].nm][1:]}")
      elif uop == UOps.ALU:
        reg = 's' if dtypes.is_float(vin[0][1]) else 'x'
        if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
          ins.append(f"ands {rtor[out[0].nm]}, {rtor[vin[0].nm]}, {rtor[vin[1].nm]}")
        elif arg == BinaryOps.MUL and vin[1].__class__ is int:
          mov_imm(vin[1], 'x15')
          ins.append(f"{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]}, x15")
        elif arg == TernaryOps.MULACC and out == vin[2]:
          ins.append(f"{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]}, {rtor[vin[1].nm]}, {rtor[vin[2].nm]}")
        elif arg == TernaryOps.WHERE:
          ins.append(f"fmov s0, #0")
          ins.append(f"{alu[arg]} {rtor[vin[0].nm]}, s0")
          ins.append(f"fcsel {rtor[out.nm]},{rtor[vin[2].nm]}, {rtor[vin[1].nm]}, eq")
        elif arg in [UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.EXP2, UnaryOps.SQRT]:
          ins.append(f"sub sp, sp, #{len(rtor)*16}")
          ins.append("stp x29, x30, [sp, #0]!")
          ins.append("mov x29, sp")
          ins.append(f"fmov s0, {rtor[vin[0].nm]}")
          for i,k in enumerate(rtor.keys()):
            ins.append(f"str {rtor[k]}, [sp, #{32 + (16*i)}]")
          ins.append(f"fmov s0, {rtor[vin[0].nm]}")
          ins.append(alu[arg])
          ins.append(f"fmov {rtor[out.nm]}, s0")
          ins.append("mov sp, x29")
          ins.append("ldp x29, x30, [sp], #0")
          for i,k in enumerate(rtor.keys()):
            if k != out.nm:
              ins.append(f"ldr {rtor[k]}, [sp, #{32 + (16*i)}]")
          ins.append(f"add sp, sp, #{len(rtor)*16}")
        elif arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
          ins.append(f"{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]}, {'#'+str(vin[1]) if vin[1].__class__ is int else rtor[vin[1].nm]}" if reg == 'x' else f"fcmp {rtor[vin[0].nm]}, {rtor[vin[1].nm]}")
        elif arg == BinaryOps.MOD:
          ins.append(f"mov x15, {vin[1]}")
          ins.append(f"udiv x14, {rtor[vin[0].nm]}, x15")
          ins.append(f"msub {rtor[out.nm]}, x14, x15 {rtor[vin[0].nm]}")
        else:
          ins.append(f"{'f' if reg == 's' else 's' if arg == BinaryOps.DIV else ''}{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]},{'#'+str(vin[1]) if vin[1].__class__ is int else rtor[vin[1].nm]}")
      elif uop == UOps.LOAD:
        if arg.__class__ in (int, float):
          mov_imm(arg, rtor[out.nm])
        else:
          #mov_imm(abs(arg[0]), "x20")
          # ins.append(f"{'sub' if arg[0] < 0 else 'add'} x1, x1, x2")
          ins.append(f"add x15, {rtor[vin[0].nm]}, #{arg[0]}")
          ins.append(f"ldr {rtor[out.nm]}, [x15]")
      elif uop == UOps.STORE:
        ins.append(f"str {rtor[vin[1].nm]}, [{rtor[vin[0].nm]}, #{arg[0]}]")
      elif uop == UOps.COND_BRANCH:
        #TODO: this is a hack it shouldn't always be a cmp before a cond branch?
        if prev_uop == UOps.LOAD:
          ins.append(f"cmp {rtor[vin[0].nm]}, #0")
        ins.append(f"b.{'lt' if arg[1] == True else 'ge'} {arg[0][1:]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg[1:]}:")
      prev_uop=uop
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".p2align 2", "_test:", "mov x19, sp"] + ins  + ["ret;"])