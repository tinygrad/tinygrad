import struct
import platform
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

#NOTE: Darwin needs lm functions to start with a "_" 
def get_op(op): return f"bl {'_' if platform.system() == 'Darwin' else ''}{op}"

class ARMCodegen(AssemblyCodegen):
  def specialize(self, asm):
    ins = [] 
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "", BinaryOps.CMPLT: "subs", BinaryOps.CMPEQ: "subs",
           UnaryOps.SIN: get_op('sin'), UnaryOps.LOG2: get_op("log2"), UnaryOps.EXP2: get_op("exp2"), UnaryOps.SQRT: get_op("sqrt"),
           FusedOps.MULACC: "fmadd"}
    reg_map = {}
    buf_map = {}
    var_size = 0
    for i, (uop, out, vin, arg) in enumerate(asm):
      if uop == UOps.DEFINE_REGISTER: 
        for i in range(arg[2]):
          var_size += 16
          reg_map[f"%{arg[1]}{i}"] = f"[sp, #{var_size}]"
      elif uop == UOps.SPECIAL:
        buf_map[arg] = out[1] 
        if arg.startswith('buf'):
          if int(arg[3:]) >= 8: ins.append(f"ldr x1, [x19, #{(int(arg[3:]) - 8) * 8}]")
          if out[1] == dtypes.int32 and arg != "buf0":
            for i in range(out.bufsize):
              ins.append(f"ldr s0, [x{'1' if int(arg[3:]) >= 8 else arg[3:]}, #{i*4}]")
              ins.append("scvtf s0, s0")
              ins.append(f"str s0, [x{'1' if int(arg[3:]) >= 8 else arg[3:]}, #{i*4}]")
          ins.append(f"str x{'1' if int(arg[3:]) >= 8 else arg[3:]}, {reg_map[out.nm]}")
      elif uop == UOps.CONST:
        ins.append(f"mov w0, #{arg & 0xffff if arg > 65535 else arg}" if arg.__class__ is int else f"mov x0, 0x{float_to_hex(arg)}")
        if arg.__class__ is int and arg > 65535:
          ins.append(f"movk w0, #{(arg >> 16) & 0xffff}, lsl #16")
          ins.append("sxtw x0, w0")
        elif arg.__class__ is float or dtypes.is_float(out.dtype):
          ins.append("fmov s0, w0")
        ins.append(f"str {'s' if arg.__class__ is float else 'x'}0, {reg_map[out.nm]}")
      elif uop == UOps.CAST:
        if arg == BinaryOps.CMPEQ:
          ins.append("cset w0, eq")
          ins.append("scvtf s0, w0")
        else:
          ins.append(f"ldr w0, {reg_map[vin[0].nm]}")
          ins.append("sxtw x0, w0")
        ins.append(f"str {'s' if dtypes.is_float(out[1]) else 'x'}0, {reg_map[out.nm]}")
      elif uop == UOps.ALU:
        reg = 's' if dtypes.is_float(vin[0][1]) else 'x'
        ins.append(f"ldr {reg}0, {reg_map[vin[0].nm]}")
        if len(vin) >= 2:
          if vin[1].__class__ is int and vin[1] > 65535:
            ins.append(f"mov w2, #{vin[1] & 0xffff}")
            ins.append(f"movk w2, #{(vin[1] >> 16) & 0xffff}, lsl #16")
            ins.append("sxtw x1, w2")
          else:
            ins.append(f"{f'mov {reg}1, #{str(vin[1])}' if vin[1].__class__ is int else f'ldr {reg}1, {reg_map[vin[1].nm]}'}")
        if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
          ins.append("ands x0, x0, x1;")
        elif arg == FusedOps.MULACC and out == vin[2]:
          ins.append(f"ldr s2, {reg_map[vin[2].nm]}")
          ins.append(f"{alu[arg]} s0, s0, s1, s2")
        elif arg in [UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.EXP2, UnaryOps.SQRT]:
          ins.append("stp x29, x30, [sp, #0]!")
          ins.append("mov x29, sp")
          ins.append("fcvt d0, s0")
          ins.append(alu[arg])
          ins.append("fcvt s0, d0")
          ins.append("mov sp, x29")
          ins.append("ldp x29, x30, [sp], #0")
        elif arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
          ins.append(f"{alu[arg]} {reg}0, {reg}0, {reg}1" if reg == 'x' else f"fcmp {reg}0, {reg}1")
        elif arg == BinaryOps.MOD:
          ins.append("udiv x2, x0, x1")
          ins.append("msub x2, x2, x1, x0")
        else:
          ins.append(f"{'f' if reg == 's' else 's' if arg == BinaryOps.DIV else ''}{alu[arg]} {reg}0, {reg}0, {reg}1")
        ins.append(f"str {reg}{'2' if arg == BinaryOps.MOD else '0'}, {reg_map[out.nm]}")
      elif uop == UOps.LOAD:
        reg = 's0' if dtypes.is_float(out[1]) else 'x0'
        ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
        if reg == 's0' and arg[0] < -255:
          ins.append(f"mov x2, #{abs(arg[0])}")
          ins.append("sub x1, x1, x2")
          ins.append(f"ldr {reg}, [x1]")
        else:
          ins.append(f"ldr {reg}, [x1, #{arg[0]}]")
        ins.append(f"str {reg}, {reg_map[out.nm]}")
      elif uop == UOps.STORE:
        reg = 's0' if dtypes.is_float(vin[1][1]) else 'x0'
        ins.append(f"ldr {reg}, {reg_map[vin[1].nm]}")
        if buf_map["buf0"] == dtypes.int32 and buf_map["buf1"] == dtypes.float:
          ins.append(f"fcvtzs w0, {reg}")
          reg = 'w0'
        elif buf_map["buf0"] == dtypes.float and buf_map["buf1"] == dtypes.int32:
          ins.append(f"scvtf {reg}, {reg}")
        ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
        ins.append(f"str {reg}, [x1, #{arg[0]}]")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"b.{'lt' if arg[1]==True else 'ge'} {arg[0][1:]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg[1:]}:")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".p2align 2", "_test:", "mov x19, sp",f"sub sp, sp, #{var_size}"] + ins  + [f"add sp, sp, #{var_size}","ret;"])
