import struct
from itertools import count
from tinygrad.codegen.assembly import AssemblyCodegen, Register, AssemblyInstruction
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
class ARMCodegen(AssemblyCodegen):
  def specialize(self, asm):
    ins = [] 
    #NOTE: MACOS needs lm func to start with _, linux doesn't need it 
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "", BinaryOps.CMPLT: "cmp", BinaryOps.CMPEQ: "cmp",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "bl _sin", UnaryOps.LOG2: "bl _log2", UnaryOps.EXP2: "bl _exp2",
           FusedOps.MULACC: "fmadd"}
    reg_map = {}
    buf_map = {}
    var_size = 32
    for i, (uop, out, vin, arg) in enumerate(asm):
      #print(asm[i])
      if uop == UOps.DEFINE_REGISTER: 
      #https://developer.arm.com/documentation/den0024/a/The-ABI-for-ARM-64-bit-Architecture/Register-use-in-the-AArch64-Procedure-Call-Standard/Parameters-in-general-purpose-registers
        for i in range(arg[2]):
          #NOTE: more than 8 args and it will go into the stack 
          reg_map[f"%{arg[1]}{i}"] = f"[sp, #{var_size}]" 
          var_size += 16
      elif uop == UOps.SPECIAL:
        buf_map[arg] = out[1] 
        if arg.startswith('buf'):
          if arg in ['buf8', 'buf9', 'buf10']: 
            ins.append(f"ldr x1, [x19, #{(int(arg[3:]) - 8) * 8}]")
            ins.append(f"str x1, {reg_map[out.nm]}")
          else:
            ins.append(f"str x{arg[3:]}, {reg_map[out.nm]}")
      elif uop == UOps.CONST:
        if arg.__class__ is float:
          ins.append(f"mov x0, 0x{float_to_hex(arg)}")
          ins.append(f"fmov s0, w0")
          ins.append(f"str s0, {reg_map[out.nm]}")
        else:
          ins.append(f"mov x0, #{arg}")
          ins.append(f"str x0, {reg_map[out.nm]}")
      elif uop == UOps.CAST:
        if arg == BinaryOps.CMPEQ:
          ins.append(f"cset w0, eq")
          ins.append(f"scvtf s0, w0")
          ins.append(f"str s0, {reg_map[out.nm]}")
        else:
          ins.append(f"ldr w0, {reg_map[vin[0].nm]}")
          ins.append(f"sxtw x0, w0")
          ins.append(f"str x0, {reg_map[out.nm]}")
      elif uop == UOps.ALU:
        if arg == FusedOps.MULACC and out == vin[2]:
          ins.append(f"ldr s0, {reg_map[vin[0].nm]}")
          ins.append(f"ldr s1, {reg_map[vin[1].nm]}")
          ins.append(f"ldr s2, {reg_map[vin[2].nm]}")
          ins.append(f"{alu[arg]} s0, s0, s1, s2")
          ins.append(f"str s0, {reg_map[out.nm]}")
        elif arg in [UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.EXP2]:
          ins.append(f"stp x29, x30, [sp, #0]!")
          ins.append(f"mov x29, sp")
          ins.append(f"ldr s0, {reg_map[vin[0].nm]}")
          ins.append(f"fcvt d0, s0")
          ins.append(f"{alu[arg]}")
          ins.append(f"fcvt s0, d0")
          ins.append(f"str s0, {reg_map[out.nm]}")
          ins.append(f"mov sp, x29")
          ins.append(f"ldp x29, x30, [sp], #0")
        elif arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
          reg = 's' if dtypes.is_float(vin[0][1]) else 'x'
          ins.append(f"ldr {reg}0, {reg_map[vin[0].nm]}")
          ins.append(f"{f'mov {reg}1, #' + str(vin[1]) if vin[1].__class__ is int else f'ldr {reg}1, ' + reg_map[vin[1].nm]}")
          ins.append(f"{'f' if reg == 's' else ''}{alu[arg]} {reg}0, {reg}1")
        elif arg == BinaryOps.MOD:
          ins.append(f"ldr x0, {reg_map[vin[0].nm]}")
          ins.append(f"{'mov x1, #' + str(vin[1]) if vin[1].__class__ is int else 'ldr x1, ' + reg_map[vin[1].nm]}")
          ins.append(f"udiv x2, x0, x1")
          ins.append(f"msub x2, x2, x1, x0")
          ins.append(f"str x2, {reg_map[out.nm]}")
        else:
          reg = 's' if dtypes.is_float(out[1]) else 'x'
          ins.append(f"ldr {reg}0, {reg_map[vin[0].nm]}")
          ins.append(f"{f'mov {reg}1, #' + str(vin[1]) if vin[1].__class__ is int else f'ldr {reg}1, ' + reg_map[vin[1].nm]}")
          ins.append(f"{'f' if reg == 's' else 's' if arg==BinaryOps.DIV else ''}{alu[arg]} {reg}0, {reg}0, {reg}1")
          ins.append(f"str {reg}0, {reg_map[out.nm]}")
      elif uop == UOps.LOAD:
        reg = 's0' if dtypes.is_float(out[1]) else 'x0'
        ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
        ins.append(f"ldr {reg}, [x1, #{arg[0]}]")
        ins.append(f"str {reg}, {reg_map[out.nm]}")
      elif uop == UOps.STORE:
        reg = 's0' if dtypes.is_float(vin[1][1]) else 'x0'
        ins.append(f"ldr {reg}, {reg_map[vin[1].nm]}")
        #TODO: ugly refactor this
        if "buf0" in buf_map and buf_map["buf0"] == dtypes.int32 and reg == 's0': 
          ins.append(f"fcvtzs w0, s0")
          reg = 'w0'
        ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
        ins.append(f"str {reg}, [x1, #{arg[0]}]")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"{'b.ne' if arg[1] else 'beq'} {arg[0][1:]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg[1:]}:")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".p2align 2"] + ["_test:"] + ["mov x19, sp",f"sub sp, sp, #{var_size}"] + ins  + [f"add sp, sp, #{var_size}"]+ ["ret;"])
