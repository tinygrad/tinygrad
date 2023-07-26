import struct
from platform import system
from extra.assembly.assembly import AssemblyCodegen, Register
from typing import Tuple, Set, Dict
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.codegen.linearizer import UOps, ConstOp
from tinygrad.helpers import dtypes

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
def compute_offsets(total):
    quotient, remainder = divmod(total, 4096)
    return [4096]*quotient + [remainder] if remainder else [4096]*quotient
rtor:Dict[Register, str] = {}
pend_regs:Set[Register] = set()
x_regs = ['x' + str(i) for i in reversed(range(15))]
s_regs = ['s' + str(i) for i in reversed(range(2,20))]
def alloc_reg(x):
  global x_regs, s_regs
  available_regs = s_regs if dtypes.is_float(x[1]) else x_regs
  if len(available_regs) == 0:
    var_name = max(filter(lambda x: x[0] != 'x', rtor.keys()), key = lambda k: rtor[k])
    available_regs.append(rtor[var_name])
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
           UnaryOps.SIN: get_op('sin'), UnaryOps.LOG2: get_op("log2"), UnaryOps.EXP2: get_op("exp2"), UnaryOps.SQRT: get_op("sqrt"),
           TernaryOps.MULACC: "fmadd", TernaryOps.WHERE: "fcmp"}
    reg_map = {}
    var_size = 0
    def mov_imm(value, to):
        # Manually move value into reg if vin[1] can't fit
        if value > 65535:
          ins.append(f"mov w2, #{value & 0xffff}")
          ins.append(f"movk w2, #{(value >> 16) & 0xffff}, lsl #16")
          ins.append(f"sxtw {to}, w2")
        else:
          ins.append(f"mov {to}, {'#' + str(value) if value.__class__ is int else '0x' + float_to_hex(arg)}")

    for i, (uop, out, vin, arg) in enumerate(asm):
      if out is not None and out.nm not in rtor:
        alloc_reg(out)
        if vin.__class__ is not int:
          for v in vin:
            if v.__class__ is not int and v.nm not in rtor:
              alloc_reg(v)
      print(rtor)
      # if uop == UOps.DEFINE_REGISTER:
      #   for i in range(arg[2]):
      #     var_size += 16
      #     #TODO: Find a way to use less memory lookups. Graph coloring?
      #     reg_map[f"%{arg[1]}{i}"] = f"[sp, #{var_size}]"
      #     #regs(f"%{arg[1]}{i}", arg[0][0], var_size)
      # elif uop == UOps.DEFINE_GLOBAL:
      #   if arg.startswith('data'):
      #     if int(arg[4:]) >= 8: ins.append(f"ldr x1, [x19, #{(int(arg[4:]) - 8) * 8}]")
      #     ins.append(f"str x{'1' if int(arg[4:]) >= 8 else int(arg[4:])}, {reg_map[out.nm]}")
      if uop == UOps.CAST:
        if arg == BinaryOps.CMPEQ:
          ins.append("cset w0, eq")
          ins.append("scvtf s0, w0")
        else:
          ins.append(f"sxtw {rtor[vin[0].nm]}, w{rtor[vin[0].nm][1:]}")
          ins.append(f"mov {rtor[out.nm]}, {rtor[vin[0].nm]}")
#        ins.append(f"str {'s' if dtypes.is_float(out[1]) else 'x'}0, {reg_map[out.nm]}")
      elif uop == UOps.ALU:
        reg = 's' if dtypes.is_float(vin[0][1]) else 'x'
        #for i in range(len(vin)):
        #   if vin[i].__class__ is not int: ins.append(f"ldr {reg}{i}, {reg_map[vin[i].nm]}")
        #   else: mov_imm(vin[1], f"{reg}{i}")
        if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
          ins.append("ands x0, x0, x1;")
        elif arg == BinaryOps.MUL and vin[1].__class__ is int:
          mov_imm(vin[1], 'x15')
          ins.append(f"{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]}, x15")
        elif arg == TernaryOps.MULACC and out == vin[2]:
          ins.append(f"{alu[arg]} s0, s0, s1, s2")
        elif arg == TernaryOps.WHERE:
          ins.append(f"fmov s3, #0")
          ins.append(f"{alu[arg]} s0, s3")
          ins.append(f"fcsel s0, s2, s1, eq")
        elif arg in [UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.EXP2, UnaryOps.SQRT]:
          ins.append("stp x29, x30, [sp, #0]!")
          ins.append("mov x29, sp")
          ins.append(f"fmov s0, {rtor[vin[0].nm]}")
          ins.append(f"fcvt d0, s0")
          ins.append(alu[arg])
          ins.append(f"fcvt s0, d0")
          ins.append(f"fmov {rtor[out.nm]}, s0")
          ins.append("mov sp, x29")
          ins.append("ldp x29, x30, [sp], #0")
        elif arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
#          ins.append(f"{alu[arg]} {reg}0, {reg}0, {reg}1" if reg == 'x' else f"fcmp {reg}0, {reg}1")
          ins.append(f"{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]}, {'#'+str(vin[1]) if vin[1].__class__ is int else rtor[vin[1].nm]}" if reg == 'x' else f"fcmp {reg}0, {reg}1")
        elif arg == BinaryOps.MOD:
          ins.append("udiv x2, x0, x1")
          ins.append("msub x2, x2, x1, x0")
        else:
          ins.append(f"{'f' if reg == 's' else 's' if arg == BinaryOps.DIV else ''}{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]},{'#'+str(vin[1]) if vin[1].__class__ is int else rtor[vin[1].nm]}")
          #ins.append(f"{'f' if reg == 's' else 's' if arg == BinaryOps.DIV else ''}{alu[arg]} {reg}0, {reg}0, {reg}1")
        #ins.append(f"str {reg}{'2' if arg == BinaryOps.MOD else '0'}, {reg_map[out.nm]}")
      elif uop == UOps.LOAD:
        if arg.__class__ in (int, float):
          mov_imm(arg, rtor[out.nm])
          # mov_imm(arg,"x0")
          # ins.append(f"str x0, {reg_map[out.nm]}") 
        else:
          #reg_out = 's0' if dtypes.is_float(out[1]) else 'x0'
          #reg_out = rtor[out.nm] 
          #reg_in = type_to_reg[arg[2] if arg[2] is not None else out[1]] + '0'
          ins.append(f"ldr {rtor[out.nm]}, [{rtor[vin[0].nm]}, #{arg[0]}]")
          # Manually offset in case it can't fix in imm
          # mov_imm(abs(arg[0]), "x2")
          # ins.append(f"{'sub' if arg[0] < 0 else 'add'} x1, x1, x2")
          # ins.append(f"ldr{'sb' if arg[2] is not None and arg[2] in (dtypes.int8, dtypes.uint8) else ''} {reg_in}, [x1]")
          # if arg[2] is not None: ins.append(f"{'fcvt' if arg[2] == dtypes.half else 'scvtf'} s0, {reg_in}")
          #ins.append(f"str {reg_out}, {reg_map[out.nm]}")
      elif uop == UOps.STORE:
        shifts = {dtypes.int64: "#3", dtypes.half: "#1", dtypes.int8:"#2", dtypes.uint8: "#2"}
        #ins.append(f"ldr s0, {reg_map[vin[1].nm]}")
        #reg_out = (type_to_reg[arg[2]] if arg[2] is not None else "s") + '0'
        #if arg[2] is not None: ins.append(f"fcvt{'zs' if arg[2] != dtypes.half else '' } {reg_out}, s0")
        #ins.append(f"mov x3, #{arg[0]}")
        #ins.append(f"ldr x2, {reg_map[vin[0].nm]}")
        ins.append(f"str {rtor[vin[1].nm]}, [{rtor[vin[0].nm]}, #{arg[0]}]")
        #ins.append(f"str {reg_out}, [x2, x3, lsl {shifts[arg[2]] if arg[2] is not None and arg[2] in shifts else '#0'}]")
      elif uop == UOps.COND_BRANCH:
        #TODO: this is a hack it shouldn't always be a cmp before a cond branch?
        if prev_uop == UOps.LOAD:
          ins.append(f"ldr x1, {reg_map[vin[0].nm]}")
          ins.append(f"cmp x1, #0")
        ins.append(f"b.{'lt' if arg[1] == True else 'ge'} {arg[0][1:]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg[1:]}:")
      prev_uop=uop
   #[f"sub sp, sp, #{offset}" for offset in compute_offsets(var_size)] 
   #+ [f"add sp, sp, #{offset}" for offset in compute_offsets(var_size)] 
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".p2align 2", "_test:", "mov x19, sp"]+[f"sub sp, sp, #{offset}" for offset in compute_offsets(32)] + ins + [f"add sp, sp, #{offset}" for offset in compute_offsets(32)] + ["ret;"])