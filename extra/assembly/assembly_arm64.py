import struct
from platform import system
from extra.assembly.assembly import AssemblyCodegen, Register
from typing import Tuple, Set, Dict
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.codegen.linearizer import UOps, ConstOp
from tinygrad.helpers import dtypes

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
pend_regs:Set[Register] = set()
def compute_offsets(total):
  quotient, remainder = divmod(total, 4096)
  return [4096]*quotient + [remainder] if remainder else [4096]*quotient

#NOTE: Darwin needs lm functions to start with a "_" 
def get_op(op): return f"bl {'_' if system() == 'Darwin' else ''}{op}"
type_to_reg = {dtypes.half: 'h', dtypes.float32: 's', dtypes.bool: 'x',dtypes.int8:'w', dtypes.int32: 'w', dtypes.int64: 'x', dtypes.uint8:'w', dtypes.uint32: 'w', dtypes.uint64: 'x'}
class ARM64Codegen(AssemblyCodegen):
  def specialize(self, asm):
    rtor:Dict[Register, str] = {}
    var_size = 0
    prev_uop = None
    ins = [] 
    x_regs = ['x' + str(i) for i in reversed(range(29)) if i not in (9,10,11,12,13,14,15,16,17,18,19,20)]
    s_regs = ['s' + str(i) for i in reversed(range(3,30))]
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "", BinaryOps.CMPLT: "subs", BinaryOps.CMPEQ: "subs",
           UnaryOps.SIN: get_op('sinf'), UnaryOps.LOG2: get_op("log2f"), UnaryOps.EXP2: get_op("exp2f"), UnaryOps.SQRT: get_op("sqrtf"),
           TernaryOps.MULACC: "madd", TernaryOps.WHERE: "fcmp"}
    def mov_imm(value, to):
        # Manually move value into reg if vin[1] can't fit
        if value > 65535:
          ins.append(f"movz w15, #{value & 0xffff}")
          ins.append(f"movk w15, #{(value >> 16) & 0xffff}, lsl #16")
          ins.append(f"sxtw {to}, w15")
        else: 
          #NOTE: value comes as int when it should be float
          ins.append(f"mov x15, #{('0x' + float_to_hex(value)) if to[0] == 's' else value}")
          ins.append(f"{'f' if to[0] == 's' else ''}mov {to}, {'w' if to[0] == 's' else 'x'}15")

    # Get variables intervals
    live_range = {}
    for i, (uop, out, vin, arg) in enumerate(asm):
      for var in ([v for v in [out] + vin if v is not None and v.__class__ is not int]):
        live_range[var.nm] = [i,i] if var.nm not in live_range else [live_range[var.nm][0], i]

    temp_floats = ['s0', 's1', 's2']
    temp_ints = ['x11', 'x12', 'x13']
    
    mem_vars = {} 
    def allocate_regs(vars): 
      nonlocal var_size
      for v in [v for v in vars if v is not None and v.__class__ is not int and v.nm not in rtor]:
        available_regs = s_regs if dtypes.is_float(v[1]) else x_regs
        #NOTE: Very simple spill, everything that don't fit in regs goes to mem
        if len(available_regs) == 0:
          var_size += 16
          available_regs.append('s0' if dtypes.is_float(out[1]) else 'x11')
          mem_vars[v.nm] = f"[sp, #{var_size}]"
        rtor[v.nm] = available_regs.pop()

    for i, (uop, out, vin, arg) in enumerate(asm):
      # Clear regs out of interval
      for var, reg in list(rtor.items()):
        available_regs = s_regs if reg[0] == 's' else x_regs
        if var[1] != 'B' and var not in mem_vars and i > live_range[var][1]:
          available_regs.append(rtor.pop(var))
      # Assign a registers to the variables using live ranges.
      allocate_regs([out] + vin)

      for i, v in enumerate([v for v in vin if v.__class__ is not int and v.nm in mem_vars]):
        rtor[v.nm] = temp_floats[i] if dtypes.is_float(v[1]) else temp_ints[i] 
        ins.append(f"ldr {rtor[v.nm]}, {mem_vars[v.nm]}")

      if uop == UOps.DEFINE_GLOBAL:
        if arg.startswith('data'):
          # args 8 onward goes into the stack, so we move them into regs 
          if int(arg[4:]) >= 8:
            ins.append(f"ldr x15, [x19, #{(int(arg[4:]) - 8) * 8}]")
            ins.append(f"mov {rtor[out.nm]}, x15")
      elif uop == UOps.CAST:
        if arg == BinaryOps.CMPEQ:
          ins.append("mov x15, xzr")
          ins.append("cset w15, eq")
          ins.append(f"scvtf {rtor[out.nm]}, w15")
        else:
          ins.append(f"sxtw {rtor[out.nm]}, w{rtor[vin[0].nm][1:]}")
      elif uop == UOps.ALU:
        reg = 's' if dtypes.is_float(vin[0][1]) else 'x'
        if len(vin)==2 and vin[1].__class__ is int: mov_imm(vin[1], 'x15')
        if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
          ins.append(f"ands {rtor[out.nm]}, {rtor[vin[0].nm]}, {'x15' if vin[1].__class__ is int else rtor[vin[1].nm]}")
        elif arg == TernaryOps.WHERE:
          ins.append(f"fmov s0, #0")
          ins.append(f"{alu[arg]} {rtor[vin[0].nm]}, s0")
          ins.append(f"fcsel {rtor[out.nm]},{rtor[vin[2].nm]}, {rtor[vin[1].nm]}, eq")
        elif arg in [UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.EXP2, UnaryOps.SQRT]:
          save_regs = [k for k in rtor.keys() if k != out.nm and k not in mem_vars]
          ins.append(f"sub sp, sp, #{(len(save_regs))*16}")
          for i,k in enumerate(save_regs,1):
            ins.append(f"str {rtor[k]}, [sp, #{16*i}]")
          ins.append("stp x29, x30, [sp, #0]!")
          ins.append("mov x29, sp")
          ins.append(f"fmov s0, {rtor[vin[0].nm]}")
          ins.append(alu[arg])
          ins.append(f"fmov {rtor[out.nm]}, s0")
          ins.append("mov sp, x29")
          ins.append("ldp x29, x30, [sp], #0")
          for i,k in enumerate(save_regs,1):
            ins.append(f"ldr {rtor[k]}, [sp, #{16*i}]")
          ins.append(f"add sp, sp, #{len(save_regs)*16}")
        elif arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT]:
          ins.append(f"{alu[arg]} {rtor[out.nm]}, {rtor[vin[0].nm]}, {'x15' if vin[1].__class__ is int else rtor[vin[1].nm]}" if reg == 'x' else f"fcmp {rtor[vin[0].nm]}, {rtor[vin[1].nm]}")
        elif arg == BinaryOps.MOD:
          ins.append(f"udiv x14, {rtor[vin[0].nm]}, x15")
          ins.append(f"msub {rtor[out.nm]}, x14, x15, {rtor[vin[0].nm]}")
        else:
          ins.append(f"{'f' if reg == 's' else 's' if arg == BinaryOps.DIV else ''}{alu[arg]} {','.join('x15' if v.__class__ is int else rtor[v.nm] for v in [out] + vin)}")
      elif uop == UOps.LOAD:
        if arg.__class__ in (int, float):
          mov_imm(arg, rtor[out.nm])
        else:
          #NOTE: if need casting load var in s/h0 or x/w12 temp regs
          reg_in = type_to_reg[arg[2]] + ('0' if dtypes.is_float(arg[2]) else '12') if arg[2] is not None else rtor[out.nm]
          mov_imm(arg[0], "x15")
          ins.append(f"add x15, {rtor[vin[0].nm]}, x15")
          ins.append(f"ldr{'sb' if arg[2] is not None and arg[2] in (dtypes.int8, dtypes.uint8) else ''} {reg_in}, [x15]")
          if arg[2] is not None: ins.append(f"{'fcvt' if arg[2] == dtypes.half else 'scvtf'} {rtor[out.nm]}, {reg_in}")
      elif uop == UOps.STORE:
        shifts = {dtypes.int64: "#3", dtypes.half: "#1", dtypes.int8:"#2", dtypes.uint8: "#2"}
        #NOTE: if need casting load var in s/h0 or x/w12 temp regs
        reg_out = (type_to_reg[arg[2]] + ('0' if dtypes.is_float(arg[2]) else '12') if arg[2] is not None else rtor[vin[1].nm])
        if arg[2] is not None: ins.append(f"fcvt{'zs' if arg[2] != dtypes.half else '' } {reg_out}, {rtor[vin[1].nm]}")
        ins.append(f"mov x15, #{arg[0]}")
        ins.append(f"str {reg_out}, [{rtor[vin[0].nm]}, x15, lsl {shifts[arg[2]] if arg[2] is not None and arg[2] in shifts else '#0'}]")
      elif uop == UOps.COND_BRANCH:
        #TODO: this is a hack it shouldn't always be a cmp before a cond branch?
        if prev_uop == UOps.LOAD:
          ins.append(f"cmp {rtor[vin[0].nm]}, #0")
        ins.append(f"b.{'lt' if arg[1] == True else 'ge'} {arg[0][1:]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg[1:]}:")
      prev_uop=uop
      if out is not None and out.nm in mem_vars: ins.append(f"str {rtor[out.nm]}, {mem_vars[out.nm]}")
    return "test", "\n".join([".arch armv8-a",".text", ".global _test",".p2align 2", "_test:", "mov x19, sp"] + [f"sub sp, sp, #{offset}" for offset in compute_offsets(var_size)]+ ins  + [f"add sp, sp, #{offset}" for offset in compute_offsets(var_size)] +["ret;"])