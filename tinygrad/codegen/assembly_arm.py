import struct
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

dtype_to_nvtype = {dtypes.float32: "f32", dtypes.float16: "u16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.bool: "pred", dtypes.uint64: "u64", dtypes.uint32: "u32"}
def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

class ARMCodegen(AssemblyCodegen):
# AssemblyInstruction(op=<UOps.DEFINE_REGISTER: 10>, out=None, vin=[], arg=(dtypes.uint64, 'A', 7))
# AssemblyInstruction(op=<UOps.DEFINE_REGISTER: 10>, out=None, vin=[], arg=(dtypes.int, 'i', 2))
# AssemblyInstruction(op=<UOps.DEFINE_REGISTER: 10>, out=None, vin=[], arg=(dtypes.float, 'f', 3))
# AssemblyInstruction(op=<UOps.SPECIAL: 9>, out=%A0, vin=[], arg='buf0')
# AssemblyInstruction(op=<UOps.SPECIAL: 9>, out=%A1, vin=[], arg='buf1')
# AssemblyInstruction(op=<UOps.SPECIAL: 9>, out=%A2, vin=[], arg='buf2')
# AssemblyInstruction(op=<UOps.SPECIAL: 9>, out=%i0, vin=[], arg='gid0')
# AssemblyInstruction(op=<UOps.CONST: 5>, out=%i1, vin=[], arg=0)
# AssemblyInstruction(op=<UOps.CAST: 8>, out=%A3, vin=[%i1], arg=None)
# AssemblyInstruction(op=<UOps.ALU: 4>, out=%A4, vin=[%A3, %A1], arg=<BinaryOps.ADD: 1>)
# AssemblyInstruction(op=<UOps.LOAD: 3>, out=%f0, vin=[%A4], arg=(0, 'global'))
# AssemblyInstruction(op=<UOps.ALU: 4>, out=%A5, vin=[%A3, %A2], arg=<BinaryOps.ADD: 1>)
# AssemblyInstruction(op=<UOps.LOAD: 3>, out=%f1, vin=[%A5], arg=(0, 'global'))
# AssemblyInstruction(op=<UOps.ALU: 4>, out=%f2, vin=[%f0, %f1], arg=<BinaryOps.ADD: 1>)
# AssemblyInstruction(op=<UOps.ALU: 4>, out=%A6, vin=[%A3, %A0], arg=<BinaryOps.ADD: 1>)
# AssemblyInstruction(op=<UOps.STORE: 7>, out=None, vin=[%A6, %f2], arg=(0, 'global'))

#         ldr     s0, [x1]
#         ldr     s1, [x2]
#         fadd    s0, s0, s1
#         str     s0, [x0]
#         ret

  def specialize(self, asm):
    ins = [".global _Z4testPfS_S_","_Z4testPfS_S_:"] 


    alu = {BinaryOps.ADD: "fadd", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", BinaryOps.CMPEQ: "setp.eq",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fma.rn"}
    def parse_reg (x): 
        return  f"{'x' if x[1].lower() in ['a','i'] else 's'}{x[2:]}" 
    arm_regs = {}
    for i, (uop, out, vin, arg) in enumerate(asm):
        print(asm[i])
        if uop == UOps.SPECIAL:
            if arg.startswith('buf'):
                i = int(arg[3:])
                arm_regs[out[0]] = (f"x{i}", out[1])
        elif uop == UOps.ALU:
            ins.append(f"{alu[arg]} {parse_reg(out[0])}, {', '.join(str(parse_reg(x[0])) for x in vin)};")
        elif uop == UOps.CONST:
            ins.append(f"mov {parse_reg(out[0])}, {arg};")
        elif uop == UOps.LOAD:
            ins.append(f"ldr {parse_reg(out[0])}, [{parse_reg(vin[0][0])}]")
        elif uop== UOps.CAST:
            ins.append(f"mov {parse_reg(out[0])}, {parse_reg(vin[0][0])};")
        elif uop == UOps.STORE:
            ins.append(f"str {parse_reg(vin[0][0])}, {parse_reg(vin[1][0])}")
        


    ins += ["ret;"]
    return "test", '\n'.join(ins)
