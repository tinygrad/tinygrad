import struct
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

dtype_to_nvtype = {dtypes.float32: "f32", dtypes.float16: "u16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.bool: "pred", dtypes.uint64: "u64", dtypes.uint32: "u32"}
def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

# https://docs.nvidia.com/cuda/parallel-thread-execution/#
class PTXCodegen(AssemblyCodegen):
  #supports_constant_folding: bool = True

  def specialize(self, asm):
    ins = [".version 7.8", ".target sm_86", ".address_size 64",
           f".visible .entry test({', '.join(f'.param .u64 buf{i}' for i in range(len(self.bufs)))}) {{"]

    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", BinaryOps.CMPEQ: "setp.eq",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fma.rn"}

    for uop, out, vin, arg in asm:
      if uop == UOps.DEFINE_REGISTER:
        ins.append(f".reg .{dtype_to_nvtype[arg[0]]} %{arg[1]}<{arg[2]}>;",)
      elif uop == UOps.DEFINE_LOCAL:
        ins.append(f".shared .align 4 .b8 {arg[0]}[{arg[1]*4}];")
      elif uop == UOps.SPECIAL:
        if arg.startswith('buf'):
          ins.append(f"ld.param.u64 {out}, [{arg}];")
          # TODO: is this needed?
          #ins.append(f"cvta.to.global.u64 {out}, {out};")
        elif arg.startswith('gid'):
          #ins.append(f"mov.u32 {out}, %ctaid.{'xyz'[int(arg[3:])]};")
          ins.append("{ .reg .b32 %tmp<3>;")
          l = 'xyz'[int(arg[3:])]
          ins.append(f"mov.u32 %tmp0, %ctaid.{l};")
          ins.append(f"mov.u32 %tmp1, %ntid.{l};")
          ins.append(f"mov.u32 %tmp2, %tid.{l};")
          ins.append(f"mad.lo.s32 {out}, %tmp0, %tmp1, %tmp2; }}")
        elif arg.startswith('lid'):
          ins.append(f"mov.u32 {out}, %tid.{'xyz'[int(arg[3:])]};")
      elif uop == UOps.ALU:
        if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
          ins.append(f"and.pred {out}, {', '.join(str(x) for x in vin)};")
        else:
          otype = vin[0].dtype if arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT] else out.dtype
          ins.append(f"{alu[arg]}{'.lo' if arg == BinaryOps.MUL and out.dtype != dtypes.float32 else ''}{'.rn' if arg == BinaryOps.DIV and out.dtype == dtypes.float32 else ''}.{dtype_to_nvtype[otype]} {out}, {', '.join(str(x) for x in vin)};")
      elif uop == UOps.LOAD:
        ins.append(f"ld.{arg[1]}.{dtype_to_nvtype[out.dtype]} {out}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
      elif uop == UOps.STORE:
        ins.append(f"st.{arg[1]}.{dtype_to_nvtype[vin[1].dtype]} [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {vin[1]};")
      elif uop == UOps.CAST:
        if vin[0].dtype == dtypes.bool:
          ins.append(f"selp.{dtype_to_nvtype[out.dtype]} {out}, 0f3F800000, 0f00000000, {vin[0]};")
        else:
          ins.append(f"cvt.{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[vin[0].dtype]} {out}, {vin[0]};")
      elif uop == UOps.CONST:
        ins.append(f"mov.{dtype_to_nvtype[out.dtype]} {out}, {'0f'+float_to_hex(arg) if dtypes.is_float(out.dtype) else arg};")
      elif uop == UOps.LABEL:
        ins.append(f"{arg}:")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"@{'!' if not arg[1] else ''}{vin[0]} bra {arg[0]};")

    ins += ["ret;", "}"]
    return "test", '\n'.join(ins)
