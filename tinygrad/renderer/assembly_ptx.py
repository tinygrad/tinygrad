from typing import List
import struct
from tinygrad.renderer.assembly import uops_to_asmstyle, AssemblyLanguage
from tinygrad.codegen.linearizer import UOps, UOp, ConstOp, MemOp
from tinygrad.helpers import dtypes
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.runtime.ops_cuda import arch

dtype_to_nvtype = {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.int8: "s8", dtypes.bool: "pred", dtypes.uint64: "u64", dtypes.uint32: "u32",
                   dtypes.uint16: "u16", dtypes.uint8: "u8"}
def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

# https://docs.nvidia.com/cuda/parallel-thread-execution/#

class PTXLanguage(AssemblyLanguage):
  supports_constant_folding: bool = True

def specialize_to_ptx(lang, function_name, asm):
  ins = [".version 8.2", ".target " + arch(), ".address_size 64",
         f".visible .entry {function_name}({', '.join(f'.param .u64 data{i}' for i in range(lang.bufs_cnt))}) {{"]

  alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
         BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", BinaryOps.CMPEQ: "setp.eq", UnaryOps.SQRT: "sqrt.approx",
         UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
         TernaryOps.MULACC: "fma.rn"}

  for uop, out, vin, arg in asm:
    if uop == UOps.ENDLOOP:
      ins.append("bar.sync 0;")
    elif uop == UOps.DEFINE_REGISTER:
      ins.append(f".reg .{dtype_to_nvtype[arg[0][0]]} %{arg[1]}<{arg[2]}>;",)
    elif uop == UOps.DEFINE_LOCAL:
      ins.append(f".shared .align 4 .b8 {arg[0]}[{arg[1]*4}];")
    elif uop == UOps.SPECIAL:
      if arg.startswith('data'):
        ins.append(f"ld.param.u64 {out}, [{arg}];")
        # TODO: is this needed?
        ins.append(f"cvta.to.global.u64 {out}, {out};")
      elif arg.startswith('gid'):
        ins.append(f"mov.u32 {out}, %ctaid.{'xyz'[int(arg[3:])]};")
      elif arg.startswith('lid'):
        ins.append(f"mov.u32 {out}, %tid.{'xyz'[int(arg[3:])]};")
    elif uop == UOps.ALU:
      if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
        ins.append(f"and.pred {out}, {', '.join(str(x) for x in vin)};")
      else:
        otype = vin[0].dtype if arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT] else out.dtype
        ins.append(f"{alu[arg]}{'.lo' if arg == BinaryOps.MUL and out.dtype != dtypes.float32 else ''}{'.rn' if arg == BinaryOps.DIV and out.dtype == dtypes.float32 else ''}.{dtype_to_nvtype[otype]} {out}, {', '.join(str(x) for x in vin)};")
    elif uop == UOps.LOAD:
      if isinstance(arg, ConstOp):
        ins.append(f"mov.{dtype_to_nvtype[out.dtype]} {out}, {'0f'+float_to_hex(arg.value) if dtypes.is_float(out.dtype) else arg.value};")
      else: # memop
        # ins.append(f"ld.{arg[1]}.{dtype_to_nvtype[out.dtype]} {out}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
        if arg[2] == "bits16":
          ins.append(f"ld.{arg[1]}.b16 {out}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
        else:
          ins.append(f"ld.{arg[1]}.{dtype_to_nvtype[arg[2]]} {out}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
    elif uop == UOps.STORE:
      ins.append(f"st.{arg[1]}.{dtype_to_nvtype[arg[2]]} [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {vin[1]};")
    elif uop == UOps.CAST:
      if vin[0].dtype == dtypes.bool:
        ins.append(f"selp.{dtype_to_nvtype[out.dtype]} {out}, 0f3F800000, 0f00000000, {vin[0]};")
      # Rounding modifier is mandatory in all of the following cases:
      #    float-to-float conversions, when destination type is smaller than source type
      #    All float-to-int conversions
      #    All int-to-float conversions
      #    All conversions involving .f16x2, .e4m3x2, .e5m2x2,.bf16x2 and .tf32 instruction types.
      # These only differ in rounding modifier
      # FIXME: cleanup, add remaining rounding modifier cases
      elif (dtypes.is_int(out.dtype) and dtypes.is_float(vin[0].dtype)):
        ins.append(f"cvt.rzi.{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[vin[0].dtype]} {out}, {vin[0]};")
      elif dtypes.is_int(vin[0].dtype) and dtypes.is_float(out.dtype):
        ins.append(f"cvt.rz.{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[vin[0].dtype]} {out}, {vin[0]};")
      else:
        ins.append(f"cvt.{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[vin[0].dtype]} {out}, {vin[0]};")
    elif uop == UOps.LABEL:
      ins.append(f"{arg}:")
    elif uop == UOps.COND_BRANCH:
      ins.append(f"@{'!' if not arg[1] else ''}{vin[0]} bra {arg[0]};")

  ins += ["ret;", "}"]
  return '\n'.join(ins)
  # return ASTRunner(name, asm,
  #   global_size[::-1], local_size[::-1],
  #   op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True})

def uops_to_ptx_asm(function_name:str, uops:List[UOp]):
  lang = PTXLanguage()
  global_size, local_size = uops_to_asmstyle(lang, function_name, uops)
  return specialize_to_ptx(lang, function_name, lang.ins), global_size[::-1], local_size[::-1]
