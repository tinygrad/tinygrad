from typing import List
import struct
from tinygrad.renderer.assembly import uops_to_asmstyle, AssemblyLanguage
from tinygrad.codegen.linearizer import UOps, UOp, ConstOp
from tinygrad.helpers import dtypes
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.runtime.ops_cuda import arch

dtype_to_nvtype = {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.int8: "s8", dtypes.bool: "pred", dtypes.uint64: "u64", dtypes.uint32: "u32",
                   dtypes.uint16: "u16", dtypes.uint8: "u8", "bits16": "b16"}
def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

# https://docs.nvidia.com/cuda/parallel-thread-execution/#

class PTXLanguage(AssemblyLanguage):
  supports_constant_folding: bool = True

def specialize_to_ptx(lang, function_name, asm):
  ins = [".version 8.2", ".target " + arch(), ".address_size 64",
         f".visible .entry {function_name}({', '.join(f'.param .u64 data{i}' for i in range(len(lang.bufs)))}) {{"]

  alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
         BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", UnaryOps.SQRT: "sqrt.approx",
         UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
         TernaryOps.MULACC: "fma.rn", TernaryOps.WHERE: "selp"}

  for uop, out, vin, arg in asm:
    if uop == UOps.ENDLOOP:
      if arg in ["local", "global+local"]:
        ins.append("bar.sync 0;")
    elif uop == UOps.DEFINE_REGISTER:
      ins.append(f".reg .{dtype_to_nvtype[arg[0][0]]} %{arg[1]}<{arg[2]}>;",)
    elif uop == UOps.DEFINE_LOCAL:
      ins.append(f".shared .align 4 .b8 {arg[0]}[{arg[1]*4}];")
    elif uop == UOps.SPECIAL:
      if arg.startswith('data'):
        ins.append(f"ld.param.u64 {out}, [{arg}];")
        # TODO: we sometimes want this to be local, nvcc converts to global most of the time, not sure when we would need to?
        # ins.append(f"cvta.to.global.u64 {out}, {out};")
      elif arg.startswith('gid'):
        ins.append(f"mov.u32 {out}, %ctaid.{'xyz'[int(arg[3:])]};")
      elif arg.startswith('lid'):
        ins.append(f"mov.u32 {out}, %tid.{'xyz'[int(arg[3:])]};")
    elif uop == UOps.ALU:
      if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
        ins.append(f"and.pred {out}, {', '.join(str(x) for x in vin)};")
      else:
        otype = vin[0].dtype if arg in [BinaryOps.CMPLT] else out.dtype
        ins.append(f"{alu[arg]}{'.lo' if arg == BinaryOps.MUL and out.dtype != dtypes.float32 else ''}{'.rn' if arg == BinaryOps.DIV and out.dtype == dtypes.float32 else ''}.{dtype_to_nvtype[otype]} {out}, {', '.join(str(x) for x in vin)};")
    elif uop == UOps.LOAD:
      if isinstance(arg, ConstOp):
        ins.append(f"mov.{dtype_to_nvtype[out.dtype]} {out}, {'0f'+float_to_hex(arg.value) if dtypes.is_float(out.dtype) else arg.value};")
      else: # memop
        ins.append(f"ld.{arg[1]}.{dtype_to_nvtype[arg[2]]} {out}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
    elif uop == UOps.STORE:
      ins.append(f"st.{arg[1]}.{dtype_to_nvtype[arg[2]]} [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {vin[1]};")
    elif uop == UOps.CAST:
      if vin[0].dtype == dtypes.bool and (dtypes.is_float(out.dtype) or dtypes.is_int(out.dtype)):
        ins.append(f"selp.{dtype_to_nvtype[out.dtype]} {out}, {'0f3F800000, 0f00000000' if dtypes.is_float(out.dtype) else '1, 0'}, {vin[0]};")
      elif out.dtype == dtypes.bool:
        ins.append(f"setp.ne.{dtype_to_nvtype[vin[0].dtype]} {out}, {'0f00000000' if dtypes.is_float(vin[0].dtype) else '0'}, {vin[0]};")
      else:
        round_mod = ".rzi" if dtypes.is_int(out.dtype) and dtypes.is_float(vin[0].dtype) else '.rz' if dtypes.is_float(out.dtype) and (dtypes.is_int(vin[0].dtype) or dtypes.is_float(vin[0].dtype) and vin[0].dtype.itemsize > out.dtype.itemsize) else ''
        ins.append(f"cvt{round_mod}.{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[vin[0].dtype]} {out}, {vin[0]};")
    elif uop == UOps.LABEL:
      ins.append(f"{arg}:")
    elif uop == UOps.COND_BRANCH:
      ins.append(f"@{'!' if not arg[1] else ''}{vin[0]} bra {arg[0]};")

  ins += ["ret;", "}"]
  return '\n'.join(ins)

def uops_to_ptx_asm(function_name:str, uops:List[UOp]):
  lang = PTXLanguage()
  global_size, local_size = uops_to_asmstyle(lang, function_name, uops, is_ptx=True)
  return specialize_to_ptx(lang, function_name, lang.ins), global_size[::-1], local_size[::-1]
