from typing import List
import struct
from tinygrad.codegen.assembly import uops_to_asmstyle, AssemblyLanguage
from tinygrad.codegen.linearizer import UOps, UOp, ConstOp
from tinygrad.helpers import dtypes
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.runtime.ops_cuda import arch

dtype_to_nvtype = {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.int8: "s8", dtypes.bool: "pred", dtypes.uint64: "u64", dtypes.uint32: "u32",
                   dtypes.uint16: "u16", dtypes.uint8: "u8", "bits16": "b16"}
def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

def ptx_needs_cast(dest_dtype, src_dtype): return dtypes.is_float(dest_dtype) and dtypes.is_int(src_dtype) or dtypes.is_int(dest_dtype) and dtypes.is_float(src_dtype) or (dtypes.is_float(src_dtype) and dtypes.is_float(dest_dtype) and dest_dtype.itemsize < src_dtype.itemsize)

def render_cast(ins, inp, out):
  print("start")
  if inp.dtype == dtypes.bool and (dtypes.is_float(out.dtype) or dtypes.is_int(out.dtype)):
    print("a")
    ins.append(f"selp.{dtype_to_nvtype[out.dtype]} {out}, {'0f3F800000, 0f00000000' if dtypes.is_float(out.dtype) else '1, 0'}, {inp};")
  elif out.dtype == dtypes.bool:
    print("b")
    ins.append(f"setp.ne.{dtype_to_nvtype[inp.dtype]} {out}, {'0f00000000' if dtypes.is_float(inp.dtype) else '0'}, {inp};")
  else:
    print("c")
    round_mod = ".rzi" if dtypes.is_int(out.dtype) and dtypes.is_float(inp.dtype) else '.rz' if dtypes.is_float(out.dtype) and (dtypes.is_int(inp.dtype) or dtypes.is_float(inp.dtype) and inp.dtype.itemsize > out.dtype.itemsize) else ''
    ins.append(f"cvt{round_mod}.{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[inp.dtype]} {out}, {inp};")


# https://docs.nvidia.com/cuda/parallel-thread-execution/#

class PTXLanguage(AssemblyLanguage):
  supports_constant_folding: bool = True

def specialize_to_ptx(lang, function_name):
  param_cnt = 0
  registers = []
  ins = []
  alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
         BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", UnaryOps.SQRT: "sqrt.approx",
         UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
         TernaryOps.MULACC: "fma.rn", TernaryOps.WHERE: "selp"}
  for uop, out, vin, arg in lang.ins:
    if uop == UOps.ENDLOOP:
      ins.append("bar.sync 0;")
    elif uop == UOps.DEFINE_REGISTER:
      registers += [arg]
      # ins.append(f".reg .{dtype_to_nvtype[arg[0][0]]} %{arg[1]}<{arg[2]}>;",)
    elif uop == UOps.DEFINE_LOCAL:
      ins.append(f".shared .align 4 .b8 {arg[0]}[{arg[1]*4}];")
    elif uop == UOps.SPECIAL:
      if arg.startswith('data'):
        param_cnt += 1
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
        if arg == TernaryOps.WHERE:
          reg = lang.newreg((vin[0], 'bool'), dtypes.bool)
          ins.append(f"setp.ne.{dtype_to_nvtype[vin[0].dtype]} {reg}, {'0f00000000' if dtypes.is_float(vin[0].dtype) else '0'}, {vin[0]};")
          vin = vin[1:] + [reg]
        ins.append(f"{alu[arg]}{'.lo' if arg == BinaryOps.MUL and out.dtype != dtypes.float32 else ''}{'.rn' if arg == BinaryOps.DIV and out.dtype == dtypes.float32 else ''}.{dtype_to_nvtype[otype]} {out}, {', '.join(str(x) for x in vin)};")
    elif uop == UOps.LOAD:
      if arg.__class__ in (int, float):
        ins.append(f"mov.{dtype_to_nvtype[out.dtype]} {out}, {'0f'+float_to_hex(arg) if dtypes.is_float(out.dtype) else int(arg)};")
      elif arg[2] is not None and (arg[2] == dtypes.bool or arg[2] != out.dtype):
        # FIXME: combine
        if arg[2] == dtypes.bool == out.dtype:
          reg = lang.newreg((out, 'u16'), dtype=dtypes.uint16)
          ins.append(f"ld.{arg[1]}.u16 {reg}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
          ins.append(f"setp.ne.u16 {out}, 0, {reg};")
        elif arg[2] == dtypes.bool:
          reg = lang.newreg((out, 'b8'), dtype=dtypes.uint8)
          ins.append(f"ld.{arg[1]}.u8 {reg}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
          ins.append(f"cvt.rz.f32.u8 {out}, {reg};")
        elif arg[2] == dtypes.half:
          reg = lang.newreg((out, 'ld'), dtype=arg[2])
          ins.append(f"ld.{arg[1]}.b16 {reg}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
          ins.append(f"cvt.f32.{dtype_to_nvtype[arg[2]]} {out}, {reg};")
        else:
          reg = lang.newreg((out, 'ld'), dtype=arg[2])
          ins.append(f"ld.{arg[1]}.{dtype_to_nvtype[arg[2]]} {reg}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
          ins.append(f"cvt.rz.f32.{dtype_to_nvtype[arg[2]]} {out}, {reg};")
      else:
        ins.append(f"ld.{arg[1]}.{dtype_to_nvtype[dtypes.float if arg[2] is None else arg[2]]} {out}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
    elif uop == UOps.STORE:
      if arg[2] != None and arg[2] != vin[1].dtype:
        if arg[2] == dtypes.bool:
          reg = lang.newreg((out, 'u32'), dtype=dtypes.uint32)
          ins.append(f"set.ne.u32.{dtype_to_nvtype[vin[1].dtype]} {reg}, {'0f00000000' if dtypes.is_float(vin[1].dtype) else '0'}, {vin[1]};")
          ins.append(f"st.{arg[1]}.b8 [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {reg};")
        elif arg[2] == dtypes.half:
          reg = lang.newreg((out, 'st'), dtype=dtypes.float16)
          ins.append(f"cvt.rz.{dtype_to_nvtype[arg[2]]}.{dtype_to_nvtype[vin[1].dtype]} {reg}, {vin[1]};")
          ins.append(f"st.{arg[1]}.b16 [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {reg};")
        else:
          reg = lang.newreg((out, 'st'), dtype=arg[2])
          ins.append(f"cvt.rzi.{dtype_to_nvtype[arg[2]]}.{dtype_to_nvtype[vin[1].dtype]} {reg}, {vin[1]};")
          ins.append(f"st.{arg[1]}.{dtype_to_nvtype[arg[2]]} [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {reg};")
      elif arg[2] == dtypes.bool:
        reg = lang.newreg((out, 'st'), dtype=dtypes.uint16)
        print(reg, vin[1])
        ins.append(f"selp.u16 {reg}, 1, 0, {vin[1]};")
        ins.append(f"st.{arg[1]}.b8 [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {reg};")
      # if ptx_needs_cast(dtypes.float if arg[2] is None else arg[2], vin[1].dtype) or arg[2] == dtypes.bool:
      #   if arg[2] == dtypes.bool != vin[1].dtype:
      #     prereg = lang.newreg((vin[1],'bool'), dtype=dtypes.bool)
      #     render_cast(ins, vin[1], prereg)
      #   else: prereg = vin[1]
      #   reg = lang.newreg((prereg, dtypes.uint16 if arg[2] == dtypes.bool else arg[2]), dtype=dtypes.uint16 if arg[2] == dtypes.bool else dtypes.float if arg[2] is None else arg[2])
      #   render_cast(ins, reg, prereg)
      #   ins.append(f"st.{arg[1]}.{dtype_to_nvtype['bits16' if arg[2] == dtypes.float16 else dtypes.uint8 if arg[2] == dtypes.bool else dtypes.float if arg[2] is None else arg[2]]} [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {reg};")
      else:
        ins.append(f"st.{arg[1]}.{dtype_to_nvtype[dtypes.float if arg[2] is None else arg[2]]} [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {vin[1]};")
    elif uop == UOps.CAST:
      render_cast(ins, vin[0], out)
    elif uop == UOps.LABEL:
      ins.append(f"{arg}:")
    elif uop == UOps.COND_BRANCH:
      ins.append(f"@{'!' if not arg[1] else ''}{vin[0]} bra {arg[0]};")

  ins_prefix = [".version 8.2", ".target " + arch(), ".address_size 64",
                f".visible .entry {function_name}({', '.join(f'.param .u64 data{i}' for i in range(param_cnt))}) {{"]

  for arg in [(dtype, lang.type_to_letter(dtype), c) for dtype,c in lang.cnts.items()]: ins_prefix.append(f".reg .{dtype_to_nvtype[arg[0][0]]} %{arg[1]}<{arg[2]}>;",)
  ins = ins_prefix + ins

  ins += ["ret;", "}"]
  return '\n'.join(ins)

def uops_to_ptx_asm(function_name:str, uops:List[UOp]):
  lang = PTXLanguage()
  global_size, local_size = uops_to_asmstyle(lang, function_name, uops)
  return specialize_to_ptx(lang, function_name), global_size[::-1], local_size[::-1], True
