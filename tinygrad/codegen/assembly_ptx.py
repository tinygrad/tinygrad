import struct
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import dtypes

dtype_to_nvtype = {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.bool: "pred", dtypes.uint64: "u64", dtypes.uint32: "u32", dtypes.uint8: 'u8', dtypes.int8: 's8'}
dtype_to_nvtypes = {dtypes.float32: "f32", dtypes.float16: "b16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.bool: "pred", dtypes.uint64: "u64", dtypes.uint32: "u32", dtypes.uint8: 'u8', dtypes.int8: 's8'}
def float_to_hex(x): return "0F%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

# https://docs.nvidia.com/cuda/parallel-thread-execution/#
class PTXCodegen(AssemblyCodegen):
  supports_constant_folding = True
  
  def specialize(self, asm):
    inst = [".version 7.8", ".target sm_86", ".address_size 64",
           f".visible .entry test({', '.join(f'.param .u64 buf{i}' for i in range(len(self.bufs)))}) {{"]

    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "rem", BinaryOps.CMPLT: "setp.lt", BinaryOps.CMPEQ: "setp.eq", UnaryOps.SQRT: "sqrt.approx",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin.approx", UnaryOps.LOG2: "lg2.approx", UnaryOps.EXP2: "ex2.approx.ftz",
           FusedOps.MULACC: "fma.rn"}

    def ins(x):
      inst.append(4*" " + x)

    old_parent = None
    for uop, out, vin, arg, parent in asm:
      if parent != old_parent:
        ins(f"// {str(parent.uop):10s}: {str(parent.out) if parent.out is not None else '':15s} {str(parent.vin):24s} {parent.arg}")
        old_parent = parent
      if uop == UOps.DEFINE_REGISTER:
        ins(f".reg .{dtype_to_nvtype[arg[0][0]]} %{arg[1]}<{arg[2]}>;",)
      elif uop == UOps.DEFINE_LOCAL:
        ins(f".shared .align 4 .b8 {arg[0]}[{arg[1]*4}];")
      elif uop == UOps.SPECIAL:
        if arg.startswith('buf'):
          ins(f"ld.param.u64 {out}, [{arg}];")
          # TODO: is this needed?
          #ins.append(f"cvta.to.global.u64 {out}, {out};")
        elif arg.startswith('gid'):
          ins(f"mov.u32 {out}, %ctaid.{'xyz'[int(arg[3:])]};")
        elif arg.startswith('lid'):
          ins(f"mov.u32 {out}, %tid.{'xyz'[int(arg[3:])]};")
      elif uop == UOps.ALU:
        if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
          ins(f"and.pred {out}, {', '.join(str(x) for x in vin)};")
        else:
          otype = vin[0].dtype if arg in [BinaryOps.CMPEQ, BinaryOps.CMPLT] else out.dtype
          ins(f"{alu[arg]}{'.lo' if arg == BinaryOps.MUL and out.dtype != dtypes.float32 else ''}{'.rn' if arg == BinaryOps.DIV and out.dtype == dtypes.float32 else ''}.{dtype_to_nvtype[otype]} {out}, {', '.join(str(x) for x in vin)};")
      elif uop == UOps.LOAD:
        
        ins(f"ld.{arg[1]}.{dtype_to_nvtypes[out.dtype]} {out}, [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}];")
      elif uop == UOps.STORE:
        # print(vin[1].dtype)
        print("store:", arg)
        ins(f"st.{arg[1]}.{dtype_to_nvtypes[vin[1].dtype]} [{vin[0]}{f'+{arg[0]}' if arg[0] is not None else ''}], {vin[1]};")
      elif uop == UOps.CAST:
        if vin[0].dtype == dtypes.bool:
          ins(f"selp.{dtype_to_nvtype[out.dtype]} {out}, 0f3F800000, 0f00000000, {vin[0]};")
        else:
          if   out.dtype == dtypes.float16 and vin[0].dtype == dtypes.float: mod = "rn." #f2f
          elif dtypes.is_integer(vin[0].dtype) and dtypes.is_integer(out.dtype): mod = ""
          elif vin[0].dtype.itemsize == 1: mod = "rn."
          elif vin[0].dtype.itemsize == 8: mod = "rz."
          elif dtypes.is_float(vin[0].dtype) ^ dtypes.is_float(out.dtype): mod = "rzi." # s2f
          else: mod = ""
          print(vin[0].dtype, "->", out.dtype)
          ins(f"cvt.{mod}{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[vin[0].dtype]} {out}, {vin[0]};")
          # ins.append(f"cvt.{dtype_to_nvtype[out.dtype]}.{dtype_to_nvtype[vin[0].dtype]} {out}, {vin[0]};")
      elif uop == UOps.CONST:
        print("[+] CONST UOP")
        ins(f"mov.{dtype_to_nvtype[out.dtype]} {out}, {float_to_hex(arg) if dtypes.is_float(out.dtype) else int(arg)};")
      elif uop == UOps.LABEL:
        ins(f"{arg}:")
      elif uop == UOps.COND_BRANCH:
        ins(f"@{'!' if not arg[1] else ''}{vin[0]} bra {arg[0]};")

    ins("bar.sync 0;")
    ins("ret;")
    inst += ["}"]
    return "test", '\n'.join(inst)
