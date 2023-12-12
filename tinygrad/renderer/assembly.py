from typing import Callable, DefaultDict, Dict, List, Tuple, Union, NamedTuple
import functools, struct
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.helpers import dtypes, DType, PtrDType

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
def double_to_hex(x): return "%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])

class AssemblyLanguage(NamedTuple):
  kernel_prefix: str = ""
  barrier: str = ""
  gid: List[str] = []
  gdim: List[str] = []
  lid: List[str] = []
  supports_half: List[Op] = []
  asm_for_op: Dict[Op, Callable[...,str]] = {}
  dtype_to_asmtype: Dict[DType, str] = {}

  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    if dtypes.is_float(var_dtype): return f"0f{float_to_hex(x)}" if var_dtype == dtypes.float32 else f"0d{double_to_hex(x)}"
    return str(int(x)) + ("U" if dtypes.is_unsigned(var_dtype) else "")

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> str:
    if bitcast: return f"mov.b{self.dtype_to_asmtype[dtype][1:]} {d}, {a};"
    no_round = ((dtypes.is_int(dtype) or dtype == dtypes.bool) and (dtypes.is_int(atype) or atype == dtypes.bool)) or ((dtypes.is_float(dtype) and dtypes.is_float(atype)) and dtype.itemsize >= atype.itemsize)
    return f"cvt{'' if no_round else '.rn' if dtypes.is_float(dtype) else '.rzi'}.{self.dtype_to_asmtype[dtype]}.{self.dtype_to_asmtype[atype]} {d}, {a};"

def uops_to_asm(lang:AssemblyLanguage, function_name:str, uops:List[UOp]) -> Tuple[str, Dict]:
  local_size: List[int] = []
  kernel:List[str] = []
  bufs = []

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, str] = {}
  def ssa(u, prefix="t", dtype=None):
    nonlocal c, r
    prefix += f"_{dtype if dtype else lang.dtype_to_asmtype[u.dtype]}_"
    c[prefix] += 1
    if u: r[u] = f"%{prefix}{c[prefix]-1}"
    return f"%{prefix}{c[prefix]-1}"

  c_label: DefaultDict[str, int] = defaultdict(int)
  r_label: Dict[UOp, str] = {}
  def ssa_label(u, prefix):
    nonlocal c_label, r_label
    c_label[prefix] += 1
    r_label[u] = f"${prefix}_{c_label[prefix]-1}"
    return r_label[u]

  def const(x:Union[float,int,bool], var_dtype, force_mov=False):
    nonlocal kernel
    if var_dtype == dtypes.half:
      kernel.extend([f"mov.f32 {(tmp:=ssa(None, 'const', 'f32'))}, {lang.render_const(x, dtypes.float)};", f"cvt.rn.f16.f32 {(out:=ssa(None, 'cast', 'b16'))}, {tmp};"])
      return out
    if force_mov:
      kernel.append(f"mov.{lang.dtype_to_asmtype[var_dtype]} {(out:=ssa(None, 'const', lang.dtype_to_asmtype[var_dtype]))}, {lang.render_const(x, var_dtype)};")
      return out
    return lang.render_const(x, var_dtype)

  def cast(a:str, dtype:DType, atype:DType, bitcast=False, u=None, pred=False):
    nonlocal kernel
    if "pred" in a: kernel.append(f"selp.{lang.dtype_to_asmtype[dtype]} {(ret:=ssa(u, 'cast', lang.dtype_to_asmtype[dtype]))}, {const(1, dtype)}, {const(0, dtype)}, {a};")
    elif pred: kernel.append(f"setp.ne.{'b16' if atype == dtypes.half else lang.dtype_to_asmtype[atype]} {(ret := ssa(u, 'cast', 'pred'))}, {a}, {const(0, atype)};")
    else: kernel.append(lang.render_cast((ret := ssa(u, 'cast', lang.dtype_to_asmtype[dtype])), a, dtype, atype, bitcast))
    return ret

  def addr(addr, offset, itemsize):
    nonlocal kernel
    if offset and offset != UOps.CONST:
      kernel.append(f"mad.wide.u32 {(loc:=ssa(None,'loc','u64'))}, {cast(r[offset], dtypes.uint32, offset.dtype)}, {const(itemsize, dtypes.uint32, force_mov=True)}, {r[addr]};")
      return f"[{loc}]"
    return f"[{r[offset]}{f'+{int(offset.arg * itemsize)}' if offset else ''}]"

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    if uop == UOps.LOOP: kernel.extend([f"mov.u32 {ssa(u, 'ridx')}, {r[vin[0]]};", f"{ssa_label(u, 'loop')}:"])
    elif uop == UOps.IF:
      assert vin[0].dtype is not None
      kernel.append(f"@!{cast(r[vin[0]], dtypes.bool, vin[0].dtype, u=u, pred=True)} bra {ssa_label(u, 'if')};")
    elif uop == UOps.END:
      if vin[0].uop == UOps.LOOP: kernel.extend([f"add.s32 {r[vin[0]]}, {r[vin[0]]}, 1;", f"setp.ne.s32 {ssa(u, 'p', 'pred')}, {r[vin[0]]}, {r[vin[0].vin[1]]};", f"@{r[u]} bra {r_label[vin[0]]};"])
      else: kernel.append(f"{r_label[vin[0]]}:")
    elif uop == UOps.BARRIER: kernel.append(lang.barrier)
    elif uop == UOps.ALU:
      assert dtype is not None and vin[0].dtype is not None
      if vin[0].dtype == dtypes.half and args not in lang.supports_half:
        kernel.append(lang.asm_for_op[args]((tmp:=ssa(None, "alu_cast", "f32")), *[cast(r[x], dtypes.float32, dtypes.half) for x in vin], lang.dtype_to_asmtype[dtypes.float32]))
        cast(tmp, dtypes.half, dtypes.float32, u=u)
      elif args == BinaryOps.CMPLT:
        kernel.append(lang.asm_for_op[args](pred:=ssa(None,'lt','pred'), *[r[x] for x in vin], lang.dtype_to_asmtype[vin[0].dtype]))
        cast(pred, dtype, dtypes.bool, u=u)
      elif args == TernaryOps.WHERE: kernel.append(lang.asm_for_op[args](ssa(u, "alu"), cast(r[vin[0]], dtypes.bool, vin[0].dtype, pred=True), r[vin[1]], r[vin[2]], lang.dtype_to_asmtype[dtype]))
      else: kernel.append(lang.asm_for_op[args](ssa(u, "alu"), *[r[x] for x in vin], lang.dtype_to_asmtype[dtype]))
    elif uop == UOps.DEFINE_ACC:
      assert dtype is not None
      kernel.append(f"mov.b{lang.dtype_to_asmtype[dtype][1:]} {ssa(u, 'acc')}, {const(args, dtype)};")
    elif uop == UOps.SPECIAL:
      if args[1].startswith("i"): kernel.extend([f"mov.u32 %{args[1]}, {lang.gid[args[0]]};", f"mov.u32 {(gdim:=ssa(None,'tmp','u32'))}, {lang.gdim[args[0]]};",
                                                 f"mov.u32 {(lid:=ssa(None,'tmp','u32'))}, {lang.lid[args[0]]};", f"mad.lo.u32 %{args[1]}, %{args[1]}, {gdim}, {lid};"])
      else: kernel.append(f"mov.u32 %{args[1]}, {(lang.gid if args[1].startswith('g') else lang.lid)[args[0]]};")
      if args[1].startswith("l"): local_size.append(args[2])
      r[u] = "%" + args[1]
      kernel = [f".reg .u32 %{args[1]};"] + kernel
    elif uop == UOps.CONST: r[u] = const(args, dtype, force_mov=True)
    elif uop == UOps.LOAD:
      assert dtype is not None and vin[1].dtype is not None
      loc = addr(vin[0], vin[1], dtype.itemsize)
      val = ssa(u, 'val')
      if dtype.itemsize == 1: tmp = ssa(None, 'tmp', 's8')
      if len(vin) > 3:
        assert vin[2].dtype is not None
        pred = cast(r[vin[2]], dtypes.bool, vin[2].dtype, pred=True)
      kernel.append(f"{f'@{pred} ' if len(vin) > 3 else ''}ld{'.shared' if vin[0].uop == UOps.DEFINE_LOCAL else ''}.{'s8' if dtype.itemsize == 1 else 'b16' if dtype == dtypes.float16 else lang.dtype_to_asmtype[dtype]} {tmp if dtype.itemsize == 1 else val}, {loc};")
      if len(vin) > 3: kernel.append(f"@!{pred} mov.b{'8' if dtype.itemsize == 1 else lang.dtype_to_asmtype[dtype][1:]} {tmp if dtype.itemsize == 1 else val}, {r[vin[3]]};")
      if dtype.itemsize == 1: kernel.append(f"cvt.{lang.dtype_to_asmtype[dtype]}.s8 {val}, {tmp};")
    elif uop == UOps.PHI:
      assert dtype is not None
      kernel.append(f"mov.b{lang.dtype_to_asmtype[dtype][1:]} {r[vin[0]]}, {r[vin[1]]};")
      r[u] = r[vin[0]]
    elif uop == UOps.STORE:
      assert vin[0].dtype is not None and vin[1].dtype is not None and vin[2].dtype is not None
      loc = addr(vin[0], vin[1], vin[0].dtype.itemsize)
      if len(vin) > 3:
        assert vin[3].dtype is not None
        pred = cast(r[vin[3]], dtypes.bool, vin[3].dtype, pred=True)
      kernel.append(f"{f'@{pred} ' if len(vin) > 3 else ''}st{'.shared' if vin[0].uop == UOps.DEFINE_LOCAL else ''}.{'s8' if vin[0].dtype.itemsize == 1 else 'b16' if vin[0].dtype == dtypes.float16 else lang.dtype_to_asmtype[vin[0].dtype]} {loc}, {r[vin[2]] if vin[0].dtype == vin[2].dtype else cast(r[vin[2]], vin[0].dtype, vin[2].dtype)};")
    elif uop == UOps.CAST and dtype is not None:
      assert vin[0].dtype is not None
      if dtype == dtypes.bool: cast(cast(r[vin[0]], dtypes.bool, vin[0].dtype, pred=True), dtypes.bool, dtypes.bool, u=u)
      else: cast(r[vin[0]], dtype, vin[0].dtype, bitcast=isinstance(args, tuple) and args[1], u=u)
    elif uop == UOps.DEFINE_LOCAL:
      assert dtype is not None
      kernel.extend([f".shared .align 4 .b8 {args[0]}[{args[1]*dtype.itemsize}];", f"mov.u64 {ssa(u, 'local', 'u64')}, {args[0]}[0];"])
    elif uop == UOps.DEFINE_GLOBAL:
      assert dtype is not None
      bufs.append((args[0], dtype))
      kernel.append(f"ld.param.{'u64' if dtype.__class__ == PtrDType else lang.dtype_to_asmtype[dtype]} {ssa(u, 'dat', dtype='u64' if dtype.__class__ == PtrDType else lang.dtype_to_asmtype[dtype])}, [{args[0]}];")
    else: raise NotImplementedError(f"no code for {uop}")

  kernel = [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in c.items()] + kernel + ["ret;"]

  ret = f"{lang.kernel_prefix} {function_name}(\n\t" + ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else lang.dtype_to_asmtype[dtype]} {name}" for name,dtype in bufs])
  return ret + "\n)\n{\n" + '\n'.join([line if line.startswith("$") else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1) for line in kernel]) + "\n}", {}

class PTXLanguage(AssemblyLanguage):
  kernel_prefix = """.version 7.8
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  gid = [f'%ctaid.{chr(120+i)}' for i in range(3)]
  gdim = [f'%nctaid.{chr(120+i)}' for i in range(3)]
  lid = [f'%tid.{chr(120+i)}' for i in range(3)]
  asm_for_op = {
    UnaryOps.NEG: lambda d,a,dtype: f"neg.{dtype} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dtype: f"ex2.approx.{dtype} {d}, {a};", UnaryOps.LOG2: lambda d,a,dtype: f"lg2.approx.{dtype} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dtype: f"sin.approx.{dtype} {d}, {a};", UnaryOps.SQRT: lambda d,a,dtype: f"sqrt.approx.{dtype} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dtype: f"{'or' if dtype == 'pred' else 'add'}.{dtype} {d}, {a}, {b};", BinaryOps.SUB: lambda d,a,b,dtype: f"sub.{dtype} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dtype: f"{'and' if dtype == 'pred' else 'mul'}{'' if dtype.startswith('f') or dtype == 'pred' else '.lo'}.{dtype} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dtype: f"xor.b{dtype[1:]} {d}, {a}, {b};", BinaryOps.DIV: lambda d,a,b,dtype: f"div{'.approx' if dtype.startswith('f') else ''}.{dtype} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dtype: f"max.{dtype} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dtype: f"rem.{dtype} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,stype: f"setp.lt.{stype} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dtype: f"{'fma.rn' if dtype.startswith('f') else 'mad.lo'}.{dtype} {d}, {a}, {b}, {c};",
    TernaryOps.WHERE: lambda d,a,b,c,dtype: f"selp.{dtype} {d}, {b}, {c}, {a};"
  }
  supports_half = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT, TernaryOps.MULACC, TernaryOps.WHERE]
  dtype_to_asmtype = {
    dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
    dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
    dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64",
    dtypes.bool: "u32", dtypes._arg_int32: "u32"
  }
PTXRenderer = functools.partial(uops_to_asm, PTXLanguage())
