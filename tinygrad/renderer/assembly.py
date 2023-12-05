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
  lid: List[str] = []
  asm_for_op: Dict[Op, Callable[...,str]] = {}
  dtype_to_asmtype: Dict[DType, str] = {}

  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    if dtypes.is_float(var_dtype): return f"0f{float_to_hex(x)}" if var_dtype == dtypes.float32 else f"0d{double_to_hex(x)}" 
    if dtypes.is_int(var_dtype): return str(int(x)) + ("U" if dtypes.is_unsigned(var_dtype) else "")
    return "1" if x else "0"

def uops_to_asm(lang:AssemblyLanguage, function_name:str, uops:List[UOp]) -> Tuple[str, Dict]:
  local_size: List[int] = []
  kernel, bufs = [], []
  label_count = 0
  labels = []

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, str] = {}
  regs: List[Tuple[str, DType]] = []
  def ssa(u, prefix="t"):
    nonlocal c, r, regs
    c[prefix] += 1
    r[u] = f"%{prefix}{c[prefix]-1}"
    regs.append((r[u], dtypes.bool if prefix == 'p' else (dtypes.uint64 if u.dtype.__class__ == PtrDType else u.dtype)))
    return r[u]

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    if uop == UOps.LOOP:
      kernel.append(f"mov.u32 {ssa(u, 'ridx')}, {r[vin[0]]};")
      kernel.append(f"$loop_{label_count}:")
      labels.append(label_count)
      label_count += 1
    elif uop == UOps.IF:
      kernel.append(f"@!{r[vin[0]]} bra $if_{label_count};")
      labels.append(label_count)
      label_count += 1
    elif uop == UOps.END:
      if vin[0].uop == UOps.LOOP:
        kernel.append(f"add.s32 {r[vin[0]]}, {r[vin[0]]}, 1;")
        kernel.append(f"setp.ne.s32 {ssa(u, 'p')}, {r[vin[0]]}, {r[vin[0].vin[1]]};")
        kernel.append(f"@{r[u]} bra $loop_{labels.pop()};")
      else:
        kernel.append(f"$if_{labels.pop()}:")
    elif uop == UOps.BARRIER: kernel.append(lang.barrier)
    elif uop == UOps.ALU:
      if args == BinaryOps.CMPLT: kernel.append(lang.asm_for_op[args](ssa(u, "p"), *[r[x] for x in vin], lang.dtype_to_asmtype[dtype], lang.dtype_to_asmtype[vin[0].dtype]))
      else: kernel.append(lang.asm_for_op[args](ssa(u, "alu"), *[r[x] for x in vin], lang.dtype_to_asmtype[dtype]))
    elif uop == UOps.DEFINE_ACC: kernel.append(f"mov.{lang.dtype_to_asmtype[dtype]} {ssa(u, 'acc')}, {lang.render_const(args, dtype)};")
    elif uop == UOps.SPECIAL:
      assert not args[1].startswith("i"), "no xid in ptx"
      xid = lang.gid if args[1].startswith("g") else lang.lid
      kernel.append(f"mov.u32 %{args[1]}, {xid[args[0]]};")
      if args[1].startswith("l"): local_size.append(args[2])
      r[u] = "%" + args[1]
      regs.append(("%" + args[1], dtypes.uint32))
    elif uop == UOps.CONST:
      r[u] = lang.render_const(args, dtype)
    elif uop == UOps.LOAD:
      if vin[0].uop == UOps.DEFINE_LOCAL:
        index = f"%index{c['index']}"
        kernel.append(f"mul.wide.{lang.dtype_to_asmtype[vin[1].dtype]} {index}, {r[vin[1]]}, {dtype.itemsize};")
        loc = f"{r[vin[0]]}[{index}]"
        regs += [(index, dtypes.uint64)]
        c["index"] += 1
      elif vin[1] and vin[1].uop != UOps.CONST:
        loc = f"%loc{c['loc']}"
        offset = f"%offset{c['offset']}"
        kernel.append(f"mul.wide.{lang.dtype_to_asmtype[vin[1].dtype]} {offset}, {r[vin[1]]}, {dtype.itemsize};")
        kernel.append(f"add.s64 {loc}, {r[vin[0]]}, {offset};")
        regs += [(loc, dtypes.uint64), (offset, dtypes.uint64)]
        c["loc"] += 1
        c["offset"] += 1
        loc = f"[{loc}]"
      else: loc = f"[{r[vin[0]]}{f'+{vin[1].arg * dtype.itemsize}' if vin[1] else ''}]"
      kernel.append(f"{f'@{r[vin[2]]} ' if len(vin) > 3 else ''}ld.{lang.dtype_to_asmtype[dtype]} {ssa(u, 'val')}, {loc};")
      if len(vin) > 3: kernel.append(f"@!{r[vin[2]]} mov.{lang.dtype_to_asmtype[dtype]} {ssa(u, 'val')}, {r[vin[3]]};")
    elif uop == UOps.PHI:
      kernel.append(f"mov.{lang.dtype_to_asmtype[dtype]} {r[vin[0]]}, {r[vin[1]]};")
      r[u] = r[vin[0]]
    elif uop == UOps.STORE:
      assert len(vin) <= 3, "no conditional store"
      if vin[0].uop == UOps.DEFINE_LOCAL:
        index = f"%index{c['index']}"
        kernel.append(f"mul.wide.{lang.dtype_to_asmtype[vin[1].dtype]} {index}, {r[vin[1]]}, {vin[0].dtype.itemsize};")
        loc = f"{r[vin[0]]}[{index}]"
        regs += [(index, dtypes.uint64)]
        c["index"] += 1
      elif vin[1].uop != UOps.CONST:
        loc = f"%loc{c['loc']}"
        offset = f"%offset{c['offset']}"
        kernel.append(f"mul.wide.{lang.dtype_to_asmtype[vin[1].dtype]} {offset}, {r[vin[1]]}, {vin[0].dtype.itemsize};")
        kernel.append(f"add.s64 {loc}, {r[vin[0]]}, {offset};")
        regs += [(loc, dtypes.uint64), (offset, dtypes.uint64)]
        c["loc"] += 1
        c["offset"] += 1
        loc = f"[{loc}]"
      else:
        loc = f"[{r[vin[0]]}{f'+{vin[1].arg * vin[0].dtype.itemsize}' if vin[1] else ''}]"
      kernel.append(f"st.{lang.dtype_to_asmtype[vin[0].dtype]} {loc}, {r[vin[2]]};")
    elif uop == UOps.CAST and dtype is not None:
      assert len(vin) == 1, "one cast input"
      if vin[0].dtype == dtypes.bool: kernel.append(f"selp.{lang.dtype_to_asmtype[dtype]} {ssa(u, 'cast')}, {lang.render_const(1, dtype)}, {lang.render_const(0, dtype)}, {r[vin[0]]};")
      else: kernel.append(f"cvt.{'rn' if dtypes.is_float(dtype) else 'rni'}.{lang.dtype_to_asmtype[dtype]}.{lang.dtype_to_asmtype[vin[0].dtype]} {ssa(u, 'cast')}, {r[vin[0]]};")
    elif uop == UOps.DEFINE_LOCAL:
      kernel.append(f".shared .align 4 .b8 {args[0]}[{args[1]*dtype.itemsize}];")
      r[u] = args[0]
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append(args)
      kernel.append(f"ld.param.u64 {ssa(u, 'data')}, [{args[0]}];")
    else: raise NotImplementedError(f"no code for {uop}")
  kernel.append("ret;")

  kernel = [f".reg .{lang.dtype_to_asmtype[dtype]} {reg};" for reg,dtype in sorted(regs, key=lambda reg: reg[0])] + kernel

  ret = f"{lang.kernel_prefix} {function_name}(\n "
  ret += ',\n\t'.join([f'.param .{lang.dtype_to_asmtype[dtypes.uint64]} {name}' for name,dtype in bufs])
  ret += "\n)\n{\n" + '\n'.join([line if line.startswith("$") else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1) for line in kernel]) + "\n}"
  return ret, {}


class PTXLanguage(AssemblyLanguage):
  kernel_prefix = """.version 7.8
.target sm_86
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  gid = [f'%ctaid.{chr(120+i)}' for i in range(3)]
  lid = [f'%tid.{chr(120+i)}' for i in range(3)]
  asm_for_op = {
    UnaryOps.NEG: lambda d,a,dtype: f"neg.{dtype} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dtype: f"ex2.approx.{dtype} {d}, {a};", UnaryOps.LOG2: lambda d,a,dtype: f"lg2.approx.{dtype} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dtype: f"sin.approx.{dtype} {d}, {a};", UnaryOps.SQRT: lambda d,a,dtype: f"sqrt.approx.{dtype} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dtype: f"{'or' if dtype == 'pred' else 'add'}.{dtype} {d}, {a}, {b};", BinaryOps.SUB: lambda d,a,b,dtype: f"sub.{dtype} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dtype: f"mul{'' if dtype.startswith('f') else '.lo'}.{dtype} {d}, {a}, {b};",
    BinaryOps.DIV: lambda d,a,b,dtype: f"div.approx.{dtype} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dtype: f"max.{dtype} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dtype: f"rem.{dtype} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dtype,stype: f"setp.lt.{stype} {d}, {a}, {b};" if dtype == "pred" else f"set.lt.{dtype}.{stype} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dtype: f"{'fma.rn' if dtype.startswith('f') else 'mad.lo'}.{dtype} {d}, {a}, {b}, {c};", # TODO: fma on int is mad
    TernaryOps.WHERE: lambda d,a,b,c,dtype: f"selp.{dtype} {d}, {b}, {c}, {a};"
  }
  dtype_to_asmtype = {
    dtypes.int8: "s8", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
    dtypes.uint8: "u8", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
    dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64",
    dtypes.bool: "pred", PtrDType: "u64"
  }
PTXRenderer = functools.partial(uops_to_asm, PTXLanguage())

