from typing import Callable, DefaultDict, Dict, List, Tuple, Union, NamedTuple
import functools, struct, re
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.helpers import dtypes, DType, PtrDType, INVERSE_DTYPES_DICT

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
def double_to_hex(x): return "%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
def trunc_float(x, fmt): return struct.unpack(fmt, struct.pack(fmt, x))[0]

def is_bool_or_unsigned(dtype: DType): return dtype == dtypes.bool or dtypes.is_unsigned(dtype)

class AssemblyLanguage(NamedTuple):
  kernel_prefix: str = ""
  barrier: str = ""
  ssa: bool = False # whether the language uses ssa (ie. LLVM)
  needs_regs: bool = True # whether registers need to be defined prior to use
  has_pred: bool = False # whether the language supports predicates on instructions
  load_global: bool = False
  label_prefix: str = ""
  gid: List[str] = []
  gdim: List[str] = []
  lid: List[str] = []
  const_requires_mov: List[DType] = [] # list of dtypes for which creating a const requires a move
  no_half_support: List[Op] = [] # list of opporations that don't support half
  asm_for_op: Dict[Op, Callable[...,str]] = {}
  types: Dict[DType, str] = INVERSE_DTYPES_DICT

  def render_const(self, x:Union[float,int,bool], dtype, mov=None) -> str: raise NotImplementedError()
  def render_local(self, dest, name, size, dtype) -> str: raise NotImplementedError()

  def render_loop(self, idx, start, label, acc=None) -> List[str]: raise NotImplementedError()
  def render_bra(self, b1, pred=None, b2=None) -> List[str]: raise NotImplementedError()
  def render_gep(self, loc, base, offset, dtype, gate=None) -> List[str]: raise NotImplementedError()
  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="") -> List[str]: raise NotImplementedError()
  def render_store(self, loc, val, dtype, gate=None) -> List[str]: raise NotImplementedError()
  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]: raise NotImplementedError()

  def render_kernel(self, kernel, function_name, bufs, regs) -> str: raise NotImplementedError()

def uops_to_asm(lang:AssemblyLanguage, function_name:str, uops:List[UOp]) -> Tuple[str, Dict]:
  local_size: List[int] = []
  kernel:List[str] = []
  bufs, loops = [], []
  accs: List[UOp] = []

  def kk(*s: str): kernel.append("\n".join(s))

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, str] = {}
  def ssa(u, prefix="t", dtype=None) -> str:
    nonlocal c, r
    prefix += f"_{dtype if dtype else lang.types[u.dtype]}_"
    c[prefix] += 1
    if u: r[u] = f"%{prefix}{c[prefix]-1}"
    return f"%{prefix}{c[prefix]-1}"

  c_label: DefaultDict[str, int] = defaultdict(int)
  r_label: Dict[UOp, str] = {}
  def ssa_label(u, prefix):
    nonlocal c_label, r_label
    c_label[prefix] += 1
    r_label[u] = f"{lang.label_prefix}{prefix}_{c_label[prefix]-1}"
    return r_label[u]

  def const(x:Union[float,int,bool], dtype, mov=False):
    if lang.ssa: return lang.render_const(x, dtype)
    if mov or dtype in lang.const_requires_mov:
      kk(lang.render_const(x, dtype, mov=(out:=ssa(None, 'const', lang.types[dtype]))))
      return out
    return lang.render_const(x, dtype)

  def cast(a:str, dtype:DType, atype:DType, bitcast=False, u=None, pred=False):
    if atype == dtype:
      if u: r[u] = a
      return a
    kk(*lang.render_cast((ret:=ssa(u, 'cast', lang.types[dtype])), a, dtype, atype, bitcast))
    return ret

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    if uop == UOps.LOOP:
      kk(*lang.render_loop(ssa(u, 'ridx'), r[vin[0]], label:=ssa_label(u, 'loop')))
      phis = []
      for acc in accs:
        assert acc.dtype is not None
        phis.append((len(kernel), acc))
        phi = f"phi {lang.types[acc.dtype]} [{r[acc]}, %pre_{label}]"
        kk(f"{ssa(acc)} = " + phi)
      loops.append(phis)
    elif uop == UOps.IF:
      assert vin[0].dtype is not None
      kk(*lang.render_bra(lb:=ssa_label(u, 'if'), cast(r[vin[0]], dtypes.bool, vin[0].dtype, u=u, pred=True), f"{lb}_true"), f"{lb}_true:")
    elif uop == UOps.END:
      if vin[0].uop == UOps.LOOP:
        kk(lang.asm_for_op[BinaryOps.ADD](upd:=f"{r[vin[0]]}_upd" if lang.ssa else r[vin[0]], r[vin[0]], "1", dtypes.int, lang.types[dtypes.int]),
           lang.asm_for_op[BinaryOps.CMPLT](pred:=ssa(None, "pred", lang.types[dtypes.bool]), upd, r[vin[0].vin[1]], dtypes.int,
                                            lang.types[dtypes.int]),
           *lang.render_bra(f"{r_label[vin[0]]}_check"), f"{r_label[vin[0]]}_check:",
           *lang.render_bra(r_label[vin[0]], pred, f"{r_label[vin[0]]}_exit"), f"{r_label[vin[0]]}_exit:")
        phis = loops.pop()
        for n, acc in phis:
          kernel[n] += f", [{r[acc]}, %{r_label[vin[0]]}_check]"
      else: kk(f"{r_label[vin[0]]}:")
    elif uop == UOps.BARRIER and lang.barrier: kk(lang.barrier)
    elif uop == UOps.ALU:
      assert dtype is not None and vin[0].dtype is not None
      if args == BinaryOps.CMPLT or args == BinaryOps.CMPEQ:
        kk(lang.asm_for_op[args](pred:=ssa(None if lang.has_pred else u,'lt','pred' if lang.has_pred else None),
                                 *[r[x] for x in vin], vin[0].dtype, lang.types[vin[0].dtype]))
        if lang.has_pred: cast(pred, dtype, dtypes.bool, u=u)
      elif args == BinaryOps.MAX and BinaryOps.MAX not in lang.asm_for_op:
        kk(lang.asm_for_op[BinaryOps.CMPLT](pred:=ssa(None, 'lt', lang.types[dtypes.bool]), r[vin[0]], r[vin[1]], dtype, lang.types[dtype]),
           lang.asm_for_op[TernaryOps.WHERE](ssa(u, "alu"), pred, r[vin[1]], r[vin[0]], dtype, lang.types[dtype]))
      elif args == TernaryOps.WHERE and TernaryOps.WHERE not in lang.asm_for_op:
        kk(lang.asm_for_op[args](ssa(u, "alu"), cast(r[vin[0]], dtypes.bool, vin[0].dtype, pred=True), r[vin[1]], r[vin[2]], lang.types[dtype]))
      elif args == TernaryOps.MULACC and TernaryOps.MULACC not in lang.asm_for_op:
        assert vin[1].dtype is not None
        kk(lang.asm_for_op[BinaryOps.MUL](tmp:=ssa(None, "tmp", lang.types[dtype]),
                                          cast(r[vin[0]], dtype, vin[0].dtype), cast(r[vin[1]], dtype, vin[1].dtype), dtype, lang.types[dtype]),
           lang.asm_for_op[BinaryOps.ADD](ssa(u, "alu"), tmp, r[vin[2]], dtype, lang.types[dtype]))
      elif vin[0].dtype == dtypes.half and args in lang.no_half_support:
        kk(lang.asm_for_op[args]((tmp:=ssa(None, "alu", lang.types[dtypes.float])), *[cast(r[x], dtypes.float, dtypes.half) for x in vin],
                                 lang.types[dtypes.float]))
        cast(tmp, dtypes.half, dtypes.float32, u=u)
      else: kk(lang.asm_for_op[args](ssa(u, "alu"), *[r[x] for x in vin], dtype, lang.types[dtype]))
    elif uop == UOps.DEFINE_ACC:
      assert dtype is not None
      if lang.ssa:
        r[u] = const(args, dtype)
        accs.append(u)
      else: kk(f"mov.b{lang.types[dtype][1:]} {ssa(u, 'acc')}, {const(args, dtype)};")
    elif uop == UOps.SPECIAL:
      if args[1][0] == "i": kk(f"mov.u32 %{args[1]}, {lang.gid[args[0]]};", f"mov.u32 {(gdim:=ssa(None,'tmp','u32'))}, {lang.gdim[args[0]]};",
                               f"mov.u32 {(lid:=ssa(None,'tmp','u32'))}, {lang.lid[args[0]]};", f"mad.lo.u32 {(tmp:=ssa(None, 'tmp', 'u32'))}, %{args[1]}, {gdim}, {lid};")
      else: kk(f"mov.u32 {(tmp:=ssa(None, 'tmp', 'u32'))}, {(lang.gid if args[1][0] == 'g' else lang.lid)[args[0]]};")
      kk(*lang.render_cast(f"%{args[1]}", tmp, dtypes.uint, dtypes.int))
      if args[1][0] == "l": local_size.append(args[2])
      r[u] = "%" + args[1]
      kernel = [f".reg .s32 %{args[1]};"] + kernel
    elif uop == UOps.CONST: r[u] = const(args, dtype, mov=True)
    elif uop == UOps.LOAD:
      assert dtype is not None and vin[1].dtype is not None
      val = ssa(u, 'val')
      if len(vin) > 3:
        assert vin[2].dtype is not None
        pred = cast(r[vin[2]], dtypes.bool, vin[2].dtype, pred=True)
        if lang.has_pred: off = cast(r[vin[1]], dtypes.uint, vin[1].dtype)
        else: kk(lang.asm_for_op[TernaryOps.WHERE](off:=ssa(None, "off", lang.types[dtypes.uint]), pred, r[vin[1]], const(0, dtypes.uint), dtypes.uint,
                                                   lang.types[dtypes.uint]))
      kk(*lang.render_gep(loc:=ssa(None,'loc',lang.types[dtypes.ulong]), r[vin[0]], off if len(vin) > 3 else cast(r[vin[1]], dtypes.uint, vin[1].dtype), dtype),
         *lang.render_load(loc, val, dtype, gate=pred if len(vin) > 3 else None, alt=r[vin[3]] if len(vin) > 3 else None))
    elif uop == UOps.PHI:
      r[u] = r[vin[1]]
      # PHI UOps can link to other PHI Uops, backtrace this to DEFINE_ACC
      backward = vin[0]
      while backward.uop == UOps.PHI: backward = backward.vin[0]
      r[backward] = r[u]
    elif uop == UOps.STORE:
      assert vin[0].dtype is not None and vin[1].dtype is not None and vin[2].dtype is not None
      kk(*lang.render_gep(loc:=ssa(None,'loc','u64'), r[vin[0]], r[vin[1]], vin[0].dtype))
      if len(vin) > 3:
        assert vin[3].dtype is not None
        pred = cast(r[vin[3]], dtypes.bool, vin[3].dtype, pred=True)
      kk(*lang.render_store(loc, r[vin[2]], vin[0].dtype, gate=pred if len(vin) > 3 else None))
    elif uop == UOps.CAST:
      assert dtype is not None and vin[0].dtype is not None
      cast(r[vin[0]], dtype, vin[0].dtype, bitcast=isinstance(args, tuple) and args[1], u=u)
    elif uop == UOps.DEFINE_LOCAL: kk(*lang.render_local(ssa(u, 'local', lang.types[dtypes.ulong]), args[0], args[1], dtype))
    elif uop == UOps.DEFINE_GLOBAL:
      assert dtype is not None
      bufs.append((args, dtype))
      r[u] = f"%{args}"
      if lang.load_global:
        dt = dtypes.ulong if dtype.__class__ == PtrDType else dtype
        kk(*lang.render_load(args, ssa(u, 'dat', dtype=lang.types[dt]), dt, ss=".param"))
    else: raise NotImplementedError(f"no code for {uop}")

  return lang.render_kernel(kernel, function_name, bufs, c.items()), {}

class PTXLanguage(AssemblyLanguage):
  kernel_prefix = """.version 7.8
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  has_pred = True
  load_global = True
  label_prefix = "$"
  gid = [f'%ctaid.{chr(120+i)}' for i in range(3)]
  gdim = [f'%nctaid.{chr(120+i)}' for i in range(3)]
  lid = [f'%tid.{chr(120+i)}' for i in range(3)]
  asm_for_op = {
    UnaryOps.NEG: lambda d,a,dtype,asmtype: f"neg.{asmtype} {d}, {a};", UnaryOps.EXP2: lambda d,a,dtype,asmtype: f"ex2.approx.{asmtype} {d}, {a};",
    UnaryOps.LOG2: lambda d,a,dtype,asmtype: f"lg2.approx.{asmtype} {d}, {a};", UnaryOps.SIN: lambda d,a,dtype,asmtype: f"sin.approx.{asmtype} {d}, {a};",
    UnaryOps.SQRT: lambda d,a,dtype,asmtype: f"sqrt.approx.{asmtype} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dtype,asmtype: f"{'or' if asmtype == 'pred' else 'add'}.{asmtype} {d}, {a}, {b};",
    BinaryOps.SUB: lambda d,a,b,dtype,asmtype: f"sub.{asmtype} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dtype,asmtype: ('and' if dtype == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dtype) else ''}.{asmtype} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dtype,asmtype: f"xor.b{dtype[1:]} {d}, {a}, {b};",
    BinaryOps.DIV: lambda d,a,b,dtype,asmtype: f"div{'.approx' if dtype[0] == 'f' else ''}.{dtype} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dtype,asmtype: f"max.{asmtype} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dtype,asmtype: f"rem.{asmtype} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dtype,asmtype: f"setp.lt.{asmtype} {d}, {a}, {b};",
    BinaryOps.CMPEQ: lambda d,a,b,dtype,asmtype: f"setp.eq.{asmtype} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dtype,asmtype: f"{'fma.rn' if dtypes.is_float(dtype) else 'mad.lo'}.{asmtype} {d}, {a}, {b}, {c};",
    TernaryOps.WHERE: lambda d,a,b,c,dtype,asmtype: f"selp.{asmtype} {d}, {b}, {c}, {a};"
  }
  supports_half = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT,
                   TernaryOps.MULACC, TernaryOps.WHERE]
  types = {
    dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
    dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
    dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64",
    dtypes.bool: "pred"
  }

  def render_const(self, x:Union[float,int,bool], dtype, mov=None) -> str:
    if dtypes.is_float(dtype): val = f"0f{float_to_hex(x)}" if dtype == dtypes.float32 else f"0d{double_to_hex(x)}"
    else: val = str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")
    return f"mov.b{self.types[dtype][1:]} {mov}, {val};" if mov else val
  
  def render_local(self, dest, name, size, dtype) -> str:
    return [f".shared .align 4 .b8 {name}[{size*dtype.itemsize}];", f"mov.u64 {dest}, {name}[0];"]
  
  def render_loop(self, idx, start, label, acc=None) -> List[str]: return [f"mov.u32 {idx}, {start};", f"{label}:"]

  def render_bra(self, b1, pred=None, b2=None) -> List[str]: return [f"@!{pred} bra {b1};", f"@{pred} bra {b2};"] if pred else [f"bra {b1};"]

  def render_gep(self, loc, base, offset, dtype, gate=None) -> List[str]: return [f"mad.wide.u32 {loc}, {offset}, {dtype.itemsize}, {base};"]

  def mem_type(self, dtype): return 's8' if dtype.itemsize == 1 else 'b16' if dtype == dtypes.float16 else self.types[dtype]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="") -> List[str]:
    if gate: return [f"@{gate} ld{ss}.{self.mem_type(dtype)} {dest}, {loc};",
                     f"@!{gate} mov.b{'8' if dtype.itemsize == 1 else self.types[dtype][1:]} {dest}, {alt};"]
    else:
      return [f"ld{ss}.{'s8' if dtype.itemsize == 1 else 'b16' if dtype == dtypes.float16 else self.types[dtype]} {dest}, [{loc}];"]

  def render_store(self, loc, val, dtype, gate=None, ss="") -> List[str]:
    if dtype == dtypes.bool: return [f".reg .s16 {val}_cast;", *self.render_cast(f"{val}_cast", val, dtypes.int16, dtype),
                                     (f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}], {val}_cast;"]
    return [(f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}], {val};"]

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]:
    if bitcast: return [f"mov.b{self.types[dtype][1:]} {d}, {a};"]
    if atype == dtypes.bool: return [f"selp.{self.types[dtype]} {d}, {self.render_const(1, dtype)}, {self.render_const(0, dtype)}, {a};"]
    rnd = ('.rzi' if dtypes.is_int(dtype) and dtypes.is_float(atype) else
           '.rn' if dtypes.is_float(dtype) and (dtype.itemsize < atype.itemsize or dtypes.is_int(atype) or atype == dtypes.bool) else '')
    return [f"cvt{rnd}.{self.types[dtype]}.{self.types[atype]} {d}, {a};"]
  
  def render_kernel(self, kernel, function_name, bufs, regs) -> str:
    kernel = [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in regs] + kernel + ["ret;"]
    return (f"{self.kernel_prefix} {function_name}(\n\t" +
            ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else self.types[dtype]} {name}" for name,dtype in bufs]) + "\n)\n{\n" +
            '\n'.join([line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1) for op in kernel for line in op.splitlines()]) +
            "\n}")

PTXRenderer = functools.partial(uops_to_asm, PTXLanguage())

LLVM_FMATH = "nsz arcp contract afn reassoc"

llvm_intrinsic_types = {dtypes.float32: "f32", dtypes.float64: "f64"}

class LLVMLanguage(AssemblyLanguage):
  kernel_prefix = """target triple = "unknown-unknown-unknown"
define void @"""
  ssa = True
  needs_regs = False
  has_max = False
  has_mulacc = False
  has_const_mov = False
  types = {**{k:f"i{k.itemsize*8}" if dtypes.is_int(k) else v for k,v in AssemblyLanguage().types.items()},
           dtypes.bool: "i1", dtypes.bfloat16: "bfloat"}
  asm_for_op = {
    UnaryOps.NEG: lambda d,a,dtype,asmtype: (f"{d} = {'xor' if dtype == dtypes.bool else 'sub' if dtypes.is_int(dtype) else f'fneg {LLVM_FMATH}'} "
    f"{asmtype} {'0, ' if dtypes.is_int(dtype) else ''}{a}{' , 1' if dtype == dtypes.bool else ''}"),
    UnaryOps.EXP2: lambda d,a,dtype,asmtype: f"{d} = call {LLVM_FMATH} {asmtype} @llvm.exp2.f{dtype.itemsize*8}({asmtype} {a})",
    UnaryOps.LOG2: lambda d,a,dtype,asmtype: f"{d} = call {LLVM_FMATH} {asmtype} @llvm.log2.f{dtype.itemsize*8}({asmtype} {a})",
    UnaryOps.SIN: lambda d,a,dtype,asmtype: f"{d} = call {LLVM_FMATH} {asmtype} @llvm.sin.f{dtype.itemsize*8}({asmtype} {a})",
    UnaryOps.SQRT: lambda d,a,dtype,asmtype: f"{d} = call {LLVM_FMATH} {asmtype} @llvm.sqrt.f{dtype.itemsize*8}({asmtype} {a})",
    BinaryOps.ADD: lambda d,a,b,dtype,asmtype:
      f"{d} = {'or' if dtype == dtypes.bool else 'add' if dtypes.is_int(dtype) else f'fadd {LLVM_FMATH}'} {asmtype} {a}, {b}",
    BinaryOps.SUB: lambda d,a,b,dtype,asmtype: f"{d} = {'fsub' if dtypes.is_float(dtype) else 'sub'} {asmtype} {a}, {b}",
    BinaryOps.MUL: lambda d,a,b,dtype,asmtype: f"{d} = {f'fmul {LLVM_FMATH}' if dtypes.is_float(dtype) else 'mul'} {asmtype} {a}, {b}",
    BinaryOps.XOR: lambda d,a,b,dtype,asmtype: f"{d} = xor {asmtype} {a}, {b}",
    BinaryOps.DIV:
      lambda d,a,b,dtype,asmtype: f"{d} = {'udiv' if is_bool_or_unsigned(dtype) else 'sdiv' if dtypes.is_int(dtype) else 'fdiv'} {asmtype} {a}, {b}",
    BinaryOps.MOD:
      lambda d,a,b,dtype,asmtype: f"{d} = {'urem' if is_bool_or_unsigned(dtype) else 'srem' if dtypes.is_int(dtype) else 'frem'} {asmtype} {a}, {b}",
    BinaryOps.CMPEQ:
      lambda d,a,b,dtype,asmtype: f"{d} = {f'fcmp {LLVM_FMATH} ueq' if dtypes.is_float(dtype) else 'icmp eq'} {asmtype} {a}, {b}",
    BinaryOps.CMPLT:
      lambda d,a,b,dtype,asmtype: (f"{d} = " +
                                   (f'fcmp {LLVM_FMATH} ult' if dtypes.is_float(dtype) else f"icmp {'ult' if is_bool_or_unsigned(dtype) else 'slt'}")
                                   + f" {asmtype} {a}, {b}"),
    TernaryOps.WHERE: lambda d,a,b,c,dtype,asmtype: f"{d} = select i1 {a}, {asmtype} {b}, {asmtype} {c};"
  }
  
  def render_const(self, x: Union[float,int,bool], dtype, mov=None) -> str:
    if not dtypes.is_float(dtype): return str(bool(x)).lower() if dtype == dtypes.bool else str(int(x))
    return f"0x{double_to_hex(x if dtype == dtypes.double else trunc_float(x, 'f') if dtype == dtypes.float else trunc_float(x, 'e'))}"

  def render_loop(self, idx, start, label, acc=None) -> List[str]:
    ret = [f"br label %pre_{label}", f"pre_{label}:", f"br label %{label}", f"{label}:",
           f"{idx} = phi i32 [{start}, %pre_{label}], [{idx}_upd, %{label}_check]"]
    return ret + ([f"{acc[0]} = phi {self.types[acc[2]]} [{acc[1]}, %pre_{label}], [{acc[0]}_upd, %{label}_check]"] if acc else [])
  
  def render_bra(self, b1, pred=None, b2=None) -> List[str]: return [f"br i1 {pred}, label %{b1}, label %{b2}"] if pred else [f"br label %{b1}"]

  def render_gep(self, loc, base, offset, dtype) -> List[str]:
    return [f"{loc} = getelementptr inbounds {self.types[dtype]}, {self.types[dtype]}* {base}, i32 {offset}"]

  def render_load(self, loc, dest, dtype, gate=None, alt=None):
    if gate: return [f"{dest}_ld = load {self.types[dtype]}, {self.types[dtype]}* {loc}",
                     f"{dest} = select i1 {gate}, {self.types[dtype]} {dest}_ld, {self.types[dtype]} {alt}"]
    else: return [f"{dest} = load {self.types[dtype]}, {self.types[dtype]}* {loc}"]

  def render_store(self, loc, val, dtype, gate=None) -> List[str]:
    if gate: [*self.render_bra(f"st_{loc}:"), gate, f"st_{loc}_end", f"st_{loc}:", f"store {self.types[dtype]} {val}, {self.types[dtype]}* {loc}",
              f"st_{loc}_end:"]
    else: return [f"store {self.types[dtype]} {val}, {self.types[dtype]}* {loc}"]

  def render_cast(self, d: str, a: str, dtype: DType, atype: DType, bitcast=False, pred=False) -> List[str]:
    instr = ""
    if dtype == dtypes.bool:
      return [f"{d} = {f'fcmp {LLVM_FMATH} une' if dtypes.is_float(atype) else 'icmp ne'} {self.types[atype]} {a}, {self.render_const(0, atype)}"]
    if bitcast: instr = "bitcast"
    elif dtypes.is_float(atype):
      if dtypes.is_float(dtype): instr = "fptrunc" if atype.itemsize > dtype.itemsize else "fpext"
      else: instr = "fptoui" if dtypes.is_unsigned(dtype) else "fptosi"
    elif is_bool_or_unsigned(atype):
      if dtypes.is_float(dtype): instr = "uitofp"
      else: instr = "zext" if atype.itemsize < dtype.itemsize or atype == dtypes.bool else "trunc" if atype.itemsize > dtype.itemsize else "bitcast"
    elif dtypes.is_int(atype):
      if dtypes.is_float(dtype): instr = "sitofp"
      else: instr = "trunc" if atype.itemsize > dtype.itemsize or dtype == dtypes.bool else "sext" if atype.itemsize < dtype.itemsize else "bitcast"
    return [f"{d} = {instr} {self.types[atype]} {a} to {self.types[dtype]}"]

  def render_kernel(self, kernel, function_name, bufs, regs) -> List[str]:
    def render_param(param, dtype): return f"{self.types[dtype] + '* noalias' if dtype.__class__ == PtrDType else self.types[dtype]} %{param}"
    kernel.append("ret void;")
    k = (f"{self.kernel_prefix}{function_name}(" + ', '.join([render_param(name, dtype) for name, dtype in bufs]) + ")\n{\n" +
         "\n".join([line if ":" in line else "  " + line for instr in kernel for line in instr.split("\n")]) + "\n}\n\n")
    return k + "\n".join([f"declare {intrin} %0)" for intrin in set(re.findall(r'[^ ]* @llvm\.[^ ]*', k))])

LLVMRenderer = functools.partial(uops_to_asm, LLVMLanguage())
