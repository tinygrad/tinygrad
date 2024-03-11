from typing import Callable, DefaultDict, Dict, List, Union, NamedTuple
import functools, struct
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, INVERSE_DTYPES_DICT
from tinygrad.codegen.uops import UOpGraph

def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
def double_to_hex(x): return "%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
def trunc_float(x, fmt): return struct.unpack(fmt, struct.pack(fmt, x))[0]

def is_bool_or_unsigned(dtype: DType): return dtype == dtypes.bool or dtypes.is_unsigned(dtype)

class AssemblyLanguage(NamedTuple):
  kernel_prefix: str = ""
  barrier: str = ""
  load_global: bool = False
  label_prefix: str = ""
  gid: List[str] = []
  gdim: List[str] = []
  lid: List[str] = []
  const_requires_mov: List[DType] = [] # list of dtypes for which creating a const requires a move
  asm_for_op: Dict[Op, Callable[...,str]] = {}
  types: Dict[DType, str] = INVERSE_DTYPES_DICT

  def render_const(self, x:Union[float,int,bool], dtype, mov=None) -> Union[List[str], str]: raise NotImplementedError()
  def render_local(self, dest, name, size, dtype) -> List[str]: raise NotImplementedError()

  def render_loop(self, idx, start, label, acc=None) -> List[str]: raise NotImplementedError()
  def render_bra(self, b1, pred=None, b2=None) -> List[str]: raise NotImplementedError()
  def render_gep(self, loc, base, offset, dtype, gate=None) -> List[str]: raise NotImplementedError()
  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="") -> List[str]: raise NotImplementedError()
  def render_store(self, loc, val, dtype, gate=None, ss="") -> List[str]: raise NotImplementedError()
  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]: raise NotImplementedError()

  def render_kernel(self, kernel, function_name, bufs, regs) -> str: raise NotImplementedError()

def uops_to_asm(lang:AssemblyLanguage, function_name:str, uops:UOpGraph) -> str:
  kernel:List[str] = []
  bufs = []

  # here we do a pretransform on UOps to fix some shortcomings of PTX
  # all uops must be a register
  # TODO: uops class should make these rewrites easier
  replace: Dict[UOp, UOp] = {}
  for u in uops:
    for o,n in replace.items():
      if o in u.vin and u is not n:
        u.vin = tuple(n if x == o else x for x in u.vin)
    if u.uop is UOps.LOAD and u.dtype is dtypes.bool:
      # rewrite load bool
      if len(u.vin) == 4:
        new = uops.add(UOps.CAST, dtypes.uint8, (u.vin[3],), insert_before=uops.uops.index(u))
        u.vin = u.vin[0:3] + (new,)
      u.dtype = dtypes.uint8
      new = uops.add(UOps.CAST, dtypes.bool, (u,), insert_before=uops.uops.index(u)+1)
      replace[u] = new
    if u.uop is UOps.ALU and u.arg in {BinaryOps.CMPEQ, BinaryOps.CMPLT} and u.vin[0].dtype is dtypes.bool:
      if u.arg == BinaryOps.CMPEQ:
        u.arg = BinaryOps.XOR
        new = uops.add(UOps.ALU, dtypes.bool, (u,), arg=UnaryOps.NEG, insert_before=uops.uops.index(u)+1)
        replace[u] = new
      if u.arg == BinaryOps.CMPLT:
        new = uops.add(UOps.ALU, dtypes.bool, (u.vin[0],), arg=UnaryOps.NEG, insert_before=uops.uops.index(u))
        u.vin = (new, u.vin[1])
        u.arg = BinaryOps.MUL
  #uops.print()

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
    if mov or dtype in lang.const_requires_mov:
      kk(*lang.render_const(x, dtype, mov=(out:=ssa(None, 'const', lang.types[dtype]))))
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
    if uop == UOps.IF:
      assert vin[0].dtype is not None
      kk(*lang.render_bra(lb:=ssa_label(u, 'if'), cast(r[vin[0]], dtypes.bool, vin[0].dtype, u=u, pred=True), f"{lb}_true"), f"{lb}_true:")
    elif uop == UOps.BARRIER and lang.barrier: kk(lang.barrier)
    elif uop == UOps.ENDLOOP:
      kk(lang.asm_for_op[BinaryOps.ADD](r[vin[0]], r[vin[0]], "1", dtypes.int, lang.types[dtypes.int]),
          lang.asm_for_op[BinaryOps.CMPLT](pred:=ssa(None, "pred", "pred"), r[vin[0]], r[vin[0].vin[1]], dtypes.int, lang.types[dtypes.int]))
      kk(*lang.render_bra(r_label[vin[0]], pred, f"{r_label[vin[0]]}_exit"), f"{r_label[vin[0]]}_exit:")
    elif uop == UOps.ENDIF:
      kk(f"{r_label[vin[0]]}:")
    elif uop == UOps.STORE:
      assert vin[0].dtype is not None and vin[1].dtype is not None and vin[2].dtype is not None
      kk(*lang.render_gep(loc:=ssa(None,'loc','u64'), r[vin[0]], r[vin[1]], vin[0].dtype))
      if len(vin) > 3:
        assert vin[3].dtype is not None
        pred = cast(r[vin[3]], dtypes.bool, vin[3].dtype, pred=True)
      kk(*lang.render_store(loc, r[vin[2]], vin[0].dtype, gate=pred if len(vin)>3 else None, ss='.shared' if vin[0].uop == UOps.DEFINE_LOCAL else ''))
    else:
      assert dtype is not None, f"None dtype for uop {uop}"
      if uop == UOps.LOOP: kk(*lang.render_loop(ssa(u, 'ridx'), r[vin[0]], ssa_label(u, 'loop')))
      elif uop == UOps.ALU:
        assert vin[0].dtype is not None
        if args == BinaryOps.CMPLT or args == BinaryOps.CMPEQ:
          # pass in the other dtype here
          kk(lang.asm_for_op[args](ssa(u, "alu"), *[r[x] for x in vin], vin[0].dtype, lang.types[vin[0].dtype]))
        else:
          kk(lang.asm_for_op[args](ssa(u, "alu"), *[r[x] for x in vin], dtype, lang.types[dtype]))
      elif uop == UOps.DEFINE_ACC: kk(f"mov.b{lang.types[dtype][1:]} {ssa(u, 'acc')}, {const(args, dtype)};")
      elif uop == UOps.SPECIAL:
        assert args[1][0] != "i", "idx not supported"
        kk(f"mov.u32 %{args[1]}, {(lang.gid if args[1][0] == 'g' else lang.lid)[args[0]]};")
        r[u] = "%" + args[1]
        kernel = [f".reg .u32 %{args[1]};"] + kernel
      elif uop == UOps.CONST: r[u] = const(args, dtype, mov=True)
      elif uop == UOps.LOAD:
        assert vin[1].dtype is not None
        val = ssa(u, 'val')
        if len(vin) > 3:
          assert vin[2].dtype is not None
          pred = cast(r[vin[2]], dtypes.bool, vin[2].dtype, pred=True)
          off = cast(r[vin[1]], dtypes.uint, vin[1].dtype)
        kk(*lang.render_gep(loc:=ssa(None,'loc',lang.types[dtypes.ulong]), r[vin[0]], off if len(vin)>3 else cast(r[vin[1]],
                                                                                                                  dtypes.uint, vin[1].dtype), dtype),
          *lang.render_load(loc, val, dtype, gate=pred if len(vin) > 3 else None,
                            alt=r[vin[3]] if len(vin) > 3 else None, ss='.shared' if vin[0].uop == UOps.DEFINE_LOCAL else ''))
      elif uop == UOps.PHI:
        kk(f"mov.b{lang.types[dtype][1:]} {r[vin[0]]}, {r[vin[1]]};")
        r[u] = r[vin[0]]
      elif uop == UOps.CAST:
        assert vin[0].dtype is not None
        cast(r[vin[0]], dtype, vin[0].dtype, bitcast=isinstance(args, tuple) and args[1], u=u)
      elif uop == UOps.DEFINE_LOCAL:
        # TODO: we should sum these, and fetch 0xC000 from somewhere
        assert args[1]*dtype.itemsize <= 0xC000, "too large local"
        kk(*lang.render_local(ssa(u, 'local', lang.types[dtypes.ulong]), args[0], args[1], dtype))
      elif uop is UOps.DEFINE_VAR:
        bufs.append((args.expr, dtype))
        r[u] = f"%{args.expr}"
        if lang.load_global:
          kk(*lang.render_load(args.expr, ssa(u, 'dat', dtype=lang.types[dtype]), dtype, ss=".param"))
      elif uop is UOps.DEFINE_GLOBAL:
        bufs.append((args[1], dtype))
        r[u] = f"%{args[1]}"
        if lang.load_global:
          dt = dtypes.ulong if dtype.__class__ == PtrDType else dtype
          kk(*lang.render_load(args[1], ssa(u, 'dat', dtype=lang.types[dt]), dt, ss=".param"))
      else: raise NotImplementedError(f"no code for {uop}")

  return lang.render_kernel(kernel, function_name, bufs, c.items())

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
    UnaryOps.NEG: lambda d,a,dt,name: f"not.pred {d}, {a};" if name == "pred" else f"neg.{name} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};",
    UnaryOps.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"{'or' if name == 'pred' else 'add'}.{name} {d}, {a}, {b};",
    BinaryOps.SUB: lambda d,a,b,dt,name: f"sub.{name} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dt,name: ('and' if dt == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if name == "pred" else f"xor.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.DIV: lambda d,a,b,dt,name: f"div{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};",
    BinaryOps.CMPEQ: lambda d,a,b,dt,name: f"setp.eq.{name} {d}, {a}, {b};",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name:
      f"@{a} mov.{name} {d}, {b};\n@!{a} mov.{name} {d}, {c};" if name == "pred" else f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
  }
  supports_half = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT, TernaryOps.WHERE]
  types = {
    dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
    dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
    dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64",
    dtypes.bool: "pred"
  }

  const_requires_mov = [dtypes.half, dtypes.bool]

  def render_const(self, x:Union[float,int,bool], dtype, mov=None) -> Union[List[str], str]:
    if dtypes.is_float(dtype): val = f"0f{float_to_hex(x)}" if dtype != dtypes.float64 else f"0d{double_to_hex(x)}"
    else: val = str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")
    if dtype == dtypes.bool: return [f"setp.ne.s16 {mov}, {val}, 0;"]
    if dtype == dtypes.half: return [f".reg .f32 {mov}_tmp;", f"mov.f32 {mov}_tmp, {val};", f"cvt.rn.f16.f32 {mov}, {mov}_tmp;"]
    return [f"mov.b{self.types[dtype][1:]} {mov}, {val};"] if mov else val

  def render_local(self, dest, name, size, dtype) -> List[str]:
    return [f".shared .align 4 .b8 {name}[{size*dtype.itemsize}];", f"mov.u64 {dest}, {name}[0];"]

  def render_loop(self, idx, start, label, acc=None) -> List[str]: return [f"mov.u32 {idx}, {start};", f"{label}:"]

  def render_bra(self, b1, pred=None, b2=None) -> List[str]: return [f"@{pred} bra {b1};", f"@!{pred} bra {b2};"] if pred else [f"bra {b1};"]

  def render_gep(self, loc, base, offset, dtype, gate=None) -> List[str]:
    # this cast is only required because of ocelot
    if "s32" in offset:
      return [f".reg .u32 {offset}_cast;", f"cvt.u32.s32 {offset}_cast, {offset};", f"mad.wide.u32 {loc}, {offset}_cast, {dtype.itemsize}, {base};"]
    else: return [f"mad.wide.u32 {loc}, {offset}, {dtype.itemsize}, {base};"]

  def mem_type(self, dtype): return 's8' if dtype.itemsize == 1 else 'b16' if dtype == dtypes.float16 else self.types[dtype]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="") -> List[str]:
    assert dtype is not dtypes.bool
    ret = []
    if gate: ret.extend([f"@{gate} ld{ss}.{self.mem_type(dtype)} {dest}, [{loc}];",
                         f"@!{gate} mov.b{self.types[dtype][1:]} {dest}, {alt};"])
    else: ret.append(f"ld{ss}.{self.mem_type(dtype)} {dest}, [{loc}];")
    return ret

  def render_store(self, loc, val, dtype, gate=None, ss="") -> List[str]:
    if dtype == dtypes.bool: return [f".reg .s16 {val}_cast;", *self.render_cast(f"{val}_cast", val, dtypes.int16, dtype),
                                     (f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}], {val}_cast;"]
    return [(f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}], {val};"]

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]:
    if bitcast: return [f"mov.b{self.types[dtype][1:]} {d}, {a};"]
    if atype == dtypes.bool: return [f"selp.{self.types[dtype]} {d}, {self.render_const(1, dtype)}, {self.render_const(0, dtype)}, {a};"]
    if dtype == dtypes.bool: return [f"setp.ne.{self.types[atype]} {d}, {a}, {self.render_const(0, atype)};"]
    rnd = ('.rzi' if dtypes.is_int(dtype) and dtypes.is_float(atype) else
           '.rn' if dtypes.is_float(dtype) and (dtype.itemsize < atype.itemsize or dtypes.is_int(atype) or atype == dtypes.bool) else '')
    return [f"cvt{rnd}.{self.types[dtype]}.{self.types[atype]} {d}, {a};"]

  def render_kernel(self, kernel, function_name, bufs, regs) -> str:
    kernel = [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in regs] + kernel + ["ret;"]
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)
    return (f"{self.kernel_prefix} {function_name}(\n\t" +
            ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else self.types[dtype]} {name}" for name,dtype in bufs]) + "\n)\n{\n" +
            '\n'.join([fmt(line) for op in kernel for line in op.splitlines()]) +
            "\n}")

PTXRenderer = functools.partial(uops_to_asm, PTXLanguage())
