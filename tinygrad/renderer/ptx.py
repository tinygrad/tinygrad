from typing import DefaultDict, Dict, List, Union, Optional, cast, Callable, Tuple
import struct
from collections import defaultdict
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import prod

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    if dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

asm_for_op: Dict[Ops, Callable] = {
  UnaryOps.RECIP: lambda d,a,dt,name: f"rcp{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a};",
  UnaryOps.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
  UnaryOps.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", UnaryOps.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
  BinaryOps.SHR: lambda d,a,b,dt,name: f"shr.{name} {d}, {a}, {b};", BinaryOps.SHL: lambda d,a,b,dt,name: f"shl.b{name[1:]} {d}, {a}, {b};",
  BinaryOps.ADD: lambda d,a,b,dt,name: f"{'or' if name == 'pred' else 'add'}.{name} {d}, {a}, {b};",
  BinaryOps.MUL: lambda d,a,b,dt,name: ('and' if dt == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
  BinaryOps.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if name == "pred" else f"xor.b{name[1:]} {d}, {a}, {b};",
  BinaryOps.AND: lambda d,a,b,dt, name: f"and.pred {d}, {a}, {b};" if name == "pred" else f"and.b{name[1:]} {d}, {a}, {b};",
  BinaryOps.OR: lambda d,a,b,dt, name: f"or.pred {d}, {a}, {b};" if name == "pred" else f"or.b{name[1:]} {d}, {a}, {b};",
  BinaryOps.IDIV: lambda d,a,b,dt,name: f"div.{name} {d}, {a}, {b};",
  BinaryOps.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
  BinaryOps.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};", BinaryOps.CMPNE: lambda d,a,b,dt,name: f"setp.ne.{name} {d}, {a}, {b};",
  TernaryOps.MULACC: lambda d,a,b,c,dt,name: f"{'fma.rn' if dtypes.is_float(dt) else 'mad.lo'}.{name} {d}, {a}, {b}, {c};",
  TernaryOps.WHERE: lambda d,a,b,c,dt,name:
    f"@{a} mov.{name} {d}, {b};\n@!{a} mov.{name} {d}, {c};" if name == "pred" else f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
}

supports_half: List[Ops] = [UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT, TernaryOps.WHERE]
doesnt_support_half: Tuple[Ops, ...] = tuple(op for op in asm_for_op.keys() if op not in supports_half)
ptx_matcher = PatternMatcher([
  # bool CMPNE is XOR, bool CMPLT is XOR+AND (universal makes this slow, this is for renderer only)
  (UPat.var('x', dtype=dtypes.bool).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtype=dtypes.bool).lt(UPat.var('y')), lambda x,y: (x^True)&y),
  # upcast to float32 all the ops that don't support half
  (UPat(doesnt_support_half, dtype=dtypes.half, name="x"),
    lambda x: (UOp(x.op, dtypes.float32, tuple(vv.cast(dtypes.float32) for vv in x.src), x.arg).cast(dtypes.half))),
  # load/store bool -> uint8
  (UPat(Ops.LOAD, dtypes.bool, src=(UPat(dtype=dtypes.int64),), name="x", allow_any_len=True),
   lambda x: UOp(x.op, dtypes.uint8, x.src[0:1] + ((x.src[1].cast(dtypes.uint8),) if len(x.src) >= 2 else ()) + x.src[2:]).cast(dtypes.bool)),
  (UPat(Ops.STORE, src=(UPat(dtype=dtypes.int64), UPat(dtype=dtypes.bool)), name="x", allow_any_len=True),
   lambda x: UOp(x.op, dtypes.void, x.src[0:1] + (x.src[1].cast(dtypes.uint8),) + x.src[2:])),
  # load/store use pointer arithmetic, and the cast does nothing
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"))), lambda buf,idx: buf.cast(dtypes.int64) + idx.cast(dtypes.int64)*buf.dtype.itemsize),
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) else None),
  # ptx shr and shl instructions require y to be uint
  (UPat.var("x") << UPat.var("y"), lambda x,y: UOp(Ops.SHL, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
  (UPat.var("x") >> UPat.var("y"), lambda x,y: UOp(Ops.SHR, x.dtype, (x,y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
])

class PTXRenderer(Renderer):
  device = "CUDA"
  suffix = "PTX"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tensor_cores = [tc for tc in CUDARenderer.tensor_cores if tc.dtype_in == dtypes.half]
  code_for_op = asm_for_op
  extra_matcher = ptx_matcher
  def __init__(self, arch:str, device="CUDA"):
    self.device, self.tensor_cores, self.arch = device, PTXRenderer.tensor_cores if int(arch[3:]) >= 80 else [], arch
  def __reduce__(self): return self.__class__, (self.arch, self.device)

  # language options
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  supports_half = supports_half
  # HACK: Use s16 and u16 for int8 and uint8 buffers. This can be wrong in cast.
  types: Dict[DType, str] = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
                              dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                              dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }

  mem_types: Dict[DType, str] =  types.copy()
  mem_types.update({dtypes.int8: "s8", dtypes.uint8: "u8", dtypes.bool: "u8", dtypes.float16: "b16"})

  const_requires_mov: List[DType] = [dtypes.half, dtypes.bool]

  def render_const(self, x:ConstType, dtype:DType, mov=None) -> Union[List[str], str]:
    val = render_val(x, dtype)
    if dtype == dtypes.bool: return [f"setp.ne.s16 {mov}, {val}, 0;"]
    return [f"mov.b{self.types[dtype][1:]} {mov}, {val};"] if mov else val

  def render_local(self, dest, name, size, dtype) -> List[str]:
    return [f".shared .align 4 .b8 {name}[{size*dtype.itemsize}];", f"mov.u64 {dest}, {name}[0];"]

  def render_loop(self, idx, start, label, acc=None) -> List[str]: return [f"mov.u32 {idx}, {start};", f"{label}:"]

  def render_bra(self, b1, pred=None, invert=False) -> List[str]:
    return [f"@{'!' if invert else ''}{pred} bra {b1};"] if pred else [f"bra {b1};"]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]:
    assert dtype != dtypes.bool
    if gate: return [f"@{gate} ld{ss}.{self.mem_types[dtype]} {dest}, [{loc}+{offset}];", f"@!{gate} mov.b{self.types[dtype][1:]} {dest}, {alt};"]
    return [f"ld{ss}.{self.mem_types[dtype]} {dest}, [{loc}+{offset}];"]

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]:
    if bitcast: return [f"mov.b{self.types[dtype][1:]} {d}, {a};"]
    if atype == dtypes.bool: return [f"selp.b{self.types[dtype][1:]} {d}, {render_val(1, dtype)}, {render_val(0, dtype)}, {a};"]
    if dtype == dtypes.bool: return [f"setp.ne.b{self.types[atype][1:]} {d}, {a}, {self.render_const(0, atype)};"]
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

  def render(self, name:str, uops:List[UOp]) -> str:
    kernel:List[str] = []
    bufs = []

    def kk(*s: str): kernel.append("\n".join(s))

    c: DefaultDict[str, int] = defaultdict(int)
    r: Dict[UOp, Union[List[str], str]] = {}
    def ssa(prefix:str, u:Optional[UOp]=None, dtype:Optional[str]=None) -> str:
      nonlocal c, r
      prefix += f"_{dtype if dtype is not None else self.types[cast(UOp, u).dtype]}_"
      c[prefix] += 1
      if u is not None: r[u] = f"%{prefix}{c[prefix]-1}"
      return f"%{prefix}{c[prefix]-1}"

    def const(x:ConstType, dtype:DType, mov=False):
      if mov or dtype in self.const_requires_mov:
        kk(*self.render_const(x, dtype, mov=(out:=ssa('const', dtype=self.types[dtype]))))
        return out
      return self.render_const(x, dtype)

    def _cast(a, dtype:DType, atype:DType, bitcast=False, u=None, pred=False):
      if atype == dtype or isinstance(atype, PtrDType):
        if u is not None: r[u] = a
        return a
      kk(*self.render_cast((ret:=ssa('cast', u, self.types[dtype])), a, dtype, atype, bitcast))
      return ret

    for u in uops:
      uop,dtype,src,args = u.op,u.dtype,u.src,u.arg
      if uop is Ops.IF:
        pred_reg = _cast(r[src[0]], dtypes.bool, src[0].dtype, u=u, pred=True)
        kk(*self.render_bra(f"IF_{r[src[0]][1:]}_{uops.index(u)}", pred_reg, invert=True))
      elif uop is Ops.BARRIER and self.barrier: kk(self.barrier)
      elif uop is Ops.ENDRANGE:
        kk(self.code_for_op[BinaryOps.ADD](r[src[0]], r[src[0]], "1", dtypes.int, self.types[dtypes.int]),
            self.code_for_op[BinaryOps.CMPLT](pred:=ssa("pred", dtype="pred"), r[src[0]], r[src[0].src[1]], dtypes.int, self.types[dtypes.int]))
        kk(*self.render_bra(f"LOOP_{r[src[0]][1:]}", pred))
      elif uop is Ops.ENDIF:
        kk(f"IF_{r[src[0].src[0]][1:]}_{uops.index(src[0])}:")
      elif uop is Ops.STORE:
        assert src[0].dtype == dtypes.int64, "store isn't int64"
        mem_type = '.shared' if src[0].op is Ops.DEFINE_LOCAL or any(x.op is Ops.DEFINE_LOCAL for x in src[0].parents) else '.global'
        gate = f"@{r[src[2]]} " if len(src)>2 and src[2].op is not Ops.IF else ""
        if src[1].dtype.count > 1:
          kk(gate + f"st{mem_type}.v{src[1].dtype.count}.{self.mem_types[src[1].dtype.scalar()]} [{r[src[0]]}+0], {{{', '.join(r[src[1]])}}};")
        else:
          kk(gate + f"st{mem_type}.{self.mem_types[src[1].dtype]} [{r[src[0]]}+0], {r[src[1]]};")
      else:
        if uop is Ops.RANGE: kk(*self.render_loop(loop:=ssa('ridx', u), r[src[0]], "LOOP_"+loop[1:]))
        elif uop in GroupOp.ALU:
          src_dtype = src[0].dtype if uop in {BinaryOps.CMPLT, BinaryOps.CMPNE} else dtype
          kk(self.code_for_op[uop](ssa("alu", u), *[r[x] for x in src], src_dtype, self.types[src_dtype]))
        elif uop is Ops.DEFINE_ACC:
          if dtype.count > 1:
            r[u] = [ssa('acc', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            for uu in r[u]: kk(f"mov.b{self.types[dtype.scalar()][1:]} {uu}, {const(src[0].src[0].arg, dtype.scalar())};")
          else: kk(f"mov.{f'b{self.types[dtype][1:]}' if dtype != dtypes.bool else 'pred'} {ssa('acc', u)}, {const(src[0].arg, dtype)};")
        elif uop is Ops.SPECIAL:
          assert args[0][0] != "i", "idx not supported"
          kk(f"mov.u32 %{args[0]}, %{'ctaid' if args[0][0] == 'g' else 'tid'}.{chr(120+int(args[0][-1]))};")
          r[u] = "%" + args[0]
          kernel = [f".reg .u32 %{args[0]};"] + kernel
        elif uop is Ops.DEFINE_VAR:
          bufs.append((args[0], dtype))
          r[u] = f"%{args[0]}"
          kk(*self.render_load(args[0], ssa('dat', u, self.types[dtype]), dtype, ss=".param"))
        elif uop is Ops.CONST: r[u] = const(args, dtype, mov=True)
        elif uop is Ops.GEP:
          assert len(u.arg) == 1
          r[u] = r[src[0]][u.arg[0]]
        elif uop is Ops.LOAD:
          assert src[0].dtype == dtypes.int64, "load isn't int64"
          mem_type = '.shared' if src[0].op is Ops.DEFINE_LOCAL or any(x.op is Ops.DEFINE_LOCAL for x in src[0].parents) else '.global'
          has_gate = len(src) > 2 and src[2].op in GroupOp.ALU
          if dtype.count > 1:
            r[u] = [ssa('val', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            if has_gate:
              for v in r[u]: kk(f"mov.{self.mem_types[dtype.scalar()]} {v}, {render_val(0, dtype.scalar())};")
            kk((f"@{r[src[2]]}" if has_gate else "")
              + f" ld{mem_type}.v{dtype.count}.{self.mem_types[dtype.scalar()]} {{{', '.join(r[u])}}}, [{r[src[0]]}+0];")
          else:
            kk(*self.render_load(r[src[0]], ssa('val', u), dtype, gate=r[src[2]] if has_gate else None,
                                alt=r[src[1]] if has_gate else None, ss=mem_type, offset=0))
        elif uop is Ops.ASSIGN:
          if dtype.count > 1:
            for x0, x1 in zip(r[src[0]], r[src[1]]): kk(f"mov.b{self.types[dtype.scalar()][1:]} {x0}, {x1};")
          else: kk(f"mov.{f'b{self.types[dtype][1:]}' if dtype != dtypes.bool else 'pred'} {r[src[0]]}, {r[src[1]]};")
          r[u] = r[src[0]]
        # NOTE: casting to str is fine because you can't vectorize a vectorize
        elif uop is Ops.VECTORIZE: r[u] = [cast(str,r[x]) for x in src]
        elif uop in {Ops.CAST, Ops.BITCAST}:
          _cast(r[src[0]], dtype, src[0].dtype, bitcast=uop is Ops.BITCAST, u=u)
        elif uop is Ops.DEFINE_LOCAL:
          # TODO: we should sum these, and fetch 0xC000 from somewhere
          assert args[1]*dtype.itemsize <= 0xC000, "too large local"
          kk(*self.render_local(ssa('local', u, self.types[dtypes.ulong]), args[0], args[1], dtype))
        elif uop is Ops.DEFINE_GLOBAL:
          bufs.append((nm:=f"data{args}", dtype))
          r[u] = f"%{nm}"
          dt = dtypes.ulong if dtype.__class__ == PtrDType else dtype
          kk(*self.render_load(nm, ssa('dat', u, self.types[dt]), dt, ss=".param"))
        elif uop is Ops.WMMA:
          _, (N, M, K), dtype_in, _, _, _, upcast_axes, _ = args
          wmma, n_operands = [], tuple(prod(sz for _, sz in upc)*dtype_in.itemsize//4 for upc in upcast_axes[:2])
          dt_map = { dtypes.half: "f16" }
          for vv in src[:2]:
            for i in range(0, len(r[vv]), 2):
              wmma.append(ssa("wmma", dtype="b32"))
              kk(f'mov.b32 {wmma[-1]}, {{{", ".join(r[vv][i:i+2])}}};')
          r[u] = [ssa("wmma", dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
          kk(f'mma.sync.aligned.m{M}n{N}k{K}.row.col.f32.{dt_map[dtype_in]}.{dt_map[dtype_in]}.f32\
            {{{", ".join(r[u])}}}, {{{", ".join(wmma[:n_operands[0]])}}}, {{{", ".join(wmma[-n_operands[1]:])}}}, {{{", ".join(r[src[2]])}}};')
        else: raise NotImplementedError(f"no code for {uop}")

    return self.render_kernel(kernel, name, bufs, c.items())

