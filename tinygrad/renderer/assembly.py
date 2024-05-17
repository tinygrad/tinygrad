from typing import DefaultDict, Dict, List, Union, Optional, cast, Callable
import struct, copy
from collections import defaultdict
from tinygrad.helpers import DEBUG
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType
from tinygrad.codegen.uops import UOpGraph, PatternMatcher
from tinygrad.renderer import Renderer, TensorCore

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    if dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

class PTXRenderer(Renderer):
  device = "CUDA"
  suffix = "PTX"
  global_max = [65535, 65535, 2147483647]
  local_max = [64, 1024, 1024]
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(0,2)], thread_local_sizes=[[2,2,2],[2,2],[2,2]], thread_local_aliases=[ [[0],[0],[5],[-2],[0],[-1,1,2,-3],[3,4]], [[3],[4],[0],[0],[5],[-1,1,2,-2],[0]], [[-1],[1],[5],[-2],[2],[0],[3,4]] ], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float)])] # noqa: E501
  def __init__(self, arch:str): self.tensor_cores = PTXRenderer.tensor_cores if int(arch[3:]) >= 80 else []

  # language options
  kernel_prefix = """.version VERSION
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
  asm_for_op: Dict[Op, Callable] = {
    UnaryOps.NEG: lambda d,a,dt,name: f"not.pred {d}, {a};" if name == "pred" else f"neg.{name} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", UnaryOps.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"{'or' if name == 'pred' else 'add'}.{name} {d}, {a}, {b};",
    BinaryOps.SUB: lambda d,a,b,dt,name: f"sub.{name} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dt,name: ('and' if dt == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if name == "pred" else f"xor.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.DIV: lambda d,a,b,dt,name: f"div{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};",
    BinaryOps.CMPEQ: lambda d,a,b,dt,name: f"setp.eq.{name} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dt,name: f"fma.rn.{name} {d}, {a}, {b}, {c};",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name:
      f"@{a} mov.{name} {d}, {b};\n@!{a} mov.{name} {d}, {c};" if name == "pred" else f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
  }
  supports_half: List[Op] = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT,
                             TernaryOps.WHERE]
  # HACK: Use s16 and u16 for int8 and uint8 buffers. This can be wrong in cast.
  types: Dict[DType, str] = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
                              dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                              dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }

  const_requires_mov: List[DType] = [dtypes.half, dtypes.bool]

  def render_const(self, x:ConstType, dtype:DType, mov=None) -> Union[List[str], str]:
    val = render_val(x, dtype)
    if dtype == dtypes.bool: return [f"setp.ne.s16 {mov}, {val}, 0;"]
    return [f"mov.b{self.types[dtype][1:]} {mov}, {val};"] if mov else val

  def render_local(self, dest, name, size, dtype) -> List[str]:
    return [f".shared .align 4 .b8 {name}[{size*dtype.itemsize}];", f"mov.u64 {dest}, {name}[0];"]

  def render_loop(self, idx, start, label, acc=None) -> List[str]: return [f"mov.u32 {idx}, {start};", f"{label}:"]

  def render_bra(self, b1, pred=None, b2=None) -> List[str]: return [f"@{pred} bra {b1};", f"@!{pred} bra {b2};"] if pred else [f"bra {b1};"]

  def mem_type(self, dtype): return 's8' if dtype.itemsize == 1 else 'b16' if dtype == dtypes.float16 else self.types[dtype]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]:
    assert dtype is not dtypes.bool
    if gate: return [f"@{gate} ld{ss}.{self.mem_type(dtype)} {dest}, [{loc}+{offset}];", f"@!{gate} mov.b{self.types[dtype][1:]} {dest}, {alt};"]
    return [f"ld{ss}.{self.mem_type(dtype)} {dest}, [{loc}+{offset}];"]

  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]:
    return [(f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}+{offset}], {val};"]

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]:
    if bitcast: return [f"mov.b{self.types[dtype][1:]} {d}, {a};"]
    if atype == dtypes.bool: return[f"selp.b{self.types[dtype][1:]} {d}, {render_val(1, dtype)}, {render_val(0, dtype)}, {a};"]
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

  def render(self, name:str, _uops:UOpGraph) -> str:
    # editing the uops breaks beam search
    uops = copy.deepcopy(_uops)
    kernel:List[str] = []
    bufs = []

    uops.linearize(ptx_matcher)
    if DEBUG >= 4: uops.print()

    def kk(*s: str): kernel.append("\n".join(s))

    c: DefaultDict[str, int] = defaultdict(int)
    r: Dict[UOp, Union[List[str], str]] = {}
    def ssa(prefix:str, u:Optional[UOp]=None, dtype:Optional[str]=None) -> str:
      nonlocal c, r
      prefix += f"_{dtype if dtype is not None else self.types[cast(DType, cast(UOp, u).dtype)]}_"
      c[prefix] += 1
      if u is not None: r[u] = f"%{prefix}{c[prefix]-1}"
      return f"%{prefix}{c[prefix]-1}"

    c_label: DefaultDict[str, int] = defaultdict(int)
    r_label: Dict[UOp, str] = {}
    def ssa_label(prefix:str, u:UOp):
      nonlocal c_label, r_label
      c_label[prefix] += 1
      r_label[u] = f"{self.label_prefix}{prefix}_{c_label[prefix]-1}"
      return r_label[u]

    def const(x:ConstType, dtype:DType, mov=False):
      if mov or dtype in self.const_requires_mov:
        kk(*self.render_const(x, dtype, mov=(out:=ssa('const', dtype=self.types[dtype]))))
        return out
      return self.render_const(x, dtype)

    def _cast(a, dtype:DType, atype:DType, bitcast=False, u=None, pred=False):
      if atype == dtype or isinstance(atype, PtrDType):
        if u: r[u] = a
        return a
      kk(*self.render_cast((ret:=ssa('cast', u, self.types[dtype])), a, dtype, atype, bitcast))
      return ret

    for u in uops:
      uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
      if uop is UOps.IF:
        assert vin[0].dtype is not None
        kk(*self.render_bra(lb:=ssa_label('if', u), _cast(r[vin[0]], dtypes.bool, vin[0].dtype, u=u, pred=True), f"{lb}_true"), f"{lb}_true:")
      elif uop is UOps.BARRIER and self.barrier: kk(self.barrier)
      elif uop is UOps.ENDLOOP:
        kk(self.asm_for_op[BinaryOps.ADD](r[vin[0]], r[vin[0]], "1", dtypes.int, self.types[dtypes.int]),
            self.asm_for_op[BinaryOps.CMPLT](pred:=ssa("pred", dtype="pred"), r[vin[0]], r[vin[0].vin[1]], dtypes.int, self.types[dtypes.int]))
        kk(*self.render_bra(r_label[vin[0]], pred, f"{r_label[vin[0]]}_exit"), f"{r_label[vin[0]]}_exit:")
      elif uop is UOps.ENDIF:
        kk(f"{r_label[vin[0]]}:")
      elif uop is UOps.STORE:
        assert vin[0].dtype is not None and vin[2].dtype is not None
        assert vin[0].dtype is dtypes.int64, "store isn't int64"
        assert vin[1].uop is UOps.CONST, f"store isn't const {u}"
        mem_type = '.shared' if vin[0].uop is UOps.DEFINE_LOCAL or any(x.uop is UOps.DEFINE_LOCAL for x in vin[0].parents) else '.global'
        if vin[2].dtype.count > 1:
          kk((f"@{r[vin[3]]} " if len(vin)>3 else "") + \
              f"st{mem_type}.v{vin[2].dtype.count}.{self.mem_type(vin[2].dtype.scalar())} [{r[vin[0]]}+{vin[1].arg}], {{{', '.join(r[vin[2]])}}};")
        else:
          kk(*self.render_store(r[vin[0]], r[vin[2]], vin[2].dtype, gate=r[vin[3]] if len(vin)>3 else None, ss=mem_type, offset=vin[1].arg))
      else:
        assert dtype is not None, f"None dtype for uop {uop}"
        if uop is UOps.LOOP: kk(*self.render_loop(ssa('ridx', u), r[vin[0]], ssa_label('loop', u)))
        elif uop is UOps.ALU:
          assert vin[0].dtype is not None
          if args is BinaryOps.CMPLT or args is BinaryOps.CMPEQ:
            # pass in the other dtype here
            kk(self.asm_for_op[args](ssa("alu", u), *[r[x] for x in vin], vin[0].dtype, self.types[vin[0].dtype]))
          else:
            kk(self.asm_for_op[args](ssa("alu", u), *[r[x] for x in vin], dtype, self.types[dtype]))
        elif uop is UOps.DEFINE_ACC:
          if dtype.count > 1:
            r[u] = [ssa('acc', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            for uu in r[u]: kk(f"mov.b{self.types[dtype.scalar()][1:]} {uu}, {const(args[0], dtype.scalar())};")
          else: kk(f"mov.b{self.types[dtype][1:]} {ssa('acc', u)}, {const(args[0], dtype)};")
        elif uop is UOps.SPECIAL:
          assert args[1][0] != "i", "idx not supported"
          kk(f"mov.u32 %{args[1]}, {(self.gid if args[1][0] == 'g' else self.lid)[args[0]]};")
          r[u] = "%" + args[1]
          kernel = [f".reg .u32 %{args[1]};"] + kernel
        elif uop is UOps.CONST:
          if dtype.count > 1: r[u] = [const(args, dtype.scalar(), mov=True) for _ in range(dtype.count)]
          else: r[u] = const(args, dtype, mov=True)
        elif uop is UOps.GEP: r[u] = r[vin[0]][u.arg]
        elif uop is UOps.LOAD:
          assert vin[0].dtype is dtypes.int64, "load isn't int64"
          assert vin[1].uop is UOps.CONST, f"load isn't const {u}"
          mem_type = '.shared' if vin[0].uop is UOps.DEFINE_LOCAL or any(x.uop is UOps.DEFINE_LOCAL for x in vin[0].parents) else '.global'
          if dtype.count > 1:
            r[u] = [ssa('val', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            if(len(vin)>3):
              for v in r[u]: kk(f"mov.{self.mem_type(dtype.scalar())} {v}, {render_val(0, dtype.scalar())};")
            kk((f"@{r[vin[2]]}"if len(vin) > 3 else "")
              + f" ld{mem_type}.v{dtype.count}.{self.mem_type(dtype.scalar())} {{{', '.join(r[u])}}}, [{r[vin[0]]}+{vin[1].arg}];")
          else:
            kk(*self.render_load(r[vin[0]], ssa('val', u), dtype, gate=r[vin[2]] if len(vin) > 3 else None,
                                alt=r[vin[3]] if len(vin) > 3 else None, ss=mem_type, offset=vin[1].arg))
        elif uop is UOps.PHI:
          if dtype.count > 1:
            for x0, x1 in zip(r[vin[0]], r[vin[1]]): kk(f"mov.b{self.types[dtype.scalar()][1:]} {x0}, {x1};")
          else:
            kk(f"mov.b{self.types[dtype][1:]} {r[vin[0]]}, {r[vin[1]]};")
          r[u] = r[vin[0]]
        elif uop in {UOps.CAST, UOps.BITCAST}:
          assert vin[0].dtype is not None
          if dtype.count>1: r[u] = [r[x] for x in vin] # type: ignore
          else: _cast(r[vin[0]], dtype, vin[0].dtype, bitcast=uop is UOps.BITCAST, u=u)
        elif uop is UOps.DEFINE_LOCAL:
          # TODO: we should sum these, and fetch 0xC000 from somewhere
          assert args[1]*dtype.itemsize <= 0xC000, "too large local"
          kk(*self.render_local(ssa('local', u, self.types[dtypes.ulong]), args[0], args[1], dtype))
        elif uop is UOps.DEFINE_VAR:
          bufs.append((args.expr, dtype))
          r[u] = f"%{args.expr}"
          if self.load_global: kk(*self.render_load(args.expr, ssa('dat', u, self.types[dtype]), dtype, ss=".param"))
        elif uop is UOps.DEFINE_GLOBAL:
          bufs.append((nm:=f"data{args[0]}", dtype))
          r[u] = f"%{nm}"
          if self.load_global:
            dt = dtypes.ulong if dtype.__class__ == PtrDType else dtype
            kk(*self.render_load(nm, ssa('dat', u, self.types[dt]), dt, ss=".param"))
        elif uop is UOps.WMMA:
          wmma = []
          for vv in vin[:2]:
            for i in range(0, len(r[vv]), 2):
              wmma.append(ssa("wmma", dtype="b32"))
              kk(f'mov.b32 {wmma[-1]}, {{{", ".join(r[vv][i:i+2])}}};')
          r[u] = r[vin[2]]
          kk(f'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\
            {{{", ".join(r[u])}}}, {{{", ".join(wmma[:4])}}}, {{{", ".join(wmma[4:])}}}, {{{", ".join(r[u])}}};')
        else: raise NotImplementedError(f"no code for {uop}")

    return self.render_kernel(kernel, name, bufs, c.items())

ptx_matcher = PatternMatcher([
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPEQ, "vin": ({"dtype": dtypes.bool},{})},
  lambda root: UOp(UOps.ALU, dtypes.bool, (UOp(root.uop, root.dtype, root.vin, BinaryOps.XOR),), UnaryOps.NEG)),
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({"__name__": "x", "dtype": dtypes.bool},{"__name__": "y"})},
  lambda root,x,y: UOp(root.uop, root.dtype, (UOp(UOps.ALU, dtypes.bool, (x,), UnaryOps.NEG), y), BinaryOps.MUL)),
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.ADD, "dtype": set([dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]),
    "vin": [{"__name__": "non_muls"}, {"__name__": "muls", "uop": UOps.ALU, "arg": BinaryOps.MUL}]},
    lambda root, muls, non_muls: UOp(UOps.ALU, root.dtype, muls.vin + (non_muls,), TernaryOps.MULACC)),
  *[({"__name__": "x", "uop": UOps.ALU, "dtype": dtypes.half, "arg": op},
    lambda x: UOp(UOps.CAST, dtypes.half, (UOp(x.uop, dtypes.float32, tuple([UOp(UOps.CAST, dtypes.float32, (vv,)) for vv in x.vin]), x.arg),)))
    for op in PTXRenderer.asm_for_op.keys() if op not in PTXRenderer.supports_half],
  ({"__name__": "root", "uop": UOps.LOAD, "dtype": dtypes.bool,
    "vin": ({"__name__": "x"},{"__name__": "y"},{"__name__": "z"},{"__name__": "k"})},
  lambda root,x,y,z,k: UOp(UOps.CAST, dtypes.bool, (UOp(root.uop, dtypes.int8, (x,y,z,UOp(UOps.CAST, dtypes.uint8, (k,)))),), root.arg)),
  ({"__name__": "root", "uop": UOps.LOAD,"dtype": dtypes.bool, "vin": ({},{})},
  lambda root: UOp(UOps.CAST, dtypes.bool, (UOp(root.uop, dtypes.uint8, root.vin, root.arg),))),
  ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{"__name__": "z","dtype": dtypes.bool}, {})},
  lambda root,z: UOp(root.uop, root.dtype, root.vin[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,)),), root.arg)),
  ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{"__name__": "z","dtype": dtypes.bool})},
  lambda root,z: UOp(root.uop, root.dtype, root.vin[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,)),), root.arg)),
  ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{},{"__name__": "g", "dtype": dtypes.int})},
  lambda root,g: UOp(root.uop, root.dtype, root.vin[:3] + (UOp(UOps.CAST, dtypes.bool, (g,)),), root.arg)),
  # ptr_ar (load/store)
  ({"__name__": "root", "uop": {UOps.LOAD, UOps.STORE}, "__allow_len__":[2,3,4,5], "vin": ({"uop":{UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}},
                                                                              {"__name__": "const", "uop":UOps.CONST})},
    lambda root, const: UOp(root.uop, root.dtype, (root.vin[0].cast(dtypes.int64),
                                UOp.const(dtypes.int64, const.arg * root.vin[0].dtype.itemsize),
                                                  )+root.vin[2:])),
  ({"__name__": "root", "uop": {UOps.LOAD, UOps.STORE}, "__allow_len__":[2,3,4,5], "vin": ({"uop":{UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}},
                                                                              {"__name__": "alu"})},  # no const here
    lambda root, alu: UOp(root.uop, root.dtype,
      (alu.cast(dtypes.int64)*UOp.const(dtypes.int64, root.vin[0].dtype.itemsize)+root.vin[0].cast(dtypes.int64),
        UOp.const(dtypes.int64, 0))+root.vin[2:])),
])
