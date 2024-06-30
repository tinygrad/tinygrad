from typing import DefaultDict, Dict, List, Union, Optional, cast, Callable
import struct, math
from collections import defaultdict
from tinygrad.helpers import DEBUG
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType
from tinygrad.codegen.uops import UOps, UOp, UOpGraph, PatternMatcher, UPat
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
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(0,2)], thread_local_sizes=[[2,2,2],[2,2],[2,2]], thread_local_aliases=[ [[0],[0],[5],[-2],[0],[-1,1,2,-3],[3,4]], [[3],[4],[0],[0],[5],[-1,1,2,-2],[0]], [[-1],[1],[5],[-2],[2],[0],[3,4]] ], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float)])] # noqa: E501
  def __init__(self, arch:str, device="CUDA"): self.device, self.tensor_cores = device, PTXRenderer.tensor_cores if int(arch[3:]) >= 80 else []

  # language options
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  gid = [f'%ctaid.{chr(120+i)}' for i in range(3)]
  gdim = [f'%nctaid.{chr(120+i)}' for i in range(3)]
  lid = [f'%tid.{chr(120+i)}' for i in range(3)]
  asm_for_op: Dict[Op, Callable] = {
    UnaryOps.NEG: lambda d,a,dt,name: f"not.pred {d}, {a};" if name == "pred" else f"sub.{name} {d}, 0, {a};" if dtypes.is_unsigned(dt) \
      else f"neg.{name} {d}, {a};",
    UnaryOps.RECIP: lambda d,a,dt,name: f"rcp{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", UnaryOps.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
    BinaryOps.SHR: lambda d,a,b,dt,name: f"shr.{name} {d}, {a}, {b};", BinaryOps.SHL: lambda d,a,b,dt,name: f"shl.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"{'or' if name == 'pred' else 'add'}.{name} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dt,name: ('and' if dt == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if name == "pred" else f"xor.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.IDIV: lambda d,a,b,dt,name: f"div.{name} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};",
    BinaryOps.CMPNE: lambda d,a,b,dt,name: f"setp.ne.{name} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dt,name: f"{'fma.rn' if dtypes.is_float(dt) else 'mad.lo'}.{name} {d}, {a}, {b}, {c};",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name:
      f"@{a} mov.{name} {d}, {b};\n@!{a} mov.{name} {d}, {c};" if name == "pred" else f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
  }
  supports_half: List[Op] = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT,
                             TernaryOps.WHERE]
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

  def render_bra(self, b1, pred=None) -> List[str]: return [f"@{pred} bra {b1};"] if pred else [f"bra {b1};"]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]:
    assert dtype != dtypes.bool
    if gate: return [f"@{gate} ld{ss}.{self.mem_types[dtype]} {dest}, [{loc}+{offset}];", f"@!{gate} mov.b{self.types[dtype][1:]} {dest}, {alt};"]
    return [f"ld{ss}.{self.mem_types[dtype]} {dest}, [{loc}+{offset}];"]

  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]:
    return [(f"@{gate} " if gate else "") + f"st{ss}.{self.mem_types[dtype]} [{loc}+{offset}], {val};"]

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

  def render(self, name:str, uops:UOpGraph) -> str:
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
      uop,dtype,src,args = u.op,u.dtype,u.src,u.arg
      if uop is UOps.IF:
        assert src[0].dtype is not None
        kk(*self.render_bra(f"IF_{r[src[0]][1:]}_{cast(List, uops._uops).index(u)}", _cast(r[src[0]], dtypes.bool, src[0].dtype, u=u, pred=True)))
      elif uop is UOps.BARRIER and self.barrier: kk(self.barrier)
      elif uop is UOps.ENDRANGE:
        kk(self.asm_for_op[BinaryOps.ADD](r[src[0]], r[src[0]], "1", dtypes.int, self.types[dtypes.int]),
            self.asm_for_op[BinaryOps.CMPLT](pred:=ssa("pred", dtype="pred"), r[src[0]], r[src[0].src[1]], dtypes.int, self.types[dtypes.int]))
        kk(*self.render_bra(f"LOOP_{r[src[0]][1:]}", pred))
      elif uop is UOps.ENDIF:
        kk(f"IF_{r[src[0].src[0]][1:]}_{cast(List, uops._uops).index(src[0])}:")
      elif uop is UOps.STORE:
        assert src[0].dtype is not None and src[2].dtype is not None
        assert src[0].dtype == dtypes.int64, "store isn't int64"
        assert src[1].op is UOps.CONST, f"store isn't const {u}"
        mem_type = '.shared' if src[0].op is UOps.DEFINE_LOCAL or any(x.op is UOps.DEFINE_LOCAL for x in src[0].parents) else '.global'
        if src[2].dtype.count > 1:
          kk((f"@{r[src[3]]} " if len(src)>3 else "") + \
              f"st{mem_type}.v{src[2].dtype.count}.{self.mem_types[src[2].dtype.scalar()]} [{r[src[0]]}+{src[1].arg}], {{{', '.join(r[src[2]])}}};")
        else:
          kk(*self.render_store(r[src[0]], r[src[2]], src[2].dtype, gate=r[src[3]] if len(src)>3 else None, ss=mem_type, offset=src[1].arg))
      else:
        assert dtype is not None, f"None dtype for uop {uop}"
        if uop is UOps.RANGE: kk(*self.render_loop(loop:=ssa('ridx', u), r[src[0]], "LOOP_"+loop[1:]))
        elif uop is UOps.ALU:
          assert src[0].dtype is not None
          if args is BinaryOps.CMPLT or args is BinaryOps.CMPNE:
            # pass in the other dtype here
            kk(self.asm_for_op[args](ssa("alu", u), *[r[x] for x in src], src[0].dtype, self.types[src[0].dtype]))
          else:
            kk(self.asm_for_op[args](ssa("alu", u), *[r[x] for x in src], dtype, self.types[dtype]))
        elif uop is UOps.DEFINE_ACC:
          if dtype.count > 1:
            r[u] = [ssa('acc', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            for uu in r[u]: kk(f"mov.b{self.types[dtype.scalar()][1:]} {uu}, {const(src[0].arg, dtype.scalar())};")
          else: kk(f"mov.b{self.types[dtype][1:]} {ssa('acc', u)}, {const(src[0].arg, dtype)};")
        elif uop is UOps.SPECIAL:
          assert args[1][0] != "i", "idx not supported"
          kk(f"mov.u32 %{args[1]}, {(self.gid if args[1][0] == 'g' else self.lid)[args[0]]};")
          r[u] = "%" + args[1]
          kernel = [f".reg .u32 %{args[1]};"] + kernel
        elif uop is UOps.CONST:
          if dtype.count > 1: r[u] = [const(args, dtype.scalar(), mov=True) for _ in range(dtype.count)]
          else: r[u] = const(args, dtype, mov=True)
        elif uop is UOps.GEP: r[u] = r[src[0]][u.arg]
        elif uop is UOps.LOAD:
          assert src[0].dtype == dtypes.int64, "load isn't int64"
          assert src[1].op is UOps.CONST, f"load isn't const {u}"
          mem_type = '.shared' if src[0].op is UOps.DEFINE_LOCAL or any(x.op is UOps.DEFINE_LOCAL for x in src[0].parents) else '.global'
          if dtype.count > 1:
            r[u] = [ssa('val', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            if(len(src)>3):
              for v in r[u]: kk(f"mov.{self.mem_types[dtype.scalar()]} {v}, {render_val(0, dtype.scalar())};")
            kk((f"@{r[src[2]]}"if len(src) > 3 else "")
              + f" ld{mem_type}.v{dtype.count}.{self.mem_types[dtype.scalar()]} {{{', '.join(r[u])}}}, [{r[src[0]]}+{src[1].arg}];")
          else:
            kk(*self.render_load(r[src[0]], ssa('val', u), dtype, gate=r[src[2]] if len(src) > 3 else None,
                                alt=r[src[3]] if len(src) > 3 else None, ss=mem_type, offset=src[1].arg))
        elif uop is UOps.PHI:
          if dtype.count > 1:
            for x0, x1 in zip(r[src[0]], r[src[1]]): kk(f"mov.b{self.types[dtype.scalar()][1:]} {x0}, {x1};")
          else:
            kk(f"mov.b{self.types[dtype][1:]} {r[src[0]]}, {r[src[1]]};")
          r[u] = r[src[0]]
        elif uop in {UOps.CAST, UOps.BITCAST}:
          assert src[0].dtype is not None
          if dtype.count>1: r[u] = [r[x] for x in src] # type: ignore
          else: _cast(r[src[0]], dtype, src[0].dtype, bitcast=uop is UOps.BITCAST, u=u)
        elif uop is UOps.DEFINE_LOCAL:
          # TODO: we should sum these, and fetch 0xC000 from somewhere
          assert args[1]*dtype.itemsize <= 0xC000, "too large local"
          kk(*self.render_local(ssa('local', u, self.types[dtypes.ulong]), args[0], args[1], dtype))
        elif uop is UOps.DEFINE_VAR:
          bufs.append((args.expr, dtype))
          r[u] = f"%{args.expr}"
          kk(*self.render_load(args.expr, ssa('dat', u, self.types[dtype]), dtype, ss=".param"))
        elif uop is UOps.DEFINE_GLOBAL:
          bufs.append((nm:=f"data{args[0]}", dtype))
          r[u] = f"%{nm}"
          dt = dtypes.ulong if dtype.__class__ == PtrDType else dtype
          kk(*self.render_load(nm, ssa('dat', u, self.types[dt]), dt, ss=".param"))
        elif uop is UOps.WMMA:
          wmma = []
          for vv in src[:2]:
            for i in range(0, len(r[vv]), 2):
              wmma.append(ssa("wmma", dtype="b32"))
              kk(f'mov.b32 {wmma[-1]}, {{{", ".join(r[vv][i:i+2])}}};')
          r[u] = [ssa("wmma", dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
          kk(f'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\
            {{{", ".join(r[u])}}}, {{{", ".join(wmma[:4])}}}, {{{", ".join(wmma[4:])}}}, {{{", ".join(r[src[2]])}}};')
        else: raise NotImplementedError(f"no code for {uop}")

    return self.render_kernel(kernel, name, bufs, c.items())

ptx_matcher = PatternMatcher([
  (UPat(UOps.ALU, BinaryOps.MUL, name="root", dtype=set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
      src=[UPat(UOps.CONST, set([2**i for i in range(64)]), name="const"), UPat(name="mul")]),
    lambda root, mul, const: UOp(UOps.ALU, root.dtype, (mul, UOp.const(dtypes.int, int(math.log2(const.arg)))), BinaryOps.SHL)),
  (UPat(UOps.ALU, BinaryOps.IDIV, name="root", dtype=set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
      src=[UPat(UOps.CONST, set([2**i for i in range(64)]), name="const"), UPat(name="div")]),
    lambda root, div, const: UOp(UOps.ALU, root.dtype, (div, UOp.const(dtypes.int, int(math.log2(const.arg)))), BinaryOps.SHR)),
  (UPat(UOps.ALU, BinaryOps.CMPNE, (UPat(dtype=dtypes.bool),UPat()), "root"), lambda root: UOp(root.op, root.dtype, root.src, BinaryOps.XOR)),
  (UPat(UOps.ALU, BinaryOps.CMPLT, (UPat(name="x", dtype=dtypes.bool),UPat(name="y")), "root"),
    lambda root,x,y: UOp(root.op, root.dtype, (UOp(UOps.ALU, dtypes.bool, (x,), UnaryOps.NEG), y), BinaryOps.MUL)),
  (UPat(UOps.ALU, BinaryOps.ADD,
    [UPat(name="non_muls"), UPat(UOps.ALU, BinaryOps.MUL, name="muls")], "root"),
    lambda root, muls, non_muls: UOp(UOps.ALU, root.dtype, muls.src + (non_muls,), TernaryOps.MULACC)),
  *[(UPat(UOps.ALU, op, dtype=dtypes.half, name="x"),
    lambda x: UOp(UOps.CAST, dtypes.half, (UOp(x.op, dtypes.float32, tuple([UOp(UOps.CAST, dtypes.float32, (vv,)) for vv in x.src]), x.arg),)))
    for op in PTXRenderer.asm_for_op.keys() if op not in PTXRenderer.supports_half],
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(name="x"),UPat(name="y"),UPat(name="z"),UPat(name="k"))),
    lambda root,x,y,z,k: UOp(UOps.CAST, dtypes.bool, (UOp(root.op, dtypes.int8, (x,y,z,UOp(UOps.CAST, dtypes.uint8, (k,)))),), root.arg)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())),
    lambda root: UOp(UOps.CAST, dtypes.bool, (UOp(root.op, dtypes.uint8, root.src, root.arg),))),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool), UPat())),
    lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,)),), root.arg)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool))),
    lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,)),), root.arg)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(),UPat(name="g", dtype=dtypes.int))),
    lambda root,g: UOp(root.op, root.dtype, root.src[:3] + (UOp(UOps.CAST, dtypes.bool, (g,)),), root.arg)),
  # ptr_ar (load/store)
  (UPat({UOps.LOAD, UOps.STORE}, name="root", allow_len={2,3,4,5}, src=(UPat({UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}),
                               UPat(UOps.ALU, BinaryOps.ADD, src=[UPat(name="alu"), UPat(UOps.CONST, name="const")]))),
    lambda root, alu, const: UOp(root.op, root.dtype,
      (alu.cast(dtypes.int64)*UOp.const(dtypes.int64, root.src[0].dtype.itemsize)+root.src[0].cast(dtypes.int64),
       UOp.const(const.dtype, root.src[0].dtype.itemsize)*const)+root.src[2:])),
  (UPat({UOps.LOAD, UOps.STORE}, name="root", allow_len={2,3,4,5}, src=(UPat({UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}),
                                                                              UPat(UOps.CONST, name="const"))),
    lambda root, const: UOp(root.op, root.dtype, (root.src[0].cast(dtypes.int64),
                                UOp.const(dtypes.int64, const.arg * root.src[0].dtype.itemsize),
                                                  )+root.src[2:])),
  (UPat({UOps.LOAD, UOps.STORE}, name="root", allow_len={2,3,4,5}, src=(UPat({UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}),
                                                                              UPat(name="alu"))),  # no const here
    lambda root, alu: UOp(root.op, root.dtype,
      (alu.cast(dtypes.int64)*UOp.const(dtypes.int64, root.src[0].dtype.itemsize)+root.src[0].cast(dtypes.int64),
        UOp.const(dtypes.int64, 0))+root.src[2:])),
])

# Begin RDNA Renderer
class RDNARenderer(Renderer):
  device = "AMD"
  suffix = "RDNA"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 65536
  tensor_cores: List[TensorCore] = []  # RDNA3 doesn't have tensor cores like CUDA

  def __init__(self, arch:str):
    self.arch = arch

  kernel_prefix = """
.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.text
.p2align 8
.global KERNEL_NAME
.type KERNEL_NAME,@function
KERNEL_NAME:"""

  barrier = "s_barrier"
  gid = [f'v[0]', f'v[1]', f'v[2]']
  lid = [f'v[3]', f'v[4]', f'v[5]']

  asm_for_op: Dict[Op, Callable] = {
    UnaryOps.NEG: lambda d,a,dt,name: f"v_sub_f32 {d}, 0, {a}" if dtypes.is_float(dt) else f"v_sub_i32 {d}, 0, {a}",
    UnaryOps.RECIP: lambda d,a,dt,name: f"v_rcp_f32 {d}, {a}",
    UnaryOps.EXP2: lambda d,a,dt,name: f"v_exp_f32 {d}, {a}",
    UnaryOps.LOG2: lambda d,a,dt,name: f"v_log_f32 {d}, {a}",
    UnaryOps.SIN: lambda d,a,dt,name: f"v_sin_f32 {d}, {a}",
    UnaryOps.SQRT: lambda d,a,dt,name: f"v_sqrt_f32 {d}, {a}",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"v_add_{'f' if dtypes.is_float(dt) else 'i'}32 {d}, {a}, {b}",
    BinaryOps.SUB: lambda d,a,b,dt,name: f"v_sub_{'f' if dtypes.is_float(dt) else 'i'}32 {d}, {a}, {b}",
    BinaryOps.MUL: lambda d,a,b,dt,name: f"v_mul_{'f' if dtypes.is_float(dt) else 'i'}32 {d}, {a}, {b}",
    BinaryOps.DIV: lambda d,a,b,dt,name: f"v_div_scale_f32 {d}, vcc, {b}, {b}, {a}\n\tv_div_fmas_f32 {d}, {d}, {b}, {a}\n\tv_div_fixup_f32 {d}, {d}, {b}, {a}",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"v_max_{'f' if dtypes.is_float(dt) else 'i'}32 {d}, {a}, {b}",
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"v_cmp_lt_{'f' if dtypes.is_float(dt) else 'i'}32 {d}, {a}, {b}",
    TernaryOps.MULACC: lambda d,a,b,c,dt,name: f"v_fma_f32 {d}, {a}, {b}, {c}" if dtypes.is_float(dt) else f"v_mad_i32_i24 {d}, {a}, {b}, {c}",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name: f"v_cndmask_b32 {d}, {c}, {b}, {a}",
  }

  types: Dict[DType, str] = {
    dtypes.int8: "s8", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
    dtypes.uint8: "u8", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
    dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "b32"
  }

  def render_const(self, x:ConstType, dtype:DType, mov=None) -> Union[List[str], str]:
    val = render_val(x, dtype)
    if dtype == dtypes.bool:
      return [f"v_cmp_eq_u32 {mov}, 1, {val}"]
    return [f"v_mov_b32 {mov}, {val}"]

  def render_local(self, name:str, size:int, dtype:DType) -> List[str]:
    return [f".shared .align {dtype.itemsize} .b8 {name}[{size*dtype.itemsize}];"]

  def render_loop(self, idx:str, start:str, label:str) -> List[str]:
    return [f"v_mov_b32 {idx}, {start}", f"{label}:"]

  def render_bra(self, b1:str, pred:Optional[str]=None) -> List[str]:
    return [f"s_cbranch_vccnz {b1}" if pred else f"s_branch {b1}"]

  def render_load(self, lsrc:str, ldst:str, dtype:DType, offset:Union[str,int]=0) -> List[str]:
    return [f"global_load_{'dword' if dtype.itemsize <= 4 else 'dwordx2'} {ldst}, {lsrc}, off offset:{offset}"]

  def render_store(self, lsrc:str, ldst:str, dtype:DType, offset:Union[str,int]=0) -> List[str]:
    return [f"global_store_{'dword' if dtype.itemsize <= 4 else 'dwordx2'} {ldst}, {lsrc}, off offset:{offset}"]

  def render_cast(self, src:str, dst:str, dtype:DType, stype:DType) -> List[str]:
    if dtype == stype: return [f"v_mov_b32 {dst}, {src}"]
    if dtypes.is_float(dtype) and dtypes.is_int(stype):
      return [f"v_cvt_f32_i32 {dst}, {src}"]
    if dtypes.is_int(dtype) and dtypes.is_float(stype):
      return [f"v_cvt_i32_f32 {dst}, {src}"]
    if dtype == dtypes.bool:
      return [f"v_cmp_ne_u32 {dst}, 0, {src}"]
    raise NotImplementedError(f"cast from {stype} to {dtype} not implemented")

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[str]) -> str:
    kernel_prefix = self.kernel_prefix.replace("KERNEL_NAME", function_name)
    kernel = [f".reg .{self.types[dtype]} %{reg};" for reg,dtype in bufs] + kernel
    return kernel_prefix + '\n' + '\n'.join([f"\t{line}" for line in kernel if line])

def render(self, name:str, uops:UOpGraph) -> str:
        kernel: List[str] = []
        bufs: []

        uops.linearize()
        if DEBUG >= 4: uops.print()

        def kk(*s: str): kernel.extend(s)

        c: DefaultDict[str, int] = defaultdict(int)
        r: Dict[UOp, Union[List[str], str]] = {}
        
        def ssa(prefix:str, u:Optional[UOp]=None, dtype:Optional[DType]=None) -> str:
            nonlocal c, r
            c[prefix] += 1
            ret = f"%{prefix}{c[prefix]}"
            if u is not None: r[u] = ret
            return ret

        for u in uops:
            uop, dtype, src, args = u.op, u.dtype, u.src, u.arg
            if uop == UOps.CONST:
                r[u] = ssa('const')
                kk(*self.render_const(args, dtype, r[u]))
            elif uop == UOps.LOAD:
                r[u] = ssa('load')
                kk(*self.render_load(r[src[0]], r[u], dtype, src[1].arg if src[1].op == UOps.CONST else r[src[1]]))
            elif uop == UOps.STORE:
                kk(*self.render_store(r[src[2]], r[src[0]], src[2].dtype, src[1].arg if src[1].op == UOps.CONST else r[src[1]]))
            elif uop == UOps.ALU:
                r[u] = ssa('alu')
                kk(self.asm_for_op[args](r[u], *[r[x] for x in src], dtype, self.types[dtype]))
            elif uop == UOps.DEFINE_GLOBAL:
                bufs.append((f"data{args[0]}", dtype))
                r[u] = f"%data{args[0]}"
            elif uop == UOps.SPECIAL:
                r[u] = ssa('special')
                if args[1].startswith('g'):
                    kk(f"v_mov_b32 {r[u]}, {self.gid[args[0]]}")
                elif args[1].startswith('l'):
                    kk(f"v_mov_b32 {r[u]}, {self.lid[args[0]]}")
            elif uop == UOps.CAST:
                r[u] = ssa('cast')
                kk(*self.render_cast(r[src[0]], r[u], dtype, src[0].dtype))
            elif uop == UOps.PHI:
                r[u] = r[src[0]]
            elif uop == UOps.LABEL:
                kk(f"{args}:")
            elif uop == UOps.BARRIER:
                kk(self.barrier)
            elif uop == UOps.DEFINE_ACC:
                r[u] = ssa('acc')
                kk(f"v_mov_b32 {r[u]}, 0")
            elif uop == UOps.LOOP:
                idx = ssa('loop_idx')
                kk(*self.render_loop(idx, r[src[0]], f"LOOP_{idx[1:]}"))
                r[u] = idx
            elif uop == UOps.END:
                kk(f"s_branch LOOP_{r[src[0]][1:]}")
                kk(f"ENDLOOP_{r[src[0]][1:]}:")
            elif uop == UOps.COND_BRANCH:
                kk(*self.render_bra(args[0], r[src[0]]))
            else:
                raise NotImplementedError(f"unhandled op {uop}")

        # Generate kernel prologue
        prologue = [
            self.kernel_prefix.replace("KERNEL_NAME", name),
            *(f".reg .{self.types[dtype]} %{reg};" for reg,dtype in bufs)
        ]

        # Combine prologue, kernel body, and epilogue
        full_kernel = prologue + kernel + ["s_endpgm"]
        
        return '\n'.join(full_kernel)

def uops_to_rdna_asm(function_name:str, uops:UOpGraph) -> str:
  renderer = RDNARenderer("gfx1100")  # Assuming RDNA3 architecture
  kernel:List[str] = []
  bufs:List[str] = []

  uops.linearize()
  if DEBUG >= 4: uops.print()

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, Union[List[str], str]] = {}

  def ssa(prefix:str, u:Optional[UOp]=None, dtype:Optional[DType]=None) -> str:
    nonlocal c, r
    c[prefix] += 1
    if u is not None:
      r[u] = f"%{prefix}{c[prefix]}"
    return f"%{prefix}{c[prefix]}"

  for u in uops:
    uop, dtype, src, args = u.op, u.dtype, u.src, u.arg

    if uop == UOps.CONST:
      r[u] = ssa('const')
      kernel.extend(renderer.render_const(args, dtype, r[u]))
    elif uop == UOps.LOAD:
      r[u] = ssa('load')
      kernel.extend(renderer.render_load(r[src[0]], r[u], dtype, src[1].arg if src[1].op == UOps.CONST else r[src[1]]))
    elif uop == UOps.STORE:
      kernel.extend(renderer.render_store(r[src[2]], r[src[0]], src[2].dtype, src[1].arg if src[1].op == UOps.CONST else r[src[1]]))
    elif uop == UOps.ALU:
      r[u] = ssa('alu')
      kernel.append(renderer.asm_for_op[args](r[u], *[r[x] for x in src], dtype, renderer.types[dtype]))
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append((f"data{args[0]}", dtype))
      r[u] = f"%data{args[0]}"
    elif uop == UOps.SPECIAL:
      r[u] = ssa('special')
      if args[1].startswith('g'):
        kernel.append(f"v_mov_b32 {r[u]}, {renderer.gid[args[0]]}")
      elif args[1].startswith('l'):
        kernel.append(f"v_mov_b32 {r[u]}, {renderer.lid[args[0]]}")
    elif uop == UOps.CAST:
      r[u] = ssa('cast')
      kernel.extend(renderer.render_cast(r[src[0]], r[u], dtype, src[0].dtype))
    elif uop == UOps.PHI:
      r[u] = r[src[0]]
    elif uop == UOps.LABEL:
      kernel.append(f"{args}:")
    elif uop == UOps.BARRIER:
      kernel.append(renderer.barrier)
    elif uop == UOps.COND_BRANCH:
      kernel.extend(renderer.render_bra(args[0], r[src[0]]))
    else:
      raise NotImplementedError(f"unhandled op {uop}")

  return renderer.render_kernel(function_name, kernel, bufs)

rdna_matcher = PatternMatcher([
  # todo we'll put specific RDNA matching here
])

def uops_to_asm(function_name:str, uops:UOpGraph, architecture:str) -> str:
  if architecture.startswith("cuda"):
    return uops_to_ptx_asm(function_name, uops)
  elif architecture.startswith("amd"):
    return uops_to_rdna_asm(function_name, uops)
  else:
    raise ValueError(f"Unsupported architecture: {architecture}")
