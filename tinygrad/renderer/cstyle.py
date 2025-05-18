from typing import Literal, Callable, cast
import os, math, sys
from collections import defaultdict, Counter
from tinygrad.ops import GroupOp, Ops, UOp, PatternMatcher, UPat
from tinygrad.helpers import strip_parens, getenv, prod, dedup, AMX
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.codegen.devectorizer import no_vectorized_alu

base_rewrite = PatternMatcher([
  (UPat(Ops.DEFINE_ACC, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat(Ops.ASSIGN, name="x"), lambda ctx,x: f"{ctx[x.src[0]]} = {ctx[x.src[1]]};"),
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"if ({ctx[x.src[0]]}) {{"),
  (UPat((Ops.ENDIF, Ops.ENDRANGE)), lambda ctx: "}"),
  (UPat(Ops.WMMA, name="x"), lambda ctx,x: f"__{x.arg[0]}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),
  # r method accesses
  (UPat(Ops.RANGE, name="x"),
   lambda ctx,x: f"for ({ctx.render_dtype(x.dtype)} {ctx[x]} = 0; {ctx[x]} < {ctx[x.src[0]]}; {ctx[x]}++) {{"),
  (UPat(Ops.VECTORIZE, name="x"),
   lambda ctx,x: f"{ctx.float4.replace('float4', ctx.render_dtype(x.dtype))}" + \
    f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x:
    f"__builtin_convertvector({ctx[x.src[0]]}, {ctx.render_dtype(x.dtype)})" if x.dtype.count > 1 and not isinstance(x.dtype, PtrDType) else None),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx[x.src[0]])})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"(*(({ctx.buffer_prefix}{ctx.render_dtype(x.dtype)}*)&{ctx[x.src[0]]}))"),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"{ctx.smem_align}{ctx.smem_prefix}{ctx.render_dtype(x.dtype.base)} {ctx[x]}[{x.dtype.size}];"),
  (UPat(Ops.BARRIER), lambda ctx: ctx.barrier),
  (UPat(Ops.NOOP, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f"{ctx.code_for_workitem[x.arg[0][0]](x.arg[0][-1])}; /* {x.arg[1]} */"),
  # const
  (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, ctx.infinity)})"),
  (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"({ctx.render_cast(x.dtype, f'-{ctx.infinity}')})"),
  (UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, ctx.nan)})" if math.isnan(x.arg) else None),
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{x.arg}f"),
  (UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}ll"),
  (UPat(Ops.CONST, dtype=dtypes.uint64, name="x"), lambda ctx,x: f"{x.arg}ull"),
  (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), lambda ctx,x: f"{x.arg}u"),
  (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "1" if x.arg else "0"),
  # consts are rendered to larger type and casted
  (UPat(Ops.CONST, (dtypes.bfloat16, dtypes.half), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}f')})"),
  (UPat(Ops.CONST, (dtypes.uint8, dtypes.uint16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, f'{x.arg}u')})"),
  (UPat(Ops.CONST, (dtypes.int8, dtypes.int16), name="x"), lambda ctx,x: f"({ctx.render_cast(x.dtype, x.arg)})"),
  # default const render
  (UPat(Ops.CONST, name="x"), lambda ctx,x: str(x.arg)),
  # new load/store
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
   lambda ctx,buf,idx: f"({ctx[buf]}+{strip_parens(ctx[idx]) if idx.arg == Ops.ADD else ctx[idx]})"),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var")), allow_any_len=True),
   lambda ctx,bidx,var,gate: f"({ctx[gate]}?*{ctx[bidx]}:{ctx[var]})"),
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx,bidx: f"*{ctx[bidx]}"),
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx,bidx,var: f"*{ctx[bidx]} = {ctx[var]};"),
  # alu/gep
  (UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
    *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, Ops.XOR} else ctx[v] for v in x.src]), x.dtype)),
  (UPat(Ops.GEP, name="x"), lambda ctx,x: ctx[x.src[0]] + \
    (f"[{x.arg[0]}]" if x.src[0].dtype.count > ctx.gep_arr_threshold else f".{'xyzwabcd'[x.arg[0]]}")),
  # custom passes through with format
  (UPat((Ops.CUSTOM, Ops.CUSTOMI), name="x"), lambda ctx,x: x.arg.format(*[ctx[y] for y in x.src])),
])

extra_pm = PatternMatcher([
  # insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
  (UPat(Ops.BITCAST, name="x"),
   lambda x: UOp(Ops.BITCAST, x.dtype, (UOp(Ops.NOOP, x.src[0].dtype, x.src),)) if x.src[0].op not in {Ops.NOOP, Ops.LOAD, Ops.CUSTOM} else None),
  # rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  # devectorize any bools
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.INDEX), dtype=dtypes.bool, name="alu"), no_vectorized_alu),
  # CAST (from bool) can't be vectorized
  (UPat(Ops.CAST, src=(UPat(dtype=dtypes.bool),), name="alu"), no_vectorized_alu),
  # WHERE can't be vectorized
  (UPat(Ops.WHERE, name="alu"), no_vectorized_alu),
])

def uops_to_dtypes(uops:list[UOp]) -> list[DType]: return dedup(u.dtype for u in uops if not isinstance(u.dtype, (ImageDType, PtrDType)))

class CStyleLanguage(Renderer):
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: dict[Literal["g", "l", "i"], Callable] = {}
  extra_args: list[str] = []
  float4: str|None = None
  float4_style: tuple[str, str] = ('(', ')')
  gep_arr_threshold: int = 4
  type_map: dict[DType, str] = {}
  infinity: str = "INFINITY"
  nan: str = "NAN"
  code_for_op: dict = {
    Ops.SQRT: lambda x,dtype: f"sqrt({x})", Ops.RECIP: lambda x,dtype: f"(1/{x})", Ops.NEG: lambda x,dtype: f"-{x}",
    Ops.EXP2: lambda x,dtype: f"exp2({x})", Ops.LOG2: lambda x,dtype: f"log2({x})", Ops.SIN: lambda x,dtype: f"sin({x})",
    Ops.AND: lambda a,b,dtype: f"({a}&{b})", Ops.XOR: lambda a,b,dtype: f"({a}^{b})", Ops.OR: lambda a,b,dtype: f"({a}|{b})",
    Ops.ADD: lambda a,b,dtype: f"({a}+{b})", Ops.SUB: lambda a,b,dtype: f"({a}-{b})", Ops.MUL: lambda a,b,dtype: f"({a}*{b})",
    Ops.MOD: lambda a,b,dtype: f"({a}%{b})", Ops.IDIV: lambda a,b,dtype: f"({a}/{b})", Ops.CMPNE: lambda a,b,dtype: f"({a}!={b})",
    Ops.SHR: lambda a,b,dtype: f"({a}>>{b})", Ops.SHL: lambda a,b,dtype: f"({a}<<{b})", Ops.CMPLT: lambda a,b,dtype: f"({a}<{b})",
    Ops.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})" }

  string_rewrite = base_rewrite
  extra_matcher = extra_pm

  def get_kernel_modifier(self, uops:list[UOp]) -> str: return ""
  def render_kernel(self, function_name:str, kernel:list[str], bufs:list[tuple[str,tuple[DType,bool]]], uops:list[UOp], prefix=None) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501
    buftypes = [(name, self.render_dtype(dtype, mutable)+self.buffer_suffix if isinstance(dtype, (ImageDType, PtrDType)) else
                self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]
    prg = ''.join([f"{self.kernel_prefix}void {self.get_kernel_modifier(uops)}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_cast(self, dt:DType, val: str) -> str: return f"({self.render_dtype(dt)})({val})"
  def render_dtype(self, dt:DType, mutable=True) -> str:
    if isinstance(dt, ImageDType): return f"{'write_only' if mutable else 'read_only'} image2d_t"
    if isinstance(dt, PtrDType):
      return (self.smem_prefix if dt.local and self.smem_prefix_for_cast else self.buffer_prefix) + self.render_dtype(dt.base) + "*"
    if dt.count > 1: return self.type_map.get(scalar:=dt.scalar(), scalar.name).replace(" ", "_") + str(dt.count)
    return self.type_map.get(scalar:=dt.scalar(), scalar.name)

  def __getitem__(self, key): return self.r[key]  # hacky helper
  def _render(self, uops:list[UOp]) -> tuple[str, list[str], list[tuple[str,tuple[DType,bool]]]]:
    r: dict[UOp, str] = {}
    self.r = r

    child_count = Counter(v for ru in uops for v in ru.src)
    bufs: dict[UOp, tuple[str, tuple[DType, bool]]] = {}
    kernel = []
    depth = 1
    c: defaultdict[str, int] = defaultdict(int)
    name = "test"
    for u in uops:
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.name
        continue
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else u.arg[0]
        bufs[u] = (r[u], (u.dtype, False))
        continue

      # mark buffers that we store to writable
      if u.op is Ops.STORE:
        for up in u.src[0].toposort():
          if up.op is Ops.DEFINE_GLOBAL: bufs[up] = (bufs[up][0], (bufs[up][1][0], True))

      # naming
      prefix = None
      if u.op is Ops.SPECIAL: r[u] = u.arg[0]
      elif u.op is Ops.RANGE: r[u] = f"ridx{u.arg}"
      else:
        prefix = {Ops.WMMA: "wmma", Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const",
                  Ops.CAST: "cast", Ops.BITCAST: "cast", Ops.GEP: "gep", Ops.VECTORIZE: "cast", Ops.NOOP: "precast",
                  Ops.INDEX: "bidx", Ops.DEFINE_ACC: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {Ops.ENDIF, Ops.ENDRANGE}: depth -= 1
      if (u.op is not Ops.CAST or u.dtype.vcount == 1) and (u.op in {Ops.CONST, Ops.GEP, Ops.INDEX, Ops.CUSTOMI} or \
        (u.op in {Ops.VECTORIZE, *(GroupOp.ALU-{Ops.WHERE}), Ops.CAST, Ops.BITCAST} and child_count[u] == 1 and not getenv("EXPAND_SSA"))):
        r[u] = l
      else:
        if u.op in {Ops.RANGE, Ops.ASSIGN, Ops.DEFINE_LOCAL} or u.dtype == dtypes.void:
          if u.op is Ops.ASSIGN: r[u] = r[u.src[0]]
        else:
          l = f"{self.render_dtype(u.dtype)} {r[u]} = {l}" + (";" if u.op is not Ops.SPECIAL else "")
        kernel.append("  "*depth + l)
        if prefix: c[prefix] += 1  # if it was used, increment
      if u.op in {Ops.IF, Ops.RANGE}: depth += 1
    del self.r

    # NOTE: this relies on bufs dict preserving order
    return (name, kernel, list(bufs.values()))
  def render(self, uops:list[UOp]) -> str: return self.render_kernel(*self._render(uops), uops)

class ClangRenderer(CStyleLanguage):
  device = "CPU"
  float4 = "(float4)"
  float4_style = ('{', '}')
  gep_arr_threshold = 0
  has_local = False
  global_max = None
  infinity = "__builtin_inff()"
  nan = '__builtin_nanf("")'
  amx_tc = [TensorCore(dims=(sz,sz,1), threads=1, elements_per_thread=(sz,sz,sz*sz), dtype_in=dt, dtype_out=dt, swizzle=(None,((),(4,5,6,7,0,1,2,3))),
                      opts=("u0","u0","u0","u0","u1","u1","u1","u1")) for dt,sz in [(dt, 64 // dt.itemsize) for dt in [dtypes.float]]]
  if AMX: tensor_cores = amx_tc

  # language options
  buffer_suffix = " restrict"
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}
  code_for_op = {**({k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.EXP2, Ops.SIN, Ops.LOG2]}),
                 Ops.SQRT: lambda x,dtype: f"__builtin_sqrt({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})"}
  # LLVM legalizes double => half cast on systems that don't support it natively (like x86 cpus without AVX512-FP16) into a compiler-rt libcall.
  extra_matcher = PatternMatcher([(UPat.var("x", dtypes.float64).cast(dtypes.float16), lambda x: x.cast(dtypes.float32).cast(dtypes.float16)),
    (UPat(Ops.SQRT, name="alu"), no_vectorized_alu),]) + CStyleLanguage.extra_matcher

  if sys.platform == 'win32':
    kernel_prefix = "__attribute__((ms_abi)) "
  def render_vector_prefix(self, dt:DType) -> str:
    # round (down) to power of two (this is actually the default clang behavior)
    alignment = 2**int(math.log2(dt.itemsize)) if getenv("ALIGNED", 1) else 1
    return f"typedef {self.render_dtype(dt.scalar())} {self.render_dtype(dt)} __attribute__((aligned({alignment}),vector_size({dt.itemsize})));"

  def _render_defines(self, uops) -> list[str]:
    prefix = [self.render_vector_prefix(dt) for dt in uops_to_dtypes(uops) if dt.count > 1]
    # https://github.com/corsix/amx
    for name, (N, M, _), dtype_in, _, _, _, _, _ in dedup([uop.arg for uop in uops if uop.op is Ops.WMMA]):
      prefix += [
        '#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")',
        '#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")',
      ]
      # 'static' in C roughly means that function symbol isn't exported. LLVM puts those symbols at the end of object file which allows Clang JIT
      # to just jump at the start of a shellcode whithout having to deal with symbols or trampolines at all. This is better than having to inline
      # wmma function every time it is called or wasting complexity on a symbol parsing and a memory page on trampoline.
      prefix += [f"""static {(out := self.render_dtype(dtype_in.vec(N*N)))} __{name}({self.render_dtype(dtype_in.vec(N))} data1, {self.render_dtype(dtype_in.vec(M))} data2, {out} data0){{
  AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}
  AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}"""] # noqa: E501
    return prefix
  def _render_body(self, function_name, kernel, bufs, uops, pref=None) -> str: return super().render_kernel(function_name, kernel, bufs, uops, pref)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[DType,bool]]]) -> str: return ""

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    defines = '\n'.join(self._render_defines(uops))
    return defines + "\n" + self._render_body(function_name, kernel, bufs, uops, prefix) + "\n" + self._render_entry(function_name, bufs)

class MetalRenderer(CStyleLanguage):
  device = "METAL"
  shared_max = 32768
  tensor_cores = [TensorCore(dims=(8,8,8), threads=32, elements_per_thread=(2,2,2), dtype_in=di, dtype_out=do, opts=("u0","l0","l1","l1","l0","l1"),
    swizzle=(((6,1,2,7,4),(8,0,3,5)), ((0,5,6,3,7),(1,2,4,8)))) for di,do in [(dtypes.float,dtypes.float),(dtypes.half,dtypes.float),
    (dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]
  def __init__(self): self.tensor_cores = MetalRenderer.tensor_cores if hasattr(os, 'uname') and os.uname().machine == "arm64" else []

  # language options
  kernel_prefix = "kernel "
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}
  # uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  type_map = {dtypes.bfloat16: "bfloat"}

  # precise::sin
  code_for_op = {**CStyleLanguage.code_for_op, Ops.SIN: lambda x,dtype: f"precise::sin({x})"}

  # upcast to float32 all the ops that don't support bfloat16
  extra_matcher = PatternMatcher([
    # NOTE: this is copied from PTX
    (UPat((Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN), dtype=dtypes.bfloat16, name="x"),
      lambda x: (UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(dtypes.bfloat16))),
  ]) + extra_pm

  string_rewrite = PatternMatcher([
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"as_type<{ctx.render_dtype(x.dtype)}>({ctx[x.src[0]]})"),
  ]) + base_rewrite

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix, wmma_args = ["#include <metal_stdlib>","using namespace metal;"], set([uop.arg for uop in uops if uop.op is Ops.WMMA])
    for arg in wmma_args: prefix.append(
  f"""{(dtype_out:=self.render_dtype(arg[3].vec(2)))} __{arg[0]}({(dtype_in:=self.render_dtype(arg[2].vec(2)))} a, {dtype_in} b, {dtype_out} c){{
  simdgroup_{self.render_dtype(arg[2])}8x8 mat_a, mat_b; simdgroup_{self.render_dtype(arg[3])}8x8 mat_c;
  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return {dtype_out}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)