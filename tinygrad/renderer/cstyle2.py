from __future__ import annotations
from typing import List, Dict, DefaultDict, cast, Tuple
import math, os
from collections import defaultdict
from tinygrad.helpers import getenv, dedup
from tinygrad.dtype import dtypes, DType, PtrDType, ImageDType
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.ops import UOp, PatternMatcher, UPat, UOps, UnaryOps, BinaryOps, TernaryOps

base_pm = PatternMatcher([
  (UPat(UOps.CONST, arg=math.inf), lambda r: "INFINITY"),
  (UPat(UOps.CONST, arg=-math.inf), lambda r: "-INFINITY"),
  (UPat(UOps.CONST, dtype=dtypes.bool, name="x"), lambda r,x: "1" if x.arg else "0"),
  (UPat(UOps.CONST, dtype=dtypes.float, name="x"), lambda r,x: f"{x.arg}f" if not math.isnan(x.arg) else "NAN"),
  (UPat(UOps.CONST, dtype=dtypes.half, name="x"), lambda r,x: f"(half)({x.arg}f)" if not math.isnan(x.arg) else "NAN"),
  (UPat(UOps.CONST, name="x"), lambda r,x: str(x.arg)),
  (UPat(UOps.LOAD, src=(UPat(name="idx"),)), lambda r,idx: f"*{r[idx]}"),
  (UPat(UOps.LOAD, src=(UPat(name="idx"), UPat(UOps.IF))), lambda r,idx: f"*{r[idx]}"),
  (UPat(UOps.LOAD, src=(UPat(name="idx"), UPat(name="alt"), UPat(name="gate"))), lambda r,idx,alt,gate: f"{r[gate]}?(*{r[idx]}):{r[alt]}"),
  (UPat(UOps.ALU, arg=UnaryOps.EXP2, name="x"), lambda r,x: f"exp2({r[x.src[0]]})"),
  (UPat(UOps.ALU, arg=UnaryOps.LOG2, name="x"), lambda r,x: f"log2({r[x.src[0]]})"),
  (UPat(UOps.ALU, arg=UnaryOps.SIN, name="x"), lambda r,x: f"sin({r[x.src[0]]})"),
  (UPat(UOps.ALU, arg=UnaryOps.SQRT, name="x"), lambda r,x: f"sqrt({r[x.src[0]]})"),
  (UPat(UOps.ALU, arg=UnaryOps.RECIP, name="x"), lambda r,x: f"1/{r[x.src[0]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.MUL, name="x"), lambda r,x: f"{r[x.src[0]]}*{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.ADD, name="x"), lambda r,x: f"{r[x.src[0]]}+{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.IDIV, name="x"), lambda r,x: f"{r[x.src[0]]}/{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.MOD, name="x"), lambda r,x: f"{r[x.src[0]]}%{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.AND, name="x"), lambda r,x: f"{r[x.src[0]]}&{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.XOR, name="x"), lambda r,x: f"{r[x.src[0]]}^{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.OR, name="x"), lambda r,x: f"{r[x.src[0]]}|{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.MAX, name="x"), lambda r,x: f"max({r[x.src[0]]},{r[x.src[1]]})"),
  (UPat(UOps.ALU, arg=BinaryOps.CMPLT, name="x"), lambda r,x: f"{r[x.src[0]]} < {r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.CMPNE, name="x"), lambda r,x: f"{r[x.src[0]]} != {r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=TernaryOps.WHERE, name="x"), lambda r,x: f"{r[x.src[0]]} ? {r[x.src[1]]} : {r[x.src[2]]}"),
  (UPat(UOps.INDEX, name="x"), lambda r,x: f"({r[x.src[0]]}+{r[x.src[1]]})"),
  (UPat(UOps.STORE, name="x"), lambda r,x: f"*{r[x.src[0]]} = {r[x.src[1]]};"),
  (UPat(UOps.DEFINE_ACC, name="x"), lambda r,x: r[x.src[0]]),
  (UPat(UOps.ASSIGN, name="x"), lambda r,x: f"{r[x.src[0]]} = {r[x.src[1]]};"),
  (UPat(UOps.IF, name="x"), lambda r,x: f"if ({r[x.src[0]]}) {{"),
  (UPat((UOps.ENDIF, UOps.ENDRANGE)), lambda r: "}"),
  (UPat(UOps.WMMA, name="x"), lambda r,x: f"__{x.arg[0]}({r[x.src[0]]}, {r[x.src[1]]}, {r[x.src[2]]})"),
  (UPat(UOps.CAST, name="x"), lambda r,x: f"({r.render_dtype(x.dtype)}){r[x.src[0]]}"),
  (UPat(UOps.VECTORIZE, name="x"), lambda r,x: f"({r.render_dtype(x.dtype)}){{" + ','.join(r[u] for u in x.src) + "}"),
  (UPat(UOps.RANGE, name="x"), lambda r,x: f"for ({r.render_dtype(x.dtype)} {r[x]} = {r[x.src[0]]}; {r[x]} < {r[x.src[1]]}; {r[x]}++) {{"),
])

# TODO: this use of INDEX should be universal and this should be removed
prepm = PatternMatcher([
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), name='buf'), UPat.var('idx')), allow_any_len=True, name="ld"),
   lambda buf,idx,ld:
    UOp.load(buf.index(idx).cast(PtrDType(ld.dtype, buf.dtype.local)) if ld.dtype.count > 1 else buf.index(idx), *ld.src[2:], dtype=ld.dtype)),
  (UPat(UOps.STORE, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), name='buf'), UPat.var('idx'), UPat.var('val')), allow_any_len=True, name="st"),
   lambda buf,idx,val,st:
    UOp.store(buf.index(idx).cast(PtrDType(val.dtype, buf.dtype.local)) if val.dtype.count > 1 else buf.index(idx), val, *st.src[3:])),
])

class CStyle2Language(Renderer):
  extra_matcher = prepm
  render_pm = base_pm

  # RenderContext is used by the functions in the PatternMatcher
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  type_map = {}
  extra_args: List[str] = []

  def get_kernel_modifier(self, uops:List[UOp]) -> str: return ""
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501
    buftypes = [(name,f"{'write_only' if mutable else 'read_only'} image2d_t" if dtype.name.startswith('image') else
                ("" if mutable else "const ")+self.render_dtype(dtype, hdr=True)+self.buffer_suffix if isinstance(dtype, PtrDType) else
                self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]
    prg = ''.join([f"{self.kernel_prefix}void {self.get_kernel_modifier(uops)}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  def render_dtype(self, x:DType, hdr=False):
    name = self.type_map.get(x, x.name)
    if isinstance(x, PtrDType): return f"{name}*"
    return name

  # hacky helper
  def __getitem__(self, key): return self.r[key]
  def render(self, name:str, uops:List[UOp]) -> str:
    r: Dict[UOp, str] = {}
    self.r = r

    # get children
    children = defaultdict(list)
    for u in uops:
      for v in u.src:
        # BITCAST is a double child so that it always renders the input
        if u.op is UOps.BITCAST: children[v].append(u)
        children[v].append(u)

    bufs: Dict[UOp, Tuple[str, Tuple[DType, bool]]] = {}
    lines = []
    depth = 1
    c: DefaultDict[str, int] = defaultdict(int)
    for u in uops:
      if u.op is UOps.DEFINE_GLOBAL:
        r[u] = f"data{u.arg}"
        bufs[u] = (r[u], (u.dtype, True))
        continue
      if u.op is UOps.DEFINE_VAR:
        r[u] = u.arg[0]
        bufs[u] = (r[u], (u.dtype, False))
        continue

      # naming
      prefix = {UOps.RANGE: "ridx", UOps.ALU: "alu", UOps.WMMA: "wmma", UOps.DEFINE_LOCAL: "local",
                UOps.DEFINE_ACC: "acc", UOps.SPECIAL: "idx", UOps.LOAD: "val"}.get(u.op, "unk")
      r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.render_pm.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {UOps.ENDIF, UOps.ENDRANGE}: depth -= 1
      if u.op in {UOps.INDEX, UOps.CONST} or (u.op is UOps.ALU and len(children[u]) == 1 and u.arg is not BinaryOps.MAX and not getenv("EXPAND_SSA")):
        r[u] = "("+l+")" if u.op is UOps.ALU else l
      else:
        if u.op in {UOps.RANGE, UOps.ASSIGN, UOps.DEFINE_LOCAL} or u.dtype == dtypes.void:
          if u.op is UOps.ASSIGN: r[u] = r[u.src[0]]
        else:
          l = f"{self.render_dtype(u.dtype)} {r[u]} = {l};"
        lines.append("  "*depth + l)
        c[prefix] += 1  # if it was used, increment
      if u.op in {UOps.IF, UOps.RANGE}: depth += 1

    del self.r
    return self.render_kernel(name, lines, list(bufs.values()), uops)

# ***** device specific renderers *****

class MetalRenderer(CStyle2Language):
  device = "METAL"
  shared_max = 32768
  tensor_cores = [TensorCore(dims=(8,8,8),threads=[(0,2),(1,4),(0,2),(1,2)],expanded_shape=(2,2,2,2),upcast_axes=([(1,2)],[(1,2)],[(1,2)]),
    st1_pattern=(((1,1),(0,1),(1,0),(0,3)),((0,0),(0,2),(1,3),(1,2))),st2_pattern=(((0,0),(1,1),(1,2),(0,2),(1,0)),((0,1),(0,3),(1,3))),
    dtype_in=di,dtype_out=do,reduce_axes=[(0,8)]) for di,do in [(dtypes.float,dtypes.float),(dtypes.half,dtypes.float),(dtypes.half,dtypes.half)]]
  kernel_prefix = "kernel "
  buffer_prefix = "device "
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  def __init__(self): self.tensor_cores = MetalRenderer.tensor_cores if os.uname().machine == "arm64" else []

  @classmethod
  def render_dtype(cls, x:DType, hdr=False):
    if isinstance(x, PtrDType): return f"device {x.name}*" if not x.local else f"threadgroup {x.name}*"
    if hdr: return f"constant {x.name}&"
    return x.name

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix, wmma_args = ["#include <metal_stdlib>","using namespace metal;"], set([uop.arg for uop in uops if uop.op is UOps.WMMA])
    for arg in wmma_args: prefix.append(f"""{arg[3].name}2 __{arg[0]}({arg[2].name}2 m, {arg[2].name}2 n, {arg[3].name}2 o) {{
  simdgroup_{arg[3].name}8x8 a,b,c; a.thread_elements()[0] = m.x; a.thread_elements()[1] = m.y; b.thread_elements()[0] = n.x;
  b.thread_elements()[1] = n.y; c.thread_elements()[0] = o.x; c.thread_elements()[1] = o.y; simdgroup_multiply_accumulate(c, a, b, c);
  return {arg[3].name}2(c.thread_elements()[0], c.thread_elements()[1]);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  # language options
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  uses_ptr_arithmetic = True
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}
  # uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']

  render_pm = PatternMatcher([
    (UPat(UOps.GEP, name="x"), lambda r,x: f"{r[x.src[0]]}.{'xyzw'[x.arg[0]]}"),
    (UPat(UOps.BARRIER), lambda r: "threadgroup_barrier(mem_flags::mem_threadgroup);"),
    (UPat(UOps.DEFINE_LOCAL, name="x"), lambda r,x: f"threadgroup {x.dtype.name} {r[x]}[{x.arg[1]}];"),
    (UPat(UOps.BITCAST, name="x"), lambda r,x: f"as_type<{r.render_dtype(x.dtype)}>({r[x.src[0]]})"),
    (UPat(UOps.SPECIAL, name="x"),
     lambda r,x: {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}[x.arg[0][0]](x.arg[0][-1])),
  ]) + base_pm

class ClangRenderer(CStyle2Language):
  device = "CLANG"
  has_local = False
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}

  def render_vector_prefix(self, dt:DType) -> str:
    return f"typedef {self.render_dtype(dt.scalar())} {self.render_dtype(dt)} __attribute__((aligned({(sz:=dt.itemsize)}),vector_size({sz})));"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix, macros = [self.render_vector_prefix(dt.base if isinstance(dt, PtrDType) else dt) for dt in dedup(uop.dtype for uop in uops if uop.dtype.count>1)], []
    # https://github.com/corsix/amx
    for name, (N, M, _), dtype_in, _, _, _, _, _ in dedup([uop.arg for uop in uops if uop.op is UOps.WMMA]):
      macros = [
        '#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")',
        '#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")',
      ]
      prefix += [f"""{(out := self.render_dtype(dtype_in.vec(N*N)))} __{name}({self.render_dtype(dtype_in.vec(N))} data1, {self.render_dtype(dtype_in.vec(M))} data2, {out} data0){{
  AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}
  AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}"""] # noqa: E501
    return super().render_kernel(function_name, kernel, bufs, uops, macros + prefix)

  render_pm = PatternMatcher([
    (UPat(UOps.ALU, arg=BinaryOps.MAX, name="x"), lambda r,x: f"(({r[x.src[0]]}>{r[x.src[1]]})?{r[x.src[0]]}:{r[x.src[1]]})"),
    (UPat(UOps.GEP, name="x"), lambda r,x: f"{r[x.src[0]]}[{x.arg[0]}]"),
  ]) + base_pm
