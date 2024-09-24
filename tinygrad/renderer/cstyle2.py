from typing import List, Dict, DefaultDict, cast
import math
from collections import defaultdict, Counter
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp, PatternMatcher, UPat, UOps, UnaryOps, BinaryOps, TernaryOps

def render_dtype(x:DType):
  if isinstance(x, PtrDType): return f"device {x.name}*" if not x.local else f"threadgroup {x.name}*"
  return x.name

pm = PatternMatcher([
  (UPat(UOps.CONST, arg=math.inf), lambda r: "INFINITY"),
  (UPat(UOps.CONST, arg=-math.inf), lambda r: "-INFINITY"),
  (UPat(UOps.CONST, dtype=dtypes.bool, name="x"), lambda r,x: "1" if x.arg else "0"),
  (UPat(UOps.CONST, dtype=dtypes.float, name="x"), lambda r,x: f"{x.arg}f" if not math.isnan(x.arg) else "NAN"),
  (UPat(UOps.CONST, name="x"), lambda r,x: str(x.arg)),
  (UPat(UOps.CAST, name="x"), lambda r,x: f"({render_dtype(x.dtype)}){r[x.src[0]]}"),
  (UPat(UOps.BITCAST, name="x"), lambda r,x: f"as_type<{render_dtype(x.dtype)}>({r[x.src[0]]})"),
  (UPat(UOps.GEP, name="x"), lambda r,x: f"{r[x.src[0]]}.{'xyzw'[x.arg[0]]}"),
  (UPat(UOps.SPECIAL, name="x"),
   lambda r,x: {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}[x.arg[0][0]](x.arg[0][-1])),
  (UPat(UOps.LOAD, src=(UPat(name="idx"),)), lambda r,idx: f"*{r[idx]}"),
  (UPat(UOps.LOAD, src=(UPat(name="idx"), UPat(UOps.IF))), lambda r,idx: f"*{r[idx]}"),
  (UPat(UOps.LOAD, src=(UPat(name="idx"), UPat(name="alt"), UPat(name="gate"))), lambda r,idx,alt,gate: f"{r[gate]}?(*{r[idx]}):{r[alt]}"),
  (UPat(UOps.ALU, arg=UnaryOps.SQRT, name="x"), lambda r,x: f"metal::sqrt({r[x.src[0]]})"),
  (UPat(UOps.ALU, arg=UnaryOps.RECIP, name="x"), lambda r,x: f"1/{r[x.src[0]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.MUL, name="x"), lambda r,x: f"{r[x.src[0]]}*{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.ADD, name="x"), lambda r,x: f"{r[x.src[0]]}+{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.IDIV, name="x"), lambda r,x: f"{r[x.src[0]]}/{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.MOD, name="x"), lambda r,x: f"{r[x.src[0]]}%{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.AND, name="x"), lambda r,x: f"{r[x.src[0]]}&{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.XOR, name="x"), lambda r,x: f"{r[x.src[0]]}^{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.OR, name="x"), lambda r,x: f"{r[x.src[0]]}|{r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.MAX, name="x"), lambda r,x: f"metal::max({r[x.src[0]]},{r[x.src[1]]})"),
  (UPat(UOps.ALU, arg=BinaryOps.CMPLT, name="x"), lambda r,x: f"{r[x.src[0]]} < {r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=BinaryOps.CMPNE, name="x"), lambda r,x: f"{r[x.src[0]]} != {r[x.src[1]]}"),
  (UPat(UOps.ALU, arg=TernaryOps.WHERE, name="x"), lambda r,x: f"{r[x.src[0]]} ? {r[x.src[1]]} : {r[x.src[2]]}"),
  (UPat(UOps.INDEX, name="x"), lambda r,x: f"({r[x.src[0]]}+{r[x.src[1]]})"),
  (UPat(UOps.STORE, name="x"), lambda r,x: f"*{r[x.src[0]]} = {r[x.src[1]]};"),
  (UPat(UOps.VECTORIZE, name="x"), lambda r,x: f"({render_dtype(x.dtype)}){{" + ','.join(r[u] for u in x.src) + "}"),
  (UPat(UOps.DEFINE_ACC, name="x"), lambda r,x: r[x.src[0]]),
  (UPat(UOps.RANGE, name="x"), lambda r,x: f"for ({render_dtype(x.dtype)} {r[x]} = {r[x.src[0]]}; {r[x]} < {r[x.src[1]]}; {r[x]}++) {{"),
  (UPat(UOps.ASSIGN, name="x"), lambda r,x: f"{r[x.src[0]]} = {r[x.src[1]]};"),
  (UPat(UOps.DEFINE_LOCAL, name="x"), lambda r,x: f"threadgroup {x.dtype.name} {r[x]}[{x.arg[1]}];"),
  (UPat(UOps.IF, name="x"), lambda r,x: f"if ({r[x.src[0]]}) {{"),
  (UPat((UOps.ENDIF, UOps.ENDRANGE)), lambda r: "}"),
  (UPat(UOps.BARRIER), lambda r: "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"),
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

  def render(self, name:str, uops:List[UOp]) -> str:
    # register assignment
    c: DefaultDict[str, int] = defaultdict(int)
    r: Dict[UOp, str] = {}

    child_count = Counter(v for ru in uops for v in ru.src)

    bufs = []
    lines = []
    depth = 1
    for u in uops:
      if u.op is UOps.DEFINE_GLOBAL:
        r[u] = f"data{u.arg}"
        bufs.append(u)
        continue
      # naming
      if u.op is UOps.DEFINE_LOCAL:
        # these shouldn't be named here
        r[u] = u.arg[0]
      else:
        prefix = {UOps.RANGE: "ridx", UOps.ALU: "alu", UOps.WMMA: "wmma",
                  UOps.DEFINE_ACC: "acc", UOps.SPECIAL: "idx", UOps.LOAD: "val"}.get(u.op, "unk")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, pm.rewrite(u, ctx=r))
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {UOps.ENDIF, UOps.ENDRANGE}: depth -= 1
      if u.op in {UOps.INDEX, UOps.CONST} or (u.op is UOps.ALU and child_count[u] == 1 and u.arg is not BinaryOps.MAX):
        r[u] = l
      else:
        if u.op in {UOps.RANGE, UOps.ASSIGN, UOps.DEFINE_LOCAL} or u.dtype == dtypes.void:
          if u.op is UOps.ASSIGN: r[u] = r[u.src[0]]
        else:
          l = f"{render_dtype(u.dtype)} {r[u]} = {l};"
        lines.append("  "*depth + l)
        c[prefix] += 1  # if it was used, increment
      if u.op in {UOps.IF, UOps.RANGE}: depth += 1

    return f"kernel void {name}(" + ', '.join([f"{render_dtype(u.dtype)} {r[u]}" for u in bufs]) + \
      ", uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n" + '\n'.join(lines) + "\n}"
