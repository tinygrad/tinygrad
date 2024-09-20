from typing import List, Dict
import math
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp, PatternMatcher, UPat, UOps, print_uops, UnaryOps, BinaryOps, TernaryOps

#def render_store(r, x:UOp) -> str:
#  if x.dtype.count > 1:
#    if x.src[0].op is UOps.DEFINE_LOCAL
#    return f"*(({prefix}{self.render_dtype(var_dtype)}*)({buf_name}+{idx})) = {var_name};"
#  return f"*({r[x.src[0]]}+{r[x.src[1]]}) = {r[x.src[2]]}"

def render_dtype(x:DType, local=False):
  if isinstance(x, PtrDType): return f"device {x.name}*" if not local else f"threadgroup {x.name}*"
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
  (UPat(UOps.VECTORIZE, name="x"), lambda r,x: f"(float{len(x.src)}){{" + ','.join(r[u] for u in x.src) + "}"),
  (UPat(UOps.DEFINE_ACC, name="x"), lambda r,x: r[x.src[0]]),
  (UPat(UOps.RANGE, name="x"), lambda r,x: f"for ({render_dtype(x.dtype)} {r[x]} = {r[x.src[0]]}; {r[x]} < {r[x.src[1]]}; {r[x]}++) {{"),
  (UPat(UOps.ASSIGN, name="x"), lambda r,x: f"{r[x.src[0]]} = {r[x.src[1]]};"),
  (UPat(UOps.DEFINE_LOCAL, name="x"), lambda r,x: f"threadgroup {x.dtype.name} {r[x]}[{x.arg[1]}];"),
  (UPat(UOps.IF, name="x"), lambda r,x: f"if ({r[x.src[0]]}) {{"),
  (UPat((UOps.ENDIF, UOps.ENDRANGE)), lambda r: "}"),
  (UPat(UOps.BARRIER), lambda r: "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"),
])

# NOTE: this should be universal
prepm = PatternMatcher([
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), name='buf'), UPat.var('idx')), name="ld"),
   lambda buf,idx,ld: UOp.load(buf.index(idx).cast(PtrDType(ld.dtype)) if ld.dtype.count > 1 else buf.index(idx), dtype=ld.dtype)),
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), name='buf'), UPat.var('idx'), UPat(UOps.IF, name='gate')), name="ld"),
   lambda buf,idx,ld,gate: UOp.load(buf.index(idx).cast(PtrDType(ld.dtype)) if ld.dtype.count > 1 else buf.index(idx), gate, dtype=ld.dtype)),
  (UPat(UOps.LOAD, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), name='buf'), UPat.var('idx'), UPat(name='alt'), UPat(name='gate')), name="ld"),
   lambda buf,idx,ld,alt,gate: UOp.load(buf.index(idx).cast(PtrDType(ld.dtype)) if ld.dtype.count > 1 else buf.index(idx), alt, gate, dtype=ld.dtype)),
  (UPat(UOps.STORE, src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), name='buf'), UPat.var('idx'), UPat.var('val'))),
   lambda buf,idx,val: UOp.store(buf.index(idx).cast(PtrDType(val.dtype)) if val.dtype.count > 1 else buf.index(idx), val)),
])

class CStyle2Language(Renderer):
  extra_matcher = prepm

  def render(self, name:str, uops:List[UOp]) -> str:
    # register assignment
    r: Dict[UOp, str] = {x:f"v{i}" for i,x in enumerate(uops)}

    #print_uops(uops)

    bufs = []
    lll = []
    depth = 1
    for u in uops:
      if u.op is UOps.DEFINE_GLOBAL:
        bufs.append(u)
        continue

      l = pm.rewrite(u, ctx=r)
      assert l is not None, f"failed to render {u.op} {u.dtype} {[(x.op,x.dtype) for x in u.src]} {u.arg}"

      if u.op in {UOps.RANGE, UOps.ASSIGN, UOps.DEFINE_LOCAL} or u.dtype == dtypes.void:
        if u.op in {UOps.ENDIF, UOps.ENDRANGE}: depth -= 1
        if u.op is UOps.ASSIGN: r[u] = r[u.src[0]]
        lll.append("  "*depth + l)
      elif u.op in {UOps.INDEX, UOps.CONST}:
        r[u] = l
      else:
        lll.append("  "*depth + f"{render_dtype(u.dtype)} {r[u]} = {l};")
      if u.op in {UOps.IF, UOps.RANGE}: depth += 1

    return f"kernel void {name}(" + ', '.join([f"{render_dtype(u.dtype)} {r[u]}" for u in bufs]) + \
      ", uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n" + '\n'.join(lll) + "\n}"
