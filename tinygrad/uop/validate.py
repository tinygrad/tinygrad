from typing import Callable, cast
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, python_alu
from tinygrad.dtype import ImageDType, dtypes, Invalid
from tinygrad.helpers import IGNORE_OOB, cpu_profile

try:
  import z3
  # older versions of z3 dont have some operators like & overloaded
  if z3.get_version() < (4, 12, 4, 0): raise ImportError

  # IDIV is truncated division but z3 does euclidian division (floor if b>0 ceil otherwise); mod by power of two sometimes uses Ops.AND
  def z3_cdiv(a, b):return z3.If((a<0), z3.If(0<b, (a+(b-1))/b, (a-(b+1))/b), a/b)
  def z3_xor(a,b):
    if isinstance(a, z3.BoolRef): return a^b
    assert a==-1 or b==-1, "xor can only be used in indexing if one of the aruments is -1"
    return -a-1 if b==-1 else -b-1
  z3_alu: dict[Ops, Callable] = python_alu | {Ops.MOD: lambda a,b: a-z3_cdiv(a,b)*b, Ops.IDIV: z3_cdiv, Ops.SHR: lambda a,b: a/(2**b.as_long()),
    Ops.SHL: lambda a,b: a*(2**b.as_long()), Ops.AND: lambda a,b: a%(b+1) if isinstance(b, z3.ArithRef) else a&b, Ops.WHERE: z3.If, Ops.XOR: z3_xor,
    Ops.MAX: lambda a,b: z3.If(a<b, b, a),}
  def create_bounded(name:str, vmin, vmax, solver:z3.Solver) -> tuple[z3.ArithRef, z3.BoolRef]:
    return (s:=z3.Int(name, ctx=solver.ctx)), (vmin <= s)&(s <= vmax)

  z3_renderer = PatternMatcher([
    (UPat.var("cond").where(UPat.var("x"), UPat.const(dtypes.index, Invalid)), lambda x,cond,ctx: (ctx[1][x], ctx[1][cond])),
    # variables
    (UPat(Ops.SPECIAL, name="x"), lambda x,ctx: create_bounded(x.arg, 0, ctx[1][x.src[0]]-1, ctx[0])),
    (UPat(Ops.DEFINE_VAR, name="x"), lambda x,ctx: create_bounded(x.arg[0], x.arg[1], x.arg[2], ctx[0])),
    (UPat(Ops.RANGE, name="x"), lambda x,ctx: create_bounded(f"r{x.arg}", 0, ctx[1][x.src[0]]-1, ctx[0])),
    # loads are variables bounded by the min/max of the dtype
    (UPat(Ops.LOAD, dtypes.ints+(dtypes.index,), name="x"), lambda x,ctx: create_bounded(f"load{len(ctx[1])}", x.dtype.min, x.dtype.max, ctx[0])),
    (UPat(Ops.LOAD, dtypes.bool, name="x"), lambda x,ctx: (z3.Bool(f"load{len(ctx[1])}", ctx=ctx[0].ctx), None)),
    # constants
    (UPat(Ops.CONST, arg=Invalid, name="x"), lambda x,ctx: (z3.Int("Invalid", ctx=ctx[0].ctx), None)),
    (UPat(Ops.CONST, dtypes.ints+(dtypes.index,), name="x"), lambda x,ctx: (z3.IntVal(x.arg, ctx=ctx[0].ctx), None)),
    (UPat(Ops.CONST, dtypes.bool, name="x"), lambda x,ctx: (z3.BoolVal(x.arg, ctx=ctx[0].ctx), None)),
    # casts from floats create new variables
    (UPat(Ops.CAST, dtypes.bool, src=(UPat(dtype=dtypes.floats),), name="x"), lambda x,ctx: (z3.Bool(f"cast{len(ctx[1])}",ctx=ctx[0].ctx), None)),
    (UPat(Ops.CAST, dtypes.ints+(dtypes.index,), src=(UPat(dtype=dtypes.floats),), name="x"), lambda x,ctx:
      create_bounded(f"cast{len(ctx[1])}", x.dtype.min, x.dtype.max, ctx[0])),
    # A comparison between floats introduces a new bool variable
    (UPat(GroupOp.Comparison, src=UPat(dtype=dtypes.floats), name="x"), lambda x,ctx: (z3.Bool(f"float_cmp{len(ctx[1])}", ctx=ctx[0].ctx), None)),
    # casts from bool/int to int/bool
    (UPat(Ops.CAST, dtypes.ints+(dtypes.index,),src=(UPat.var("x", dtypes.bool),), name="c"), lambda x,c,ctx: (z3.If(ctx[1][x], 1, 0), None)),
    (UPat(Ops.CAST, dtypes.ints+(dtypes.index,), src=(UPat.var("x", dtypes.ints+(dtypes.index,)),), name="c"), lambda x,c,ctx: (ctx[1][x], None)),
    (UPat(Ops.CAST, dtypes.bool, name="x"), lambda x,ctx: (ctx[1][x.src[0]]!=0, None)),
    (UPat(GroupOp.ALU, name="x"), lambda x,ctx: (z3_alu[x.op](*(ctx[1][s] for s in x.src)), None)),
  ])

  def uops_to_z3(solver, *uops: UOp) -> list[z3.ExprRef]:
    lst = list(UOp.sink(*uops).toposort(gate=lambda x: x.dtype.scalar() in dtypes.ints+(dtypes.bool, dtypes.index) or x.op is Ops.SINK))[:-1]
    z3map: dict[UOp, z3.ExprRef] = {}
    for i,u in enumerate(lst):
      new_u, constraint = cast(tuple[z3.ArithRef, z3.BoolRef|None], z3_renderer.rewrite(u, ctx=(solver, z3map)))
      if constraint is not None: solver.add(constraint)
      z3map[u] = new_u
    assert all(u in z3map for u in uops), "UOp failed to rewrite to z3!"
    return [z3map[u] for u in uops]

  z3_imported = True
except (ImportError, AttributeError): z3_imported = False

def validate_index(buf:UOp, idx:UOp, gate:UOp|None=None):
  if idx.op is Ops.CONST and idx.arg is Invalid: return True
  if gate is None: gate = UOp.const(dtypes.bool, True)
  # TODO: check for overflow
  if IGNORE_OOB or isinstance(buf.dtype, ImageDType) or (sz := buf.ptrdtype.size) == -1: return True
  # We can use UOp min/max to do a faster check, but it can give false positive since its not an exact bound and doesn't consider the mask
  if 0<=idx.vmin and idx.vmax<sz: return True

  # WEBGPU has a BITCAST in the index. TODO: fix
  if any(x.op is Ops.BITCAST for x in idx.toposort()): return True

  if not z3_imported: raise ImportError("z3 >= 4.12.4 is required for bounds checking, try IGNORE_OOB=0 or \"pip install 'z3-solver>=4.12.4\"")
  solver = z3.Solver(ctx=z3.Context())
  z3_idx, z3_mask = uops_to_z3(solver, idx, gate)
  solver.add(z3_mask)
  with cpu_profile("validate index with z3", "TINY"):
    match solver.check((z3_idx<0)|(sz<=z3_idx)):
      case z3.unsat: return True
      case z3.sat: print(f"# OUT OF BOUNDS ACCESS: at {solver.model()} INDEX not in 0 - {sz}\nconstraints = {solver}")
      case z3.unknown: print(f"# UNKNOWN RESULT FROM Z3: {solver.reason_unknown()}\nconstraints = {solver}")
  print(f"idx={idx.render(simplify=False)}")
  print(f"mask={gate.render(simplify=False)}")
  return False
