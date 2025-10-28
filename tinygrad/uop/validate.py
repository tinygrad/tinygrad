from typing import Callable
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, python_alu, graph_rewrite
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.helpers import IGNORE_OOB, Context, cpu_profile

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
    Ops.MAX: lambda a,b: z3.If(a<b, b, a), Ops.TRUNC: lambda a: a if a.is_int() else z3.ToReal(z3.If(a >= 0, z3.ToInt(a), -z3.ToInt(-a)))}
  def create_bounded(name:str, vmin, vmax, solver:z3.Solver) -> z3.ArithRef:
    s = z3.Int(name, ctx=solver.ctx)
    solver.add(vmin <= s, s <= vmax)
    return s

  # ctx is (solver, load_number_dict)
  # each uop gets rewritten to NOOP(arg=(solver, z3_object)), the arg has the solver first due to UOpMetaClass caching. z3 objects from different
  # contexts can have the same hash but error on comparison
  z3_renderer = PatternMatcher([
    (UPat(Ops.SPECIAL, src=UPat(Ops.NOOP), name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=(ctx[0],create_bounded(x.arg, 0, x.src[0].arg[1]-1, ctx[0])))),
    (UPat(Ops.DEFINE_VAR, name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=(ctx[0],create_bounded(x.arg[0], x.arg[1], x.arg[2], ctx[0])))),
    (UPat(Ops.RANGE, name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=(ctx[0],create_bounded(f"ridx{x.arg}", 0, x.src[0].arg[1]-1, ctx[0])))),
    # loaded bools become a z3 int with min max of 0-1
    (UPat(Ops.LOAD, dtypes.ints+(dtypes.bool,), name="x"), lambda x,ctx:
      UOp(Ops.NOOP, arg=(ctx[0],create_bounded(f"load{ctx[1].setdefault(x, len(ctx[1]))}", x.dtype.min, x.dtype.max, ctx[0]))).cast(x.dtype)),
    (UPat(Ops.CONST, dtype=dtypes.ints+(dtypes.bool,dtypes.index), name="x"),
      lambda x,ctx: UOp(Ops.NOOP, arg=(ctx[0],(z3.BoolVal if dtypes.is_bool(x.dtype) else z3.IntVal)(x.arg, ctx=ctx[0].ctx)))),
    # z3 can cast from bool to int automatically
    (UPat(Ops.CAST, dtype=dtypes.ints+(dtypes.index,), src=UPat(Ops.NOOP), name="x"), lambda x: x.src[0]),
    (UPat(Ops.CAST, dtype=dtypes.bool, src=UPat(Ops.NOOP), name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=(ctx[0], x.src[0].arg[1]!=0))),
    # if the source of the cast is not a noop it means that it is a float and so we create a new variable
    (UPat(Ops.CAST, dtype=dtypes.ints+(dtypes.index,), name="x"), lambda x,ctx:
      UOp(Ops.NOOP, arg=(ctx[0], create_bounded(f"cast{ctx[1].setdefault(x, len(ctx[1]))}", x.dtype.min, x.dtype.max, ctx[0])))),
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x,ctx:
      UOp(Ops.NOOP, arg=(ctx[0], z3.Bool(f"cast{ctx[1].setdefault(x, len(ctx[1]))}",ctx=ctx[0].ctx)))),
    (UPat(GroupOp.ALU, src=UPat(Ops.NOOP), name="x"), lambda x,ctx: UOp(Ops.NOOP, arg=(ctx[0], z3_alu[x.op](*(s.arg[1] for s in x.src))))),
    # A comparison between floats introduces a new bool variable
    (UPat(GroupOp.Comparison, src=UPat(dtype=dtypes.floats), name="x"), lambda x,ctx:
      UOp(Ops.NOOP, arg=(ctx[0], z3.Bool(f"float_cmp{ctx[1].setdefault(x, len(ctx[1]))}",ctx=ctx[0].ctx)))),
  ])

  def uops_to_z3(solver, *uops: UOp) -> 'list[z3.ExprRef]':
    with Context(TRACK_MATCH_STATS=0, SPEC=0):  # cant pickle z3 objects, and these UOps don't follow spec
      return [s.arg[1] for s in graph_rewrite(uops[0].sink(*uops[1:]), z3_renderer, ctx=(solver, {})).src]

  z3_imported = True
except (ImportError, AttributeError): z3_imported = False

def validate_index(idx:UOp, gate:UOp|None=None):
  if gate is None: gate = UOp.const(dtypes.bool, True)
  # TODO: check for overflow
  if IGNORE_OOB or isinstance(idx.dtype, ImageDType) or (sz := idx.src[0].ptrdtype.size) == -1: return True
  # We can use UOp min/max to do a faster check, but it can give false positive since its not an exact bound and doesn't consider the mask
  if 0<=idx.src[1].vmin and idx.src[1].vmax<sz: return True

  # WEBGPU has a BITCAST in the index. TODO: fix
  if any(x.op is Ops.BITCAST for x in idx.toposort()): return True

  if not z3_imported: raise ImportError("z3 >= 4.12.4 is required for bounds checking, try IGNORE_OOB=0 or \"pip install 'z3-solver>=4.12.4\"")
  solver = z3.Solver(ctx=z3.Context())
  z3_idx, z3_mask = uops_to_z3(solver, idx.src[1], gate)
  solver.add(z3_mask)
  with cpu_profile("validate index with z3", "TINY"):
    if solver.check((z3_idx<0)|(sz<=z3_idx)) == z3.sat:
      print(f"idx={idx.src[1].render(simplify=False)}")
      print(f"gate={gate.render(simplify=False)}")
      print(f"# OUT OF BOUNDS ACCESS: at {solver.model()} INDEX not in 0 - {sz}\nconstraints = {solver}")
      return False
  return True
