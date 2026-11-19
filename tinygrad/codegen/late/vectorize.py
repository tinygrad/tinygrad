# the devectorizer splits every op into scalar lanes. for renderers with vector ALU ops, rebuild the vector ops from the lanes:
# a STACK of the same op on each lane is that op on the STACK of each of its srcs
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import all_same

def lane_of(u:UOp) -> tuple[UOp, int]|None:
  # u is lane i of the vector value x if it's x.index(i)
  if u.op is not Ops.INDEX or len(u.src) != 2 or u.src[0].addrspace is not AddrSpace.ALU or len(u.src[0].shape) != 1: return None
  i = u.src[1]
  while i.op is Ops.CAST: i = i.src[0]
  return (u.src[0], i.arg) if i.op is Ops.CONST else None

def vectorize_stack(x:UOp) -> UOp|None:
  if len(x.src) < 2 or any(s.shape != () for s in x.src) or all_same(x.src): return None
  # a stack of all the lanes of x in order is x
  if all((l:=lane_of(s)) is not None and l == (x.src[0].src[0], i) for i,s in enumerate(x.src)) and x.src[0].src[0].shape == (len(x.src),):
    return x.src[0].src[0]
  # a where selecting a const when true selects it when false (a pad and a valid are both selects of a const)
  def is_const(u:UOp) -> bool: return u.op is Ops.CONST or (u.op is Ops.CAST and is_const(u.src[0]))
  if any(s.op is Ops.WHERE and is_const(s.src[1]) and not is_const(s.src[2]) for s in x.src):
    return UOp.stack(*[s.src[0].ne(True).where(s.src[2], s.src[1]) if s.op is Ops.WHERE and is_const(s.src[1]) and not is_const(s.src[2]) else s
                       for s in x.src])
  # a lane that isn't a where of the stack's other lanes' else value is a where that's always true (the valid of a merged load)
  wheres = [s for s in x.src if s.op is Ops.WHERE]
  if wheres and len(wheres) != len(x.src) and all_same([w.src[2] for w in wheres]):
    return UOp.stack(*[s if s.op is Ops.WHERE else UOp.const(True, wheres[0].src[0].dtype).where(s, wheres[0].src[2]) for s in x.src])
  # a const lane among casts is the cast of a const (a pad value among the loaded values)
  casts = [s for s in x.src if s.op is Ops.CAST and not is_const(s)]
  if casts and len(casts) != len(x.src) and all_same([c.src[0].dtype for c in casts]) and all(s in casts or is_const(s) for s in x.src):
    def const_val(u:UOp): return u.arg if u.op is Ops.CONST else const_val(u.src[0])
    sd = casts[0].src[0].dtype
    if all(s in casts or (isinstance(v:=const_val(s), int) and sd.min <= v <= sd.max) for s in x.src):
      return UOp.stack(*[s if s in casts else UOp.const(const_val(s), sd).cast(x.dtype) for s in x.src])
  # a stack of the same op is the op on stacks
  s0 = x.src[0]
  if s0.op not in GroupOp.ALU|{Ops.CAST, Ops.BITCAST}: return None
  if not all(s.op is s0.op and s.dtype == s0.dtype and s.arg == s0.arg for s in x.src): return None
  cols = list(zip(*[s.src for s in x.src]))
  if not all(all_same([s.dtype for s in col]) for col in cols): return None
  # a src that's the same on every lane is broadcast
  return s0.replace(src=tuple(col[0] if all_same(col) else UOp.stack(*col) for col in cols))

pm_vectorize = PatternMatcher([(UPat(Ops.STACK, name="x"), vectorize_stack)])
