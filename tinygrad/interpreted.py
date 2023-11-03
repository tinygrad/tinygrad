import math, functools
from typing import Tuple, List, Dict, Callable
from tinygrad.helpers import dedup, dtypes, DEBUG
from tinygrad.ops import MovementOps, LazyOp, MemBuffer, ConstBuffer, Op, TernaryOps, ReduceOps, BinaryOps, BufferOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

def to_movement_ops(shape:ShapeTracker) -> List[Tuple[MovementOps, Tuple]]:
  to_apply:List[Tuple[MovementOps, Tuple]] = []
  for v in shape.views:
    real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
    real_offset = v.offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
    # first, we apply the offset
    # then, we make it the correct shape
    # then, we apply permutations
    # TODO: don't use as_strided
    to_apply.append((MovementOps.AS_STRIDED, ([s if st != 0 else 1 for s,st in zip(real_shape, v.strides)], v.strides, real_offset)))
    # then, we apply pre expand pads
    if v.mask is not None:
      pre_expand_pads = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      post_expand_pads = tuple((x,s-y) if st == 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      if any(x != (0,0) for x in pre_expand_pads):
        to_apply.append((MovementOps.PAD, pre_expand_pads))
        real_shape = tuple(x+s[0]+s[1] for x,s in zip(real_shape, pre_expand_pads))
    # then, we do any expands
    if any(s != 1 and st == 0 for s,st in zip(real_shape, v.strides)): to_apply.append((MovementOps.EXPAND, real_shape))
    # lastly, we apply post expand pads
    if v.mask is not None and any(x != (0,0) for x in post_expand_pads): to_apply.append((MovementOps.PAD, post_expand_pads))
  return to_apply

def ast_to_python(ast:LazyOp, fxn_for_op: Dict[Op, Callable]):
  lines: List[Tuple[str, List[str], str]] = []
  last_use: Dict[str, int] = {}
  tglob = {"dtypes": dtypes, "inf": math.inf, "nan": math.nan, "MemBuffer": MemBuffer, "ConstBuffer": ConstBuffer, "ShapeTracker": ShapeTracker, "View": View}
  def prep_op(op):
    ret = str(op).replace('.', '_')
    tglob[ret] = fxn_for_op[op]
    return ret
  expand_shapetracker = MovementOps.AS_STRIDED in fxn_for_op
  @functools.lru_cache(None)
  def _compile_ast(ast:LazyOp) -> str:
    nonlocal lines
    if TernaryOps.MULACC in fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)
    inp = [_compile_ast(src) for src in ast.src]
    for ip in inp: last_use[ip] = len(lines)
    real_inp = inp[:]
    if expand_shapetracker and ast.op == BufferOps.MEM: inp += [f"inputs[{ast.arg.idx-1}]"]
    elif expand_shapetracker and ast.op == BufferOps.CONST: inp += [str(ast.arg.val), str(ast.arg.dtype)]
    elif ast.arg: inp += [str(ast.arg)]
    ret = f"a{len(lines)}"
    lines.append((ret, dedup(real_inp), f"{prep_op(ast.op)}({', '.join(inp)})"))
    if expand_shapetracker and ast.op in BufferOps:
      for mop,arg in to_movement_ops(ast.arg.st):
        lines.append((ret, [ret], f"{prep_op(mop)}({ret}, {str(arg)})"))
    return ret
  ret = _compile_ast(ast)
  src = ['def run(inputs):']
  for i,(x,inp,y) in enumerate(lines):
    src.append(f"  {x} = {y}")
    to_del = [ip for ip in inp if last_use[ip] == i]
    if to_del and i != len(lines)-1: src.append(f"  del {','.join(to_del)}")
  src.append(f"  return {ret}")
  ssrc = '\n'.join(src)
  if DEBUG >= 4: print(ssrc)
  # pylint: disable-next=exec-used
  exec(compile(ssrc, "<ast>", "exec"), tglob)
  return tglob['run']
