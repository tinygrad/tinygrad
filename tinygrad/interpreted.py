import functools
from typing import Tuple, List, Dict, Callable, Any
from tinygrad.helpers import DEBUG
from tinygrad.ops import MovementOps, LazyOp, Op, TernaryOps, ReduceOps, BinaryOps, BufferOps
from tinygrad.shape.shapetracker import ShapeTracker

def to_movement_ops(shape:ShapeTracker) -> List[Tuple[MovementOps, Tuple]]:
  to_apply:List[Tuple[MovementOps, Tuple]] = []
  for v in shape.views:
    real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
    real_offset = v.offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
    # first, we apply the offset
    # then, we make it the correct shape
    # then, we apply permutations
    # TODO: don't use as_strided
    to_apply.append((MovementOps.AS_STRIDED, (tuple([s if st != 0 else 1 for s,st in zip(real_shape, v.strides)]), v.strides, real_offset)))
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

def ast_to_python(ast:LazyOp, f:Dict[Op, Callable]):
  tglob: Dict[str, Any] = {}
  lines: List[str] = []

  @functools.lru_cache(None)
  def gstr(x:Any, nm=None) -> str:
    ret = str(nm).replace(".", "_") if nm else f"m{len(tglob):04d}"
    tglob[ret] = x
    return ret

  @functools.lru_cache(None)
  def _compile_ast(ast:LazyOp) -> str:
    if TernaryOps.MULACC in f and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)
    inp = [_compile_ast(src) for src in ast.src]

    if MovementOps.AS_STRIDED in f and ast.op in BufferOps:
      tmp = f"{gstr(f[ast.op], ast.op)}({gstr(ast.arg.val)}, {gstr(ast.arg.dtype)})" if ast.op == BufferOps.CONST else f"{gstr(f[ast.op], ast.op)}(inputs[{ast.arg.idx-1}])"
      for mop,arg in to_movement_ops(ast.arg.st): tmp = f"{gstr(f[mop], mop)}({tmp}, {gstr(arg)})"
    else:
      tmp = f"{gstr(f[ast.op], ast.op)}({', '.join(inp + ([gstr(ast.arg)] if ast.arg else []))})"

    ret = f"a{len(lines)}"
    lines.append(f"  {ret} = {tmp}")
    return ret

  ret = _compile_ast(ast)
  src = ['def run(inputs):'] + lines + [f"  return {ret}"]
  ssrc = '\n'.join(src)
  if DEBUG >= 4: print(ssrc)
  exec(compile(ssrc, "<ast>", "exec"), tglob) # pylint: disable=exec-used
  return tglob['run']
