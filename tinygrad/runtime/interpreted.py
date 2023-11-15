from typing import Callable, Optional, Dict, List, Any
import functools, re
from tinygrad.helpers import DEBUG
from tinygrad.ops import LazyOp, TernaryOps, ReduceOps, BinaryOps, BufferOps, Op
from tinygrad.shape.symbolic import Variable

def interpret_ast(fxn_for_op:Dict[Op, Callable], from_underlying:Optional[Callable], ast:LazyOp) -> Callable:
  if DEBUG >= 3:
    from tinygrad.graph import print_tree
    print_tree(ast)
  tglob: Dict[str, Any] = {"Variable": Variable}
  lines: List[str] = []

  @functools.lru_cache(None)
  def gstr(x:Any, nm=None) -> str:
    if ('Variable' in (str_arg := repr(x)) or 'NumNode' in str_arg):
      str_arg = re.sub(r'Variable\(.*?\)', lambda m: f'var_vals[{str(m.group(0))}]', str_arg)
      # TODO: (Variable - Variable) might create NumNode. can we remove it?
      return re.sub(r'NumNode\((.*?)\)', r'\1', str_arg)
    ret = str(nm).replace(".", "_") if nm else f"m{len(tglob):04d}"
    tglob[ret] = x
    return ret

  @functools.lru_cache(None)
  def _interpret_ast(ast:LazyOp) -> str:
    if TernaryOps.MULACC in fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)

    if ast.op in BufferOps:
      tmp = f"{gstr(fxn_for_op[ast.op], ast.op)}({gstr(ast.arg.val)}, {gstr(ast.arg.dtype)})" if ast.op == BufferOps.CONST else f"{gstr(fxn_for_op[ast.op], ast.op)}(inputs[{ast.arg.idx-1}])"
      for mop,arg in ast.arg.st.to_movement_ops(): tmp = f"{gstr(fxn_for_op[mop], mop)}({tmp}, {gstr(arg)})"
    else:
      inp = [_interpret_ast(src) for src in ast.src]
      tmp = f"{gstr(fxn_for_op[ast.op], ast.op)}({', '.join(inp + ([gstr(ast.arg)] if ast.arg else []))})"

    ret = f"a{len(lines)}"
    lines.append(f"  {ret} = {tmp}")
    return ret

  ret = _interpret_ast(ast)
  src = '\n'.join(['def run(inputs, var_vals):'] + lines + [f"  return {gstr(from_underlying, 'from_underlying')}({ret})" if from_underlying is not None else f"  return {ret}"])
  if DEBUG >= 4: print(functools.reduce(lambda x,y: (x.replace(y[0], str(y[1])) if y[0][0:2] == "m0" else x), tglob.items(), src))
  exec(compile(src, "<ast>", "exec"), tglob) # pylint: disable=exec-used
  return tglob['run']