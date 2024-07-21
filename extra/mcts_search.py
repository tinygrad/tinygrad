from typing import List, Optional
import math, functools, time, random, statistics
from tinygrad.helpers import DEBUG, getenv, CACHELEVEL, diskcache_get, diskcache_put
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Buffer, Device
from tinygrad.engine.search import _ensure_buffer_alloc, get_kernel_actions, _try_compile_linearized_w_idx, _time_program

class MCTSNode:
  def __init__(self, kernel, parent=None):
    self.kernel = kernel
    self.t = math.inf
    self.n = 0
    self.parent: Optional[MCTSNode] = parent
    self.children: Optional[List[MCTSNode]] = None

def expand_node(node:MCTSNode):
  assert node.children is None
  node.children = [MCTSNode(x, node) for x in get_kernel_actions(node.kernel, include_0=False).values()]
  random.shuffle(node.children)

C = math.sqrt(2)
def mcts_search(lin:Kernel, rawbufs:List[Buffer], amt:int) -> Kernel:
  # TODO: copied from BEAM
  key = {"ast": lin.ast.key, "amt": amt, "device": lin.opts.device, "suffix": lin.opts.suffix}
  if not getenv("IGNORE_MCTS_CACHE") and CACHELEVEL >= 1 and (val:=diskcache_get("mcts_search", key)) is not None:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k:(k.max+k.min)//2 for k in lin.ast.vars()}
  dev = Device[lin.opts.device]
  root = MCTSNode(lin)
  _compile_fn = functools.partial(_try_compile_linearized_w_idx, compiler=dev.compiler)

  def remove_node(node):
    if node.parent is not None:
      assert node.parent.children is not None
      node.parent.children.remove(node)

  st = time.perf_counter()
  best, best_idx, best_tm = lin, 0, math.inf
  for i in range(amt):
    # tree traversal
    node = root
    while node.children is not None and len(node.children) != 0:
      #if DEBUG>=2: print(f"{(node.t/node.n)/best_tm:6.2f} value {node.n:3d}", node.kernel.name)
      ucb = sorted([(math.inf if child.n == 0 else -child.t/best_tm + C*math.sqrt(math.log(node.n)/child.n), child)
                    for child in node.children], key=lambda x: x[0], reverse=True) # pylint: disable=not-an-iterable
      node = ucb[0][1]

    if node.children is not None: break  # no more nodes?

    # node expansion
    expand_node(node)

    # rollout
    _, compile_ret = _compile_fn((0, node.kernel))
    if compile_ret is None:
      remove_node(node)
      tm = math.inf
    else:
      p, lib, _ = compile_ret
      try: tm = statistics.median(_time_program(p, lib, var_vals, rawbufs, cnt=5, early_stop=best_tm*10/1e6))*1e6
      except RuntimeError:
        remove_node(node)
        tm = math.inf

    if tm < best_tm: best, best_idx, best_tm = node.kernel, i, tm
    if DEBUG>=2: print(f"\r{time.perf_counter() - st:7.2f}s: {tm:12.2f} us     best: {best_tm:12.2f} us @ {best_idx+1:4d}        {i+1:4d}/{amt:4d}         {node.kernel.colored_shape()}\033[K", end="")  # noqa: E501

    # backprop
    bnode: Optional[MCTSNode] = node
    while bnode is not None:
      if bnode.t > tm: bnode.t = tm
      bnode.n += 1
      bnode = bnode.parent

  if DEBUG>=2: print()
  if CACHELEVEL >= 1: diskcache_put("mcts_search", key, best.applied_opts)
  return best
