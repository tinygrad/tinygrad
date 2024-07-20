from typing import List, Optional
import math, functools
from tinygrad.helpers import DEBUG
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Buffer, Device
from tinygrad.engine.search import _ensure_buffer_alloc, get_kernel_actions, _try_compile_linearized_w_idx, _time_program

class MCTSNode:
  def __init__(self, kernel, parent=None):
    self.kernel = kernel
    self.t = 0
    self.n = 0
    self.parent = parent
    self.children: Optional[List[MCTSNode]] = None

def expand_node(node:MCTSNode):
  node.children = [MCTSNode(x, node) for x in get_kernel_actions(node.kernel, include_0=False).values()]

C = math.sqrt(2)
def mcts_search(lin:Kernel, rawbufs:List[Buffer], amt:int) -> Kernel:
  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k:(k.max+k.min)//2 for k in lin.ast.vars()}
  dev = Device[lin.opts.device]
  root = MCTSNode(lin)
  _compile_fn = functools.partial(_try_compile_linearized_w_idx, compiler=dev.compiler)

  best, best_tm = lin, math.inf
  for _ in range(amt):
    # tree traversal
    node = root
    while node.children is not None:
      if len(node.children) == 0: break
      #if DEBUG>=2: print(f"{node.t/node.n*1e6:6.2f} us {node.n:3d}", node.kernel.name)
      ucb = sorted([(math.inf if child.n == 0 else (child.t/child.n) + C*math.sqrt(math.log(node.n)/child.n), child) for child in node.children],
                   key=lambda x: x[0], reverse=True)  # big number good
      node = ucb[0][1]

    # node expansion
    expand_node(node)

    # rollout
    _, (p, lib, compile_et) = _compile_fn((0, node.kernel))
    tm = min(_time_program(p, lib, var_vals, rawbufs))
    if DEBUG>=3: print(f"{tm*1e6:6.2f} us", node.kernel.name, f"compile {compile_et*1e6:6.2f} us")
    if tm < best_tm: best, best_tm = node.kernel, tm

    # backprop
    bnode = node
    while bnode is not None:
      bnode.t += -tm
      bnode.n += 1
      bnode = bnode.parent

  return best
