from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
np.set_printoptions(suppress=True)
import math, functools, time, random, statistics
from tinygrad.helpers import DEBUG, getenv, CACHELEVEL, diskcache_get, diskcache_put, flatten
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Buffer, Device
from tinygrad.engine.search import _ensure_buffer_alloc, get_kernel_actions, _try_compile_linearized_w_idx, _time_program

class MCTSNode:
  def __init__(self, kernel, parent=None):
    self.kernel = kernel
    self.t = math.inf
    self.n = 0
    self.tm = math.inf
    self.i = -1
    self.parents: List[MCTSNode] = [parent] if parent is not None else []
    self.children: Optional[List[MCTSNode]] = None

def expand_node(node:MCTSNode) -> MCTSNode:
  assert node.children is None
  node.children = [MCTSNode(x, node) for x in get_kernel_actions(node.kernel, include_0=False).values()]
  return random.choice(node.children)

def sample_tree(node:MCTSNode, best_tm:float, temperature:float=1.0) -> MCTSNode:
  if node.children is None or len(node.children) == 0: return node
  unexplored_children = []
  explored_children = []
  ucb_explored_children = []
  for child in node.children:
    if child.n == 0: unexplored_children.append(child)
    else:
      explored_children.append(child)
      ucb_explored_children.append(-child.t/best_tm + C*math.sqrt(math.log(node.n)/child.n))
  if len(unexplored_children): return random.choice(unexplored_children)
  ucb_exp = np.exp(np.array(ucb_explored_children)/temperature)
  return sample_tree(np.random.choice(explored_children, p=ucb_exp/np.sum(ucb_exp)), best_tm, temperature)

def backprop(bnode:MCTSNode, tm, strength=1.0):
  if bnode.t > tm: bnode.t = tm
  bnode.n += strength
  for parent in bnode.parents: backprop(parent, tm, strength/len(bnode.parents))

graph_mcts_cnt = 0
C = math.sqrt(2)
def mcts_search(lin:Kernel, rawbufs:List[Buffer], amt:int) -> Kernel:
  global graph_mcts_cnt
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

  st = time.perf_counter()
  best, best_idx, best_tm = lin, 0, math.inf
  seen_libs: Dict[bytes, MCTSNode] = {}
  for i in range(amt):
    # tree traversal
    node = sample_tree(root, best_tm, temperature=0.5)
    if node.children is not None: break  # no more nodes?

    # node expansion
    if node.n != 0: node = expand_node(node)
    node.i = i  # when was node explored

    # rollout
    _, compile_ret = _compile_fn((0, node.kernel))
    if compile_ret is None:
      tm = math.inf
    else:
      p, lib, _ = compile_ret
      if (sibling_node:=seen_libs.get(lib, None)) is not None:
        # remove this node, it's a duplicate
        for parent in node.parents:
          assert parent.children is not None
          parent.children.remove(node)
        tm = sibling_node.t
      else:
        seen_libs[lib] = node
        try: tm = statistics.median(_time_program(p, lib, var_vals, rawbufs, cnt=5, early_stop=best_tm*10/1e6))*1e6
        except RuntimeError:
          tm = math.inf
        node.tm = tm

    if tm < best_tm: best, best_idx, best_tm = node.kernel, i, tm
    if DEBUG>=2: print(f"\r{time.perf_counter() - st:7.2f}s: {tm:12.2f} us     best: {best_tm:12.2f} us @ {best_idx+1:4d}        {i+1:4d}/{amt:4d}         {node.kernel.colored_shape()}\033[K", end="")  # noqa: E501

    # backprop
    backprop(node, tm)
  if DEBUG>=2: print()

  if getenv("MCTSGRAPH"):
    from tinygrad.engine.graph import nx, save_graph, GRAPHPATH
    G = nx.DiGraph()
    def add_node(node:MCTSNode):
      if node.n == 0: return
      for parent in node.parents: G.add_edge(parent, node)
      gopts = node.kernel.applied_opts
      edge_lbl = f"{str(gopts[-1].op)[7:]} {gopts[-1].axis} {gopts[-1].amt}" if len(gopts) else "ROOT"
      G.add_node(node, label=f"{node.i+1}\n{node.tm:.2f} us\n{edge_lbl}\nt {node.t:.2f}\nn {node.n}",
                 fillcolor="#80ff8080" if node.tm == best_tm else "#ffff8080", style='filled' if node.t == best_tm else '')
      if node.children is not None:
        for child in node.children: add_node(child)
    add_node(root)
    save_graph(G, f"{GRAPHPATH}.{graph_mcts_cnt}.mcts", '-Grankdir=LR')
    graph_mcts_cnt += 1

  if CACHELEVEL >= 1: diskcache_put("mcts_search", key, best.applied_opts)
  return best
