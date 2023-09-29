import os, atexit
try:
  import networkx as nx  # type: ignore
except ImportError:
  nx = None # graph won't work
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps, Op, OpType, LazyOp
from tinygrad.helpers import GRAPH, GRAPHPATH, DEBUG, GlobalCounters
from tinygrad.runtime.lib import RawConst

if TYPE_CHECKING: from tinygrad.lazy import LazyBuffer

# **** debugging and graphing ****

G = nx.DiGraph() if nx is not None else None
cnts: Dict[OpType, int] = defaultdict(int)
if DEBUG >= 2:
  def print_globalcounters():
    if GlobalCounters.time_sum_s == 0: return
    print(f"avg: {GlobalCounters.global_ops*1e-9/GlobalCounters.time_sum_s:8.2f} GFLOPS {GlobalCounters.global_mem*1e-9/GlobalCounters.time_sum_s:8.2f} GB/s",
          f"{' '*10}total: {GlobalCounters.kernel_count:5d} kernels {GlobalCounters.global_ops*1e-9:8.2f} GOPS {GlobalCounters.global_mem*1e-9:8.2f} GB {GlobalCounters.time_sum_s*1e3:8.2f} ms")
  atexit.register(print_globalcounters)
if GRAPH:
  def save_graph_exit():
    for k,v in cnts.items(): print(k, v)
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.dot')
    # -Gnslimit=100 can make it finish, but you won't like results
    os.system(f'dot -Tsvg {GRAPHPATH}.dot -o {GRAPHPATH}.svg')
  atexit.register(save_graph_exit)

node_count = 0
def nm(x):
  global node_count
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', node_count)
    node_count += 1
  return x.node_id

def get_sop(op: List[Op]):
  if len(op) <= 2: return '.'.join([str(y).split(".")[1] for y in op][::-1])
  if len(op) <= 4: return '.'.join([str(y).split(".")[1][0:3] for y in op][::-1])
  return str(len(op))

def str_dtype(dtyp):
  ret = str(dtyp)[7:]
  return "" if ret == 'float' else f"\n{ret}"

def log_op(ret: 'LazyBuffer', ast: LazyOp, show_graph: Optional[bool] = None, phantom=False):
  if show_graph is None: show_graph = bool(GRAPH)
  if not DEBUG and not show_graph: return
  if ast.op == LoadOps.CONST: return
  op: List[Op] = [x.op for x in ast.get_lazyops()]
  inp: List['LazyBuffer'] = [x for x in ast.buffers if (not isinstance(x.realized, RawConst) and (hasattr(x.base, 'op') and not x.base.op.op == LoadOps.CONST)) or GRAPH > 1]
  oporder = [LoadOps, TernaryOps, ReduceOps, BinaryOps, UnaryOps, MovementOps]
  optype = type(sorted(op, key=lambda x: oporder.index(type(x)))[0])
  cnts[optype] += 1
  if DEBUG >= 6: print(f"{op} : {', '.join([f'{x.shape}-<{nm(x)}>' for x in inp])} -> {ret.shape}-<{nm(ret)}>")
  if show_graph:
    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", TernaryOps: "#c0c0c0"}
    dashed = optype == LoadOps or (hasattr(ret, "st") and not ret.st.contiguous)  # type: ignore

    for x in inp:
      if x.base != x:
        # view node
        G.add_node(nm(x), label=f"{x.st.shape} {x.st.real_strides()} {x.st.views[-1].mask if x.st.views[-1].mask else ''}", fillcolor="#80ff8080", style='filled')
        G.add_edge(nm(x.base), nm(x), color="#408040")
      G.add_edge(nm(x), nm(ret), label=get_sop(op), color='#00000060' if phantom else 'black')
      if 'label' not in G.nodes[nm(x.base)]:
        G.nodes[nm(x.base)]['label'] = str(x.shape)+str_dtype(ret.dtype)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))

    G.nodes[nm(ret)]['label'] = (str(set(x.shape for x in inp))+"\n"+str(ret.shape) if optype == ReduceOps else str(ret.shape))+str_dtype(ret.dtype)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('60' if phantom else ('80' if dashed else str()))) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['color'] = 'white' if phantom else 'black'
    G.nodes[nm(ret)]['style'] = ('filled, dashed' if dashed else 'filled')
    G.nodes[nm(ret)]['prunable'] = optype in [LoadOps, MovementOps]
