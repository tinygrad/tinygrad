import os, atexit, itertools
try:
  import networkx as nx  # type: ignore
except ImportError:
  nx = None # graph won't work
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps, Op, OpType, LazyOp
from tinygrad.helpers import GRAPH, GRAPHPATH, PRUNEGRAPH, DEBUG, GlobalCounters
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
    if PRUNEGRAPH: prune_graph()
    print("saving", G)
    a = r'(bmp|cgimage|canon|dot|gv|xdot|eps|exr|fig|gd|gd2|gif|gtk|ico|imap|imap_np|ismap|cmap|cmapx|cmapx_np|jpg|jpeg|jpe|jp2|json|json0|dot_json|xdot_json|pdf|pic|pct|pict|plain|plain-ext|png|pov|ps|ps2|psd|sgi|svg|svgz|tga|tif|tiff|tk|vml|vmlz|vrml|wbmp|webp|xlib|x11)' # dot -Tv
    p = os.path.dirname(GRAPHPATH)+"/"
    f = [i for i in re.split(r'[^\w\-_]+', GRAPHPATH.replace(p,"")) if i != ""]
    fn = f[0] if len(f)>0 and not re.fullmatch(a, f[0]) else "tiny_graph" 
    nx.drawing.nx_pydot.write_dot(G, f'{p}{fn}.dot')
    # -Gnslimit=100 can make it finish, but you won't like results
    ex = [os.system(f'dot -T{fmt} {p}{fn}.dot -o {p}{fn}.{fmt}') if re.fullmatch(a, fmt.lower()) else 0 for fmt in f ]
    if len(f) <= 1 or len(ex) == len(f): os.system(f'dot -Tsvg {p}{fn}.dot -o {p}{fn}.svg')#to keep legacy behavior and a default option if no match
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
  op: List[Op] = [x.op for x in ast.get_lazyops()]
  inp: List['LazyBuffer'] = [x for x in ast.buffers if not isinstance(x.realized, RawConst) or GRAPH > 1]
  oporder = [LoadOps, TernaryOps, ReduceOps, BinaryOps, UnaryOps, MovementOps]
  optype = type(sorted(op, key=lambda x: oporder.index(type(x)))[0])
  cnts[optype] += 1
  if DEBUG >= 6: print(f"{op} : {', '.join([f'{x.shape}-<{nm(x)}>' for x in inp])} -> {ret.shape}-<{nm(ret)}>")
  if show_graph:
    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", TernaryOps: "#ff8080"}
    dashed = (optype == LoadOps and hasattr(ret, "_backing")) or (hasattr(ret, "st") and not ret.st.contiguous)  # type: ignore

    for x in inp:
      G.add_edge(nm(x), nm(ret), label=get_sop(op), color='#00000060' if phantom else 'black')
      if 'label' not in G.nodes[nm(x)]:
        G.nodes[nm(x)]['label'] = str(x.shape)+str_dtype(ret.dtype)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))

    G.nodes[nm(ret)]['label'] = (str(set(x.shape for x in inp))+"\n"+str(ret.shape) if optype == ReduceOps else str(ret.shape))+str_dtype(ret.dtype)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('60' if phantom else ('80' if dashed else str()))) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['color'] = 'white' if phantom else 'black'
    G.nodes[nm(ret)]['style'] = ('filled, dashed' if dashed else 'filled')
    G.nodes[nm(ret)]['prunable'] = optype in [LoadOps, MovementOps]

# prune movementops and loadops
def prune_graph():
  dead_nodes = []
  for n in G.nodes:
    if 'prunable' in G.nodes[n] and G.nodes[n]['prunable']:
      G.add_edges_from([(x, y) for (x,_),(_,y) in itertools.product(G.in_edges(n), G.out_edges(n))])
      dead_nodes.append(n)
  G.remove_nodes_from(dead_nodes)
