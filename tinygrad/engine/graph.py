import os, atexit, functools, contextlib
from collections import defaultdict
from typing import List, Any, DefaultDict
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps, UOps, UOp
from tinygrad.device import Device
from tinygrad.helpers import GRAPHPATH, DEBUG, GlobalCounters, word_wrap
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.viz.serve import uops_colors

with contextlib.suppress(ImportError): import networkx as nx

# **** debugging and graphing ****

def print_globalcounters():
  if GlobalCounters.time_sum_s == 0: return
  print(f"avg: {GlobalCounters.global_ops*1e-9/GlobalCounters.time_sum_s:8.2f} GFLOPS {GlobalCounters.global_mem*1e-9/GlobalCounters.time_sum_s:8.2f} GB/s",  # noqa: E501
        f"{' '*10}total: {GlobalCounters.kernel_count:5d} kernels {GlobalCounters.global_ops*1e-9:8.2f} GOPS {GlobalCounters.global_mem*1e-9:8.2f} GB {GlobalCounters.time_sum_s*1e3:8.2f} ms")  # noqa: E501
if DEBUG >= 2: atexit.register(print_globalcounters)

def save_graph(G, fn, opt=""):
  print("saving", G, f"to {fn}.svg")
  nx.drawing.nx_pydot.write_dot(G, f'{fn}.dot')
  os.system(f'dot {opt} -Tsvg {fn}.dot -o {fn}.svg')

G:Any = None
def init_graph():
  global G
  if G is not None: return
  G = nx.DiGraph()
  atexit.register(functools.partial(save_graph, G, GRAPHPATH)) # -Gnslimit=100 can make it finish, but you won't like results

counts: DefaultDict[type, int] = defaultdict(int)
def nm(x):
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', counts[type(x)])
    counts[type(x)] += 1
  return x.node_id

def realized_lazybuffer(lb:'LazyBuffer', num):
  init_graph()
  G.nodes[nm(lb)]['style'] = '"filled,bold"'
  G.nodes[nm(lb)]['fillcolor'] = G.nodes[nm(lb)]['fillcolor'][:-2]
  G.nodes[nm(lb)]['label'] = '"' + G.nodes[nm(lb)]["label"].replace('"', '') + f'\nK:{num}"'

top_colors = {MetaOps: '#FFFFa0', UnaryOps: "#c0c0c0", ReduceOps: "#FFA0A0", BinaryOps: "#c0c0c0", TernaryOps: "#c0c0c0"}
def log_lazybuffer(lb:'LazyBuffer', scheduled=False):
  init_graph()
  if lb.base.realized is None and lb.base.op is MetaOps.CONST: return
  if lb.base != lb:
    offset = tuple(x.offset for x in lb.st.views if x.offset != 0)
    label = f"{lb.st.shape}\n{lb.st.real_strides()}" + (f"\n{offset}" if len(offset) else "")
    G.add_node(nm(lb), style='"filled,dashed"', fillcolor="#80ff8080", color="black", label=label)
    G.add_edge(nm(lb.base), nm(lb), color='#00000060')
    lb = lb.base
  if lb.realized is None:
    label_append = []
    for idx,x in enumerate(lb.srcs):
      if nm(x) not in G.nodes: log_lazybuffer(x)
      if x.base.realized is None and x.base.op is MetaOps.CONST:
        label_append.append(f"\nCONST{idx} {x.base.arg:g}")
      else:
        G.add_edge(nm(x), nm(lb), color='#a0a0a0')
    label = '"' + \
      (str(set(x.shape for x in lb.srcs))+"\n"+str(lb.shape) if lb.op in ReduceOps else str(lb.shape)) + \
      (f"\n{lb.dtype.name}" if lb.dtype.name != "float" else "")+f"\n{lb.op}"+(f"\n{lb.arg}" if lb.op in {MetaOps.CONST, UnaryOps.CAST} else "") + \
      (f"\n{lb.device[:15]}" if lb.device != Device.DEFAULT else "") + ''.join(label_append) + f'\n{lb.metadata}"'
    G.add_node(nm(lb), style='"filled,dashed"', fillcolor=[v for k,v in top_colors.items() if lb.op in k][0] + "80", color="black", label=label)
    if scheduled: G.nodes[nm(lb)]['shape'] = 'box'
  else:
    if nm(lb) not in G.nodes:
      # realized but unseen?
      G.add_node(nm(lb), label=f'"{str(lb.base.realized)[5:-1].replace(" ", chr(10))}\nb:{nm(lb.realized)}"', style='filled', fillcolor="#f0c08080")

graph_uops_cnt = 0
def graph_uops(uops:List[UOp]):
  global graph_uops_cnt
  G = nx.DiGraph()
  for u in uops:
    if u.op in {UOps.ENDRANGE, UOps.ENDIF}: continue
    G.add_node(uops.index(u), label=f"{str(u.op)[5:]}{(' '+word_wrap(str(u.arg).replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}",
               style="filled", fillcolor=uops_colors.get(u.op, "#ffffff"))
    for v in u.src: G.add_edge(uops.index(v), uops.index(u))
  save_graph(G, f'{GRAPHPATH}.{graph_uops_cnt}.uops', '-Grankdir=LR')
  graph_uops_cnt += 1
def graph_uop(uop:UOp): return graph_uops(list(uop.sparents))
