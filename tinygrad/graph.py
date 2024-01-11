import os, atexit
from collections import defaultdict
from typing import List, Any, DefaultDict
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, BufferOps, TernaryOps, Op, LazyOp, GlobalCounters
from tinygrad.device import Device
from tinygrad.helpers import GRAPHPATH, DEBUG, getenv
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.shape.symbolic import NumNode

# **** debugging and graphing ****

if DEBUG >= 2:
  def print_globalcounters():
    if GlobalCounters.time_sum_s == 0: return
    print(f"avg: {GlobalCounters.global_ops*1e-9/GlobalCounters.time_sum_s:8.2f} GFLOPS {GlobalCounters.global_mem*1e-9/GlobalCounters.time_sum_s:8.2f} GB/s",  # noqa: E501
          f"{' '*10}total: {GlobalCounters.kernel_count:5d} kernels {GlobalCounters.global_ops*1e-9:8.2f} GOPS {GlobalCounters.global_mem*1e-9:8.2f} GB {GlobalCounters.time_sum_s*1e3:8.2f} ms")  # noqa: E501
  atexit.register(print_globalcounters)

G:Any = None
def init_graph():
  global G
  if G is not None: return
  import networkx as nx
  G = nx.DiGraph()
  def save_graph_exit():
    print("saving", G, f"to {GRAPHPATH}.svg")
    nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.dot')
    # -Gnslimit=100 can make it finish, but you won't like results
    os.system(f'dot -Tsvg {GRAPHPATH}.dot -o {GRAPHPATH}.svg')
  atexit.register(save_graph_exit)

counts: DefaultDict[type, int] = defaultdict(int)
def nm(x):
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', counts[type(x)])
    counts[type(x)] += 1
  return x.node_id

def get_sop(op: List[Op]):
  op = [x for x in op if x not in BufferOps]
  if len(op) <= 2: return '.'.join([str(y).split(".")[1] for y in op][::-1])
  if len(op) <= 6: return '.'.join([str(y).split(".")[1][0:3] for y in op][::-1])
  return str(len(op))

def str_dtype(dtyp):
  ret = str(dtyp)[7:]
  return "" if ret == 'float' else f"\n{ret}"

def realized_lazybuffer(lb, num):
  init_graph()
  G.nodes[nm(lb)]['style'] = '"filled,bold"'
  G.nodes[nm(lb)]['fillcolor'] = G.nodes[nm(lb)]['fillcolor'][:-2]
  G.nodes[nm(lb)]['label'] = '"' + G.nodes[nm(lb)]["label"].replace('"', '') + f'\nK:{num} b:{"FAKE" if lb.realized is None else nm(lb.realized)}"'

top_colors = {LoadOps: '#FFFFa0', UnaryOps: "#c0c0c0", ReduceOps: "#FFA0A0", BinaryOps: "#c0c0c0",
              MovementOps: "#80ff80", TernaryOps: "#c0c0c0", BufferOps: '#a0a0ff'}
def log_lazybuffer(lb, scheduled=False):
  init_graph()
  if lb.base != lb:
    offset = lb.st.expr_node(NumNode(0))[0]
    label = f"{lb.st.shape}\n{lb.st.real_strides()}" + (f"\n{offset}" if offset != 0 else "")
    G.add_node(nm(lb), style='"filled,dashed"', fillcolor="#80ff8080", color="black", label=label)
    G.add_edge(nm(lb.base), nm(lb), color='#00000060')
    lb = lb.base
  if lb.realized is None:
    for x in lb.srcs:
      if nm(x) not in G.nodes: log_lazybuffer(x)
      G.add_edge(nm(x), nm(lb), color='#a0a0a0')
    label = '"' + \
      (str(set(x.shape for x in lb.srcs))+"\n"+str(lb.shape) if lb.op in ReduceOps else str(lb.shape)) + \
      str_dtype(lb.dtype)+f"\n{lb.op}"+(f"\n{lb.arg}" if lb.op in {LoadOps.CONST, UnaryOps.CAST} else "") + \
      (f"\n{lb.device}" if lb.device != Device.DEFAULT else "") + '"'
    G.add_node(nm(lb), style='"filled,dashed"', fillcolor=[v for k,v in top_colors.items() if lb.op in k][0] + "80", color="black", label=label)
    if scheduled: G.nodes[nm(lb)]['shape'] = 'box'
  else:
    if nm(lb) not in G.nodes:
      # realized but unseen?
      G.add_node(nm(lb), label=f'"{str(lb.base.realized)[5:-1].replace(" ", chr(10))}\nb:{nm(lb.realized)}"', style='filled', fillcolor="#f0c08080")

def _tree(lazydata, cycles, cnt, prefix=""):
  cnt[0] += 1
  if len(lazydata.src) == 0: return [f"━━ {prefix}{lazydata.op.name} {lazydata.arg if lazydata.arg else ''}"]
  if (lid := id(lazydata)) in cycles and cycles[lid][1] > (tcnt := getenv("TREE_CYCLE_CNT", 5)) and tcnt >= 0:
    return [f"━⬆︎ goto {cycles[id(lazydata)][0]}: {lazydata.op.name}"]
  cycles[lid] = (cnt[0], 1 if lid not in cycles else cycles[lid][1]+1)
  lines = [f"━┳ {prefix}{lazydata.op.name} {lazydata.arg if lazydata.arg else ''}"]
  childs = [_tree(c, cycles, cnt) for c in lazydata.src[:]]
  for c in childs[:-1]: lines += [f" ┣{c[0]}"] + [f" ┃{l}" for l in c[1:]]
  return lines + [" ┗"+childs[-1][0]] + ["  "+l for l in childs[-1][1:]]

def print_tree(lazydata:LazyOp): print("\n".join([f"{str(i).rjust(3)} {s}" for i,s in enumerate(_tree(lazydata, {}, [-1]))]))

def graph_uops(uops:List[UOp]):
  import networkx as nx
  colors = {UOps.ALU: "#ffffc0", UOps.LOAD: "#ffc0c0", UOps.STORE: "#c0ffc0", UOps.SPECIAL: "#c0c0ff", UOps.CONST: "#e0e0e0",
            UOps.DEFINE_GLOBAL: "#ffe0b0", UOps.DEFINE_LOCAL: "#ffe0d0", UOps.DEFINE_ACC: "#f0ffe0",
            UOps.LOOP: "#c8a0e0", UOps.PHI: "#e0ffc0", UOps.BARRIER: "#ff8080", UOps.IF: "#c8b0c0"}
  G = nx.DiGraph()
  for u in uops:
    if u.uop == UOps.END: continue
    G.add_node(uops.index(u), label=f"{str(u.uop)[5:]}{(' '+str(u.arg)) if u.arg is not None else ''}\n{str(u.dtype)}", style="filled", fillcolor=colors.get(u.uop, "#ffffff"))  # noqa: E501
    for v in u.vin: G.add_edge(uops.index(v), uops.index(u))
  nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.uops.dot')
  os.system(f'dot -Grankdir=LR -Tsvg {GRAPHPATH}.uops.dot -o {GRAPHPATH}.uops.svg')
