from enum import Enum
from tinygrad.helpers import prod
UnaryOps = Enum("UnaryOps", ["NOOP", "RELU", "EXP", "LOG", "NEG", "SIGN"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SLICE", "EXPAND", "FLIP"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])

import os
DEBUG = int(os.getenv("DEBUG", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
from collections import defaultdict
cnts = defaultdict(int)

import atexit
if DEBUG:
  def debug_exit():
    for k,v in cnts.items():
      print(k, v)
    print(f"GFLOP: {Ops.flops*1e-9:.2f} MEMBW {Ops.mem*1e-9:.2f} GB")
  atexit.register(debug_exit)

if GRAPH:
  import networkx as nx
  G = nx.DiGraph()
  def save_graph_exit():
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, '/tmp/net.dot')
    os.system('dot -Tsvg /tmp/net.dot -o /tmp/net.svg')
  atexit.register(save_graph_exit)

global_num_max = 0
def log_op(optype, op, ret, inp):
  cnts[optype] += 1
  if DEBUG >= 2: print(f"{op} : {', '.join([str(x.shape) for x in inp])} -> {ret.shape}")
  if GRAPH:
    def nm(x):
      global global_num_max
      if getattr(x, 'global_num', None) is None:
        setattr(x, 'global_num', global_num_max)
        global_num_max += 1
      return f"<<< {x.global_num} >>>"

    top_colors = {UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}

    for x in inp:
      if not isinstance(op, list): op = [op]
      if GRAPH == 2: sop = '.'.join([str(y).split(".")[1] for y in op][::-1])
      else: sop = str(len(op))
      G.add_edge(nm(x), nm(ret), label=sop)
      if 'label' not in G.nodes[nm(x)]: G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))
    st = getattr(ret, "st", None)
    non_contiguous = st is not None and not st.contiguous
    G.nodes[nm(ret)]['label'] = str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('80' if non_contiguous else '')) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if non_contiguous else 'filled'

class Ops:
  flops = 0
  mem = 0

  def unary_op(ctx, op:UnaryOps, x):
    ret = x.unary_op(op)
    if 'LAZY' not in ctx.device: log_op(UnaryOps, op, ret, [x])
    Ops.flops += prod(x.shape)
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == x.shape
    return ret

  def reduce_op(ctx, op:ReduceOps, x, new_shape):
    ret = x.reduce_op(op, tuple(new_shape))
    if 'LAZY' not in ctx.device: log_op(ReduceOps, op, ret, [x])
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == tuple(new_shape)
    return ret

  def binary_op(ctx, op:BinaryOps, x, y):
    assert x.shape == y.shape
    ret = x.binary_op(op, y)
    if 'LAZY' not in ctx.device: log_op(BinaryOps, op, ret, [x, y])
    Ops.flops += prod(x.shape)*2
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == x.shape
    return ret

  def movement_op(ctx, op:MovementOps, x, arg):
    ret = x.movement_op(op, tuple(arg))
    if 'LAZY' not in ctx.device: log_op(MovementOps, op, ret, [x])
    assert isinstance(ret, ctx.buffer)
    # this check is slow
    #assert ret.shape == ShapeTracker(x.shape).movement_op(op, arg).shape
    return ret

  def processing_op(ctx, op:ProcessingOps, x, y, C):
    ret = x.processing_op(op, y, C)
    if 'LAZY' not in ctx.device: log_op(ProcessingOps, op, ret, [x, y])
    Ops.flops += C.bs*C.cout*C.oy*C.ox*C.cin*C.H*C.W*2
    Ops.mem += C.bs*C.cout*C.oy*C.ox + C.cout*C.cin*C.H*C.W + C.bs*C.cin*C.iy*C.ix
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == C.out_shape
    return ret