# TODO: move Device to here and proxy buffer call
from enum import Enum
UnaryOps = Enum("UnaryOps", ["NOOP", "RELU", "EXP", "LOG", "NEG", "SIGN"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SLICE", "EXPAND", "FLIP"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU"])

from tinygrad.shapetracker import ShapeTracker

import os
DEBUG = int(os.getenv("PRINT_LLOPS", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
from collections import defaultdict
cnts = defaultdict(int)
if GRAPH:
  import atexit
  import networkx as nx
  G = nx.DiGraph()
  def save_graph_exit():
    for k,v in cnts.items():
      print(k, v)
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, '/tmp/net.dot')
    os.system('dot -Tsvg /tmp/net.dot -o /tmp/net.svg')
  atexit.register(save_graph_exit)

global_num_max = 0
def log_op(optype, op, ret, inp, dashed=False):
  cnts[optype] += 1
  if DEBUG: print(f"{op} : {', '.join([str(x.shape) for x in inp])} -> {ret.shape}")
  if GRAPH:
    def nm(x):
      global global_num_max
      if getattr(x, 'global_num', None) is None:
        setattr(x, 'global_num', global_num_max)
        global_num_max += 1
      return f"<<< {x.global_num} >>>"

    top_colors = {LoadOps: "#ffffff", UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}

    for x in inp:
      if not isinstance(op, list): op = [op]
      sop = '.'.join([str(y).split(".")[1] for y in op][::-1])
      G.add_edge(nm(x), nm(ret), label=sop, color='#808080' if dashed else '', style='dashed' if dashed else '')
      if 'label' not in G.nodes[nm(x)]: G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))
    st = getattr(ret, "st", None)
    non_contiguous = st is not None and not st.contiguous
    G.nodes[nm(ret)]['label'] = str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = top_colors[optype] + ('80' if non_contiguous else '')
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if non_contiguous else 'filled'

class Ops:
  def unary_op(ctx, op:UnaryOps, x):
    ret = ctx.op.unary_op(op, x)
    if 'LAZY' not in ctx.device: log_op(UnaryOps, op, ret, [x])
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == x.shape
    return ret

  def reduce_op(ctx, op:ReduceOps, x, new_shape):
    ret = ctx.op.reduce_op(op, x, tuple(new_shape))
    if 'LAZY' not in ctx.device: log_op(ReduceOps, op, ret, [x])
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == tuple(new_shape)
    return ret

  def binary_op(ctx, op:BinaryOps, x, y):
    assert x.shape == y.shape
    ret = ctx.op.binary_op(op, x, y)
    if 'LAZY' not in ctx.device: log_op(BinaryOps, op, ret, [x, y])
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == x.shape
    return ret

  def movement_op(ctx, op:MovementOps, x, arg):
    ret = ctx.op.movement_op(op, x, tuple(arg))
    if 'LAZY' not in ctx.device: log_op(MovementOps, op, ret, [x])
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == ShapeTracker(x.shape).movement_op(op, arg).shape
    return ret

  def processing_op(ctx, op:ProcessingOps, x, y, C):
    ret = ctx.op.processing_op(op, x, y, C)
    if 'LAZY' not in ctx.device: log_op(ProcessingOps, op, ret, [x, y])
    assert isinstance(ret, ctx.buffer)
    assert ret.shape == C.out_shape
    return ret