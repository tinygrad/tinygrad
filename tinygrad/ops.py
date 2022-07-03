from __future__ import annotations
from ast import UnaryOp
from enum import Enum
from typing import Tuple, NamedTuple, Union, Any, List
import functools, operator
from tinygrad.helpers import ConvArgs
from tinygrad.shapetracker import ShapeTracker
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SLICE", "EXPAND", "FLIP"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU"])
Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]

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
  atexit.register(debug_exit)

if GRAPH:
  import networkx as nx
  G = nx.DiGraph()
  def save_graph_exit():
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, '/tmp/net.dot')
    # -Gnslimit=100 can make it finish, but you won't like results
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

    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}

    for x in inp:
      if not isinstance(op, list): op = [op]
      if len(op) <= 2: sop = '.'.join([str(y).split(".")[1] for y in op][::-1])
      elif len(op) <= 4: sop = '.'.join([str(y).split(".")[1][0:2] for y in op][::-1])
      else: sop = str(len(op))
      G.add_edge(nm(x), nm(ret), label=sop)
      if 'label' not in G.nodes[nm(x)]: G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))
    st = getattr(ret, "st", None)
    non_contiguous = st is not None and not st.contiguous
    if non_contiguous:
      G.nodes[nm(ret)]['label'] = str(tuple(x[0] if x[1]!=0 else 0 for x in st.views[-1].shape_strides))
    else:
      G.nodes[nm(ret)]['label'] = str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('80' if non_contiguous else '')) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if non_contiguous else 'filled'

# **** enumerate supported devices ****

import importlib, inspect
class Device:
  _ops = sorted(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "llops")))
  DEFAULT = None
  buffers = {}
  for i,op in enumerate([os.path.splitext(x)[0] for x in _ops if x.startswith("ops_")]):
    name = op[len("ops_"):].upper()
    vars()[name] = name 
    DEFAULT = name if os.environ.get(name, 0) == "1" else DEFAULT
    try:
      def find_buffer(llo, name): return [cls for cname, cls in inspect.getmembers(llo, inspect.isclass) if (cname.upper() == name + "BUFFER")][0]
      buffers[name] = find_buffer(importlib.import_module('tinygrad.llops.'+op), name)
    except ImportError as e:
      print(op, "not available", e)
  DEFAULT = CPU if DEFAULT is None else DEFAULT

# TODO: get device buffer types
DeviceBuffer = Any

def _realize(self:LazyBuffer) -> DeviceBuffer:
  if self.optype == LoadOps and self.op.op == LoadOps.FROMCPU:
    return Device.buffers[self.device].fromCPU(self.op.arg), []
  elif self.optype == ReduceOps:
    real_src = self.op.src[0].realize(self.device)
    return real_src.reduce_op(self.op.op, self.op.arg), [real_src]
  elif self.optype == MovementOps:
    real_src = self.op.src[0].realize(self.device)
    return real_src.movement_op(self.op.op, self.op.arg), [real_src]
  elif self.optype == UnaryOps:
    real_src_x = self.op.src[0].realize(self.device)
    return real_src_x.unary_op(self.op.op), [real_src_x]
  elif self.optype == BinaryOps:
    real_src_x = self.op.src[0].realize(self.device)
    real_src_y = self.op.src[1].realize(self.device)
    return real_src_x.binary_op(self.op.op, real_src_y), [real_src_x, real_src_y]
  elif self.optype == ProcessingOps:
    real_src_x = self.op.src[0].realize(self.device)
    real_src_w = self.op.src[1].realize(self.device)
    return real_src_x.processing_op(self.op.op, real_src_w, self.op.arg), [real_src_x, real_src_w]

# **** lazy operations ****

class LazyOp(NamedTuple):
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer]]
  arg: Any = None

def get_lazybuffers(op:LazyOp) -> List[LazyBuffer]: return functools.reduce(operator.add, [get_lazybuffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])

LAZY = int(os.getenv("LAZY", "0"))

class LazyBuffer:
  def __init__(self, device, shape:Union[ShapeTracker, Tuple[int]], optype:Op, op:LazyOp):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self.optype, self.op = optype, op
    self.realized = None
    self.device = device
    if not LAZY: self.realize()

  # this produces a device buffer
  def realize(self:LazyBuffer, required_device=None) -> DeviceBuffer:
    if required_device is not None: assert required_device == self.device
    if self.realized is None:
      # we haven't realized the Buffer yet
      self.realized, real_srcs = _realize(self)
      if DEBUG or GRAPH:
        # in lazy mode, we don't log until we realize
        log_op(self.optype, [x.op for x in get_lazyops(self.op)], self.realized, real_srcs)
      del self.op

    assert self.realized.shape == self.shape
    assert isinstance(self.realized, Device.buffers[self.device])
    return self.realized

  @staticmethod
  def fromCPU(x, device):
    # TODO: is there a better place to put this?
    if x.shape == tuple(): x = x.reshape((1,))
    return LazyBuffer(device, x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x))

class Ops:
  def unary_op(ctx, op:UnaryOps, x:LazyBuffer) -> LazyBuffer:
    return LazyBuffer(x.device, x.shape, UnaryOps, LazyOp(op, (x,)))

  def binary_op(ctx, op:BinaryOps, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    return LazyBuffer(x.device, x.shape, BinaryOps, LazyOp(op, (x,y)))

  def reduce_op(ctx, op:ReduceOps, x:LazyBuffer, new_shape:Tuple[int]) -> LazyBuffer:
    return LazyBuffer(x.device, tuple(new_shape), ReduceOps, LazyOp(op, (x,), tuple(new_shape)))

  def movement_op(ctx, op:MovementOps, x:LazyBuffer, arg) -> LazyBuffer:
    return LazyBuffer(x.device, ShapeTracker(x.st).movement_op(op, arg), MovementOps, LazyOp(op, (x,), arg))

  def processing_op(ctx, op:ProcessingOps, x:LazyBuffer, w:LazyBuffer, C:ConvArgs) -> LazyBuffer:
    return LazyBuffer(x.device, C.out_shape, ProcessingOps, LazyOp(op, (x, w), C))

