from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple, NamedTuple, Union, Any, List, Dict, Type
from copy import copy
import os, sys, functools, operator
from tinygrad.helpers import ConvArgs
from tinygrad.shapetracker import ShapeTracker

# lazy can recurse a lot
sys.setrecursionlimit(10000)

# these are the llops your accelerator must implement
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SLICE", "EXPAND", "FLIP"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

# TODO: get device buffer types
DeviceBuffer = Any

DEBUG = int(os.getenv("DEBUG", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
OPT = int(os.getenv("OPT", "1"))

MERGE_MOVEMENT_OPS, REMOVE_MOVEMENT_NOPS, MERGE_UNARY_OPS = OPT>=1, OPT>=1, OPT>=1
MERGE_ELEMENTWISE_OPS, SHUFFLE_MOVEMENT_OPS = OPT>=2, OPT>=2
SHUFFLE_SLICE_OPS = OPT>=3  # NOTE: 0/0 is NaN if you slice, so this can change the output

from collections import defaultdict
cnts : Dict[OpType, int] = defaultdict(int)

import atexit
if GRAPH:
  import networkx as nx  # type: ignore
  G = nx.DiGraph()
  def save_graph_exit():
    for k,v in cnts.items(): print(k, v)
    print("saving", G)
    nx.drawing.nx_pydot.write_dot(G, '/tmp/net.dot')
    # -Gnslimit=100 can make it finish, but you won't like results
    os.system('dot -Tsvg /tmp/net.dot -o /tmp/net.svg')
  atexit.register(save_graph_exit)

global_num_max = 0
def log_op(optype : OpType, op : List[Op], ret : DeviceBuffer, inp : List[DeviceBuffer]):
  cnts[optype] += 1
  if DEBUG >= 1: print(f"{op} : {', '.join([str(x.shape) for x in inp])} -> {ret.shape}")
  if GRAPH:
    def nm(x):
      global global_num_max
      if getattr(x, 'global_num', None) is None:
        setattr(x, 'global_num', global_num_max)
        global_num_max += 1
      return f"<<< {x.global_num} >>>"

    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}

    for x in inp:
      if len(op) <= 2: sop = '.'.join([str(y).split(".")[1] for y in op][::-1])
      elif len(op) <= 4: sop = '.'.join([str(y).split(".")[1][0:2] for y in op][::-1])
      else: sop = str(len(op))
      G.add_edge(nm(x), nm(ret), label=sop)
      if 'label' not in G.nodes[nm(x)]: G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))
    st = getattr(ret, "st", None)
    non_contiguous = st is not None and not st.contiguous
    if non_contiguous and st is not None:   # checked twice to make type checker happy
      G.nodes[nm(ret)]['label'] = str(tuple(x[0] if x[1]!=0 else 0 for x in st.views[-1].shape_strides))
    else:
      G.nodes[nm(ret)]['label'] = str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('80' if non_contiguous else '')) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if non_contiguous else 'filled'

# **** enumerate supported devices ****

import importlib, inspect
def find_buffer(llo, name):
  return [cls for cname, cls in inspect.getmembers(llo, inspect.isclass) if (cname.upper() == name + "BUFFER")][0]
class Device:
  _ops = sorted(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "llops")))
  DEFAULT = None
  _buffers = {}
  for i,op in enumerate([os.path.splitext(x)[0] for x in _ops if x.startswith("ops_")]):
    name = op[len("ops_"):].upper()
    vars()[name] = name 
    DEFAULT = name if os.environ.get(name, 0) == "1" else DEFAULT
    try: _buffers[name] = find_buffer(importlib.import_module('tinygrad.llops.'+op), name)
    except ImportError as e:
      print(op, "not available", e)
  DEFAULT = "CPU" if DEFAULT is None else DEFAULT

# TODO: make a _realize function for each type called by realize
def _realize(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer]]:
  if self.optype == LoadOps and self.op.op == LoadOps.FROMCPU:
    return Device._buffers[self.device].fromCPU(self.op.arg), []
  elif self.optype == ReduceOps:
    real_src = self.op.src[0].realize(self.device)
    return real_src.reduce_op(self.op.op, self.op.arg), [real_src]
  elif self.optype == MovementOps:
    real_src = get_lazybuffers(self.op)[0].realize(self.device)
    if getattr(real_src, "shapeTrackerView", None) is not None:
      return real_src.shapeTrackerView(self.st), [real_src]
    else:
      return functools.reduce(lambda x,o: x.movement_op(o.op, o.arg), get_lazyops(self.op)[::-1], real_src), [real_src]
  elif self.optype == BinaryOps:
    real_srcs : Dict[LazyBuffer, DeviceBuffer] = {}
    [real_srcs.setdefault(x,x.realize(self.device)) for x in get_lazybuffers(self.op) if x not in real_srcs]
    def ast_eval(x: Union[LazyBuffer, LazyOp]) -> DeviceBuffer:
      if isinstance(x, LazyBuffer): return real_srcs[x]
      if isinstance(x.op, UnaryOps): return ast_eval(x.src[0]).unary_op(x.op)
      if isinstance(x.op, BinaryOps): return ast_eval(x.src[0]).binary_op(x.op, ast_eval(x.src[1]))
    return ast_eval(self.op), list(real_srcs.values())
  elif self.optype == ProcessingOps:
    real_src_x = self.op.src[0].realize(self.device)
    real_src_w = self.op.src[1].realize(self.device)
    return real_src_x.processing_op(self.op.op, real_src_w, self.op.arg), [real_src_x, real_src_w]
  else: raise NotImplementedError(f"can't handle optype {self.optype}")

# **** lazy operations ****

class LazyOp(NamedTuple):
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

def get_lazybuffers(op:LazyOp) -> List[LazyBuffer]: return functools.reduce(operator.add, [get_lazybuffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])

LAZY = int(os.getenv("LAZY", "0"))

class LazyBuffer:
  def __init__(self, device, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self.optype, self.op = optype, op
    self.realized : Optional[DeviceBuffer] = None
    self.device = device
    if not LAZY: self.realize()

  # this produces a device buffer
  def realize(self:LazyBuffer, required_device=None) -> DeviceBuffer:
    if required_device is not None: assert required_device == self.device
    if self.realized is None:
      # we haven't realized the Buffer yet
      self.realized, real_srcs = _realize(self)
      # in lazy mode, we don't log until we realize
      log_op(self.optype, [x.op for x in get_lazyops(self.op)], self.realized, real_srcs)
      # no need to keep the op after realization
      del self.op

    assert self.realized.shape == self.shape
    assert isinstance(self.realized, Device._buffers[self.device])
    return self.realized

  @staticmethod
  def fromCPU(x, device):
    return LazyBuffer(device, x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x.copy()))
  
  def toCPU(x):
    return x.realize().toCPU()

  def unary_op(x:LazyBuffer, op:UnaryOps) -> LazyBuffer: return elementwise_op(op, (x,))
  def binary_op(x:LazyBuffer, op:BinaryOps, y:LazyBuffer) -> LazyBuffer: return elementwise_op(op, (x,y))

  def reduce_op(x:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    return LazyBuffer(x.device, tuple(new_shape), ReduceOps, LazyOp(op, (x,), tuple(new_shape)))

  def movement_op(x:LazyBuffer, op:MovementOps, arg) -> LazyBuffer:
    # TODO: look into why that copy is needed
    arg = copy(arg)

    # TODO: SHUFFLE_SLICE_OPS is okay if it's a shrink
    if SHUFFLE_MOVEMENT_OPS and x.optype == BinaryOps and x.realized is None and (SHUFFLE_SLICE_OPS or op != MovementOps.SLICE):
      # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
      def replace_with_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
        if isinstance(y, LazyBuffer): return y.movement_op(op, arg)
        assert isinstance(y.op, BinaryOps) or isinstance(y.op, UnaryOps)
        return elementwise_op(y.op, tuple(replace_with_movement_op(z) for z in y.src))
      return replace_with_movement_op(x.op)

    # if a MovementOp is applied to a MovementOp, merge them and use one buffer
    ret = LazyBuffer(x.device, ShapeTracker(x.st).movement_op(op, arg), MovementOps,
            LazyOp(op, (x.op if MERGE_MOVEMENT_OPS and x.optype == MovementOps and x.realized is None else x,), arg))

    if REMOVE_MOVEMENT_NOPS and x.realized is None and ret.st.contiguous:
      root = get_lazybuffers(ret.op)[0]
      if ret.st.shape == root.shape:
        return root

    return ret

  def processing_op(x:LazyBuffer, op:ProcessingOps, w:LazyBuffer, C:ConvArgs) -> LazyBuffer:
    return LazyBuffer(x.device, C.out_shape, ProcessingOps, LazyOp(op, (x, w), C))

def elementwise_op(op:Union[UnaryOps, BinaryOps], srcs:Tuple[LazyBuffer, ...]) -> LazyBuffer:
  out_device, out_shape = srcs[0].device, srcs[0].shape

  if (MERGE_UNARY_OPS and len(srcs) == 1) or MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any BinaryOps that feed into this
    srcs = tuple(x.op if x.optype == BinaryOps and x.realized is None else x for x in srcs)  # type: ignore

  return LazyBuffer(out_device, out_shape, BinaryOps, LazyOp(op, srcs))
