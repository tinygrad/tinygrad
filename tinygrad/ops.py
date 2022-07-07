from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple, NamedTuple, Union, Any, List, Dict, Type
from copy import copy
import os, sys, functools, operator, weakref
from tinygrad.helpers import ConvArgs, get_available_llops
from tinygrad.shapetracker import ShapeTracker

# lazy can recurse a lot
sys.setrecursionlimit(10000)

# these are the llops your accelerator must implement, along with toCpu
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SLICE", "EXPAND", "FLIP"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

DEBUG = int(os.getenv("DEBUG", "0"))
GRAPH = int(os.getenv("GRAPH", "0"))
OPT = int(os.getenv("OPT", "1"))

# TODO: why does REMOVE_MOVEMENT_NOPS break things if MERGE_MOVEMENT_OPS isn't used?
MERGE_MOVEMENT_OPS, REMOVE_MOVEMENT_NOPS, MERGE_UNARY_OPS = OPT>=1, OPT>=1, OPT>=1
MERGE_ELEMENTWISE_OPS = OPT>=2
SHUFFLE_MOVEMENT_OPS, MERGE_ELEMENTWISE_INTO_CONV_OUTPUTS = OPT>=3, OPT>=3
SHUFFLE_SLICE_OPS = OPT>=4  # NOTE: 0/0 is NaN if you slice, so this can change the output

# this is doing something, but missing some
# TODO: see the test for why, double reshape isn't cached
def lazycache(func):
  cache = weakref.WeakValueDictionary()
  def wrapper(*args):
    weakargs = tuple(weakref.ref(x) if isinstance(x, LazyBuffer) else x for x in args)
    # NOTE: even though we don't use ref, we need to keep it around or it will be deleted
    if weakargs not in cache: cache[weakargs] = ret = func(*args)
    return cache[weakargs]
  return wrapper

# **** enumerate supported devices ****

class Device:
  _buffers, DEFAULT = get_available_llops()
  for name in _buffers.keys():
    vars()[name] = name

# TODO: get device buffer types
DeviceBuffer = Any

# **** debugging and graphing ****

import atexit
from collections import defaultdict
cnts : Dict[OpType, int] = defaultdict(int)
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
  if DEBUG >= 3: print(f"{op} : {', '.join([str(x.shape) for x in inp])} -> {ret.shape}")
  if GRAPH:
    def nm(x):
      global global_num_max
      if getattr(x, 'global_num', None) is None:
        setattr(x, 'global_num', global_num_max)
        global_num_max += 1
      return f"<<< {x.global_num} >>>"

    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}

    dashed = optype == LoadOps and getattr(ret, "_backing", None) is not None

    for x in inp:
      if len(op) <= 2: sop = '.'.join([str(y).split(".")[1] for y in op][::-1])
      elif len(op) <= 4: sop = '.'.join([str(y).split(".")[1][0:2] for y in op][::-1])
      else: sop = str(len(op))
      G.add_edge(nm(x), nm(ret), label=sop)
      if 'label' not in G.nodes[nm(x)]: G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes: G.add_node(nm(ret))
    if getattr(ret, "st", None) is not None and not ret.st.contiguous:   # checked twice to make type checker happy
      G.nodes[nm(ret)]['label'] = str(tuple(x[0] if x[1]!=0 else 0 for x in ret.st.views[-1].shape_strides))
      dashed = True
    else:
      G.nodes[nm(ret)]['label'] = str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('80' if dashed else '')) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if dashed else 'filled'

# **** realize functions ****

def _realize_loadops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  assert self.op.op == LoadOps.FROMCPU
  return Device._buffers[self.device].fromCPU(self.op.arg), [], LoadOps

def _realize_reduceops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src = self.op.src[0].realize(self.device)
  return real_src.reduce_op(self.op.op, self.op.arg), [real_src], ReduceOps

def _realize_movementops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src = get_lazybuffers(self.op)[0].realize(self.device)
  if getattr(real_src, "shapeTrackerView", None) is not None:
    return real_src.shapeTrackerView(self.st), [real_src], MovementOps
  else:
    # slow path, creates middle buffers
    return functools.reduce(lambda x,o: x.movement_op(o.op, o.arg), get_lazyops(self.op)[::-1], real_src), [real_src], MovementOps

def _realize_binaryops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_srcs : Dict[LazyBuffer, DeviceBuffer] = {x:None for x in get_lazybuffers(self.op)}
  if getattr(Device._buffers[self.device], "_processing_op", None) is not None:
    buf_names : Dict[LazyBuffer, str] = {x:f"arg_{i}" for i,x in enumerate(real_srcs.keys())}

    # if there's *one* processing op in here, we can corealize it. we can corealize binary op sibilings as well
    # TODO: we have to confirm the conv doesn't have other children
    # NOTE: if it references the same conv multiple times, they should already be merged

    conv_args : Optional[ConvArgs] = None
    psrcs = [x for x in real_srcs.keys() if x.optype == ProcessingOps and x.realized is None and len(x.children) <= 1]
    if len(psrcs) == 1 and MERGE_ELEMENTWISE_INTO_CONV_OUTPUTS:
      conv_args = psrcs[0].op.arg
      del real_srcs[psrcs[0]]
      real_srcs[psrcs[0].op.src[0]], real_srcs[psrcs[0].op.src[1]] = None, None
      buf_names[psrcs[0].op.src[0]], buf_names[psrcs[0].op.src[1]] = "input", "weight"   # NOTE: these will not be in the ast
      buf_names[psrcs[0]] = "acc"

    for x in real_srcs.keys(): real_srcs[x] = x.realize(self.device)
    # fast path, no middle buffers
    def ast_op(op, srcs_code: List[str]) -> str:
      code = Device._buffers[self.device].code_for_op[op]
      if len(srcs_code) >= 1: code = code.replace("A", srcs_code[0])
      if len(srcs_code) >= 2: code = code.replace("B", srcs_code[1])
      return code

    def _ast(x: Union[LazyBuffer, LazyOp]) -> str:
      if isinstance(x, LazyBuffer): return buf_names[x]
      return ast_op(x.op, [_ast(src) for src in x.src])

    return Device._buffers[self.device](self.shape)._processing_op([(buf_names[lb], db) for lb,db in real_srcs.items()], _ast(self.op), C=conv_args), \
      list(real_srcs.values()), ProcessingOps if conv_args is not None else BinaryOps
  else:
    for x in real_srcs.keys(): real_srcs[x] = x.realize(self.device)
    # slow path, creates middle buffers
    def ast_eval(x: Union[LazyBuffer, LazyOp]) -> DeviceBuffer:
      if isinstance(x, LazyBuffer): return real_srcs[x]
      if isinstance(x.op, UnaryOps): return ast_eval(x.src[0]).unary_op(x.op)
      if isinstance(x.op, BinaryOps): return ast_eval(x.src[0]).binary_op(x.op, ast_eval(x.src[1]))
    return ast_eval(self.op), list(real_srcs.values()), BinaryOps

def _realize_processingops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src_x = self.op.src[0].realize(self.device)
  real_src_w = self.op.src[1].realize(self.device)
  return real_src_x.processing_op(self.op.op, real_src_w, self.op.arg), [real_src_x, real_src_w], ProcessingOps

_realize = {LoadOps:_realize_loadops, ReduceOps:_realize_reduceops, MovementOps:_realize_movementops, BinaryOps:_realize_binaryops, ProcessingOps:_realize_processingops}

# **** lazy operations ****

class LazyOp(NamedTuple):
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

def get_lazybuffers(op:LazyOp) -> List[LazyBuffer]: return functools.reduce(operator.add, [get_lazybuffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])

LAZY = int(os.getenv("LAZY", "1"))

class LazyBuffer:
  def __init__(self, device, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self.optype, self.op = optype, op
    self.realized : Optional[DeviceBuffer] = None
    self.device = device
    self.children = weakref.WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    for x in get_lazybuffers(op): x.children.add(self)
    if not LAZY: self.realize()

  def __repr__(self): return f"<LB {self.shape} op:{self.op.op if self.realized is None else 'realized'}>"

  # this produces a device buffer
  def realize(self:LazyBuffer, required_device=None) -> DeviceBuffer:
    if required_device is not None: assert required_device == self.device
    if self.realized is None:
      # we haven't realized the Buffer yet
      self.realized, real_srcs, real_type = _realize[self.optype](self)
      # in lazy mode, we don't log until we realize
      log_op(real_type, [x.op for x in get_lazyops(self.op)], self.realized, real_srcs)
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

  def unary_op(x:LazyBuffer, op:UnaryOps) -> LazyBuffer: return elementwise_op(op, x)
  def binary_op(x:LazyBuffer, op:BinaryOps, y:LazyBuffer) -> LazyBuffer: return elementwise_op(op, x, y)

  @lazycache
  def reduce_op(x:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    return LazyBuffer(x.device, tuple(new_shape), ReduceOps, LazyOp(op, (x,), tuple(new_shape)))

  @lazycache
  def movement_op(x:LazyBuffer, op:MovementOps, arg) -> LazyBuffer:
    # TODO: look into why that copy is needed
    arg = copy(arg)

    # TODO: SHUFFLE_SLICE_OPS is okay if it's a shrink
    if SHUFFLE_MOVEMENT_OPS and x.optype == BinaryOps and x.realized is None and (SHUFFLE_SLICE_OPS or op != MovementOps.SLICE):
      # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
      def replace_with_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
        if isinstance(y, LazyBuffer): return y.movement_op(op, arg)
        assert isinstance(y.op, BinaryOps) or isinstance(y.op, UnaryOps)
        return elementwise_op(y.op, *[replace_with_movement_op(z) for z in y.src])
      return replace_with_movement_op(x.op)

    # if a MovementOp is applied to a MovementOp, merge them and use one buffer
    ret = LazyBuffer(x.device, ShapeTracker(x.st).movement_op(op, arg), MovementOps,
            LazyOp(op, (x.op if MERGE_MOVEMENT_OPS and x.optype == MovementOps and x.realized is None else x,), arg))

    if REMOVE_MOVEMENT_NOPS and x.realized is None and ret.st.contiguous:
      root = get_lazybuffers(ret.op)[0]
      if ret.st.shape == root.shape:
        return root

    return ret

  @lazycache
  def processing_op(x:LazyBuffer, op:ProcessingOps, w:LazyBuffer, C:ConvArgs) -> LazyBuffer:
    return LazyBuffer(x.device, C.out_shape, ProcessingOps, LazyOp(op, (x, w), C))

@lazycache
def elementwise_op(op:Union[UnaryOps, BinaryOps], *srcs:LazyBuffer) -> LazyBuffer:
  out_device, out_shape = srcs[0].device, srcs[0].shape

  if (MERGE_UNARY_OPS and len(srcs) == 1) or MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any BinaryOps that feed into this
    srcs = tuple(x.op if x.optype == BinaryOps and x.realized is None else x for x in srcs)  # type: ignore

  return LazyBuffer(out_device, out_shape, BinaryOps, LazyOp(op, srcs))
