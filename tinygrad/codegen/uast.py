from __future__ import annotations
import functools, itertools, heapq
from collections import defaultdict
from typing import NamedTuple, List, Any, Tuple, Union, Dict, cast
from tinygrad.ops import ReduceOps, BinaryOps, UnaryOps, LazyOp, TernaryOps
from tinygrad.codegen.optimizer import OptimizedKernel
from tinygrad.lazy import LazyBuffer
from tinygrad.runtime.lib import RawConst
from tinygrad.helpers import dtypes, DEBUG, DType, getenv, colored, PtrDType
from enum import Enum, auto
from tinygrad.shape.symbolic import Variable, NumNode, Node, MulNode, SumNode, DivNode, ModNode, LtNode, AndNode, sym_rename
VariableOrNum = Union[Variable, NumNode, Node]

class UOps(Enum):
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  SPECIAL = auto(); ENDLOOP = auto() # loops can be global, local, or other # noqa: E702
  CONST = auto(); LOAD = auto(); STORE = auto(); BARRIER = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto() # noqa: E702
  LOOP = auto()  # loop is last

class UOp(NamedTuple):
  uop: UOps
  dtype: DType
  vin: Tuple[UOp, ...]
  arg: Any
  def __repr__(self): return f"{self.uop} {self.dtype} {self.arg}"
  def __lt__(self, x):
    if self.uop == UOps.LOOP and x.uop == UOps.LOOP: return self.arg < x.arg
    return self.uop.value < x.uop.value
  # NOTE: if you don't cache the hash, it's really slow
  hash: int
  def __hash__(self): return self.hash

class UAst(OptimizedKernel):
  _uop_cache: Dict[Any, UOp] = {}
  def uop(self, uop:UOps, dtype:DType, vin:Tuple[UOp, ...], arg:Any=None) -> UOp:
    key = (uop, dtype, vin, arg)
    if key not in self._uop_cache:
      self._uop_cache[key] = UOp(uop, dtype, vin, arg, hash(key))
    return self._uop_cache[key]

  def uop_alu_idx(self, a:UOp, b, ops, ctx:UAst, op, dtype=dtypes.int32):
    render_b:UOp = cast(UOp, (NumNode(b) if not isinstance(b, Node) else b).render(ops, ctx))
    return self.uop(UOps.ALU, dtype, (a, render_b), op)

  def var_to_loop(self, var):
    if self.opts.has_local and (var.expr.startswith("gidx") or var.expr.startswith("lidx")):
      assert var.min == 0
      global_dims_count = self.first_reduce-self.local_dims
      if var.expr.startswith("gidx"): spec = ("global", (global_dims_count-1)-int(var.expr[4:]), var.max+1)
      elif var.expr.startswith("lidx"): spec = ("local", (self.local_dims-1)-(int(var.expr[4:])-global_dims_count), var.max+1)
      return self.uop(UOps.SPECIAL, dtypes.int32, tuple(), spec)
    return self.uop(UOps.LOOP, dtypes.int32, tuple(), (var.expr,var.min,var.max+1))

  render_ops: Any = { Variable: lambda self, ops, ctx: ctx.var_to_loop(self),
                NumNode: lambda self, ops, ctx: ctx.uop(UOps.CONST, dtypes.int32, tuple(), self.b),
                MulNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MUL),
                DivNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.DIV),
                ModNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MOD),
                LtNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.CMPLT, dtype=dtypes.bool),
    SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.ADD), self.nodes[1:], self.nodes[0].render(ops,ctx)),
    AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.MUL, dtype=dtypes.bool), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

  def get_uast(self) -> List[UOp]:
    global_bufs = [self.uop(UOps.DEFINE_GLOBAL, PtrDType(buf.dtype), tuple(), i) for i,buf in enumerate(self.arg_bufs.keys())]

    # define Variables
    global_idxs = [Variable(f"gidx{i}", 0, self.full_shape[i]-1) for i in range(0, self.first_reduce-self.local_dims)]
    local_idxs = [Variable(f"lidx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce-self.local_dims, self.first_reduce+len(self.group_for_reduce))]
    reduce_idxs = [Variable(f"ridx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce+len(self.group_for_reduce), self.shape_len-self.upcasted)]
    fake_reduce_idxs = [x*0 for x in reduce_idxs]
    full_upcast_idxs = [Variable(None, 0, s-1) for s in self.full_shape[self.shape_len-self.upcasted:]]
    upcast_idxs = [Variable(None, 0, s-1) for s in self.output_shape[self.shape_len-self.upcasted:]]

    acc_count = 0
    def ast_parse(x:Union[LazyBuffer, LazyOp], idxs) -> UOp:
      nonlocal acc_count
      if isinstance(x, LazyBuffer):
        buf_idx = self.bufs.index(x)
        idx, valid = self.sts[buf_idx].expr_idxs(idxs)
        idx_rendered = idx.render(self.render_ops, self)
        valid_rendered = valid.render(self.render_ops, self) if valid.min == 0 else None
        if isinstance(x.realized, RawConst):
          ret = self.uop(UOps.CONST, x.dtype, (), x.realized._buf)
        else:
          # TODO: gate the load
          ret = self.uop(UOps.LOAD, x.dtype, (global_bufs[self.arg_bufs_num[x.realized]], idx_rendered) + ((valid_rendered,) if valid_rendered is not None else tuple()))
        if valid_rendered is not None: ret = self.uop(UOps.ALU, x.dtype, (valid_rendered, ret, self.uop(UOps.CONST, x.dtype, (), 0)), TernaryOps.WHERE)
        return ret
      if x.op in ReduceOps:
        nidxs = global_idxs+local_idxs+reduce_idxs
        nidxs += [(i1 if i2==i3 else i2) for i1,i2,i3 in zip(idxs[len(nidxs):], full_upcast_idxs, upcast_idxs)]
        expanded_nodes = [idx.expand() for idx in nidxs]
        lreduce_idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
        vin = tuple(ast_parse(x.src[0], lidxs) for lidxs in lreduce_idxs)
        if len(reduce_idxs):
          acc = self.uop(UOps.DEFINE_ACC, dtypes.float32, (), acc_count)
          acc_count += 1
          first = functools.reduce(lambda a,b: self.uop(UOps.ALU, dtypes.int32, (a,b), BinaryOps.ADD), [self.var_to_loop(ri) for ri in reduce_idxs])
          phi = self.uop(UOps.ALU, dtypes.float32, (first, acc, self.uop(UOps.CONST, dtypes.float32, tuple(), float('-inf') if x.op == ReduceOps.MAX else 0)),TernaryOps.WHERE)
          vin += (phi, )
        # NOTE: this determines the order of these when it doesn't have to
        ret = functools.reduce(lambda a,b: self.uop(UOps.ALU, dtypes.float32, (a,b), BinaryOps.MAX if x.op == ReduceOps.MAX else BinaryOps.ADD), vin)
        if len(reduce_idxs):
          ret = self.uop(UOps.STORE, dtypes.float32, (acc, ret))
          for ri in reduce_idxs[::-1]:
            ret = self.uop(UOps.ENDLOOP, dtypes.float32, (ret, self.var_to_loop(ri)))
        return ret
      vin = tuple(ast_parse(v, idxs) for v in x.src)
      # TODO: reenable this at some point
      #assert all_same([x.dtype for x in vin])
      if x.op == UnaryOps.NOOP: return vin[0]
      if x.op == UnaryOps.CAST: return self.uop(UOps.CAST, x.arg[0], vin, x.arg[1])
      return self.uop(UOps.ALU, vin[0].dtype, vin, x.op)

    sinks = []
    expanded_nodes = [idx.expand() for idx in (global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs)]
    store_idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    for idxs in store_idxs:
      idx, valid = self.sts[0].expr_idxs(idxs)
      assert valid.min == 1
      idx_rendered = idx.render(self.render_ops, self)
      sinks.append(self.uop(UOps.STORE, self.bufs[0].dtype, (global_bufs[0], ast_parse(self.ast, idxs), idx_rendered)))

    # graph debugging
    if getenv("UASTGRAPH"):
      import networkx as nx   # type: ignore
      G = nx.DiGraph()
      def add_node_recursive(x:UOp):
        if x in G.nodes: return
        G.add_node(id(x), label=str(x.uop).replace('UOps.', '') + "\n" + (f"{x.arg}\n" if x.arg is not None else "") + str(x.dtype))
        for a in x.vin:
          add_node_recursive(a)
          G.add_edge(id(a), id(x))
      for s in sinks: add_node_recursive(s)
      import os
      from tinygrad.helpers import GRAPHPATH
      nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.dot')
      os.system(f'dot -Grankdir=LR -Tsvg {GRAPHPATH}.dot -o {GRAPHPATH}.svg')

    return sinks

  def linearize(self):
    self.process()
    if DEBUG >= 3: self.printbufs()

    # kernel name (before late upcast)
    self.function_name = ("r_" if self.reduceop else "E_") + '_'.join([str(x) if isinstance(x, int) else sym_rename(x) for x in self.full_shape])
    self.display_name = ("r_" if self.reduceop else "E_") + colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

    # get sink uops
    uops = self.get_uast()

    # first, we fetch all the uops
    seen = set()
    srcs = []
    children = defaultdict(list)
    def visit(x):
      if x in seen: return
      if len(x.vin) == 0: srcs.append(x)
      seen.add(x)
      for u in x.vin:
        children[u].append(x)
        visit(u)
    for u in uops: visit(u)

    # then we figure out which are renderable and put them in order, leaving loops for last
    heapq.heapify(srcs)
    rendered = set()
    order = []
    while len(srcs):
      u = heapq.heappop(srcs)
      if u not in rendered:
        rendered.add(u)
        order.append(u)
        #if DEBUG >= 4: print(u)
        for c in children[u]:
          if all(x in rendered for x in c.vin):
            heapq.heappush(srcs, c)

    self.uops = order  # TODO: should this be return?
