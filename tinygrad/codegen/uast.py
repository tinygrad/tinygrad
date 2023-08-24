from __future__ import annotations
import functools, math, itertools
from typing import NamedTuple, Optional, List, Any, Tuple, cast, Sequence, Union
from tinygrad.ops import ReduceOps, BinaryOps, LazyOp
from tinygrad.codegen.optimizer import OptimizedKernel
from tinygrad.lazy import LazyBuffer
from tinygrad.runtime.lib import RawConst
from tinygrad.helpers import dtypes, DEBUG, DType, all_same
from enum import Enum, auto
from tinygrad.shape.symbolic import Variable, NumNode, Node, MulNode, SumNode, DivNode, ModNode, LtNode, AndNode
VariableOrNum = Union[Variable, NumNode, Node]

class UOps(Enum):
  LOOP = auto(); ENDLOOP = auto() # loops can be global, local, or other # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  CONST = auto(); LOAD = auto(); STORE = auto(); BARRIER = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto() # noqa: E702

class UOp(NamedTuple):
  uop: UOps
  vin: Tuple[UOp]
  dtype: Optional[DType]
  arg: Any
  def __repr__(self): return str(self.uop).replace('UOps.', '') + "\n" + (f"{self.arg}\n" if self.arg else "") + str(self.dtype)

# TODO: generic visitor pattern?
def expand_node(idx:Node) -> List[Node]:
  if isinstance(idx, Variable): return [idx] if idx.expr is not None else [Variable.num(j) for j in range(idx.min, idx.max+1)]
  if isinstance(idx, NumNode): return [idx]
  if isinstance(idx, MulNode): return [x*idx.b for x in expand_node(idx.a)]
  if isinstance(idx, SumNode): return [Variable.sum(list(it)) for it in itertools.product(*[expand_node(x) for x in idx.nodes])]
  raise NotImplementedError(idx)

class UAst(OptimizedKernel):
  @functools.lru_cache(None)
  def uop(self, uop:UOps, vin:Tuple[UOp], dtype:Optional[DType]=None, arg:Any=None) -> UOp:
    print(f"{str(uop):20s}: {len(vin)} {str(dtype):20s} {arg}")
    return UOp(uop, vin, dtype, arg)

  def uop_alu_idx(self, a, b, ops, ctx:UAst, op, dtype=dtypes.int32):
    return self.uop(UOps.ALU, (a, (NumNode(b) if not isinstance(b, Node) else b).render(ops, ctx)), dtype, op)

  render_ops: Any = { Variable: lambda self, ops, ctx: ctx.uop(UOps.LOOP, tuple(), dtypes.int32, (self.expr,self.min,self.max)),
                NumNode: lambda self, ops, ctx: ctx.uop(UOps.CONST, tuple(), dtypes.int32, self.b),
                MulNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MUL),
                DivNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.DIV),
                ModNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MOD),
                LtNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.CMPLT, dtype=dtypes.bool),
    SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.ADD), self.nodes[1:], self.nodes[0].render(ops,ctx)),
    AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.MUL, dtype=dtypes.bool), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

  def global_load(self, i:int, idxs:Sequence[VariableOrNum], acc=None) -> List[UOp]:
    #const = self.bufs[i].realized._buf if isinstance(self.bufs[i].realized, RawConst) else acc
    expanded_nodes = [expand_node(idx) for idx in idxs]
    ret = []
    for _idx in [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]:
      idx, valid = self.sts[i].expr_idxs(_idx)
      idx_rendered = idx.render(self.render_ops, self)
      valid_rendered = valid.render(self.render_ops, self) if valid.min == 0 else None
      ret.append(self.uop(UOps.LOAD, (self.global_bufs[i], idx_rendered, valid_rendered)))
    return ret

  def linearize(self):
    self.process()
    if DEBUG >= 3: self.printbufs()
    global_bufs = [self.uop(UOps.DEFINE_GLOBAL, tuple(), buf.dtype, name) for buf,name in self.arg_bufs.items()]

    # define Variables
    global_idxs = [Variable(f"gidx{i}", 0, self.full_shape[i]-1) for i in range(0, self.first_reduce-self.local_dims)]
    local_idxs = [Variable(f"lidx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce-self.local_dims, self.first_reduce+len(self.group_for_reduce))]
    reduce_idxs = [Variable(f"ridx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce+len(self.group_for_reduce), self.shape_len-self.upcasted)]
    fake_reduce_idxs = [x*0 for x in reduce_idxs]
    full_upcast_idxs = [Variable(None, 0, s-1) for s in self.full_shape[self.shape_len-self.upcasted:]]
    upcast_idxs = [Variable(None, 0, s-1) for s in self.output_shape[self.shape_len-self.upcasted:]]

    def ast_parse(x:Union[LazyBuffer, LazyOp], idxs) -> UOp:
      if isinstance(x, LazyBuffer):
        buf_idx = self.bufs.index(x)
        idx, valid = self.sts[buf_idx].expr_idxs(idxs)
        idx_rendered = idx.render(self.render_ops, self)
        valid_rendered = valid.render(self.render_ops, self) if valid.min == 0 else None
        return self.uop(UOps.LOAD, (global_bufs[buf_idx], idx_rendered) + ((valid_rendered,) if valid_rendered is not None else tuple()), x.dtype)
      if x.op in ReduceOps:
        nidxs = global_idxs+local_idxs+reduce_idxs
        nidxs += [(i1 if i2==i3 else i2) for i1,i2,i3 in zip(idxs[len(nidxs):], full_upcast_idxs, upcast_idxs)]
        expanded_nodes = [expand_node(idx) for idx in nidxs]
        lreduce_idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
        vin = tuple(ast_parse(x.src[0], lidxs) for lidxs in lreduce_idxs)
      else:
        vin = tuple(ast_parse(v, idxs) for v in x.src)
      assert all_same([x.dtype for x in vin])
      return self.uop(UOps.ALU, vin, vin[0].dtype, x.op)

    sinks = []
    expanded_nodes = [expand_node(idx) for idx in (global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs)]
    store_idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    for idxs in store_idxs:
      idx, valid = self.sts[0].expr_idxs(idxs)
      assert valid.min == 1
      idx_rendered = idx.render(self.render_ops, self)
      sinks.append(self.uop(UOps.STORE, (global_bufs[0], idx_rendered, ast_parse(self.ast, idxs))))

    # graph debugging
    import networkx as nx
    G = nx.DiGraph()
    def add_node_recursive(x:UOp):
      if x in G.nodes: return
      G.add_node(id(x), label=str(x))
      for a in x.vin:
        add_node_recursive(a)
        G.add_edge(id(a), id(x))
    for s in sinks: add_node_recursive(s)
    import os
    from tinygrad.helpers import GRAPHPATH
    nx.drawing.nx_pydot.write_dot(G, f'{GRAPHPATH}.dot')
    os.system(f'dot -Grankdir=LR -Tsvg {GRAPHPATH}.dot -o {GRAPHPATH}.svg')
