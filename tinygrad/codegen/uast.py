from __future__ import annotations
import functools, math, itertools
from collections import defaultdict
from typing import NamedTuple, Optional, List, Any, Tuple, cast, Sequence, Union, Dict, Set
from tinygrad.ops import ReduceOps, BinaryOps, UnaryOps, LazyOp, TernaryOps
from tinygrad.codegen.optimizer import OptimizedKernel
from tinygrad.lazy import LazyBuffer
from tinygrad.runtime.lib import RawConst
from tinygrad.helpers import dtypes, DEBUG, DType, all_same, getenv, colored, PtrDType, partition
from enum import Enum, auto
from tinygrad.shape.symbolic import Variable, NumNode, Node, MulNode, SumNode, DivNode, ModNode, LtNode, AndNode
VariableOrNum = Union[Variable, NumNode, Node]

class UOps(Enum):
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  CONST = auto(); LOAD = auto(); STORE = auto(); BARRIER = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto() # noqa: E702
  LOOP = auto(); ENDLOOP = auto() # loops can be global, local, or other # noqa: E702
  def __lt__(self, x): return self.value < x.value

class UOp(NamedTuple):
  uop: UOps
  vin: Tuple[UOp]
  dtype: Optional[DType]
  arg: Any
  def __repr__(self): return f"{self.uop} {self.dtype} {self.arg}"

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
    #print(f"{str(uop):20s}: {len(vin)} {str(dtype):20s} {arg}")
    return UOp(uop, vin, dtype, arg)

  def uop_alu_idx(self, a, b, ops, ctx:UAst, op, dtype=dtypes.int32):
    return self.uop(UOps.ALU, (a, (NumNode(b) if not isinstance(b, Node) else b).render(ops, ctx)), dtype, op)

  def var_to_loop(self, var):
    return self.uop(UOps.LOOP, tuple(), dtypes.int32, (var.expr,var.min,var.max))

  render_ops: Any = { Variable: lambda self, ops, ctx: ctx.var_to_loop(self),
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
    # kernel name (before late upcast)
    self.function_name = ("r_" if self.reduceop else "E_") + '_'.join([str(x) if isinstance(x, int) else sym_rename(x) for x in self.full_shape])
    self.display_name = ("r_" if self.reduceop else "E_") + colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

    global_bufs = [self.uop(UOps.DEFINE_GLOBAL, tuple(), PtrDType(buf.dtype), i) for i,buf in enumerate(self.arg_bufs.keys())]

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
          ret = self.uop(UOps.CONST, (), x.dtype, x.realized._buf)
        else:
          # TODO: gate the load
          ret = self.uop(UOps.LOAD, (global_bufs[self.arg_bufs_num[x.realized]], idx_rendered) + ((valid_rendered,) if valid_rendered is not None else tuple()), x.dtype)
        if valid_rendered is not None: ret = self.uop(UOps.ALU, (valid_rendered, ret, self.uop(UOps.CONST, (), x.dtype, 0)), x.dtype, TernaryOps.WHERE)
        return ret
      if x.op in ReduceOps:
        nidxs = global_idxs+local_idxs+reduce_idxs
        nidxs += [(i1 if i2==i3 else i2) for i1,i2,i3 in zip(idxs[len(nidxs):], full_upcast_idxs, upcast_idxs)]
        expanded_nodes = [expand_node(idx) for idx in nidxs]
        lreduce_idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
        vin = tuple(ast_parse(x.src[0], lidxs) for lidxs in lreduce_idxs)
        if len(reduce_idxs):
          acc = self.uop(UOps.DEFINE_ACC, (), dtypes.float32, acc_count)
          acc_count += 1
          #first = self.uop(UOps.ALU, (self.var_to_loop(reduce_idxs[0]), self.uop(UOps.CONST, (), dtypes.int32, reduce_idxs[0].min+1)), dtypes.bool, BinaryOps.CMPLT)
          first = functools.reduce(lambda a,b: self.uop(UOps.ALU, (a,b), dtypes.int32, BinaryOps.ADD), [self.var_to_loop(ri) for ri in reduce_idxs])
          phi = self.uop(UOps.ALU, (first, acc, self.uop(UOps.CONST, tuple(), dtypes.float32, float('-inf') if x.op == ReduceOps.MAX else 0)), dtypes.float32, TernaryOps.WHERE)
          vin += (phi, )
        # NOTE: this determines the order of these when it doesn't have to
        ret = functools.reduce(lambda a,b: self.uop(UOps.ALU, (a,b), dtypes.float32, BinaryOps.MAX if x.op == ReduceOps.MAX else BinaryOps.ADD), vin)
        if len(reduce_idxs):
          ret = self.uop(UOps.STORE, (acc, ret), dtypes.float32)
          for ri in reduce_idxs:
            ret = self.uop(UOps.ENDLOOP, (ret, self.var_to_loop(ri)), dtypes.float32)
        return ret
        #return self.uop(UOps.PHI, (acc, ret, self.var_to_loop(reduce_idxs[0])), dtypes.float32) if len(reduce_idxs) else ret
      else:
        vin = tuple(ast_parse(v, idxs) for v in x.src)
        assert all_same([x.dtype for x in vin])
        if x.op == UnaryOps.NOOP: return vin[0]
        if x.op == UnaryOps.CAST: return self.uop(UOps.CAST, vin, x.arg)
        return self.uop(UOps.ALU, vin, vin[0].dtype, x.op)

    sinks = []
    expanded_nodes = [expand_node(idx) for idx in (global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs)]
    store_idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    for idxs in store_idxs:
      idx, valid = self.sts[0].expr_idxs(idxs)
      assert valid.min == 1
      idx_rendered = idx.render(self.render_ops, self)
      sinks.append(self.uop(UOps.STORE, (global_bufs[0], ast_parse(self.ast, idxs), idx_rendered)))

    # graph debugging
    if getenv("UASTGRAPH"):
      import networkx as nx
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

from tinygrad.renderer.cstyle import CStyleLanguage
def uops_to_cstyle2(function_name:str, uops:List[UOp]):
  lang = CStyleLanguage()
  r: Dict[UOp, Optional[str]] = {}
  statements: List[str] = []  # LOOP, LOAD, STORE
  globalz: List[Optional[str]] = []
  seen_end = defaultdict(int)

  c = defaultdict(int)
  def ssa(prefix="t"):
    nonlocal c
    c[prefix] += 1
    return f"{prefix}{c[prefix]-1}"
  def render_one(u:UOp) -> Optional[str]:
    nonlocal globalz
    if DEBUG >= 4: print(u.uop, u.dtype, u.arg)
    if u.uop == UOps.CONST:
      if u.arg == float("inf"): return "INFINITY"
      if u.arg == float("-inf"): return "-INFINITY"
      if math.isnan(u.arg): return "NAN"
      return str(u.arg)
    elif u.uop == UOps.LOOP:
      statements.append(f"for (int {u.arg[0]} = {u.arg[1]}; {u.arg[0]} <= {u.arg[2]}; {u.arg[0]}++) {{")
      return u.arg[0]
    elif u.uop == UOps.ALU: return lang.code_for_op[u.arg](*[r[x] for x in u.vin])
    elif u.uop == UOps.DEFINE_GLOBAL:
      globalz += [None] * (u.arg+1-len(globalz))
      globalz[u.arg] = f"float *data{u.arg}"
      return f"data{u.arg}"
    elif u.uop == UOps.ENDLOOP:
      seen_end[u.vin[1]] -= 1
      if seen_end[u.vin[1]] == 0: statements.append("}")
      return r[u.vin[0]]
    elif u.uop == UOps.DEFINE_ACC:
      tok = ssa("acc")
      #statements.append(f"{u.dtype.name} {tok} = {r[u.vin[0]]};")
      statements.append(f"{u.dtype.name} {tok};")
      return tok
    elif u.uop == UOps.LOAD:
      tok = ssa("val")
      if len(u.vin) == 3:
        # suppress the load if it's invalid
        statements.append(f"{u.dtype.name} {tok} = {r[u.vin[2]]} ? {r[u.vin[0]]}[{r[u.vin[1]]}] : 0.0;")
      else:
        statements.append(f"{u.dtype.name} {tok} = {r[u.vin[0]]}[{r[u.vin[1]]}];")
      return tok
    elif u.uop == UOps.STORE:
      if len(u.vin) == 2:
        statements.append(f"{r[u.vin[0]]} = {r[u.vin[1]]};")
      else:
        statements.append(f"{r[u.vin[0]]}[{r[u.vin[2]]}] = {r[u.vin[1]]};")
      return r[u.vin[0]]
    else:
      raise NotImplementedError(f"can't render {u.uop}")

  # first, we fetch all the uops
  seen = []
  def visit(x):
    if x in seen: return
    seen.append(x)
    for u in x.vin: visit(u)
  for u in uops: visit(u)
  in_loops = 0
  for x in seen:
    if x.uop == UOps.LOOP: in_loops += 1
    if x.uop == UOps.ENDLOOP: seen_end[x.vin[1]] += 1

  # then we figure out which are renderable and do that, leaving loops for last
  while len(seen):
    rend, seen = partition(seen, lambda x: all(u in r for u in x.vin))
    assert len(rend), "none to render"
    rend_nl, rend_l = partition(rend, lambda x: x.uop != UOps.LOOP)
    if not rend_nl:
      rend_l = sorted(rend_l, key=lambda x: x.arg)
      # we render one loop and see if it unlocks anything
      r[rend_l[0]] = render_one(rend_l[0])
      seen = rend_l[1:] + seen
    else:
      for u in rend_nl: r[u] = render_one(u)
      seen = rend_l + seen

  src = f"void {function_name}({', '.join(globalz)}) {{\n" + '\n'.join(statements)  + '\n' + '}'*(in_loops+1-len(seen_end))
  return src

