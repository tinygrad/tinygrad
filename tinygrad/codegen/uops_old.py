from __future__ import annotations
import functools, itertools
from typing import List, Set, Optional, Tuple, Any, Dict, DefaultDict, Callable, cast
from collections import defaultdict
from tinygrad.helpers import DEBUG, flatten, prod
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, exec_alu
from tinygrad.shape.symbolic import sint, Variable, Node, NumNode, MulNode, DivNode, SumNode
from tinygrad.codegen.uops import UOps, UOp

def uop_alu_resolve(u:UOp) -> sint:
  if u.uop is UOps.CONST: return u.arg
  elif u.uop is UOps.DEFINE_VAR: return u.arg
  elif u.uop is UOps.ALU and u.arg is BinaryOps.MUL: return uop_alu_resolve(u.vin[0]) * uop_alu_resolve(u.vin[1])
  elif u.uop is UOps.ALU and u.arg is BinaryOps.ADD: return uop_alu_resolve(u.vin[0]) + uop_alu_resolve(u.vin[1])
  else: raise RuntimeError(f"ALU resolve fail @ {u.uop}")

def _match(uop:UOp, pattern:Dict[str, Any], store:Dict[str, UOp]) -> bool:
  for k,v in pattern.items():
    if k == "__name__":
      if v in store and store[v] != uop: return False
      store[v] = uop
    elif k == "vin":
      # only one if it's a tuple
      # try all permutations if it's a list
      # repeat if it's a dict
      for vp in itertools.permutations(v) if isinstance(v, list) else ([v] if isinstance(v, tuple) else [(v,)*len(uop.vin)]):
        if len(uop.vin) != len(vp): return False
        new_store = store.copy()
        if all(_match(uu, vv, new_store) for uu, vv in zip(uop.vin, vp)):
          for k,v in new_store.items(): store[k] = v
          return True
      return False
    elif k == "dtype":
      if uop.__getattribute__(k) not in (v if isinstance(v, set) else set([v])): return False
    else:
      if uop.__getattribute__(k) != v: return False
  return True

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[Dict[str, Any], Any]]):
    self.patterns = patterns
    self.pdict = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns: self.pdict[(p.get("uop"), p.get("arg", None))].append((p, fxn))

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    for p,fxn in itertools.chain(self.pdict[(uop.uop, uop.arg)], self.pdict[(uop.uop, None)]):
      store: Dict[str, UOp] = {}
      if _match(uop, p, store): return fxn(**store)
    return None

  def rewrite_graph(self, uops: UOpGraph):
    replace: Dict[UOp, UOp] = {}
    seen: Set[UOp] = set()
    for u in uops:
      if u in seen: continue
      seen.add(u)
      for o,n in replace.items():
        if o in u.vin and u is not n:
          u.vin = tuple(n if x == o else x for x in u.vin)
      if rew := self.rewrite(u): replace[u] = rew

    for o,n in replace.items():
      queue = [n]
      while queue:
        if all([qq in uops.uops for qq in queue[-1].vin]):
          new = uops.add_op(q:=queue.pop(), insert_before=max([0]+[uops.uops.index(vv) for vv in q.vin])+1)
          if new != q:
            for vv in uops.uops + queue: vv.vin = tuple(new if x is q else x for x in vv.vin)
        else: queue.extend([qq for qq in queue[-1].vin if qq not in uops.uops])
      if not any([o in u.vin for u in uops.uops[uops.uops.index(o):]]): uops.uops.remove(o)

constant_folder = PatternMatcher([
  # const rules
  ({"__name__": "root", "uop": UOps.GEP, "vin": ({"__name__": "c", "uop": UOps.CONST},)}, lambda root, c: UOp.const(root.dtype, c.arg)),
  ({"__name__": "root", "uop": UOps.CAST, "vin": {"__name__": "c", "uop": UOps.CONST}}, lambda root, c: UOp.const(root.dtype, c.arg)),
  # a phi without loops (len(vin)==2) is a noop
  ({"uop": UOps.PHI, "vin": ({}, {"__name__": "x"})}, lambda x: x),
  # x+-y -> x-y
  ({"uop": UOps.ALU, "arg": BinaryOps.ADD, "vin": ({"__name__": "x"}, {"__name__": "my", "uop": UOps.ALU, "arg": UnaryOps.NEG})},
    lambda x, my: UOp(UOps.ALU, x.dtype, (x, my.vin[0]), BinaryOps.SUB)),
  # bool < False is always false, True < bool is always false
  ({"uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({}, {"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.bool, "arg": False})}, lambda x: x),
  ({"uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.bool, "arg": True}, {})},
    lambda x: UOp.const(dtypes.bool, False)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  ({"uop": UOps.ALU, "arg": TernaryOps.WHERE, "vin": ({}, {"__name__": "val"}, {"__name__": "val"})}, lambda val: val),
  ({"uop": UOps.ALU, "arg": TernaryOps.WHERE, "vin": ({"__name__": "gate", "uop": UOps.CONST}, {"__name__": "c0"}, {"__name__": "c1"})},
    lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  ({"__name__": "root", "uop": UOps.ALU, "vin": {"uop": UOps.CONST}},
    lambda root: UOp.const(root.dtype, exec_alu(root.arg, root.dtype, [x.arg for x in root.vin]))),
  # ** self folding **
  ({"uop": UOps.ALU, "arg": BinaryOps.ADD, "vin": [{"__name__": "x"}, {"uop": UOps.CONST, "arg": 0}]}, lambda x: x),   # x+0 -> x or 0+x -> x
  ({"uop": UOps.ALU, "arg": BinaryOps.MUL, "vin": [{"__name__": "x"}, {"uop": UOps.CONST, "arg": 1}]}, lambda x: x),   # x*1 -> x or 1*x -> x
  ({"uop": UOps.ALU, "arg": BinaryOps.SUB, "vin": ({"__name__": "x"}, {"uop": UOps.CONST, "arg": 0})}, lambda x: x),   # x-0 -> x
  ({"uop": UOps.ALU, "arg": BinaryOps.DIV, "vin": ({"__name__": "x"}, {"uop": UOps.CONST, "arg": 1})}, lambda x: x),   # x/1 -> x
  # ** zero folding **
  ({"uop": UOps.ALU, "arg": BinaryOps.MUL, "vin": [{}, {"__name__": "c", "uop": UOps.CONST, "arg": 0}]}, lambda c: c), # x*0 -> 0 or 0*x -> 0
  ({"uop": UOps.ALU, "arg": BinaryOps.SUB, "vin": ({"__name__": "x"}, {"__name__": "x"})}, lambda x: UOp.const(x.dtype, 0)),   # x-x -> 0
  # ** load/store folding **
  ({"uop": UOps.STORE, "vin": ({"__name__": "buf"}, {"__name__": "idx"},
                               {"uop": UOps.LOAD, "vin": ({"__name__": "buf"}, {"__name__": "idx"})})}, lambda buf, idx: UOp(UOps.NOOP)),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  ({"uop": UOps.STORE, "vin": ({"__name__": "buf"}, {"__name__": "idx"}, {"uop": UOps.ALU, "arg": TernaryOps.WHERE,
                       "vin": ({"__name__": "gate"}, {"__name__": "alt"}, {"uop": UOps.LOAD, "vin": ({"__name__": "buf"}, {"__name__": "idx"})})})},
    lambda buf, idx, gate, alt: UOp(UOps.STORE, None, (buf, idx, alt, gate))),
])

class UOpGraph:
  def __init__(self, start_uops:Optional[List[UOp]]=None):
    # list of uops
    self.uops: List[UOp] = [] if start_uops is None else start_uops

    # global uop cache
    self.saved_exprs: Dict[Tuple, UOp] = dict()

  def __iter__(self): return iter(self.uops)

  def vars(self) -> List[Variable]: return [x.arg for x in self.uops if x.uop is UOps.DEFINE_VAR]
  def globals(self) -> List[Tuple[int, bool]]: return [x.arg for x in self.uops if x.uop is UOps.DEFINE_GLOBAL]

  def graph(self):
    from tinygrad.features.graph import graph_uops
    graph_uops(self.uops)

  def print(self):
    for u in self.uops:
      print(f"{self.uops.index(u):4d} {str(u.uop):20s}: {str(u.dtype) if u.dtype is not None else '':25s} "
            f"{str([self.uops.index(x) for x in u.vin]):32s} {u.arg}")

  def add(self, uop:UOps, dtype:Optional[DType]=None, vin:Tuple[UOp, ...]=tuple(), arg:Any=None, insert_before=None, simplify=True) -> UOp:
    return self.add_op(UOp(uop, dtype, vin, arg) if uop is not UOps.CONST else UOp.const(dtype, arg), insert_before, simplify)

  def add_op(self, ret:UOp, insert_before=None, simplify=True) -> UOp:
    if simplify and (rewritten:=constant_folder.rewrite(ret)) is not None:
      if rewritten in self.uops: return rewritten
      ret = rewritten
    key = (ret.uop, ret.dtype, ret.vin, ret.arg)
    if insert_before is None: insert_before = len(self.uops)
    # check if the cached expr is valid with the given insert place.
    if (expr:=self.saved_exprs.get(key, None)) is not None and self.uops.index(expr) <= insert_before: return expr
    self.uops.insert(insert_before, ret)
    self.saved_exprs[key] = ret
    return ret

  def remove_childless(self, keep:Set[UOp]):
    while 1:
      has_child: Set[UOp] = set()
      for ru in self.uops:
        for vu in ru.vin:
          has_child.add(vu)
      nu: List[UOp] = [x for x in self.uops if x in has_child or x in keep]
      if len(nu) == len(self.uops): break
      if DEBUG >= 4: print(f"reduced UOp count from {len(self.uops)} to {len(nu)}")
      self.uops = nu
    self.saved_exprs = {k:v for k,v in self.saved_exprs.items() if v in nu}

  # optional
  def type_verify(self):
    for u in self.uops:
      uop, arg, vin, dtype = u.uop, u.arg, u.vin, u.dtype
      if uop in {UOps.CONST, UOps.DEFINE_ACC}:
        if uop is UOps.DEFINE_ACC: arg = arg[0]
        assert dtype is not None and type(arg) is type(dtypes.as_const(arg, dtype)), f"type of {arg=} does not match {dtype}"
      if uop is UOps.ALU:
        if arg in UnaryOps:
          assert dtype == vin[0].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=}"
        elif arg in (BinaryOps.CMPLT, BinaryOps.CMPEQ):
          assert dtype == dtypes.bool, f"{arg} output dtype mismatch {dtype=} != {dtypes.bool}"
          assert vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
        elif arg in BinaryOps:
          assert dtype == vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
        elif arg == TernaryOps.WHERE:
          assert vin[0].dtype == dtypes.bool, f"{arg} selector dtype mismatch {vin[0].dtype=} != {dtypes.bool}"
          assert dtype == vin[1].dtype == vin[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {vin[1].dtype=} != {vin[2].dtype=}"

  def get_recursive_children(self, x:UOp) -> Set[UOp]:
    deps = set([x])
    ssize = 0
    while ssize != len(deps):
      ssize = len(deps)
      for u in self.uops:
        if len(deps.intersection([x for x in u.vin if x.uop is not UOps.PHI])):
          deps.add(u)
    return deps

  def add_ends(self):
    for u in self.uops:
      if u.uop is UOps.LOOP:
        # add END of loops after the last thing that (recursively) depends on them
        insert_before = self.uops.index(sorted(list(self.get_recursive_children(u)), key=self.uops.index)[-1])+1
        self.add(UOps.ENDLOOP, None, (u,), insert_before=insert_before)
      elif u.uop is UOps.IF:
        # END any if statements at the end of the uops
        self.add(UOps.ENDIF, None, (u,))

  def fix_loop_scope(self, get_recursive_parents:Callable[..., Set[UOp]]):
    loop_stack: List[List[UOp]] = [[]]
    # push uops upward out of loop if it does not depend on the loop
    for u in self.uops:
      if not loop_stack[-1]: loop_stack[-1].append(u)
      elif u.uop is UOps.LOOP: loop_stack.append([u])
      elif u.uop not in [UOps.CONST, UOps.ALU, UOps.CAST, UOps.LOAD]: loop_stack[-1].append(u)
      else:
        parents = get_recursive_parents(u, with_phi=True)
        # don't push any local buffer because there might have STORE and BARRIER (not considered as parent) between DEFINE_LOCAL and here
        if any(u.uop is UOps.DEFINE_LOCAL for u in parents): loop_stack[-1].append(u)
        else:
          for i in reversed(range(len(loop_stack))):
            # check backwards and put the uop in the first encounter with some dependency
            if any(x in parents for x in loop_stack[i]) or i == 0:
              loop_stack[i].append(u)
              break
    self.uops = flatten(loop_stack)

  def replace_op(self, old, new):
    for v in self.uops: v.vin = tuple(new if x is old else x for x in v.vin)
    self.uops.remove(old)

  def simplify_phi_loops(self, get_recursive_parents):
    def alu_opposite(arg, x, y):
      if arg is BinaryOps.ADD: return x - y
      elif arg is BinaryOps.MUL: return Node.__floordiv__(x, y, False)
      else: raise RuntimeError("unhandled alu")
    def to_symbolic(u: UOp):
      if u.uop is UOps.CONST: return NumNode(int(u.arg))
      elif u.uop in {UOps.LOOP, UOps.SPECIAL}:
        if u not in seen_vars: seen_vars[u] = u.arg[1] if u.uop is UOps.SPECIAL else "loop{}".format(len(seen_vars))
        return Variable(seen_vars[u], u.vin[0].arg, u.vin[1].arg-1) if u.uop is UOps.LOOP else Variable(seen_vars[u], 0, u.arg[2]-1)
      elif u.uop is UOps.ALU and u.arg is BinaryOps.ADD: return to_symbolic(u.vin[0]) + to_symbolic(u.vin[1])
      elif u.uop is UOps.ALU and u.arg is BinaryOps.MUL: return to_symbolic(u.vin[0]) * to_symbolic(u.vin[1])
      else: raise RuntimeError("unhandled op: {}".format(u))
    def loop_factor(with_loop: UOp, factored: Node, loop_op, round_up=False):
      if with_loop == loop_op: return factored
      elif with_loop.uop is UOps.ALU:
        next_with_loop = next(v for v in with_loop.vin if v == loop_op or loop_op in get_recursive_parents(v))
        non_loop = to_symbolic(next(v for v in with_loop.vin if v != next_with_loop and loop_op not in get_recursive_parents(v)))
        if round_up and with_loop.arg is BinaryOps.MUL: factored = factored + (non_loop - 1)
        return loop_factor(next_with_loop, alu_opposite(with_loop.arg, factored, non_loop), loop_op)
    def const(x, insert_before=None): return self.add(UOps.CONST, dtypes.int32, tuple(), x, insert_before=insert_before)
    def neg(x): return self.add(UOps.ALU, dtypes.int32, (x,), UnaryOps.NEG)
    def max(x, y): return self.add(UOps.ALU, dtypes.int32, (x, y), BinaryOps.MAX)
    def uop_alu_idx(a: UOp, b, op, dtype=dtypes.int32):
      render_b: UOp = cast(UOp, (NumNode(b) if not isinstance(b, Node) else b).render(render_ops))
      return self.add(UOps.ALU, dtype, (a, render_b), op)
    seen_vars: Dict[UOp,str] = {}
    render_ops = {Variable: lambda self, ops, _: next(op for op, name in seen_vars.items() if name == self.expr),
                  NumNode: lambda self, ops, _: const(self.b),
                  MulNode: lambda self, ops, _: uop_alu_idx(self.a.render(ops, self), self.b, BinaryOps.MUL),
                  DivNode: lambda self, ops, _: uop_alu_idx(self.a.render(ops, self), self.b, BinaryOps.DIV),
                  SumNode: lambda self, ops, _:
                  functools.reduce(lambda a, b: uop_alu_idx(a, b, BinaryOps.ADD), self.nodes[1:], self.nodes[0].render(ops, self))}

    allowed_ops = {UOps.CONST, UOps.SPECIAL, UOps.ALU, UOps.LOOP, UOps.DEFINE_ACC}
    allowed_alus = {BinaryOps.MUL, BinaryOps.ADD, BinaryOps.CMPLT, TernaryOps.WHERE}
    for loop_op in reversed([op for op in self.uops if op.uop is UOps.LOOP]):
      phis = set([u for u in self.get_recursive_children(loop_op) if u.uop is UOps.PHI])
      wheres = set([u for phi in phis for u in get_recursive_parents(phi) if u.arg == TernaryOps.WHERE])
      if (any([u.uop is not UOps.CONST for u in loop_op.vin])
        or any([u.uop not in allowed_ops or (u.uop is UOps.ALU and u.arg not in allowed_alus) for phi in phis for u in get_recursive_parents(phi)])
        or any([where.vin[2].arg != 0 or where.vin[0].vin[1].uop is not UOps.CONST for where in wheres])
        or any(len([op for op in get_recursive_parents(where) if op.uop is UOps.LOOP]) == 0 for where in wheres)): continue
      if DEBUG >= 4 and (len(phis) > 0 or len(wheres) > 0): print("simplified {} PHI and {} WHERE in loop".format(len(phis), len(wheres)))
      loop_length = loop_op.vin[1].arg - loop_op.vin[0].arg
      for u in self.uops:
        if u.arg is BinaryOps.ADD and len(wheres.intersection(get_recursive_parents(u))) and len(phis.intersection(self.get_recursive_children(u))):
          u.vin = tuple([const(vin.arg*loop_length, insert_before=self.uops.index(u)) if vin.uop is UOps.CONST else vin for vin in list(u.vin)])
      for where in sorted(wheres, key=lambda x: self.uops.index(x)):
        comp_lt, comp_gt = where.vin[0].vin[0], where.vin[0].vin[1]
        factored = loop_factor(comp_lt, NumNode(int(comp_gt.arg)), loop_op, round_up=(comp_gt.arg > 0))
        final_value = factored - NumNode(loop_op.vin[0].arg) if (comp_gt.arg > 0) else NumNode(loop_op.vin[1].arg-1) - factored
        self.uops, after_split_ops = self.uops[:(where_index:=self.uops.index(where))], self.uops[where_index:]
        rendered = final_value.render(render_ops)
        min_clamped = max(rendered, const(0)) if (final_value.min < 0) else rendered
        max_clamped = neg(max(const(-1*loop_length), neg(min_clamped))) if (final_value.max > loop_length) else min_clamped
        maybe_cast = self.add(UOps.CAST, where.dtype, (max_clamped,)) if where.dtype != dtypes.int32 else max_clamped
        final_op = self.add(UOps.ALU, where.dtype, (maybe_cast, where.vin[1]), BinaryOps.MUL)
        self.uops = self.uops + after_split_ops
        self.replace_op(where, final_op)
      for phi in phis:
        self.replace_op(phi, phi.vin[1])
        self.uops.remove((accumulator:=phi.vin[0]))
        for alu_with_accum in [op for op in self.uops if accumulator in op.vin]:
          self.replace_op(alu_with_accum, next(op for op in alu_with_accum.vin if op != accumulator))
      get_recursive_parents.cache_clear()

  def fix_to_store_directly(self):
    replaced_stores: Dict[UOp,UOp] = {}
    for u in self.uops:
      if u.uop is not UOps.STORE or (val:=u.vin[-1]).uop is not UOps.CAST or cast(DType,val.dtype).count == 1: continue

      vins = val.vin
      while all(el.uop is UOps.PHI for el in vins): vins = tuple([el.vin[0] for el in vins])
      if all(el.uop is UOps.GEP for el in vins) and len(set(el.vin[0] for el in vins)) == 1 and val.dtype == vins[0].vin[0].dtype:
        # Check that accesses are in order.
        if all(i==el.arg for i,el in enumerate(vins)):
          replaced_stores[u] = vins[0].vin[0]

    for prev,new in replaced_stores.items():
      try: self.uops.remove(prev.vin[-1])  # remove the old upcast NOTE: the upcast's vins become childless now
      except ValueError: pass  # already removed
      self.uops[self.uops.index(prev)].vin = (prev.vin[0],prev.vin[1],new) # replace with the float4 value

  def uops_optimization(self, get_recursive_parents):
    for u in self.uops:
      if u.uop is UOps.PHI and len(u.vin) == 3:
        # if the parents of the PHI node don't have the LOOP in their parents, it can be folded
        # TODO: ADD becomes a MUL, MAX can just become nothing
        # NOTE: ADD -> MUL does not fold, this maintains original MULACC code path
        if all(x.uop is not UOps.LOOP for x in get_recursive_parents(UOp(u.uop, u.dtype, u.vin[0:2], u.arg))) \
          and u.vin[1].arg is BinaryOps.ADD and u.vin[1].vin[0].arg is not BinaryOps.MUL:
          if DEBUG >= 4: print(f"removing PHI node {u}")
          del self.saved_exprs[(u.uop, u.dtype, u.vin, u.arg)]
          # NOTE: assuming u.vin[2].vin[1] and u.vin[2].vin[0] have the same dtype
          loop_len = self.add(UOps.ALU, u.vin[2].vin[1].dtype, (u.vin[2].vin[1], u.vin[2].vin[0]), BinaryOps.SUB,
                              insert_before=self.uops.index(u))
          if loop_len.dtype != u.dtype: loop_len = self.add(UOps.CAST, u.dtype, (loop_len,),
                                                            insert_before=self.uops.index(u))
          new = self.add(UOps.ALU, u.dtype, (u.vin[1], loop_len,), BinaryOps.MUL, insert_before=self.uops.index(u))
          self.replace_op(u, new)
          return True

  def optimize_loops(self):
    # get PHI node loop scope, link anything using a DEFINE_ACC to the loop as a "parent"
    acc_scope: DefaultDict[UOp, List[UOp]] = defaultdict(list)
    for u in self.uops:
      if u.uop is UOps.PHI: acc_scope[u.vin[0]] += u.vin[2:]

    # graph helper functions
    @functools.lru_cache(None)
    def get_recursive_parents(x:UOp, with_phi=False) -> Set[UOp]:
      return set.union(set(x.vin), *[get_recursive_parents(p, with_phi) for p in x.vin], set(acc_scope[x]) if with_phi else set())

    # fix loop scope, push uops upward out of loop if it does not depend on the loop
    self.fix_loop_scope(get_recursive_parents)

    # uops optimization
    while self.uops_optimization(get_recursive_parents): pass
    self.simplify_phi_loops(get_recursive_parents)

  def uoptimize(self):
    self.optimize_loops()

    # (recursively) remove childless uops
    self.remove_childless(set(x for x in self.uops if x.uop is UOps.STORE))

    # store float4 upcasts directly if possible
    self.fix_to_store_directly()

    # add UOps.END*
    self.add_ends()

    # verify the uop types
    self.type_verify()

  def flops_mem(self) -> Tuple[sint, sint]:
    flops: sint = 0
    mem: sint = 0
    mults: sint = 1
    mult_stack = []
    for u in self.uops:
      if u.uop is UOps.LOOP:
        mult_stack.append(mults)
        mults *= uop_alu_resolve(u.vin[1])
      elif u.uop is UOps.ENDLOOP:
        mults = mult_stack.pop(-1)
      elif u.uop is UOps.ALU:
        flops += mults
      elif u.uop is UOps.LOAD:
        assert u.dtype is not None
        mem += u.dtype.itemsize * mults
      elif u.uop is UOps.STORE:
        assert u.vin[2].dtype is not None
        mem += u.vin[2].dtype.itemsize * mults
      elif u.uop is UOps.WMMA:
        assert u.arg[1] is not None
        flops += 2 * prod(u.arg[1]) // 32 * mults
    return flops, mem
