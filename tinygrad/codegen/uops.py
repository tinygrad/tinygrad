from __future__ import annotations
import functools, math, operator
from typing import List, Set, Optional, Tuple, Any, Dict, DefaultDict, Callable, cast
from collections import defaultdict
from tinygrad.helpers import DEBUG, flatten, all_same
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.shape.symbolic import sint, Variable
from enum import Enum, auto
from dataclasses import dataclass

# bottom ones are asm only
class UOps(Enum):
  LOOP = auto(); IF = auto(); ENDLOOP = auto(); ENDIF = auto(); SPECIAL = auto() # loops can be global, local, or other # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  LOAD = auto(); STORE = auto(); CONST = auto(); BARRIER = auto(); PHI = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto(); GEP = auto() # noqa: E702

@dataclass(eq=False)
class UOp:
  uop: UOps
  dtype: Optional[DType]
  vin: Tuple[UOp, ...]
  arg: Any
  def __repr__(self):
    return f"{str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.uop for x in self.vin]):32s} {self.arg}"

def hook_overflow(dv, fxn):
  def wfxn(*args):
    try: return fxn(*args)
    except OverflowError: return dv
  return wfxn

python_alu = {
  UnaryOps.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan,
  UnaryOps.EXP2: hook_overflow(math.inf, lambda x: math.exp(x*math.log(2))),
  UnaryOps.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, UnaryOps.SIN: math.sin,
  UnaryOps.NEG: lambda x: (not x) if isinstance(x, bool) else -x,
  BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.XOR: operator.xor,
  BinaryOps.MAX: max, BinaryOps.CMPEQ: operator.eq, BinaryOps.CMPLT: operator.lt, BinaryOps.MOD: operator.mod,
  BinaryOps.DIV: lambda x,y: int(x/y) if isinstance(x, int) else (x/y if y != 0 else math.nan),
  TernaryOps.WHERE: lambda x,y,z: y if x else z}

truncate: Dict[DType, Callable] = {
  dtypes.bool: lambda x: bool(x),
  **{dt:lambda x: x for dt in dtypes.fields().values() if dtypes.is_float(dt)},
  **{dt:functools.partial(lambda vv,x: x&vv, (1 << (dt.itemsize*8))-1) for dt in dtypes.fields().values() if dtypes.is_unsigned(dt)},
  **{dt:functools.partial(lambda vv,aa,x: ((x+aa)&vv)-aa, (1 << (dt.itemsize*8))-1, 1 << (dt.itemsize*8-1)) \
     for dt in dtypes.fields().values() if dtypes.is_int(dt) and not dtypes.is_unsigned(dt)}}
def exec_alu(arg, dtype, p): return truncate[dtype](python_alu[arg](*p))

def uop_alu_resolve(u:UOp) -> sint:
  if u.uop is UOps.CONST: return u.arg
  elif u.uop is UOps.DEFINE_VAR: return u.arg
  elif u.uop is UOps.ALU and u.arg == BinaryOps.MUL: return uop_alu_resolve(u.vin[0]) * uop_alu_resolve(u.vin[1])
  elif u.uop is UOps.ALU and u.arg == BinaryOps.ADD: return uop_alu_resolve(u.vin[0]) + uop_alu_resolve(u.vin[1])
  else: raise RuntimeError(f"ALU resolve fail @ {u.uop}")

def phi_resolve_acc(u:UOp) -> UOp: return u if u.uop is UOps.DEFINE_ACC else phi_resolve_acc(u.vin[0])

class UOpGraph:
  def __init__(self, start_uops:Optional[List[UOp]]=None):
    # list of uops
    self.uops: List[UOp] = [] if start_uops is None else start_uops

    # global uop cache
    self.saved_exprs: Dict[Tuple, UOp] = dict()

  def __iter__(self): return iter(self.uops)

  def vars(self) -> List[Variable]: return [x.arg for x in self.uops if x.uop is UOps.DEFINE_VAR]

  def graph(self):
    from tinygrad.features.graph import graph_uops
    graph_uops(self.uops)

  def print(self):
    for u in self.uops:
      print(f"{self.uops.index(u):4d} {str(u.uop):20s}: {str(u.dtype) if u.dtype is not None else '':25s} "
            f"{str([self.uops.index(x) for x in u.vin]):32s} {u.arg}")

  def add(self, uop:UOps, dtype:Optional[DType]=None, vin:Tuple[UOp, ...]=tuple(), arg:Any=None, cachable=True, insert_before=None,
          simplify=True) -> UOp:
    if simplify:
      if uop is UOps.PHI and len(vin) == 2: return vin[1]   # a phi without loops is a noop
      if uop is UOps.GEP and vin[0].uop is UOps.CONST: return self.add(UOps.CONST, dtype, arg=vin[0].arg, insert_before=insert_before)
      if uop is UOps.CAST and all(x.uop is UOps.CONST for x in vin) and all_same([x.arg for x in vin]):
        return self.add(UOps.CONST, dtype, arg=vin[0].arg, insert_before=insert_before)
      if uop is UOps.ALU:
        # rewrites. NOTE: the rewritten NEG op is still around...
        if arg is BinaryOps.ADD and vin[1].uop is UOps.ALU and vin[1].arg is UnaryOps.NEG:
          return self.add(UOps.ALU, dtype, (vin[0], vin[1].vin[0]), BinaryOps.SUB, cachable, insert_before)
        # constant folding
        if arg is TernaryOps.WHERE and vin[1] == vin[2]: return vin[1] # a conditional with the same results either way is a noop
        if arg is TernaryOps.WHERE and vin[0].uop is UOps.CONST: return vin[1] if vin[0].arg else vin[2]
        if all(x.uop is UOps.CONST for x in vin):
          return self.add(UOps.CONST, dtype, arg=exec_alu(arg, dtype, [x.arg for x in vin]), insert_before=insert_before)
        # zero folding
        for x in [0,1]:
          if arg is BinaryOps.ADD and vin[x].uop is UOps.CONST and vin[x].arg == 0.0: return vin[1-x]
          if arg is BinaryOps.MUL and vin[x].uop is UOps.CONST and vin[x].arg == 1.0: return vin[1-x]
          if arg is BinaryOps.MUL and vin[x].uop is UOps.CONST and vin[x].arg == 0.0: return vin[x]
        if arg is BinaryOps.SUB and vin[1].uop is UOps.CONST and vin[1].arg == 0.0: return vin[0]
        if arg is BinaryOps.DIV and vin[1].uop is UOps.CONST and vin[1].arg == 1.0: return vin[0]

    key = (uop, dtype, vin, arg)
    if insert_before is None: insert_before = len(self.uops)
    # check if the cached expr is valid with the given insert place.
    if cachable and (expr:=self.saved_exprs.get(key, None)) is not None and self.uops.index(expr) <= insert_before: return expr
    ret = UOp(uop, dtype, vin, arg)
    self.uops.insert(insert_before, ret)
    if cachable: self.saved_exprs[key] = ret
    return ret

  def remove_childless(self):
    UOPS_W_SIDE_EFFECTS = {UOps.DEFINE_GLOBAL, UOps.STORE}

    while 1:
      has_child: Set[UOp] = set()
      for ru in self.uops:
        for vu in ru.vin:
          has_child.add(vu)
      nu: List[UOp] = [x for x in self.uops if x in has_child or x.uop in UOPS_W_SIDE_EFFECTS]
      if len(nu) == len(self.uops): break
      if DEBUG >= 4: print(f"reduced UOp count from {len(self.uops)} to {len(nu)}")
      self.uops = nu

  # optional
  def type_verify(self):
    for u in self.uops:
      uop, arg, vin, dtype = u.uop, u.arg, u.vin, u.dtype
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
        if len(deps.intersection([x for x in u.vin if x.uop != UOps.PHI])):
          deps.add(u)
    return deps

  def add_ends(self):
    for u in self.uops:
      if u.uop is UOps.LOOP:
        # add END of loops after the last thing that (recursively) depends on them
        insert_before = self.uops.index(sorted(list(self.get_recursive_children(u)), key=self.uops.index)[-1])+1
        self.add(UOps.ENDLOOP, None, (u,), cachable=False, insert_before=insert_before)
      elif u.uop is UOps.IF:
        # END any if statements at the end of the uops
        self.add(UOps.ENDIF, None, (u,), cachable=False)

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

  def uoptimize(self):
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
    changed_something = True
    while changed_something:
      changed_something = False
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
            loop_len = self.add(UOps.ALU, u.vin[2].vin[1].dtype, (u.vin[2].vin[1], u.vin[2].vin[0]), BinaryOps.SUB, insert_before=self.uops.index(u))
            if loop_len.dtype != u.dtype: loop_len = self.add(UOps.CAST, u.dtype, (loop_len,), insert_before=self.uops.index(u))
            new = self.add(UOps.ALU, u.dtype, (u.vin[1], loop_len,), BinaryOps.MUL, insert_before=self.uops.index(u))
            # replace u with new
            for v in self.uops: v.vin = tuple(new if x is u else x for x in v.vin)
            self.uops.remove(u)
            changed_something = True

    # (recursively) remove childless uops
    self.remove_childless()

    # store float4 upcasts directly if possible
    replaced_stores: Dict[UOp,UOp] = {}
    for u in self.uops:
      if u.uop is not UOps.STORE or (val:=u.vin[-1]).uop is not UOps.CAST or cast(DType,val.dtype).count == 1: continue
      if all(el.uop is UOps.GEP for el in val.vin): replaced_stores[u] = val.vin[0].vin[0]
      elif all(el.uop is UOps.PHI for el in val.vin): replaced_stores[u] = phi_resolve_acc(val)
    for prev,new in replaced_stores.items():
      try: self.uops.remove(prev.vin[-1])  # remove the old upcast NOTE: the upcast's vins become childless now
      except ValueError: pass  # already removed
      self.uops[self.uops.index(prev)].vin = (prev.vin[0],prev.vin[1],new) # replace with the float4 value

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
        if u.arg.startswith("__metal_wmma"): flops += 2*(8*8*8)//32 * mults
        elif u.arg == "__hip_wmma_f16_f16" or u.arg == "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32": flops += 2*(16*16*16)//32 * mults
        elif u.arg == "__cuda_mma_m16n8k16_f16_f32": flops += 2*(8*16*16)//32 * mults
        else: raise Exception("not implemented")
    return flops, mem
