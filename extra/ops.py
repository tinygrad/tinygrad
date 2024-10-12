from __future__ import annotations
from typing import Dict, Union, Tuple, Any, List, cast
import functools, hashlib
from enum import Enum, auto
from dataclasses import dataclass
from tinygrad.helpers import dedup, prod
from tinygrad.ops import ReduceOps, UnaryOps, BinaryOps, TernaryOps, UOp, UOps, pretty_print
from tinygrad.dtype import ImageDType, PtrDType, dtypes, DType, ConstType
from tinygrad.ops import Variable, sint
from tinygrad.shape.shapetracker import ShapeTracker

# these ops are deleted after AST is UOp
class BufferOps(Enum): LOAD = auto(); CONST = auto(); STORE = auto() # noqa: E702
class MetaOps(Enum): KERNEL = auto();
Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps, BufferOps]

@dataclass(frozen=True)
class MemBuffer:
  idx: int
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ConstBuffer:
  val: ConstType | Variable
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True, eq=False)
class LazyOp:
  op: Op
  src: Tuple[LazyOp, ...] = ()
  arg: Any = None
  def cached_compare(self, x, context):
    if id(self) == id(x): return True
    if self.op != x.op or self.arg != x.arg or len(self.src) != len(x.src): return False
    if (key := (id(self), id(x))) in context: return context[key]
    ret = context[key] = all(a.cached_compare(b, context) for a,b in zip(self.src, x.src))
    return ret
  def __eq__(self, x): return self.cached_compare(x, context={})
  def __repr__(self:LazyOp): return pretty_print(self, lambda x: f'LazyOp({x.op}, arg={x.arg}, src=(%s))')
  @functools.cached_property
  def dtype(self) -> DType:
    if self.op in BufferOps: return self.arg.dtype
    if self.op in [UnaryOps.CAST, UnaryOps.BITCAST]: return self.arg
    return dtypes.bool if self.op in {BinaryOps.CMPLT, BinaryOps.CMPNE} else self.src[-1].dtype
  @functools.cached_property
  def full_shape(self) -> Tuple[sint, ...]:
    if len(self.src) == 0 and self.op in BufferOps: return self.arg.st.shape
    return tuple(max(x) for x in zip(*[x.full_shape for x in self.src]))
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(functools.reduce(lambda x,y: x+y, [s.key for s in self.src], str((self.op, self.arg)).encode())).digest()
  @functools.cached_property
  def hash(self): return hash((self.op, self.src, self.arg))
  def __hash__(self): return self.hash
  @functools.cached_property
  def lazyops(self) -> List[LazyOp]: return dedup([self] + [item for x in self.src for item in x.lazyops])
  def vars(self) -> List[Variable]:
    extract_vars = [x.arg.st.vars() for x in self.lazyops if x.op in BufferOps]
    const_vars = [x.arg.val for x in self.lazyops if x.op is BufferOps.CONST and isinstance(x.arg.val, Variable)]
    return sorted(set.union(*extract_vars, set(const_vars)), key=lambda v: v.expr)
  def __add__(self, x:LazyOp): return LazyOp(BinaryOps.ADD, (self, x))
  def __sub__(self, x:LazyOp): return LazyOp(BinaryOps.ADD, (self, -x))
  def __mul__(self, x:LazyOp): return LazyOp(BinaryOps.MUL, (self, x))
  def ne(self, x:LazyOp): return LazyOp(BinaryOps.CMPNE, (self, x))
  def eq(self, x:LazyOp): return -self.ne(x)
  def __neg__(self): return LazyOp(UnaryOps.NEG, (self,))
  @staticmethod
  def const(val, dtype:DType, shape:Tuple[sint, ...]):
    return LazyOp(BufferOps.CONST, (), ConstBuffer(val, dtype, ShapeTracker.from_shape(()).reshape((1,)*len(shape)).expand(shape)))

# the living definition of LazyOps
def verify_lazyop(ast:LazyOp) -> Dict[LazyOp, ShapeTracker]:
  assert ast.op is MetaOps.KERNEL, "must be SINK"
  sts: Dict[LazyOp, ShapeTracker] = {}
  def assert_valid(op:LazyOp, st:ShapeTracker):
    if op in sts: return
    # restore globals from the two stage reduce
    if op.op is BufferOps.LOAD and op.arg.idx < 0:
      assert_valid(local_reduce:=op.src[0].src[0], op.arg.st)
      return sts.setdefault(op, sts[local_reduce])
    for x in op.src: assert_valid(x, st)
    # only reduceop is allowed to change shape, limited to turning n to 1
    if op.op in ReduceOps:
      axis = op.arg
      assert isinstance(axis, tuple) and all(isinstance(i, int) for i in axis), f"reduceop must have axis {op.arg}"
      st = ShapeTracker.from_shape(sts[op.src[0]].reduce(axis))
    else:
      # movementops are pushed to the edges with LOAD
      # elementwise inherits shape
      st = op.arg.st if op.op in BufferOps else sts[op.src[0]]
      for x in op.src:
        if sts[x].shape != st.shape:
          if prod(sts[x].shape) == prod(st.shape): raise AssertionError(f"found implicit reshape {x.op} {op.op} {sts[x].shape} != {st.shape}")
          raise AssertionError(f"found implicit expand {x.op} {sts[x].shape} != {op.op} {st.shape} {prod(sts[x].shape)} != {prod(st.shape)}")
    sts[op] = st
  for i, out in enumerate(ast.src):
    assert out.arg.idx == i, f"unexpected output buffer idx {out.arg.idx} != {i}"
    assert out.op is BufferOps.STORE, f"kernels must have stores as the output, got {out.op}"
    assert out.arg.st.size == ast.src[-1].arg.st.size, f"outputs must have the same size, got {out.arg.st.size}"
    assert_valid(out, out.arg.st)
  shape_dims = [sorted(dedup(dims)) for dims in zip(*[x.shape for x in sts.values()])]
  assert all(len(x) == 1 or (len(x) == 2 and x[0] == 1) for x in shape_dims), f"shapes must have either 1 or n in each dimension, {shape_dims}"
  return sts

def to_uop(*a) -> UOp:
  assert isinstance(a[0], LazyOp), f"{a} must be a LazyOp ast"
  if a[0].op is BufferOps.STORE: ast = LazyOp(MetaOps.KERNEL, a)
  else:
    assert a[0].op is MetaOps.KERNEL
    ast = a[0]
  verify_lazyop(ast)
  @functools.lru_cache(None)
  def create_uop(lop:LazyOp) -> UOp:
    if lop.op in BufferOps:
      st_uop = lop.arg.st.to_uop()
      membuf_dtype: DType = lop.arg.dtype
      dtype = membuf_dtype.base if isinstance(membuf_dtype, ImageDType) else membuf_dtype
      if lop.op is BufferOps.CONST:
        return UOp(UOps.CONST, dtype, (st_uop,), lop.arg.val)
      buf = UOp(UOps.DEFINE_GLOBAL, membuf_dtype if isinstance(membuf_dtype, ImageDType) else PtrDType(membuf_dtype), (), lop.arg.idx)
      if lop.op is BufferOps.LOAD: return UOp(UOps.LOAD, dtype, (buf, st_uop))
      return UOp(UOps.STORE, dtypes.void, (buf, st_uop, create_uop(lop.src[0])))
    src = tuple(create_uop(x) for x in lop.src)
    if lop.op is MetaOps.KERNEL: return UOp(UOps.SINK, dtypes.void, src)
    if lop.op in ReduceOps:
      alu_op = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.PROD:BinaryOps.MUL, ReduceOps.MAX:BinaryOps.MAX}[cast(ReduceOps, lop.op)]
      return UOp(UOps.REDUCE_AXIS, src[0].dtype, src, (alu_op, lop.arg))
    if lop.op is UnaryOps.CAST: return UOp(UOps.CAST, lop.arg.scalar(), src)
    if lop.op is UnaryOps.BITCAST: return UOp(UOps.BITCAST, lop.arg.scalar(), src)
    return src[0].alu(lop.op, *src[1:])
  ret = create_uop(ast)
  #with open("/tmp/ast", "w") as f: f.write(str(ret))
  return ret
