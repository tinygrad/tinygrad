from __future__ import annotations
from typing import Union, Tuple, Any, List, Dict, Callable
import functools, hashlib, math, operator, ctypes, struct
from enum import Enum, auto
from dataclasses import dataclass
from tinygrad.helpers import prod, dedup
from tinygrad.dtype import dtypes, DType, ConstType
from tinygrad.shape.symbolic import Variable, sint
from tinygrad.shape.shapetracker import ShapeTracker

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: many GPUs don't have DIV, but UnaryOps.RECIP doesn't work for integer division
class UnaryOps(Enum):
  """A -> A (elementwise)"""
  EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto(); RECIP = auto() # noqa: E702
class BinaryOps(Enum):
  """A + A -> A (elementwise)"""
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto() # noqa: E702
class TernaryOps(Enum):
  """A + A + A -> A (elementwise)"""
  WHERE = auto(); MULACC = auto() # noqa: E702
class ReduceOps(Enum):
  """A -> B (reduce)"""
  SUM = auto(); MAX = auto(); WMMA = auto() # noqa: E702
class BufferOps(Enum): LOAD = auto(); CONST = auto(); STORE = auto() # noqa: E702
class MetaOps(Enum): SINK = auto()
class LoadOps(Enum): EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); ASSIGN = auto(); VIEW = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, LoadOps, TernaryOps, BufferOps, MetaOps]

# do not preserve f(0) = 0
UNSAFE_PAD_OPS = {UnaryOps.RECIP, UnaryOps.LOG2, UnaryOps.EXP2, BinaryOps.IDIV}

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
  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  @functools.cached_property
  def dtype(self) -> DType:
    if self.op in BufferOps: return self.arg.dtype
    if self.op is ReduceOps.WMMA: return self.arg[3]   # WMMA can change the type
    if self.op in [UnaryOps.CAST, UnaryOps.BITCAST]: return self.arg
    return dtypes.bool if self.op in {BinaryOps.CMPLT, BinaryOps.CMPNE} else self.src[-1].dtype
  @functools.cached_property
  def full_shape(self):
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

  # TODO: support non-lazyop
  def __add__(self, x:LazyOp): return LazyOp(BinaryOps.ADD, (self, x))
  def __sub__(self, x:LazyOp): return LazyOp(BinaryOps.ADD, (self, -x))
  def __mul__(self, x:LazyOp): return LazyOp(BinaryOps.MUL, (self, x))
  def __neg__(self): return LazyOp(UnaryOps.NEG, (self,))

# **************** independent FlopCounter ****************

@dataclass
class FlopCounter:
  shape: Tuple[int, ...]
  flops: sint
  mem: Dict[int, int]
  @property
  def mem_estimate(self): return sum(self.mem.values())
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret

InterpretedFlopCounter: Dict[Op, Callable] = {
  BufferOps.LOAD: lambda arg: FlopCounter(arg.st.shape, 0, {arg.idx: arg.dtype.itemsize * arg.st.real_size()}),
  BufferOps.CONST: lambda arg: FlopCounter(arg.st.shape, 0, {}),
  BufferOps.STORE: lambda self,arg: FlopCounter(arg.st.shape, self.consume_flops(), {**self.mem, arg.idx: arg.dtype.itemsize * arg.st.real_size()}),
  UnaryOps.CAST: lambda self,arg: FlopCounter(self.shape, self.consume_flops(), self.mem),   # cast uses no flops
  UnaryOps.BITCAST: lambda self,arg: FlopCounter(self.shape, self.consume_flops(), self.mem),   # bitcast uses no flops
  **{op:lambda self: FlopCounter(self.shape, self.consume_flops() + prod(self.shape), self.mem) for op in UnaryOps if op not in {UnaryOps.CAST, UnaryOps.BITCAST}},  # noqa: E501
  **{op:lambda self,y: FlopCounter(self.shape, self.consume_flops() + y.consume_flops() + prod(self.shape), {**self.mem, **y.mem}) for op in BinaryOps},  # noqa: E501
  **{op:lambda self,axis: FlopCounter(tuple(1 if i in axis else s for i,s in enumerate(self.shape)), self.consume_flops() + prod(self.shape), self.mem) for op in ReduceOps},  # noqa: E501
  TernaryOps.WHERE: lambda self,y,z: FlopCounter(self.shape, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape), {**self.mem, **y.mem, **z.mem})}  # noqa: E501

@functools.lru_cache(None)
def get_lazyop_info(ast:LazyOp) -> FlopCounter:
  @functools.lru_cache(None) # NOTE: this cache needs to be recreated for new ASTs
  def run_ast(ast): return InterpretedFlopCounter[ast.op](*([run_ast(x) for x in ast.src]+([ast.arg] if ast.arg is not None else [])))
  return run_ast(ast)

# **************** ops in python ****************

def hook_overflow(dv, fxn):
  def wfxn(*args):
    try: return fxn(*args)
    except OverflowError: return dv
  return wfxn

python_alu = {
  UnaryOps.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan,
  UnaryOps.EXP2: hook_overflow(math.inf, lambda x: 2**x),
  UnaryOps.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan,
  UnaryOps.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan,
  UnaryOps.RECIP: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  UnaryOps.NEG: lambda x: (not x) if isinstance(x, bool) else -x,
  BinaryOps.SHR: operator.rshift, BinaryOps.SHL: operator.lshift,
  BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add,
  BinaryOps.XOR: operator.xor, BinaryOps.MAX: max, BinaryOps.CMPNE: operator.ne, BinaryOps.CMPLT: operator.lt,
  BinaryOps.OR: operator.or_, BinaryOps.AND: operator.and_,
  BinaryOps.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0], BinaryOps.IDIV: lambda x, y: int(x/y) if y != 0 else x*math.inf,
  TernaryOps.MULACC: lambda x,y,z: (x*y)+z,
  TernaryOps.WHERE: lambda x,y,z: y if x else z}

def truncate_fp16(x):
  try:
    x = float(x)
    struct.pack("@e", x)
    return x
  except OverflowError: return math.copysign(math.inf, x)

truncate: Dict[DType, Callable] = {dtypes.bool: bool,
  # TODO: bfloat16
  dtypes.float16: truncate_fp16, dtypes.float32: lambda x: ctypes.c_float(x).value, dtypes.float64: lambda x: ctypes.c_double(x).value,
  dtypes.uint8: lambda x: ctypes.c_uint8(x).value, dtypes.uint16: lambda x: ctypes.c_uint16(x).value,
  dtypes.uint32: lambda x: ctypes.c_uint32(x).value, dtypes.uint64: lambda x: ctypes.c_uint64(x).value,
  dtypes.int8: lambda x: ctypes.c_int8(x).value, dtypes.int16: lambda x: ctypes.c_int16(x).value,
  dtypes.int32: lambda x: ctypes.c_int32(x).value, dtypes.int64: lambda x: ctypes.c_int64(x).value, dtypes.bigint: lambda x: x }

def exec_alu(op:Op, dtype:DType, operands): return truncate.get(dtype, lambda x: x)(python_alu[op](*operands))

# the living definition of LazyOps
def verify_lazyop(*ast:LazyOp) -> Dict[LazyOp, ShapeTracker]:
  sts: Dict[LazyOp, ShapeTracker] = {}
  def dfs(op:LazyOp, st:ShapeTracker):
    if op in sts: return
    for x in op.src: dfs(x, st)
    # only reduceop is allowed to change shape, limited to turning n to 1
    if op.op in ReduceOps:
      assert isinstance(op.arg, tuple)
      st = ShapeTracker.from_shape(tuple(1 if i in op.arg else s for i,s in enumerate(sts[op.src[0]].shape)))
    else:
      # movementops are pushed to the edges with LOAD
      if op.op in BufferOps: st = op.arg.st
      else: st = sts[op.src[0]]
      for x in op.src: assert sts[x].shape == st.shape, f"found implicit movement op {x.op} {sts[x].shape} != {op.op} {st.shape}"
    sts[op] = st
  for i, out in enumerate(ast):
    assert out.arg.idx == i, f"unexpected output buffer idx {out.arg.idx} != {i}"
    assert out.op is BufferOps.STORE, f"kernels must have stores as the output, got {out.op}"
    assert out.arg.st.size == ast[-1].arg.st.size, f"outputs must have the same size, got {out.arg.st.size}"
    dfs(out, out.arg.st)
  return sts
