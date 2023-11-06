from __future__ import annotations
import importlib, inspect, functools, pathlib
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Optional, Dict, Callable, Mapping
from tinygrad.helpers import prod, DEBUG, getenv, DType
from tinygrad.runtime.lib import RawBuffer
from tinygrad.shape.symbolic import Variable
from dataclasses import dataclass

from tinygrad.helpers import GlobalCounters  # noqa: F401

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: rdna3 only has RECIP and not DIV. DIV and POW are on the chopping block
class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class BufferOps(Enum): MEM = auto(); CONST = auto() # noqa: E702
# Ops below this line are not allowed in ASTs
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto(); AS_STRIDED = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker
  from tinygrad.lazy import LazyBuffer

@dataclass(frozen=True)
class MemBuffer:
  idx: int
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ConstBuffer:
  val: Any
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ScheduleItem:
  ast: LazyOp
  out: LazyBuffer
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

@dataclass(frozen=True)
class LazyOp:
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]
  arg: Any = None
  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  @property
  def buffers(self):
    buffers: Tuple[Union[LazyOp, LazyBuffer], ...] = ()
    try:  # NOTE: the linearizer's key function maps the buffers to ints, and LOCAL_BUFFER is used. we don't care about buffers in these cases
      for x in self.src: buffers += x.buffers
    except AttributeError: buffers = ()
    return buffers

  @property
  def key(self): return (self.op, tuple(map(lambda x: getattr(x, "key", x), self.src)), getattr(self.arg, "key", self.arg))

  def map_buffers(self, real_srcs: Mapping[Any, Union[LazyBuffer, LazyOp]]) -> LazyOp: return LazyOp(self.op, tuple([y.map_buffers(real_srcs) if y not in real_srcs else real_srcs[y] for y in self.src]), self.arg)
  def get_lazyops(self) -> List[LazyOp]: return [self] + [item for x in self.src for item in x.get_lazyops()]

  def replace_with_movement_ops(self:LazyOp, ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> 'LazyBuffer':
    assert self.op in BinaryOps or self.op in UnaryOps or self.op in TernaryOps
    srcs = [z.replace_with_movement_ops(ops) for z in self.src]
    return srcs[0].e(self.op, *srcs[1:], arg=self.arg)   # type: ignore

  @property
  def st(self): raise NotImplementedError
  @property
  def realized(self): raise NotImplementedError
  @property
  def children(self): raise NotImplementedError

  # movement ops
  def reshape(self, _): raise NotImplementedError
  def pad(self, _): raise NotImplementedError
  def expand(self, _): raise NotImplementedError
  def permute(self, _): raise NotImplementedError
  def shrink(self, _): raise NotImplementedError
  def stride(self, _): raise NotImplementedError

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
  def canonicalize(self, device:Optional[str]) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "") if device is not None else self.DEFAULT
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __getitem__(self, x:str) -> Union[Interpreted, Compiled]:
    x = x.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "buffer") and x in self._buffers][0]
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, None)
    if device_from_env: return device_from_env
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()

# **************** for Interpreted Buffers ****************

class Interpreted:
  def __init__(self, buffer, fxn_for_op: Dict[Op, Callable], to_underlying=lambda x: x._buf, from_underlying=None):
    self.buffer, self.fxn_for_op, self.to_underlying, self.from_underlying = buffer, fxn_for_op, to_underlying, from_underlying
    self.synchronize = lambda: None
    self.codegen = None

@functools.lru_cache(None)
def interpret_ast(device:Interpreted, ast:LazyOp) -> Callable:
  tglob: Dict[str, Any] = {}
  lines: List[str] = []
  f = device.fxn_for_op

  @functools.lru_cache(None)
  def gstr(x:Any, nm=None) -> str:
    ret = str(nm).replace(".", "_") if nm else f"m{len(tglob):04d}"
    tglob[ret] = x
    return ret

  @functools.lru_cache(None)
  def _interpret_ast(ast:LazyOp) -> str:
    if TernaryOps.MULACC in f and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)

    if MovementOps.AS_STRIDED in f and ast.op in BufferOps:
      # expand the shapetracker
      tmp = f"{gstr(f[ast.op], ast.op)}({gstr(ast.arg.val)}, {gstr(ast.arg.dtype)})" if ast.op == BufferOps.CONST else f"{gstr(f[ast.op], ast.op)}(inputs[{ast.arg.idx-1}])"
      for mop,arg in ast.arg.st.to_movement_ops(): tmp = f"{gstr(f[mop], mop)}({tmp}, {gstr(arg)})"
    else:
      inp = [_interpret_ast(src) for src in ast.src]
      tmp = f"{gstr(f[ast.op], ast.op)}({', '.join(inp + ([gstr(ast.arg)] if ast.arg else []))})"

    ret = f"a{len(lines)}"
    lines.append(f"  {ret} = {tmp}")
    return ret

  ret = _interpret_ast(ast)
  src = '\n'.join(['def run(*inputs):'] + lines + [f"  return {gstr(device.from_underlying, 'from_underlying')}({ret})" if device.from_underlying else f"  return {ret}"])
  if DEBUG >= 4: print(functools.reduce(lambda x,y: (x.replace(y[0], str(y[1])) if y[0][0:2] == "m0" else x), tglob.items(), src))
  exec(compile(src, "<ast>", "exec"), tglob) # pylint: disable=exec-used
  return tglob['run']

@dataclass
class FlopCounter:
  shape: Tuple[int, ...]
  dtype: DType
  flops: int
  mem: Dict[int, int]
  @property
  def mem_estimate(self): return sum(self.mem.values()) + self.dtype.itemsize*prod(self.shape)
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
InterpretedFlopCounter = Interpreted(FlopCounter, {
  BufferOps.MEM: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {arg.idx: arg.dtype.itemsize*arg.st.size()}), BufferOps.CONST: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {}),
  UnaryOps.CAST: lambda self,arg: FlopCounter(self.shape, arg[0], self.consume_flops(), self.mem),   # cast uses no flops
  **{op:lambda self: FlopCounter(self.shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in UnaryOps if op != UnaryOps.CAST},
  **{op:lambda self,y: FlopCounter(self.shape, max(self.dtype, y.dtype), self.consume_flops() + y.consume_flops() + prod(self.shape), {**self.mem, **y.mem}) for op in BinaryOps},
  **{op:lambda self,new_shape: FlopCounter(new_shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in ReduceOps},
  TernaryOps.WHERE: lambda self,y,z: FlopCounter(self.shape, y.dtype, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape), {**self.mem, **y.mem, **z.mem})})

@functools.lru_cache(None)
def get_lazyop_info(ast:LazyOp) -> FlopCounter: return interpret_ast(InterpretedFlopCounter, ast)(None)

# **************** for Compiled Buffers ****************

@dataclass
class Compiled:
  buffer: Type[RawBuffer]
  linearizer_opts: Any
  renderer: Any
  compiler: Any
  runtime: Any
  synchronize: Any = lambda: None  # noqa: E731
