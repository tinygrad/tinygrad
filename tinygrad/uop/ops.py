from __future__ import annotations
from typing import Any, Callable, cast, TYPE_CHECKING, Type, Sequence
import sys, time, functools, itertools, math, operator, hashlib, os, types, pickle, pathlib, inspect, weakref, collections
from dataclasses import dataclass
from enum import Enum, auto
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.mathtraits import MathTrait
from tinygrad.dtype import ConstType, ImageDType, dtypes, DType, truncate, PtrDType, least_upper_dtype, Invalid, InvalidType
from tinygrad.helpers import ContextVar, all_int, prod, getenv, all_same, Context, partition, temp, unwrap, T, argfix, Metadata, flatten, TRACEMETA
from tinygrad.helpers import PICKLE_BUFFERS, PROFILE, dedup, cdiv, cmod, diskcache_put, to_function_name, cpu_profile, TracingKey, VIZ, SPEC
from tinygrad.helpers import strip_parens
if TYPE_CHECKING:
  from tinygrad.device import Buffer, MultiBuffer

class AxisType(Enum):
  def __repr__(self): return str(self)
  GLOBAL = auto(); WARP = auto(); LOCAL = auto(); LOOP = auto(); GROUP_REDUCE = auto(); REDUCE = auto(); UPCAST = auto(); UNROLL = auto() # noqa: E702
  THREAD = auto()

range_start = {Ops.BUFFERIZE: 1, Ops.REDUCE: 1, Ops.STORE: 2, Ops.WMMA: 3}

# https://en.wikipedia.org/wiki/Identity_element
def identity_element(op:Ops, dt:DType) -> ConstType: return dtypes.as_const({Ops.ADD:0, Ops.MUL:1, Ops.MAX:dtypes.min(dt)}[op], dt)

# With True as the default, this matches the old symbolic behavior
def resolve(x:UOp|bool, default:bool=True):
  if isinstance(x, bool): return x
  assert x.dtype == dtypes.bool, "UOp in resolve must be bool"
  # NOTE: generating the text for the exception is expensive, so we do this
  return bool(sx.vmin) if (sx:=x.simplify()).vmin == sx.vmax else default

# smax/smin are replacements for max/min that preserve symbolic
def _suop(lst, uop_fxn, python_fxn):
  uops, nums = partition(lst, lambda x: isinstance(x, UOp))
  return ssimplify(functools.reduce(uop_fxn, uops + ([python_fxn(nums)] if nums else [])))
def smax(*lst) -> sint: return _suop(argfix(*lst), UOp.maximum, max)
def smin(*lst) -> sint: return _suop(argfix(*lst), UOp.minimum, min)
def srender(x:sint) -> str: return x.render() if isinstance(x, UOp) else str(x)

def ssimplify(uop:sint): return uop.ssimplify() if isinstance(uop, UOp) else uop
def sym_infer(uop: UOp|int, var_vals: dict[str, int]) -> int: return uop.sym_infer(var_vals) if isinstance(uop, UOp) else uop

def range_str(u:UOp) -> str: return '_'.join([str(x) if x >= 0 else "m"+str(-x) for x in u.arg[0:-1]])

# used for UOp and UPat
def pretty_print(x:Any, rep:Callable, srcfn=lambda x: x.src, cache=None, d=0)->str:
  def dfs(x:Any, cache:dict):
    for s in srcfn(x) or []:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(x, cache:={})
  if (cx:=cache.setdefault(x, [0,0,False]))[2]: return f"{' '*d} x{cx[0]}"
  cx[2], srcs = True, ('None' if srcfn(x) is None else ''.join(f'\n{pretty_print(s, rep, srcfn, cache, d+2)},' for s in srcfn(x)))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{rep(x)}" % srcs

class UOpMetaClass(type):
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:tuple[UOp,...]=tuple(), arg:Any=None, tag:Any=None,
               metadata:tuple[Metadata,...]|None=None, _buffer:Buffer|None=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg, tag), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = weakref.ref(created:=super().__call__(*key))
    if metadata is not None: all_metadata[created] = metadata
    # NOTE: this value is set by pickle when pickling a realized tensor
    if _buffer is not None:
      assert op is Ops.BUFFER, f"trying to set Buffer {_buffer} for {op}"
      buffers[created] = _buffer
    if SPEC:
      from tinygrad.uop.spec import full_spec
      with Context(IGNORE_OOB=1): ret = full_spec.rewrite(created)
      if cast(bool|None, ret) is not True: raise RuntimeError(f"SPEC ISSUE {ret}: {created}")
    return created

# some uops map to other stuff
buffers:weakref.WeakKeyDictionary[UOp, Buffer|MultiBuffer] = weakref.WeakKeyDictionary() # this maps BUFFER uops to their device Buffers
all_metadata:weakref.WeakKeyDictionary[UOp, tuple[Metadata, ...]] = weakref.WeakKeyDictionary() # TODO: should this be here?

# recursive_property replaces functools.cached_property in recursive UOp functions to prevent RecursionError
_NOT_FOUND = object()
class recursive_property(property):
  def __init__(self, fxn):
    self.fxn = fxn
    self.nm = "_RECURSIVE_PROPERTY_"+fxn.__name__
    self.__doc__ = fxn.__doc__
  def __get__(self, x:UOp|None, owner=None):
    if x is None: return self
    if (val:=x.__dict__.get(self.nm, _NOT_FOUND)) is _NOT_FOUND:
      for s in x.toposort(lambda z: not hasattr(z, self.nm)):
        s.__dict__[self.nm] = val = self.fxn(s)
    return val

# NOTE: this should be frozen, but frozen is slower
@dataclass(eq=False, slots=True)
class UOp(MathTrait, metaclass=UOpMetaClass):
  op:Ops
  dtype:DType = dtypes.void
  src:tuple[UOp, ...] = tuple()
  arg:Any = None
  tag:Any = None
  def __del__(self):
    if Ops is not None and self.op is Ops.BUFFER and (buffer:=buffers.get(self)) is not None: buffer.ref(-1)
    try: del UOpMetaClass.ucache[(self.op, self.dtype, self.src, self.arg, self.tag)]
    except AttributeError: pass
  def __reduce__(self):
    args = [self.op, self.dtype, self.src, self.arg, self.tag, self.metadata]
    if self.op is Ops.BUFFER and self.realized is not None and PICKLE_BUFFERS: args.append(self.realized)
    return UOp, tuple(args)
  def replace(self, **kwargs) -> UOp:
    new_args = (kwargs.pop("op", self.op), kwargs.pop("dtype", self.dtype), kwargs.pop("src", self.src),
                kwargs.pop("arg", self.arg), kwargs.pop("tag", self.tag))
    assert len(kwargs) == 0, f"unused kwargs in replace {list(kwargs)}"
    if (self.op, self.dtype, self.src, self.arg, self.tag) == new_args: return self
    return UOp(*new_args)
  def rtag(self, tag=True): return self.replace(tag=tag)
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(str((self.op, self.dtype, self.arg)).encode() + b"".join([s.key for s in self.src])).digest()
  def __repr__(self): return pretty_print(self, lambda x: f"{type(self).__name__}({x.op}, {x.dtype}, arg={x.argstr()}{x.tagstr()}, src=(%s))")
  def argstr(self): return f'({", ".join(map(str, self.arg))})' if self.op is Ops.REDUCE_AXIS else repr(self.arg)
  def tagstr(self): return f", tag={self.tag}" if self.tag is not None else ""

  def f(self, op, **kwargs): return UOp(op, dtype=kwargs.pop("dtype", self.dtype), src=(self,), **kwargs)

  @functools.cached_property
  def backward_slice(self:UOp) -> dict[UOp, None]:
    res: dict[UOp, None] = self.toposort()
    res.pop(self)
    return res

  @property
  def backward_slice_with_self(self:UOp) -> dict[UOp, None]: return {self:None, **self.backward_slice}
  def op_in_backward_slice_with_self(self, *ops:Ops): return any(x.op in ops for x in self.backward_slice_with_self)

  def toposort(self, gate:Callable|None=None) -> dict[UOp, None]:
    ret: dict[UOp, None] = {}
    stack: list[tuple[UOp, bool]] = [(self, False)] # each stack entry is (node, visited_flag)
    while stack:
      node, visited = stack.pop()
      if node in ret: continue
      if not visited:
        if gate is None or gate(node):
          stack.append((node, True))  # push node back on stack to process after its srcs
          for s in reversed(node.src): stack.append((s, False)) # push srcs on the stack
      else: ret[node] = None # second time i'm seeing this node, add it to returned toposort
    return ret

  # returns map of UOps to their consumers in the graph rooted by self
  def get_consumer_map(self) -> dict[UOp, dict[UOp, None]]:
    ret: dict[UOp, dict[UOp, None]] = {}
    for u in self.toposort():
      ret[u] = {}
      for s in u.src: ret[s][u] = None
    return ret

  def reverse_toposort(self, consumer_map) -> dict[UOp, None]:
    ret: dict[UOp, None] = {}
    stack: list[tuple[UOp, bool]] = [(x, False) for x in consumer_map if len(x.src) == 0]
    while stack:
      node, visited = stack.pop()
      if node in ret: continue
      if not visited:
        stack.append((node, True))  # push node back on stack to process after its srcs
        for s in consumer_map[node]: stack.append((s, False)) # push srcs on the stack
      else: ret[node] = None # second time i'm seeing this node, add it to returned toposort
    return ret

  @functools.cached_property
  def tuplize(self:UOp) -> tuple:
    return (self.op.value, self.arg, self.dtype,)+tuple([x.tuplize for x in self.src])

  @property
  def ptrdtype(self) -> PtrDType:
    if not isinstance(self.dtype, PtrDType): raise RuntimeError("ptrdtype called on UOp without PtrDType")
    return self.dtype

  # *** uop shape stuff ***

  @recursive_property
  def _shape(self) -> tuple[sint, ...]|None:
    match self.op:
      # late ops don't have shape
      case Ops.UNIQUE | Ops.DEVICE | Ops.RANGE | Ops.INDEX | Ops.LOAD | Ops.IF | Ops.BARRIER | Ops.CUSTOM | Ops.CUSTOMI | \
           Ops.VECTORIZE | Ops.VCONST | Ops.SUBSTITUTE | Ops.GEP | Ops.SPECIAL | Ops.UNROLL | Ops.PRECAST:
        return None

      # some ops init the shape
      case Ops.CONST | Ops.DEFINE_VAR | Ops.BIND: return () if self._device is not None else None
      case Ops.BUFFER: return (self.arg,)
      case Ops.BUFFER_VIEW: return (self.arg[0],)
      case Ops.BUFFERIZE: return tuple([int(r.vmax+1) for r in self.src[1:]])
      case Ops.DEFINE_GLOBAL | Ops.DEFINE_LOCAL | Ops.DEFINE_REG: return (self.ptrdtype.size,)

      # passthrough ops
      case Ops.REDUCE | Ops.MSTACK | Ops.MSELECT | Ops.DETACH | Ops.CONTIGUOUS | Ops.CONTIGUOUS_BACKWARD | Ops.FUSE: return self.src[0]._shape

      # ops with custom handling
      case Ops.KERNEL: return self.arg.ast._shape
      case Ops.STORE:
        if isinstance(self.dtype, PtrDType): return (self.ptrdtype.size,)
        if self.dtype is not dtypes.void: return self.src[0].src[0].shape
        return None

      # TODO: disallow shape changing bitcast
      case Ops.BITCAST:
        ps = self.src[0]._shape
        if ps is None: return None
        if (output_sz:=self.dtype.itemsize) != (input_sz:=self.src[0].dtype.itemsize): return ps[:-1]+(ssimplify((ps[-1]*input_sz) // output_sz),)
        return ps

      # TODO: disallow reshape from nothing. tested by TestOpenClip.test_multigpu_clip_score
      case Ops.RESHAPE:
        if self.src[0]._shape is None: return self.marg

    # movement ops change the shape. this is the logic from the old ShapeTracker
    # NOTE: ssimplify is required because the shape needs to be canonical for broadcasting and same shape checking
    if self.op in GroupOp.Movement.union({Ops.MULTI, Ops.REDUCE_AXIS, Ops.WMMA}):
      ps = self.src[0]._shape
      # TODO: WMMA is used for both axis WMMA and op WMMA. fix this and remove this hack. tested by BERT on AMD LLVM
      if ps is None and self.op is Ops.WMMA: return None
      if ps is None: raise RuntimeError(f"movement op {self.op} requires shape")
      match self.op:
        case Ops.RESHAPE:
          if not all(x >= 0 for x in self.marg): raise ValueError(f"shape can't contain negative numbers {self.marg}")
          if prod(ps) != prod(self.marg): raise ValueError(f"bad reshape: {ps} -> {self.marg}")
          return self.marg
        case Ops.EXPAND:
          if len(ps) != len(self.marg) or not all(s==ns or (s==1 and ns>=0) for s,ns in zip(ps, self.marg)):
            raise ValueError(f"bad expand: {ps} -> {self.marg}")
          return self.marg
        case Ops.PERMUTE:
          if sorted(self.marg) != list(range(len(ps))): raise ValueError(f"invalid permutation {self.marg} of len {len(ps)}")
          return tuple(ps[i] for i in self.marg)
        case Ops.PAD:
          # TODO: why do i need resolve here?
          if len(ps) != len(self.marg) or not all(resolve(b>=0) and resolve(e>=0) for b,e in self.marg): raise ValueError(f"invalid pad {self.marg}")
          return tuple(ssimplify(s+b+e) for s,(b,e) in zip(ps, self.marg))
        case Ops.SHRINK:
          # TODO: why do i need resolve here?
          if len(ps) != len(self.marg) or not all(resolve(0<=b) and resolve(b<=e) and resolve(e<=s) for s,(b,e) in zip(ps, self.marg)):
            raise ValueError(f"invalid shrink {self.marg} for {ps}")
          return tuple(ssimplify(e-s) for s,e in self.marg)
        case Ops.FLIP:
          if len(ps) != len(self.marg) or not all(isinstance(x, bool) for x in self.marg): raise ValueError(f"bad flip on {ps}, {self.marg}")
          return ps
        case Ops.MULTI: return tuple(s*len(self.device) if a == self.axis else s for a,s in enumerate(ps))
        case Ops.REDUCE_AXIS | Ops.WMMA:
          axis_arg = self.arg[1] if self.op is Ops.REDUCE_AXIS else self.arg[7]
          if not isinstance(axis_arg, tuple) or not all(isinstance(x, int) and x>=0 and x<len(ps) for x in axis_arg):
            raise ValueError(f"invalid type for axis: {axis_arg}")
          return tuple(1 if i in axis_arg else s for i,s in enumerate(ps))

    # elementwise ops keep the shape the same. all inputs with shape must match
    if self.op in (GroupOp.Elementwise-{Ops.BITCAST}).union({Ops.COPY, Ops.ASSIGN, Ops.NOOP, Ops.SINK, Ops.ALLREDUCE}):
      # TODO: remove this hack for 3 op assign
      input_shapes = [x._shape for x in (self.src[:2] if self.op is Ops.ASSIGN else self.src) if x._shape is not None]
      if len(input_shapes) == 0: return None
      if not all_same(input_shapes): raise RuntimeError(f"shape mismatch at {self.op}: {input_shapes}")
      return input_shapes[0]

    # all Ops must be explicitly handled
    raise NotImplementedError(f"no shape handling for {self.op} with {self.dtype}")

  @property
  def shape(self) -> tuple[sint, ...]:
    if (ret:=self._shape) is None: raise RuntimeError(f"shape requested, but {self.op} doesn't have a shape")
    return ret

  @property
  def size(self) -> int: return prod([int(x.vmax) if isinstance(x, UOp) else x for x in self.shape])

  # determine what ranges this is in
  @recursive_property
  def _ranges(self) -> dict[UOp, None]:
    ret: dict[UOp, None] = {}
    if self.op in range_start.keys():
      for s in self.src[:range_start[self.op]]: ret.update(s.ranges)
      for s in UOp.sink(*self.src[range_start[self.op]:]).ranges:
        if s in ret: del ret[s]
    else:
      for s in self.src: ret.update(s.ranges)
    return ret

  @property
  def ranges(self) -> dict[UOp, None]:
    if self.op is Ops.RANGE: return {self:None}
    return self._ranges

  # *** uop evaluation ***

  def simplify(self, tracked=False, full_symbolic=True):
    # late import!
    from tinygrad.uop.symbolic import symbolic, commutative
    with Context(TRACK_MATCH_STATS=0 if not tracked else TRACK_MATCH_STATS.value):
      return graph_rewrite(self, symbolic if full_symbolic else commutative, name="simplify")
  def ssimplify(self) -> UOp|ConstType: return ret.arg if (ret:=self.simplify()).op is Ops.CONST else ret
  def _eval(self, dtype, expected_type:Type[T]) -> T:
    assert self.dtype in dtype, f"eval with wrong dtype {self}"
    vmin, vmax = (simple_self:=self.simplify())._min_max
    if vmin != vmax: raise ValueError(f"eval failed to be a single number, range is {vmin} to {vmax} in {simple_self.render()}")
    assert isinstance(vmin, expected_type), f"vmin is wrong dtype {type(vmin)} != {expected_type}"
    return vmin
  def __bool__(self): return self._eval((dtypes.bool,), bool)
  def __int__(self): return self._eval(dtypes.ints, int)
  def __float__(self): return self._eval(dtypes.floats, float)
  def substitute(self, dvars:dict[UOp, UOp], name:str|None=None, extra_pm:PatternMatcher|None=None):
    dvars = {k:v for k,v in dvars.items() if k is not v}
    if len(dvars) == 0: return self
    with Context(TRACK_MATCH_STATS=(0 if name is None else TRACK_MATCH_STATS.value)):
      return graph_rewrite(self, (extra_pm+_substitute) if extra_pm is not None else _substitute, dvars, bottom_up=True, name=name)

  # *** uop tracing stuff ***

  @recursive_property
  def trace_num(self):
    num = next(ucount)
    # KERNEL also has a UOp in the arg
    arg = type(self.arg)(self.arg.ast.trace_num, self.arg.metadata) if self.op is Ops.KERNEL else self.arg
    uop_fields[num] = (self.op, self.dtype, tuple(s.trace_num for s in self.src), arg, self.tag)+((self.metadata,) if TRACEMETA>=2 else ())
    return num

  # *** uop syntactic sugar ***

  def sink(*srcs:UOp|None, **kwargs):  # pylint: disable=no-self-argument
    return UOp(Ops.SINK, dtypes.void, tuple([x for x in srcs if x is not None]), **kwargs)
  def detach(self): return UOp(Ops.DETACH, self.dtype, (self,))
  def index(self, *srcs:UOp|None, **kwargs):
    return UOp(Ops.INDEX, kwargs.pop("dtype", self.dtype), (self,)+tuple([x for x in srcs if x is not None]), **kwargs)
  def __getitem__(self, idx): return self.index(idx)
  def const_like(self, b:ConstLike):
    # constants can optionally have a DEVICE source
    return UOp.const(self.dtype, b, device=self._device, shape=self._shape)
  def broadcast(self, count:int):
    assert self.dtype.count == 1
    if count == 1: return self
    return UOp(Ops.VECTORIZE, self.dtype.vec(count), (self,)*count)
  def cast(self, dtype:DType):
    # TODO: we shouldn't have to check for dtype.count == 1 here, but CAST is misused in AMD LLVM
    if dtype.count == 1 and dtype.count != self.dtype.count: dtype = dtype.vec(self.dtype.count)
    if self.dtype == dtype: return self
    return UOp(Ops.CAST, dtype, (self,))
  def bitcast(self, dtype:DType): return UOp(Ops.BITCAST, dtype, (self,))
  def gep(self, i:tuple[int, ...]|int):
    if isinstance(i, tuple) and len(i) == 1: return self.gep(i[0])
    if isinstance(i, int):
      # NOTE: these are just shortcuts to not have to create and fold later
      if self.op is Ops.VECTORIZE: return self.src[i]
      if self.op is Ops.VCONST: return UOp.const(self.dtype.scalar(), self.arg[i])
      if self.op is Ops.CONST: return UOp.const(self.dtype.scalar(), self.arg)
      i = (i,)
    return UOp(Ops.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
  def load(self, *src:UOp, **kwargs): return UOp(Ops.LOAD, dtype=kwargs.pop("dtype", self.dtype.base), src=(self,)+src, **kwargs)
  def store(self, *src:UOp, **kwargs): return UOp(Ops.STORE, kwargs.pop("dtype", dtypes.void), (self,)+src, **kwargs)
  def assign(self, x:UOp): return UOp(Ops.ASSIGN, self.dtype, (self, x))
  def barrier(self, *src:UOp): return UOp(Ops.BARRIER, src=(self,)+src)
  def alu(self, op, *src:UOp, **kwargs):
    out_dtype = (self, *src)[-1].dtype
    if op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    return UOp(op, out_dtype, (self,)+src, **kwargs)
  @staticmethod
  def const(dtype:DType, b:ConstLike, device:str|tuple[str, ...]|None=None, shape:tuple[sint, ...]|None=None, src=None):
    if isinstance(b, UOp): return b.unbind()[0] if b.op is Ops.BIND else b
    if isinstance(b, tuple) and all_same(b): b = b[0]  # doesn't have to be a VCONST if they are all the same
    ret = UOp(Ops.VCONST if isinstance(b, tuple) else Ops.CONST, dtype, arg=dtypes.as_const(b, dtype), src=() if src is None else (src,))
    if device is not None: ret = ret.replace(src=(UOp(Ops.DEVICE, arg=device),))
    if shape is not None: ret = ret.reshape((1,)*len(shape)).expand(shape)
    return ret
  @staticmethod
  def range(end:sint, *arg):
    if len(arg) == 0: raise RuntimeError("range needs an arg")
    if len(arg) == 1: arg = arg+(AxisType.LOOP,)
    return UOp(Ops.RANGE, dtype=dtypes.index, src=(sint_to_uop(end),), arg=arg)
  def r(self, op:Ops, axis:tuple[int, ...]):
    axis = tuple(sorted([x for x in axis if resolve(self.shape[x] != 1)]))
    return UOp(Ops.REDUCE_AXIS, self.dtype, (self,), (op, axis)) if len(axis) else self
  @staticmethod
  def invalid(count=1): return UOp(Ops.CONST, dtypes.index.vec(count), src=(), arg=Invalid)
  def valid(self, cond): return self if cond.op is Ops.WHERE and cond.arg else cond.where(self, UOp.invalid(self.dtype.count))
  def get_idx(self) -> UOp:
    assert self.dtype.scalar() is dtypes.index, "Can only call get_idx on index dtype"
    return self.src[1] if self.op is Ops.WHERE and self.src[2].arg is Invalid else self
  def get_valid(self) -> UOp:
    assert self.dtype.scalar() is dtypes.index, "Can only call get_valid on index dtype"
    return self.src[0] if self.op is Ops.WHERE and self.src[2].arg is Invalid else UOp.const(dtypes.bool, self.arg is not Invalid)
  def reduce(self, *src:UOp, **kwargs): return UOp(Ops.REDUCE, kwargs.pop('dtype', self.dtype), src=(self,)+src, **kwargs)

  def is_contiguous(self):
    # TODO: this is is_realized
    if self.op is Ops.RESHAPE: return self.src[0].is_contiguous()
    return self.op is Ops.BUFFER

  def contiguous(self, *args, **kwargs):
    if self.is_contiguous(): return self
    return UOp(Ops.CONTIGUOUS, dtype=self.dtype, src=(self,)+args, **kwargs)
  def contiguous_backward(self): return self.alu(Ops.CONTIGUOUS_BACKWARD)
  def bufferize(self, *args, **kwargs): return UOp(Ops.BUFFERIZE, dtype=self.dtype, src=(self,)+args, **kwargs)
  def fuse(self): return self.alu(Ops.FUSE)
  def allreduce(self, op, device:str|tuple[str, ...]|UOp):
    assert isinstance(self.device, tuple), f"allreduce must be on tuple {self.device} isn't"
    return UOp(Ops.ALLREDUCE, self.dtype, (self, UOp(Ops.DEVICE, arg=device) if not isinstance(device, UOp) else device), op)
  def overflows(self, dtype:DType) -> bool: return self.vmin < dtype.min or dtype.max < self.vmax

  # *** ShapeTracker helpers ***

  def split_uop(self:UOp, sep:Ops):
    if self.op is sep:
      for s in self.src: yield from s.split_uop(sep)
    else: yield self

  # *** from MultiLazyBuffer ***

  def multi(self, axis:int|None):
    assert isinstance(self.device, tuple), f"multi device must be tuple, {self.device} isn't"
    assert axis is not None, "multi None is no longer supported"
    return UOp(Ops.MULTI, self.dtype, (self,), axis)

  @property
  def bounds(self):
    if self.axis is None: raise RuntimeError("bounds is not defined when axis is None")
    return tuple(itertools.pairwise(itertools.accumulate([self.src[0].shape[self.axis] for _ in self.device], initial=0)))

  @functools.cached_property
  def axis(self) -> int|None:
    if self.op is Ops.MULTI: return self.arg
    # NOTE: they all have to share an axis, we always choose [-1]
    if self.op in GroupOp.ALU: return axes[-1] if (axes := dedup([x.axis for x in self.src if x.axis is not None])) else None
    if len(self.src) == 0: return None
    src_axis = self.src[0].axis
    if self.op is Ops.REDUCE_AXIS: return None if src_axis is not None and src_axis in self.arg[1] else src_axis
    if self.op is Ops.RESHAPE:
      if src_axis is None: return None
      arg_acc:list[sint] = list(itertools.accumulate(self.marg, operator.mul, initial=1))
      # new_axis is the last one that preserves prod(prior to new_axis) and must not move items between shards
      # TODO: what to do about shrinking to self.shape[self.axis]==1 len(self.real_lbs)==1?
      return len(arg_acc) - arg_acc[::-1].index(prod(self.src[0].shape[:src_axis])) - 1
    if self.op is Ops.PERMUTE: return self.marg.index(src_axis) if src_axis is not None else None
    return src_axis

  def _unshard(self, axis:int) -> UOp:
    bsz, dcount = self.shape[axis], len(self.device)
    dnum = UOp.variable("_device_num", 0, dcount-1)
    return self.pad(tuple((0,0) if a != axis else (bsz*dnum, bsz*(dcount-1) - bsz*dnum) for a in range(len(self.shape))))

  def _shard(self, axis:int) -> UOp:
    dcount = len(self.device)
    dnum = UOp.variable("_device_num", 0, dcount-1)
    if self.shape[axis] % dcount != 0: raise RuntimeError(f"multi axis uneven: {self.shape[axis]=} {axis=} {dcount=}")
    sz = self.shape[axis] // dcount
    return self.shrink(tuple((0,s) if i != axis else (dnum*sz,dnum*sz+sz) for i,s in enumerate(self.shape)))
  def shard(self, devices:tuple[str, ...], axis:int) -> UOp: return self.copy_to_device(devices)._shard(axis).multi(axis)

  # *** from LazyBuffer ***

  def copy_to_device(self, device:str|tuple[str, ...]|UOp, arg=None):
    assert arg is None or isinstance(self.device, tuple)
    inp = self if arg is None else UOp(Ops.MSELECT, self.dtype, src=(self,), arg=arg)
    return UOp(Ops.COPY, self.dtype, (inp, UOp(Ops.DEVICE, arg=device) if not isinstance(device, UOp) else device))
  def mselect(self, arg:int) -> UOp: return UOp(Ops.MSELECT, self.dtype, (self,), arg)
  @property
  def metadata(self) -> tuple[Metadata, ...]|None: return all_metadata.get(self, None)

  # *** uop movement ops ***

  @property
  def base(self) -> UOp:
    if self.op in GroupOp.Movement: return self.src[0].base
    if self.op is Ops.MULTI: return self.src[0].base  # MULTI is really a VIEW
    return self

  # like gep, but might return an integer
  def sgep(self, i:int) -> sint:
    match self.op:
      case Ops.CONST: return self.arg
      case Ops.VCONST: return self.arg[i]
      case Ops.VECTORIZE: return cast(sint, self.src[i].ssimplify())
      case _: raise RuntimeError(f"no sgep on {self.op}")

  @functools.cached_property
  def marg(self):
    match self.op:
      case Ops.RESHAPE | Ops.EXPAND: return tuple(self.src[1].sgep(i) for i in range(self.src[1].dtype.count))
      case Ops.PAD | Ops.SHRINK: return tuple((self.src[1].sgep(i), self.src[2].sgep(i)) for i in range(self.src[1].dtype.count))
      case Ops.PERMUTE | Ops.FLIP: return self.arg
      case _: raise RuntimeError(f"{self.op} is not a MovementOp")

  def _mop(self, op:Ops, arg, same_shape_noop:bool=False) -> UOp:
    match op:
      case Ops.RESHAPE | Ops.EXPAND: src_args = [arg]
      case Ops.PAD | Ops.SHRINK: src_args = list(zip(*arg))
      case Ops.PERMUTE | Ops.FLIP: src_args = []
      case _: raise RuntimeError(f"{op} is not a MovementOp")
    usrcs = []
    for arg in src_args:
      if len(arg) == 0: usrcs.append(UOp(Ops.VECTORIZE, dtypes.index.vec(0)))
      elif all(isinstance(x, int) for x in arg): usrcs.append(UOp.const(dtypes.index.vec(len(arg)), arg))
      else: usrcs.append(UOp(Ops.VECTORIZE, dtypes.index.vec(len(arg)), tuple(UOp.const(dtypes.index, x) if isinstance(x, int) else x for x in arg)))
    ret = UOp(op, self.dtype, (self,)+tuple(usrcs), arg if len(usrcs) == 0 else None)
    # for all movement ops, we check shape property
    if ret.shape == self.shape and same_shape_noop: return self
    return ret

  # in these four, if the shape doesn't change we can return self
  def forced_reshape(self, arg:tuple[sint, ...]): return self._mop(Ops.RESHAPE, arg, same_shape_noop=False)
  def reshape(self, arg:tuple[sint, ...]): return self._mop(Ops.RESHAPE, arg, same_shape_noop=True)
  def expand(self, arg:tuple[sint, ...]): return self._mop(Ops.EXPAND, arg, same_shape_noop=True)
  def shrink(self, arg:tuple[tuple[sint, sint], ...]): return self._mop(Ops.SHRINK, arg, same_shape_noop=True)
  def pad(self, arg:tuple[tuple[sint, sint], ...]): return self._mop(Ops.PAD, arg, same_shape_noop=True)

  # in these two, we have custom logic to check if they are a no-op
  def permute(self, arg:tuple[int, ...]): return self._mop(Ops.PERMUTE, arg, same_shape_noop=False) if arg != tuple(range(len(self.shape))) else self
  def flip(self, arg:tuple[bool, ...]): return self._mop(Ops.FLIP, arg, same_shape_noop=False) if any(arg) and len(arg) == len(self.shape) else self

  # *** uop UNIQUE ***

  # TODO: use this in Buffer
  unique_num = itertools.count(0)
  @staticmethod
  def unique(): return UOp(Ops.UNIQUE, arg=next(UOp.unique_num))

  # *** uop Buffer stuff ***

  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType): return UOp(Ops.BUFFER, dtype, (UOp.unique(), UOp(Ops.DEVICE, arg=device)), size)
  @property
  def device(self) -> str|tuple[str, ...]: return cast(str|tuple[str, ...], unwrap(self._device))
  @recursive_property
  def _device(self) -> str|tuple[str, ...]|None:
    if self.op is Ops.DEVICE: return self.arg
    if self.op is Ops.BUFFERIZE: return self.arg.device
    if self.op is Ops.MSELECT:
      assert isinstance(self.src[0].device, tuple), "mselect must be on tuple device"
      return self.src[0].device[self.arg]
    if self.op is Ops.MSTACK: return tuple(cast(str, x.device) for x in self.src)
    if self.op in {Ops.COPY, Ops.BUFFER, Ops.ALLREDUCE}: return self.src[1].device
    for x in self.src:
      if x._device is not None: return x._device
    return None
  @property
  def buf_uop(self) -> UOp:
    if self.op is Ops.BUFFER: return self
    if self.op is Ops.MSELECT: return self.src[0].buf_uop.mselect(self.arg)
    if self.op is Ops.MSTACK: return UOp(Ops.MSTACK, self.dtype, src=tuple(x.buf_uop for x in self.src))
    assert self.op is Ops.ASSIGN, f"must be ASSIGN {self.op}"
    return self.src[0].base

  def as_buf(self) -> UOp:
    if self.op is Ops.MSELECT: return self.src[0].as_buf().mselect(self.arg)
    if self.op is Ops.MSTACK: return UOp(Ops.MSTACK, self.dtype, src=tuple(x.as_buf() for x in self.src))
    # TODO: this should be the only one of these. this is the one RANGEIFY uses
    s = self
    while len(s.src) and s.op not in {Ops.BUFFER, Ops.BUFFERIZE, Ops.MSTACK}: s = s.src[0]
    return s

  @property
  def buffer(self) -> Buffer|MultiBuffer:
    from tinygrad.device import Buffer, MultiBuffer
    if self is not self.base:
      assert self.op is Ops.RESHAPE, f"can only be RESHAPE {self}"
      return self.src[0].buffer
    if self.op is Ops.MSELECT:
      ret = self.src[0].buffer
      assert isinstance(ret, MultiBuffer)
      return ret.bufs[self.arg]
    if self.op is Ops.MSTACK:
      ret = MultiBuffer.__new__(MultiBuffer)
      ret.bufs = [cast(Buffer, x.buffer) for x in self.src]
      assert all_same([x.size for x in ret.bufs]) and all_same([x.dtype for x in ret.bufs]), "multibuffers mismatch buffers"
      return ret
    assert self.op is Ops.BUFFER, f"must be BUFFER {self.op}"
    if (cret:=buffers.get(self)) is not None: return cret
    rdtype = self.dtype if isinstance(self.dtype, ImageDType) else self.dtype.base
    if isinstance(self.device, tuple): ret = MultiBuffer(self.device, self.size, rdtype).ref(1)
    else: ret = Buffer(self.device, self.size, rdtype).ref(1)
    buffers[self] = ret
    return ret
  @property
  def realized(self) -> Buffer|MultiBuffer|None:
    # NOTE: this is used by the JIT to determine which inputs we capture
    return self.buffer if self.op in {Ops.BUFFER, Ops.MSTACK} and self.buffer.is_allocated() else None
  @property
  def is_realized(self) -> bool:
    return all(x.base.realized is not None for x in self.base.src) if self.base.op is Ops.MULTI else self.base.realized is not None

  # *** uop Variable stuff ***

  @staticmethod
  def variable(name:str, min_val:ConstType, max_val:ConstType, dtype:DType=dtypes.index) -> UOp:
    assert not isinstance(min_val, UOp) and not isinstance(max_val, UOp), f"can't create Variable {name} with {min_val}/{max_val}"
    return UOp(Ops.DEFINE_VAR, dtype, arg=(name, min_val, max_val))
  @property
  def expr(self) -> str:
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    return self.arg[0]
  def bind(self, val:int|UOp):
    assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    uval = self.const_like(val) if isinstance(val, int) else val
    assert self.arg[1] <= uval.vmin and uval.vmax <= self.arg[2], f"bind {val} not in range [{self.arg[1]}, {self.arg[2]}]"
    return UOp(Ops.BIND, self.dtype, (self, uval))
  def unbind(self) -> tuple[Variable, int]:
    assert self.op is Ops.BIND and self.src[0].op is Ops.DEFINE_VAR and self.src[1].op is Ops.CONST, f"can't unbind {self}"
    return self.src[0], self.src[1].arg
  def unbind_all(self) -> tuple[UOp, dict[Variable, int]]:
    ret:dict[Variable, int] = {}
    return graph_rewrite(self, pm_unbind, ctx=ret), ret
  @property
  def val(self) -> int: return self.unbind()[1]
  def vars(self) -> set[UOp]:
    bound_vars = set([x for x in self.toposort() if x.op is Ops.BIND and x.src[0].op is Ops.DEFINE_VAR])
    bound_var_base = set(x.src[0] for x in bound_vars)
    all_vars = set([x for x in self.toposort() if x.op is Ops.DEFINE_VAR])
    return bound_vars.union(set([x for x in all_vars if x not in bound_var_base]))
  def variables(self) -> list[Variable]:
    return sorted(set([x.unbind()[0] if x.op is not Ops.DEFINE_VAR else x for x in self.vars()]), key=lambda v: v.arg)

  # *** uop symbolic stuff ***

  def is_increasing(self:UOp) -> bool:
    # is f a monotonically increasing function regards its input
    if self.op in GroupOp.Irreducible: return True
    if self.op is Ops.ADD: return self.src[0].is_increasing() and self.src[1].is_increasing()
    if self.op in (Ops.MUL, Ops.IDIV) and self.src[1].op is Ops.CONST and self.src[1].arg >= 0: return self.src[0].is_increasing()
    return False  # False if not sure
  def const_factor(self) -> int:
    """largest known int that divides self"""
    # TODO: for negatives it's not the largest
    if self.op is Ops.CONST: return self.arg
    if self.op is Ops.VCONST: return math.gcd(*self.arg)
    if self.op is Ops.ADD: return math.gcd(self.src[0].const_factor(), self.src[1].const_factor())
    if self.op is Ops.MUL: return self.src[0].arg if self.src[0].op is Ops.CONST else self.src[1].arg if self.src[1].op is Ops.CONST else 1
    return 1
  def divides(self, v:int) -> UOp|None:
    if v==1: return self
    if self.op is Ops.CONST: return self.const_like(self.arg//v) if self.arg%v == 0 else None
    if self.op is Ops.VCONST: return self.const_like(tuple(x//v for x in self.arg)) if all(x%v == 0 for x in self.arg) else None
    if self.op is Ops.ADD: return d0+d1 if (d0:=self.src[0].divides(v)) is not None and (d1:=self.src[1].divides(v)) is not None else None
    if self.op is Ops.MUL:
      if (d0:=self.src[0].divides(v)) is not None: return d0 * self.src[1]
      if (d1:=self.src[1].divides(v)) is not None: return self.src[0] * d1
    return None # generic None if we aren't sure
  def pop_const(self, op=Ops.ADD) -> tuple[UOp, ConstType]:
    return (self.src[0], self.src[1].arg) if self.op is op and self.src[1].op is Ops.CONST else (self, identity_element(op, self.dtype))
  @staticmethod
  def gcd(*uops: UOp) -> UOp:
    terms, factors = zip(*[(u.divides(f:=u.const_factor()),f) for u in uops])
    count = functools.reduce(operator.and_, [collections.Counter(term.split_uop(Ops.MUL)) for term in terms])
    return math.prod([*count.elements(), terms[0].const_like(math.gcd(*factors))])  # put the const at the top
  def divide_exact(self, v:UOp) -> UOp|None:
    if self is v: return self.const_like(1)
    if self.op is Ops.ADD: return None if (s0:=self.src[0].divide_exact(v)) is None or (s1:=self.src[1].divide_exact(v)) is None else s0+s1
    if v.op is Ops.CONST: return self.divides(v.arg)
    if self.op is Ops.MUL:
      (fac, const), (div_fac, div_const) = self.pop_const(Ops.MUL), v.pop_const(Ops.MUL)
      new_count = collections.Counter(fac.split_uop(Ops.MUL))
      new_count.subtract(div_fac.split_uop(Ops.MUL))
      if const%div_const==0 and all(v>=0 for v in new_count.values()): return math.prod([*new_count.elements(), self.const_like(const//div_const)])
    return None # generic None if we aren't sure
  @property
  def vmin(self) -> ConstType: return self._min_max[0]
  @property
  def vmax(self) -> ConstType: return self._min_max[1]
  @functools.cached_property
  def _min_max(self) -> tuple[ConstType, ConstType]:
    if self.op in GroupOp.Binary and not dtypes.is_float(self.dtype):
      (s0_vmin, s0_vmax), (s1_vmin, s1_vmax) = self.src[0]._min_max, self.src[1]._min_max
      if self.op is Ops.ADD: return s0_vmin+s1_vmin, s0_vmax+s1_vmax
      if self.op is Ops.SUB: return s0_vmin-s1_vmax, s0_vmax-s1_vmin
      if self.op is Ops.AND and s1_vmin == s1_vmax and s0_vmin >= 0 and s1_vmin >= 0: return min(0, s0_vmin), min(s0_vmax, s1_vmax)
      if self.op is Ops.MUL: return min(vals:=(s0_vmin*s1_vmin, s0_vmin*s1_vmax, s0_vmax*s1_vmin, s0_vmax*s1_vmax)), max(vals)
      # SHL/SHR on consts only
      if self.op is Ops.SHL and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] << t[2], t[1] << t[2]
      if self.op is Ops.SHR and s1_vmin == s1_vmax and all_int(t:=(s0_vmin, s0_vmax, s1_vmin)): return t[0] >> t[2], t[1] >> t[2]
      if self.op is Ops.MOD:
        if s1_vmin > 0: return (0, s1_vmax-1) if s0_vmin >= 0 else (-(s1_vmax-1), 0) if s0_vmax <= 0 else (-(s1_vmax-1), s1_vmax-1)
        if s1_vmax < 0: return (0, -s1_vmin-1) if s0_vmin >= 0 else (-(-s1_vmin-1), 0) if s0_vmax <= 0 else (-(-s1_vmin-1), -s1_vmin-1)
      if self.op is Ops.IDIV:
        assert isinstance(s0_vmin, int) and isinstance(s0_vmax, int) and isinstance(s1_vmin, int) and isinstance(s1_vmax, int)
        if s1_vmin*s1_vmax>0:
          return min(vals:=(cdiv(s0_vmin, s1_vmin), cdiv(s0_vmin, s1_vmax), cdiv(s0_vmax, s1_vmin), cdiv(s0_vmax, s1_vmax))), max(vals)
      if self.op is Ops.MAX: return max(s0_vmin, s1_vmin), max(s0_vmax, s1_vmax)
      if self.op is Ops.CMPLT: return (s0_vmax<s1_vmin, s0_vmin<s1_vmax)
      if self.op is Ops.CMPNE: return ((s0_vmax < s1_vmin) or (s1_vmax < s0_vmin), not (s0_vmin == s0_vmax == s1_vmin == s1_vmax))
      if self.dtype == dtypes.bool:
        if self.op is Ops.OR: return s0_vmin or s1_vmin, s0_vmax or s1_vmax
        if self.op is Ops.AND: return s0_vmin and s1_vmin, s0_vmax and s1_vmax
    # float has NAN issue and we use explicit NAN in transcendental
    if self.op is Ops.WHERE and dtypes.is_int(self.dtype): return min(self.src[1].vmin, self.src[2].vmin), max(self.src[1].vmax, self.src[2].vmax)
    # NOTE: returned UOp is assumed to be CONST
    if self.op is Ops.DEFINE_VAR and self.arg: return self.arg[1], self.arg[2]
    if self.op in (Ops.RANGE, Ops.SPECIAL): return 0, (self.src[0]-1).vmax
    if self.op is Ops.BIND: return self.src[0]._min_max # ignore the bound value
    if self.op in {Ops.UNROLL, Ops.VECTORIZE}: return min(x.vmin for x in self.src), max(x.vmax for x in self.src)
    if self.op is Ops.CONST and self.arg is not Invalid: return self.arg, self.arg
    if self.op is Ops.VCONST and Invalid not in self.arg: return (min(self.arg), max(self.arg))
    if self.op is Ops.GEP: return self.src[0]._min_max
    # TODO: CAST to bool/unsigned is not monotone, still some case can be simplified
    if self.op is Ops.CAST and self.dtype in dtypes.floats+dtypes.sints+(dtypes.index,):
      return max(dtypes.min(self.dtype), self.src[0].vmin), min(self.src[0].vmax, dtypes.max(self.dtype))
    return dtypes.min(self.dtype), dtypes.max(self.dtype)

  @functools.cached_property
  def _sym_fxn(self):
    sself = self.simplify()
    varnames = tuple(x.arg[0] for x in sself.toposort() if x.op is Ops.DEFINE_VAR)
    # TODO: sanitize varnames, or don't use naked eval while staying fast
    return eval("lambda "+','.join(varnames)+": "+sself.render(pm=renderer_infer)), varnames  # pylint: disable=eval-used

  def sym_infer(self, var_vals:dict[str, int]):
    fxn, varnames = self._sym_fxn
    return fxn(**{k:v for k,v in var_vals.items() if k in varnames})

  def render(self, simplify=True, pm:PatternMatcher|None=None) -> str:
    with Context(TRACK_MATCH_STATS=0, SPEC=0):
      ret = graph_rewrite(self.simplify() if simplify else self, renderer if pm is None else pm)
    return ret.arg if ret.op is Ops.NOOP else str(ret)

@dataclass(frozen=True)
class KernelInfo:
  name: str = "test"            # name of the kernel
  axis_types: tuple[AxisType, ...] = tuple()
  dont_use_locals: bool = False # don't use local indexing
  applied_opts: tuple = tuple()
  opts_to_apply: tuple|None = None
  @property
  def function_name(self): return to_function_name(self.name)

# ******** ops in python ********

def safe_exp2(x):
  try: return 2 ** x
  except OverflowError: return math.inf

def safe_pow(x, y):
  try: return math.nan if isinstance(p:=pow(x, y), complex) else p
  except ZeroDivisionError: return math.inf
  except ValueError: return math.inf if x > 0 else -math.inf

python_alu: dict[Ops, Callable]  = {
  Ops.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, Ops.EXP2: safe_exp2,
  Ops.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, Ops.RECIP: lambda x: 1/x if x != 0 else math.copysign(math.inf, x),
  Ops.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan, Ops.POW: safe_pow, Ops.TRUNC: math.trunc,
  Ops.NEG: operator.neg, Ops.ADD: operator.add, Ops.SUB: operator.sub, Ops.MUL: operator.mul, Ops.CMPNE: operator.ne, Ops.CMPLT: operator.lt,
  Ops.XOR: operator.xor, Ops.OR: operator.or_, Ops.AND: operator.and_, Ops.SHR: operator.rshift, Ops.SHL: operator.lshift, Ops.MAX: max,
  Ops.MOD: cmod, Ops.IDIV: cdiv, Ops.MULACC: lambda x,y,z: (x*y)+z, Ops.WHERE: lambda x,y,z: y if x else z, Ops.CMPEQ: operator.eq}

def exec_alu(op:Ops, dtype:DType, operands, truncate_output=True):
  if dtype.count > 1:
    return tuple([exec_alu(op, dtype.scalar(), [x[i] if isinstance(x, tuple) else x for x in operands]) for i in range(dtype.count)])
  if dtype==dtypes.index and op in GroupOp.Binary and Invalid in operands: return Invalid
  alu = python_alu[op](*operands)
  return truncate.get(dtype, lambda x: x)(alu) if truncate_output else alu

# ***** uop helpers *****

def print_uops(uops:list[UOp]):
  for i,u in enumerate(uops):
    formatted_srcs = [(uops.index(x) if x.op is not Ops.CONST else f"{x.arg}") if x in uops else "--" for x in u.src]
    print(f"{i:4d} {str(u.op):20s}: {str(u.dtype):30s} " f"{str(formatted_srcs):32s} {u.arg}")

# ***** pattern matcher *****

def get_location() -> tuple[str, int]:
  frm = sys._getframe(1)
  # skip over ops.py/mathtraits.py (unless there's nothing but ops.py/mathtraits.py)
  while pathlib.Path(frm.f_code.co_filename).name in ("ops.py", "mathtraits.py") and frm.f_back is not None and \
      not frm.f_back.f_code.co_filename.startswith("<frozen"):
    frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno

@functools.cache
def lines(fn) -> list[str]:
  with open(fn) as f: return f.readlines()

def printable(loc:tuple[str, int]) -> str:
  try: return lines(loc[0])[loc[1]-1].strip()
  except FileNotFoundError: return "<missing>"

class UPat(MathTrait):
  __slots__ = ("op", "dtype", "arg", "name", "src")
  def __init__(self, op:Ops|tuple[Ops, ...]|set[Ops]|None=None, dtype:DType|tuple[DType, ...]|None=None,
               src:tuple[UPat, ...]|list[UPat]|UPat|None=None, arg:Any=None,
               name:str|None=None, allow_any_len:bool=False, custom_early_reject:set[Ops]|None=None, location=None):
    assert op is None or isinstance(op, (Ops, tuple, set)), "op must be Ops or tuple of Ops"
    self.op: tuple[Ops, ...]|None = (op,) if isinstance(op, Ops) else (tuple(op) if isinstance(op, set) else op)
    self.dtype: tuple[DType, ...]|None = (dtype,) if isinstance(dtype, DType) else dtype
    self.arg, self.name, self._in_src, self.custom_early_reject = arg, name, src, custom_early_reject
    self.src: Any = None
    assert self.name != "ctx", "UPat can't be named ctx"
    assert dtype is None or isinstance(dtype, DType) or all(isinstance(x, DType) for x in dtype), f"invalid dtype {dtype}"

    # try all permutations if it's a list
    if isinstance(src, list): self.src = list(itertools.permutations(src)) if not all_same(src) else [tuple(src)]
    # only one if it's a tuple
    elif isinstance(src, tuple): self.src = [src]
    # repeat if it's a UPat
    elif isinstance(src, UPat): self.src = [itertools.repeat(src)]

    self.strict_length = not (allow_any_len or isinstance(src, UPat) or src is None)
    self.required_len: int = 0 if isinstance(src, UPat) or src is None else len(src)
    self.location = location or get_location()

    if custom_early_reject is not None: self.early_reject = custom_early_reject
    else:
      upat_match = [src] if isinstance(src, UPat) else ([] if src is None else self.src[0])
      self.early_reject = {pp.op[0] for pp in upat_match if pp.op is not None and len(pp.op) == 1}

  def __reduce__(self):
    return UPat, (self.op, self.dtype, self._in_src, self.arg, self.name, not self.strict_length, self.custom_early_reject, self.location)
  def named(self, name:str): return UPat(self.op, self.dtype, self._in_src, self.arg, name, not self.strict_length, self.custom_early_reject)

  @staticmethod
  def any(*src): return UPatAny(src=src)
  def or_casted(self, name:str|None=None): return UPat.any(self if name is None else self.named(name), UPat(Ops.CAST, name=name, src=(self,)))

  @staticmethod
  @functools.cache
  def var(name:str|None=None, dtype:DType|tuple[DType, ...]|None=None): return UPat(dtype=dtype, name=name)
  @staticmethod
  @functools.cache
  def cvar(name:str|None=None, dtype:DType|tuple[DType, ...]|None=None, vec=True, arg=None):
    return UPat((Ops.CONST,Ops.VCONST) if vec else Ops.CONST, dtype, name=name, arg=arg)
  @staticmethod
  def const(dtype:DType|tuple[DType, ...]|None, b:ConstType|InvalidType): return UPat(Ops.CONST, dtype=dtype, arg=b)

  # lil helper
  def f(self, op, **kwargs): return UPat(op, src=(self,), **kwargs)

  # copied from UOp
  def sink(self, *srcs:UPat|None, **kwargs): return UPat(Ops.SINK, dtypes.void, (self,)+tuple([x for x in srcs if x is not None]), **kwargs)
  def index(self, idx:UPat, valid:UPat|None=None): return UPat(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
  def cast(self, dtype=None, **kwargs): return UPat(Ops.CAST, dtype, (self,), **kwargs)
  def bitcast(self, dtype=None): return UPat(Ops.BITCAST, dtype, (self,))
  def gep(self, i:int|None=None, **kwargs): return UPat(Ops.GEP, None, (self,), (i,) if i is not None else None, **kwargs)
  def load(self, *src:UPat, **kwargs): return UPat(Ops.LOAD, src=(self,)+src, **kwargs)
  def store(self, *src:UPat, **kwargs): return UPat(Ops.STORE, self.dtype, (self,)+src, **kwargs)
  def assign(self, x:UPat, **kwargs): return UPat(Ops.ASSIGN, self.dtype, (self,x), **kwargs)
  def reduce(self, *src:UPat, **kwargs): return UPat(Ops.REDUCE, self.dtype, src=(self,)+src, **kwargs)
  def fuse(self): return self.alu(Ops.FUSE)
  def broadcast(self, **kwargs): return UPat(Ops.VECTORIZE, self.dtype, src=self, **kwargs)
  def or_broadcasted(self, **kwargs): return UPat.any(self, self.broadcast(**kwargs))
  def contiguous(self, *args, **kwargs): return UPat(Ops.CONTIGUOUS, dtype=self.dtype, src=(self,)+args, **kwargs)

  def const_like(self, b:ConstLike): return UPat.const(self.dtype, cast(ConstType, b))
  def alu(self, op:Ops, *src:UPat):
    asrc = (self,)+src
    return UPat(op, dtypes.bool if op in {Ops.CMPLT, Ops.CMPNE} else asrc[-1].dtype, list(asrc) if op in GroupOp.Commutative else asrc)

  def match(self:UPat, uop:UOp, store:dict[str, UOp]) -> list[dict[str, UOp]]:
    if (self.op is not None and uop.op not in self.op) or \
       (self.name is not None and store.setdefault(self.name, uop) is not uop) or \
       (self.dtype is not None and uop.dtype not in self.dtype and uop.dtype.scalar() not in self.dtype) or \
       (self.arg is not None and self.arg != uop.arg) or \
       (len(uop.src) < self.required_len) or \
       (self.strict_length and len(uop.src) != self.required_len): return []
    if self.src is None: return [store]
    res: list[dict[str, UOp]] = []
    for vp in self.src:
      stores, new_stores = [store.copy()], []
      for uu, vv in zip(uop.src, vp):
        for s in stores: new_stores.extend(vv.match(uu, s))
        stores, new_stores = new_stores, []
      res.extend(stores)
    return res

class UPatAny(UPat):
  def match(self:UPat, uop:UOp, store:dict[str, UOp]) -> list[dict[str, UOp]]:
    matches = [x.match(uop, store.copy()) for x in self.src[0]]
    return flatten([x for x in matches if x is not None])

def deconstruct_function(fxn:Callable) -> tuple:
  new_globals = {k:v for k,v in fxn.__globals__.items() if k in fxn.__code__.co_names}
  for co in fxn.__code__.co_consts:
    if isinstance(co, types.CodeType): new_globals.update({k:v for k,v in fxn.__globals__.items() if k in co.co_names})
  # NOTE: optional round trip through pickle!
  assert fxn.__closure__ is None, "closures are not supported in pattern matchers"
  ret = fxn.__code__, new_globals, fxn.__name__, fxn.__defaults__
  return pickle.loads(pickle.dumps(ret)) if getenv("TEST_PICKLE") else ret

@functools.cache
def upat_interpret(p:UPat, fxn:Callable) -> Callable:
  real_fxn = types.FunctionType(*deconstruct_function(fxn))
  if 'ctx' in inspect.signature(real_fxn).parameters:
    def universal_match(uop, ctx):
      for match in p.match(uop, {}):
        if (ret:=real_fxn(ctx=ctx, **match)) is not None: return ret  # pylint: disable=not-callable
      return None
  else:
    def universal_match(uop, _):
      for match in p.match(uop, {}):
        if (ret:=real_fxn(**match)) is not None: return ret  # pylint: disable=not-callable
      return None
  return universal_match

class PatternMatcher:
  def __init__(self, patterns:Sequence[tuple[UPat, Callable|tuple]], compiled=bool(getenv("UPAT_COMPILE", 1))):
    if compiled: from tinygrad.uop.upat import upat_compile
    # if this comes from a pickle, we reconstruct the lambda functions here
    self.patterns:list[tuple[UPat, Callable]] = [(p,types.FunctionType(*fxn) if isinstance(fxn, tuple) else fxn) for p,fxn in patterns]
    # NOTE: use of DefaultDict here is very dangerous! all keys will live for the lifetime of the PatternMatcher!
    self.pdict: dict[Ops, list[tuple[UPat, Callable, set]]] = {}
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      if compiled and (match:=upat_compile(p, fxn)) is not None: pass # pylint: disable=E0606
      else: match = upat_interpret(p, fxn)
      for uop in p.op: self.pdict.setdefault(uop, []).append((p, match, p.early_reject))

  def __reduce__(self): return PatternMatcher, ([(x,deconstruct_function(fxn) if fxn.__name__ == "<lambda>" else fxn) for x,fxn in self.patterns],)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher) -> PatternMatcher: return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp, ctx=None) -> UOp|None:
    ler = {u.op for u in uop.src}
    for _,match,early_reject in self.pdict.get(uop.op, []):
      if not early_reject.issubset(ler): continue
      if (ret:=match(uop, ctx)) is not None and ret is not uop: return ret
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if VIZ else 0)
match_stats:dict[UPat, list[int|float]] = dict()

# TRACK_MATCH_STATS>=2 or VIZ=1 saves all matches
ucount = itertools.count()
uop_fields:dict[int, tuple] = {}

@dataclass(frozen=True)
class TrackedGraphRewrite:
  loc:tuple[str, int]                           # location that called graph_rewrite
  sink:int                                      # the sink input to graph_rewrite
  matches:list[tuple[int, int, tuple, float]]   # before/after UOp, UPat location and time
  name:str|None                                 # optional name of the rewrite
  depth:int                                     # depth if it's a subrewrite
  bottom_up:bool

tracked_keys:list[TracingKey] = []
tracked_ctxs:list[list[TrackedGraphRewrite]] = []
_name_cnt:dict[str, itertools.count] = {}

if getenv("CAPTURE_PROCESS_REPLAY"):
  replay_capture: dict[str, bytes] = {}
  import atexit
  @atexit.register
  def save_to_diskcache():
    for k,v in replay_capture.items(): diskcache_put("process_replay", k, v, prepickled=True)

def add_trace_group(kt:TracingKey) -> None:
  tracked_keys.append(kt)
  tracked_ctxs.append([])

def track_rewrites(name:Callable[..., str|TracingKey]|bool=True, replay:bool=False):
  def _decorator(func):
    def __wrapper(*args, **kwargs):
      fn = key = func.__name__
      if TRACK_MATCH_STATS >= 2: add_trace_group(key:=TracingKey(n:=f"{fn} n{next(_name_cnt.setdefault(fn, itertools.count(1)))}", (n,)))
      with cpu_profile(key, "TINY") as e:
        ret = func(*args, **kwargs)
      if TRACK_MATCH_STATS >= 2 and callable(name):
        name_ret = name(*args, **kwargs, ret=ret)
        assert isinstance(name_ret, (TracingKey, str)), f"name function returned {type(name_ret)}"
        tracked_keys[-1] = k = TracingKey(n:=tracked_keys[-1].display_name.replace(fn, name_ret), (n,)) if isinstance(name_ret, str) else name_ret
        e.name = TracingKey(k.display_name if isinstance(name_ret, str) else f"{fn} for {k.display_name}", k.keys)
      if getenv("CAPTURE_PROCESS_REPLAY") and replay:
        # find the unittest frame we're capturing in
        frm = sys._getframe(1)
        while (f_back:=frm.f_back) is not None and "unittest" not in f_back.f_code.co_filename: frm = f_back
        loc = f"{frm.f_code.co_filename.split('/')[-1]}:{frm.f_lineno} {frm.f_code.co_name}"
        # capture global context vars and all the args passed in
        with Context(PICKLE_BUFFERS=0):
          inputs = (fn, args, kwargs, ContextVar._cache)
          replay_capture[hashlib.sha256(pickle.dumps(inputs)).hexdigest()] = pickle.dumps(inputs+(loc, ret))
      return ret
    return __wrapper
  return _decorator

active_rewrites:list[TrackedGraphRewrite] = []
def track_matches(func):
  def _track_func(*args, **kwargs):
    if tracking:=(TRACK_MATCH_STATS >= 2):
      loc = ((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno)
      depth = len(active_rewrites)
      if not tracked_ctxs: add_trace_group(TracingKey(f"default {func.__name__}"))
      tracked_ctxs[-1].append(ctx:=TrackedGraphRewrite(loc, args[0].trace_num, [], kwargs.get("name", None), depth, kwargs.get("bottom_up", False)))
      active_rewrites.append(ctx)
    with cpu_profile(kwargs.get("name", "<unnamed>"), "TINY", display=tracking):
      ret = func(*args, **kwargs)
    if tracking: active_rewrites.pop()
    return ret
  return _track_func

class TrackedPatternMatcher(PatternMatcher):
  def rewrite(self, uop:UOp, ctx=None) -> UOp|None:
    ret = None
    ler = {u.op for u in uop.src}
    for p,match,early_reject in self.pdict.get(uop.op, []):
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      try: ret = match(uop, ctx)
      except Exception:
        if TRACK_MATCH_STATS >= 2 and active_rewrites:
          active_rewrites[-1].matches.append((uop.trace_num, UOp(Ops.REWRITE_ERROR,src=uop.src,arg=str(sys.exc_info()[1])).trace_num,p.location,0))
        raise
      if ret is not None and ret is not uop:
        match_stats[p][0] += 1
        match_stats[p][3] += (et:=time.perf_counter()-st)
        if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", printable(p.location))
        if TRACK_MATCH_STATS >= 2 and isinstance(ret, UOp) and active_rewrites:
          active_rewrites[-1].matches.append((uop.trace_num, ret.trace_num, p.location, et))
        return ret
      match_stats[p][2] += time.perf_counter()-st
    return None

@dataclass(frozen=True)
class RewriteTrace: keys:list[TracingKey]; rewrites:list[list[TrackedGraphRewrite]]; uop_fields:dict[int, tuple] # noqa: E702

if TRACK_MATCH_STATS or PROFILE:
  PatternMatcher = TrackedPatternMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    if TRACK_MATCH_STATS >= 2:
      with open(fn:=temp("rewrites.pkl", append_user=True), "wb") as f:
        print(f"rewrote {len(tracked_ctxs)} graphs and matched {sum(len(r.matches) for x in tracked_ctxs for r in x)} times, saved to {fn}")
        pickle.dump(RewriteTrace(tracked_keys, tracked_ctxs, uop_fields), f)
    if VIZ: return launch_viz("VIZ", temp("rewrites.pkl", append_user=True))
    if getenv("PRINT_MATCH_STATS", TRACK_MATCH_STATS.value):
      ret = [0,0,0.0,0.0]
      for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]+x[1][3]):
        loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
        if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {(v[2]+v[3])*1000.:9.2f} ms -- {loc_str:20s}", printable(k.location))
        ret = [x+y for x,y in zip(ret, v)]
      print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {(ret[2]+ret[3])*1000.:9.2f} ms -- TOTAL")
      print(f"{len(match_stats)} rules, {sum(v[0] > 0 for v in match_stats.values())} matched once")

  def launch_viz(env_str:str, data:str):
    os.environ[env_str] = "0"
    os.environ[f"{env_str}_DATA"] = data
    if not int(os.getenv("VIZ", "0")) and not int(os.getenv("PROFILE", "0")) and not int(os.getenv("SQTT", "0")):
      args = ['--kernels', getenv("VIZ_DATA", "")] if getenv("VIZ_DATA", "") else []
      args += ['--profile', getenv("PROFILE_DATA", "")] if getenv("PROFILE_DATA", "") else []
      os.execv(sys.executable, [sys.executable] + [pathlib.Path(__file__).resolve().parent.parent / "viz" / "serve.py"] + args)

# *** simple graph rewrite engine ***

with Context(SPEC=0): SENTINEL = UOp(Ops.SENTINEL)
class BottomUpGate(Exception): pass
class RewriteContext:
  def __init__(self, pm, bpm, ctx=None):
    self.pm: PatternMatcher|None = pm
    self.pm_cache: dict[UOp, UOp|None] = {}
    self.bpm: PatternMatcher|None = bpm
    self.bpm_cache: dict[UOp, UOp|None] = {}
    self.ctx = ctx
    self.replace: dict[UOp, UOp] = {}

  def cached_pm_rewrite(self, x:UOp):
    if (ret:=self.pm_cache.get(x,SENTINEL)) is not SENTINEL: return ret
    ret = self.pm_cache[x] = cast(PatternMatcher, self.pm).rewrite(x, self.ctx)
    return ret

  def cached_bpm_rewrite(self, x:UOp):
    if (ret:=self.bpm_cache.get(x,SENTINEL)) is not SENTINEL: return ret
    ret = self.bpm_cache[x] = cast(PatternMatcher, self.bpm).rewrite(x, self.ctx)
    return ret

  def unified_rewrite(self, root:UOp) -> UOp:
    stack: collections.deque[tuple[UOp, int, UOp]] = collections.deque([(root, 0, root)])
    on_stack = {root}  # all UOps either on the stack or in self.replace, i.e. dont have to be placed again
    REWRITE_STACK_LIMIT = getenv("REWRITE_STACK_LIMIT", 250000)
    while stack:
      if len(stack) > REWRITE_STACK_LIMIT: raise RuntimeError("infinite loop in graph_rewrite (stack too big)")
      n, stage, new_n = stack.pop()
      if n in self.replace: continue  # skip any nodes we have seen
      if stage == 0:
        # if bottom up, we rewrite this node early. in both cases, we add its srcs to the stack
        if self.bpm is not None:
          # apply rewrite rules until a fixed point is reached. may return `uop` itself if PatternMatcher doesn't match
          test_n: UOp|None = n
          seen = set()
          try:
            while test_n is not None:
              if test_n in seen: raise RuntimeError("infinite loop in fixed_point_rewrite")
              seen.add(test_n)
              new_n, test_n = test_n, self.cached_bpm_rewrite(test_n)
          except BottomUpGate:
            # if the bpm matching raised a gate, we are done with this node and dont continue down the srcs
            self.replace[n] = new_n
            continue
        stack.append((n, 1, new_n))
        for x in reversed(new_n.src):
          if x in on_stack: continue
          stack.append((x, 0, x))
          on_stack.add(x)
      elif stage == 1:
        tmp = []
        for x in new_n.src:
          if (rx:=self.replace.get(x, SENTINEL)) is SENTINEL:
            # if some new sources aren't ready, we try this again later. happens with on_stack, maybe should remove?
            stack.appendleft((n, 1, new_n))
            break
          tmp.append(rx)
        else:
          # in stage 1, once all srcs are rewritten, rebuild (if changed) or run top-down rewrite
          if (new_src:=tuple(tmp)) == new_n.src:
            # if top down, do the rewrite. if no rewrite or bottom up, we are done rewriting this node so we add it to the dict
            if self.pm is None or (new_src_n:=self.cached_pm_rewrite(new_n)) is None:
              self.replace[n] = new_n
              continue
          else:
            # if srcs changed from rewrites, construct a new UOp with the new srcs
            new_src_n = UOp(new_n.op, new_n.dtype, new_src, new_n.arg, new_n.tag)
          # trigger a rewrite of new_src_n, then after that rewrite is done, link it back to n
          stack.append((n, 2, new_src_n))
          stack.append((new_src_n, 0, new_src_n))
      else:
        # in stage 2, we link the result of new_n to the result of n
        if (replaced_new_n:=self.replace.get(new_n, SENTINEL)) is SENTINEL:
          # not ready, try the link later
          stack.appendleft((n, 2, new_n))
        else:
          # otherwise we are done
          self.replace[n] = replaced_new_n
    return self.replace[root]

@track_matches
def graph_rewrite(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False, name=None, bpm=None) -> UOp:
  rewrite_ctx = RewriteContext(pm if not bottom_up else None, pm if bottom_up else bpm, ctx)
  return rewrite_ctx.unified_rewrite(sink)

@track_matches
def graph_rewrite_map(sink:UOp, pm:PatternMatcher, ctx=None, bottom_up=False, name=None, bpm=None,
                      input_map:dict[UOp, UOp]|None=None, ) -> dict[UOp, UOp]:
  rewrite_ctx = RewriteContext(pm if not bottom_up else None, pm if bottom_up else bpm, ctx)
  new_map: dict[UOp, UOp] = {}
  for k in (list(sink.toposort())[::-1] if bottom_up else sink.toposort()):
    new_map[k] = v = rewrite_ctx.unified_rewrite(k)
    if k is not v and k.metadata is not None: all_metadata[v] = tuple(dedup(all_metadata.get(v, ())))+k.metadata
  if input_map is not None:
    for k,v in input_map.items(): new_map[k] = new_map.get(v,v)
  return new_map

def sint_to_uop(x:sint) -> UOp: return UOp.const(dtypes.index, x) if isinstance(x, int) else x.cast(dtypes.index)

def select_dtype(u): return (dtypes.long if u.overflows(dtypes.int32) else dtypes.int).vec(u.dtype.count)
pm_lower_index_dtype = PatternMatcher([
  # There are no Unary ops at this point in symbolic, those are introduced later
  (UPat(GroupOp.Binary, name="u", src=(UPat.var("x").cast(dtypes.index), UPat.var("y").cast(dtypes.index))), lambda u,x,y:
    x.cast(dt:=least_upper_dtype(select_dtype(u), x.dtype, y.dtype)).alu(u.op, y.cast(dt)).cast(u.dtype)),
  (UPat((Ops.CONST, Ops.VCONST), dtype=dtypes.index, name="u"), lambda u: u.replace(dtype=select_dtype(u)).cast(u.dtype) if u.arg!=Invalid else None),
  (UPat(Ops.WHERE, dtypes.index, src=(UPat.var("cond"), UPat.var("x").cast(dtypes.index), UPat.var("y").cast(dtypes.index))), lambda cond,x,y:
    cond.where(x.cast(dt:=least_upper_dtype(x.dtype, y.dtype)), y.cast(dt)).cast(dtypes.index)),
  (UPat(Ops.RANGE, src=(UPat.var("end").cast(dtypes.index)), name="r"), lambda r,end: r.replace(dtype=end.dtype, src=(end,)).cast(dtypes.index)),
  (UPat(Ops.VECTORIZE, src=UPat().cast(dtypes.index), name="v"),
    lambda v: v.replace(dtype=(dt:=select_dtype(v)), src=tuple(s.src[0].cast(dt.scalar()) for s in v.src)).cast(dtypes.index)),
  # special can only be int32
  (UPat(Ops.SPECIAL, src=(UPat.var("var").cast(dtypes.index),), name="u"), lambda u,var: u.replace(dtype=dtypes.int, src=(var,)).cast(dtypes.index)),
  (UPat(Ops.DEFINE_VAR, dtype=dtypes.index, name="u"), lambda u: u.replace(dtype=dtypes.int).cast(dtypes.index)),
  (UPat(Ops.BIND, src=(UPat.var("var").cast(dtypes.index), UPat.cvar("val").cast(dtypes.index))), lambda var,val: var.bind(val).cast(dtypes.index)),
  (UPat(Ops.CAST, src=(UPat(name="x").cast(dtypes.index),), name="c"), lambda x,c: x.cast(c.dtype)),
  # lower Invalid
  (UPat.var("buf").index(UPat.var("cond").where(UPat.var("idx"), UPat(Ops.CONST, arg=Invalid))), lambda buf,idx,cond: buf.index(idx, cond)),
  # remove hanging casts
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx", dtypes.ints).cast()),), lambda buf,idx: buf.index(idx)),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx", dtypes.ints).cast(), UPat.var("valid"))), lambda buf,idx,valid: buf.index(idx, valid)),
  (UPat((Ops.STORE, Ops.LOAD), src=(UPat(), UPat(), UPat().cast(dtypes.index)), allow_any_len=True, name="s"),
    lambda s: s.replace(src=s.src[:2]+tuple(u.src[0] for u in s.src[2:]))),
  (UPat((Ops.SINK, Ops.NOOP), src=UPat().cast(dtypes.index), name="n"), lambda n: n.replace(src=tuple(s.src[0] for s in n.src))),
])
def _index_to_concrete_int(u:UOp): return graph_rewrite(u.sink(), pm_lower_index_dtype).src[0]

_substitute = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None))])

def do_unbind(ctx:dict[Variable, int], x:UOp):
  v,i = x.unbind()
  ctx[v] = i
  return v
pm_unbind = PatternMatcher([(UPat(Ops.BIND, name="x"), do_unbind)])

# for debug
syms = { Ops.ADD: "+", Ops.SUB: "-", Ops.IDIV: "//", Ops.MOD: "%", Ops.SHL: "<<", Ops.SHR: ">>",
         Ops.MUL: "*", Ops.CMPLT: "<", Ops.CMPNE: "!=", Ops.AND: "&", Ops.OR: "|", Ops.XOR: "^"}
renderer = PatternMatcher([
  (UPat((Ops.DEFINE_VAR,), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg[0])),
  (UPat((Ops.SPECIAL), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg)),
  (UPat(Ops.RANGE, name="x"), lambda x: UOp(Ops.NOOP, arg=f"r{range_str(x)}")),
  (UPat((Ops.CONST, Ops.VCONST), name="x"), lambda x: UOp(Ops.NOOP, arg=str(x.arg))),
  (UPat(Ops.UNROLL, name="x"), lambda x: UOp(Ops.NOOP, arg=f"UNROLL({x.src[0].arg}, {x.arg})")),
  (UPat(Ops.CAST, name="x"), lambda x: UOp(Ops.NOOP, arg=f"({str(x.dtype)[7:]})({x.src[0].arg})")),
  (UPat(Ops.BIND, src=UPat(Ops.NOOP), name="x"), lambda x: x.src[0]),
  #(UPat(Ops.BIND, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"{x.src[0].arg}[={x.src[1].arg}]")),
  (UPat(Ops.NEG, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(-{x.src[0].arg})")),
  (UPat(Ops.RECIP, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"(1/{x.src[0].arg})")),
  (UPat(Ops.MAX, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"max({x.src[0].arg}, {x.src[1].arg})")),
  (UPat(Ops.MULACC, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}*{x.src[1].arg}+{x.src[2].arg})")),
  (UPat(Ops.WHERE, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[1].arg} if {x.src[0].arg} else {x.src[2].arg})")),
  (UPat(set(syms.keys()), src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"({x.src[0].arg}{syms[x.op]}{x.src[1].arg})")),
  (UPat((Ops.INDEX, Ops.BUFFERIZE), name="x"), lambda x:
   UOp(Ops.NOOP, arg=''.join([f"[{strip_parens(y.arg)}]" for y in x.src[1:]])) if all(y.op is Ops.NOOP for y in x.src[1:]) else None),
  (UPat(Ops.VECTORIZE, src=UPat(Ops.NOOP), name="x"),
   lambda x: UOp(Ops.NOOP, arg=f"{{{','.join([y.arg for y in x.src])}}}" if not all_same(x.src) else f"{{{x.src[0].arg}, ...}}")),
])
renderer_infer = PatternMatcher([
  (UPat(Ops.MOD, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"cmod({x.src[0].arg}, {x.src[1].arg})")),
  (UPat(Ops.IDIV, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=f"cdiv({x.src[0].arg}, {x.src[1].arg})")),
  *renderer.patterns
])

sugar = { Ops.SINK: "sink", Ops.STORE: "store", Ops.LOAD: "load", Ops.SQRT: "sqrt", Ops.INDEX: "index", Ops.REDUCE: "reduce",
          Ops.WHERE: "where", Ops.RECIP: "reciprocal", Ops.EXP2: "exp2", Ops.LOG2: "log2", Ops.SIN: "sin"}
pm_pyrender = PatternMatcher([
  (UPat(Ops.CONST, src=(UPat(Ops.NOOP),), name="x"), lambda x: UOp(Ops.NOOP, arg=f"UOp.const({x.dtype}, {x.arg}, src={x.src[0].arg})")),
  (UPat(Ops.CONST, name="x"), lambda x: UOp(Ops.NOOP, arg=f"UOp.const({x.dtype}, {x.arg})")),
  (UPat(Ops.CAST, src=(UPat(Ops.NOOP),), name="x"), lambda x: UOp(Ops.NOOP, arg=f"{x.src[0].arg}.cast({x.dtype})")),
  (UPat(Ops.BITCAST, src=(UPat(Ops.NOOP),), name="x"), lambda x: UOp(Ops.NOOP, arg=f"{x.src[0].arg}.bitcast({x.dtype})")),
  (UPat({Ops.MAX, Ops.THREEFRY, Ops.CMPLT, Ops.CMPNE, Ops.POW}, src=UPat(Ops.NOOP), name="x"),
   lambda x: UOp(Ops.NOOP, arg=f"{x.src[0].arg}.alu({x.op}, {x.src[1].arg})")),
  (UPat(Ops.RANGE, src=(UPat(Ops.NOOP),), name="x"), lambda x:
    UOp(Ops.NOOP, arg=f"UOp.range({x.src[0].arg}, {str(x.arg[0])}, {str(x.arg[1])})")),
  (UPat(set(sugar.keys()), src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP,
    arg=f"{x.src[0].arg}.{sugar[x.op]}({', '.join([y.arg for y in x.src[1:]] + ([f'arg={str(x.arg)}'] if x.arg is not None else []))})")),
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.NOOP),), name="x"),
   lambda x: UOp(Ops.NOOP, arg=f"{x.src[0].arg}.f({x.op}, arg=({', '.join([str(y) for y in x.arg])}))")),
])

@Context(SPEC=0)
def pyrender(ast:UOp) -> list[str]:
  cmap = ast.get_consumer_map()
  to_render = set()
  for u in ast.toposort():
    if u.op is Ops.STORE: to_render.add(u.src[1])
    if len(cmap[u]) == 1 and u.op not in {Ops.DEFINE_GLOBAL, Ops.LOAD} or u.op in {Ops.CONST}: continue
    if u.op in {Ops.SINK}:
      for s in u.src: to_render.add(s)
    to_render.add(u)
  ret: list[str] = []
  rep: dict[UOp, UOp] = {}
  for u in ast.toposort():
    if u not in to_render: continue
    ret.append(f"c{len(ret)} = {u.substitute(rep).render(simplify=False, pm=pm_pyrender+renderer)}")
    rep[u] = UOp(Ops.NOOP, arg=f"c{len(ret)-1}")
  return ret[0:-1] + ["ast ="+ret[-1].split("=", 1)[1]]

# *** what was symbolic.py ***

sint = int|UOp
Variable = UOp

ConstLike = ConstType|InvalidType|Variable|tuple[ConstType|InvalidType, ...]
