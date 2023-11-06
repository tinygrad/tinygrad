from __future__ import annotations
import importlib, inspect, functools, pathlib
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Optional, Dict, Callable, Mapping
from tinygrad.helpers import prod, DEBUG, getenv, DType, colored, ansilen, NOOPT, BEAM
from tinygrad.runtime.lib import RawBuffer
from tinygrad.shape.symbolic import Variable, sym_infer
from dataclasses import dataclass, field

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

  def interpret_ast(self:Interpreted, ast:LazyOp) -> Callable:
    tglob: Dict[str, Any] = {}
    lines: List[str] = []
    f = self.fxn_for_op

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
    src = '\n'.join(['def run(*inputs):'] + lines + [f"  return {gstr(self.from_underlying, 'from_underlying')}({ret})" if self.from_underlying else f"  return {ret}"])
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
def get_lazyop_info(ast:LazyOp) -> FlopCounter: return InterpretedFlopCounter.interpret_ast(ast)(None)

# **************** for Compiled Buffers ****************

def print_info(name, ast, var_vals, lra, et, jit=False):
  info = get_lazyop_info(ast)
  op_estimate = sym_infer(info.flops, var_vals)
  mem_estimate = sym_infer(info.mem_estimate, var_vals)
  if DEBUG >= 2:
    print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(name+' '*(37-ansilen(name)))} arg {len(info.mem):3d} sz {str(lra.get('global_size')):18s} {str(lra.get('local_size')):12s} OPs {int(op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
        (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))
  GlobalCounters.kernel_count += 1
  GlobalCounters.global_ops += op_estimate
  GlobalCounters.global_mem += mem_estimate

@dataclass(frozen=True)
class Runner:
  fxn: Callable
  name: str
  prg: str
  ast: LazyOp
  runtime_args: Dict
  # TODO: remove these
  @property
  def global_size(self): return self.runtime_args.get('global_size')
  @property
  def local_size(self): return self.runtime_args.get('local_size')
  def __call__(self, rawbufs, var_vals, jit=False):
    lra = self.runtime_args.copy()
    if DEBUG >= 2: lra['wait'] = True
    if 'global_size' in lra: lra['global_size'] = [sym_infer(sz, var_vals) for sz in lra['global_size']]
    if 'local_size' in lra: lra['local_size'] = [sym_infer(sz, var_vals) for sz in lra['local_size']]
    if et := self.fxn(*rawbufs, *var_vals.values(), **lra): GlobalCounters.time_sum_s += et
    print_info(self.name, self.ast, var_vals, lra, et, jit=jit)

@dataclass
class Compiled:
  buffer: Type[RawBuffer]
  linearizer_opts: Any
  renderer: Any
  compiler: Any
  runtime: Any
  synchronize: Any = lambda: None  # noqa: E731
  _method_cache: Dict = field(default_factory=dict)

  # rawbufs are just used for timing
  def compile_ast(self:Compiled, ast:LazyOp, rawbufs:List[RawBuffer]) -> Runner:
    if ast in self._method_cache: return self._method_cache[ast]

    # get linearizer
    from tinygrad.codegen.linearizer import Linearizer
    lin = Linearizer(ast, self.linearizer_opts)

    if not NOOPT:
      if not (used_tensor_cores:=lin.apply_tensor_cores(getenv("TC", 1))): lin.hand_coded_optimizations()
      if BEAM >= 1:
        lins = [(("tc" if used_tensor_cores else "hc"), lin)]
        # allocate a scratch buffer if output buffer is also input
        test_rawbuffers = [type(rawbufs[0])(rawbufs[0].size, rawbufs[0].dtype), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
        kb = Linearizer(ast, self.linearizer_opts)
        kb.required_optimizations()
        from tinygrad.features.search import beam_search, time_linearizer
        lins.append((f"beam{BEAM.value}", beam_search(kb, test_rawbuffers, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))))
        if used_tensor_cores:
          lins.append(("hc", Linearizer(ast, self.linearizer_opts)))
          lins[-1][1].hand_coded_optimizations()
        timed = sorted([(nm, tk, time_linearizer(tk, test_rawbuffers, allow_test_size=False, disable_cache=True, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
        if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
        lin = timed[0][1]
    else:
      lin.required_optimizations()

    # generate uops from the AST
    lin.linearize()

    # render the source code
    # TODO: move global_size and local_size to runtime_args
    src, runtime_args = self.renderer(lin.function_name, lin.uops)
    if DEBUG >= 4: print(src)

    # move this to renderer?
    if lin.global_size: runtime_args['global_size'] = lin.global_size
    if lin.local_size: runtime_args['local_size'] = lin.local_size

    # compile the source code. TODO: pass in device identifier
    lib: bytes = self.compiler.__wrapped__(src) if getenv("DISABLE_COMPILER_CACHE") else self.compiler(src)

    # get the function
    fxn = self.runtime(lin.function_name, lib)

    # local opt (TODO: confirm it's not symbolic)
    if 'global_size' in runtime_args and 'local_size' not in runtime_args:
      from tinygrad.features.search import optimize_local_size
      runtime_args['local_size'] = optimize_local_size(fxn, runtime_args['global_size'], rawbufs)

    runner = self._method_cache[ast] = Runner(fxn, lin.display_name, src, ast, runtime_args)
    return runner
