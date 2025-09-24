from typing import cast, Generator, Callable
import time, pprint, random, itertools, math, ctypes, sys
try:
  import numpy as np
except ImportError:  # pragma: no cover
  np = None  # type: ignore[assignment]
from dataclasses import dataclass, replace, field
from tinygrad.helpers import all_same, colored, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, TRACEMETA, TracingKey
from tinygrad.helpers import (
  DEVECTORIZE,
  time_to_str,
  VALIDATE_WITH_CPU,
  getenv,
  cpu_profile,
  PROFILE,
  ProfilePointEvent,
  cpu_events,
  prod,
  Context,
  unwrap,
)
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer, graph_rewrite, print_uops, track_rewrites, KernelInfo, pyrender
from tinygrad.device import Device, Buffer
from tinygrad.renderer import Renderer, ProgramSpec, Estimates
from tinygrad.dtype import dtypes
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.codegen import full_rewrite
from tinygrad.codegen.opt import Opt

# **************** Program Creation ****************

@track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, (ret.function_name, ret.ast), ret=ret), replay=True)
def get_program(ast:UOp, renderer:Renderer|None=None, opts:list[Opt]|None=None) -> ProgramSpec:
  """
  Transform an AST into a ProgramSpec. May trigger BEAM search.

  Args:
    ast: The Ops.SINK rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The ProgramSpec of the program.
  """

  if getenv("VIZ"): graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print('\n'.join(pyrender(ast)))

  # linearize
  if renderer is None: renderer = Device.default.renderer
  if opts is not None:
    assert ast.arg is None, "can't apply opts if sink has an arg"
    ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))
  try:
    uops = full_rewrite(ast, renderer)
  except RuntimeError as e:
    print("***** LINEARIZE FAILURE *****")
    print(e)
    print('\n'.join(pyrender(ast)))
    raise
  assert uops[-1].op is Ops.SINK, "last uop must be sink"

  # print and render
  if DEBUG >= 6: print_uops(uops)
  src = renderer.render(uops)

  return ProgramSpec(uops[-1].arg.name if uops[-1].arg is not None else "test", src, renderer.device, ast, uops,
                     global_size=[1,1,1] if renderer.has_local or renderer.has_threads else None,
                     local_size=[1,1,1] if renderer.has_local else None)

# **************** Runners ****************

class Runner:
  def __init__(self, display_name:str, device:str, estimates=Estimates()):
    self.first_run, self.display_name, self.device, self.estimates = True, display_name, device, estimates
  @property
  def dev(self): return Device[self.device]
  def exec(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None) -> float|None:
    return self(rawbufs, {} if var_vals is None else var_vals)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    raise NotImplementedError("override this")

def optimize_local_size(_prg:Callable, global_size:list[int], rawbufs:list[Buffer]) -> list[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try:
      return _prg(*[x._buf for x in test_rawbuffers],global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)],
                  local_size=local_size, wait=True)
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

class CompiledRunner(Runner):
  def __init__(self, p:ProgramSpec, precompiled:bytes|None=None, prg=None):
    if DEBUG >= 4: print(p.src)
    self.p:ProgramSpec = p
    if prg is not None:
      self.lib = precompiled
      self._prg = prg
    else:
      if precompiled is not None: self.lib = precompiled
      else:
        with cpu_profile(TracingKey(f"compile {p.name}", (p.function_name,)), "TINY"):
          self.lib = Device[p.device].compiler.compile_cached(p.src)
      if DEBUG >= 7: Device[p.device].compiler.disassemble(self.lib)
      self._prg = Device[p.device].runtime(p.function_name, self.lib)
    super().__init__(p.name, p.device, p.estimates)

  def __reduce__(self): return self.__class__, (self.p, self.lib)

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    has_local = Device[self.p.device].renderer.has_local
    global_size, local_size = self.p.launch_dims(var_vals)
    if has_local and global_size is not None and local_size is None and all_int(self.p.global_size): # type: ignore[arg-type]
      local_size = optimize_local_size(self._prg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      self.p = replace(self.p, global_size=global_size, local_size=local_size)
    lra = {}
    if global_size:
      lra['global_size'] = tuple(global_size)
      assert len(global_size) == 3, "global size must have len 3"
    if local_size:
      lra['local_size'] = tuple(local_size)
      assert len(local_size) == 3, "local size must have len 3"
    return self._prg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k.expr] for k in self.p.vars), wait=wait)

class ViewOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"

class BufferCopy(Runner):
  def __init__(self, total_sz, dest_device, src_device):
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, Estimates(lds=total_sz, mem=total_sz))
  def copy(self, dest, src):
    disk_supports_fast_copyout = src.device.startswith("DISK") and hasattr(src.allocator.dev, 'io_uring') and \
      getattr(src.allocator.dev, 'fd', None) is not None and dest.allocator.supports_copy_from_disk
    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_disk') and disk_supports_fast_copyout and src.nbytes >= 4096:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif src.device.startswith("DISK") and hasattr(dest.allocator, '_as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    if wait:
      Device[dest.device].synchronize()
      return time.perf_counter() - st

class BufferXfer(BufferCopy):
  def copy(self, dest, src): dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)

# **************** method cache ****************

method_cache: dict[tuple[str, type, bytes, tuple[int, ...], bool], CompiledRunner] = {}

_vdsp_handle:ctypes.CDLL|None = None

def _get_vdsp() -> ctypes.CDLL|None:
  global _vdsp_handle
  if _vdsp_handle is not None: return _vdsp_handle
  if sys.platform != "darwin": return None
  try:
    lib = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Accelerate.framework/Accelerate')
  except OSError:
    return None
  lib.vDSP_sve.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
  lib.vDSP_sve.restype = None
  _vdsp_handle = lib
  return _vdsp_handle

def _ndindex(shape:tuple[int, ...]):
  if len(shape) == 0:
    yield ()
    return
  ranges = [range(s) for s in shape]
  for idx in itertools.product(*ranges): yield idx

def _contiguous_reduce_block(shape:tuple[int, ...], strides:tuple[int, ...], reduce_axes:tuple[int, ...]) -> bool:
  if len(reduce_axes) == 0: return False
  expected = tuple(range(len(shape)-len(reduce_axes), len(shape)))
  if reduce_axes != expected: return False
  acc = 1
  for axis in reversed(reduce_axes):
    if strides[axis] != acc: return False
    acc *= shape[axis]
  return True

def _match_cpu_fast_sum(ast:UOp):
  if sys.platform != "darwin" or _get_vdsp() is None: return None
  if ast.op is not Ops.SINK or len(ast.src) != 1: return None
  store = ast.src[0]
  if store.op is not Ops.STORE or len(store.src) != 2: return None
  dest_view, red = store.src
  if dest_view.op is not Ops.VIEW or red.op is not Ops.REDUCE_AXIS: return None
  if red.arg[0] is not Ops.ADD or red.dtype != dtypes.float32: return None

  dest_base = dest_view.src[0]
  if dest_base.op is Ops.VIEW or dest_base.op is not Ops.DEFINE_GLOBAL: return None
  dest_id = dest_base.arg

  src_uop = red.src[0]
  if src_uop.op is Ops.LOAD:
    base = src_uop.src[0]
    if base.op is not Ops.DEFINE_GLOBAL: return None
    src_id = base.arg
    if src_uop.st is None: return None
    src_view = src_uop.view(src_uop.st_arg)
  elif src_uop.op is Ops.VIEW and src_uop.src[0].op is Ops.DEFINE_GLOBAL:
    src_id = src_uop.src[0].arg
    src_view = src_uop
  elif src_uop.op is Ops.VIEW and src_uop.src[0].op is Ops.LOAD and src_uop.src[0].src[0].op is Ops.DEFINE_GLOBAL:
    src_id = src_uop.src[0].src[0].arg
    src_view = src_uop
  else:
    return None

  src_st = unwrap(src_view.st)
  if any(v.mask is not None for v in src_st.views): return None
  try:
    shape = tuple(int(sym_infer(s, {})) for s in src_st.shape)
    strides_t = src_st.real_strides(ignore_valid=True)
    strides = []
    for s in strides_t:
      if isinstance(s, UOp): strides.append(int(sym_infer(s, {})))
      elif s is None: strides.append(None)
      else: strides.append(int(s))
    strides = tuple(strides)
  except Exception:
    return None
  if any(s is None for s in strides): return None
  reduce_axes:tuple[int, ...] = red.arg[1]
  if not _contiguous_reduce_block(shape, strides, reduce_axes): return None

  dest_st = unwrap(dest_view.st)
  if any(v.mask is not None for v in dest_st.views): return None
  try:
    dest_shape = tuple(int(sym_infer(s, {})) for s in dest_st.shape)
    dest_strides_t = dest_st.real_strides(ignore_valid=True)
    dest_strides = tuple(0 if s is None else (int(sym_infer(s, {})) if isinstance(s, UOp) else int(s)) for s in dest_strides_t)
  except Exception:
    return None

  reduce_size = math.prod(shape[a] for a in reduce_axes)
  outer_axes = [i for i in range(len(shape)) if i not in reduce_axes]
  outer_shape = [shape[i] for i in outer_axes]
  outer_strides = [strides[i] for i in outer_axes]

  if outer_axes and np is None: return None

  return {
    "src_id": src_id,
    "dest_id": dest_id,
    "shape": shape,
    "strides": strides,
    "dest_shape": dest_shape,
    "dest_strides": dest_strides,
    "reduce_axes": reduce_axes,
    "reduce_size": reduce_size,
    "outer_axes": outer_axes,
    "outer_shape": outer_shape,
    "outer_strides": outer_strides,
  }

def get_runner(device:str, ast:UOp) -> CompiledRunner:
  # TODO: this should be all context relevant to rendering
  context = (BEAM.value, NOOPT.value, DEVECTORIZE.value)
  ckey = (device, type(Device[device].compiler), ast.key, context, False)
  if cret:=method_cache.get(ckey): return cret
  bkey = (device.split(":")[0], type(Device[device].compiler), ast.key, context, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device), bret.lib)
  else:
    fast_info = _match_cpu_fast_sum(ast) if device.split(":")[0] == "CPU" else None
    if fast_info is not None:
      runner = FastSumRunner(device, fast_info, ast)
      method_cache[ckey] = method_cache[bkey] = runner
      return runner
    prg: ProgramSpec = get_program(ast, Device[device].renderer)
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
  return ret

# **************** lowering functions ****************

@dataclass(frozen=True)
class ExecItem:
  prg: Runner
  bufs: list[Buffer|None]
  metadata: tuple[Metadata, ...]|None = None
  fixedvars: dict[str, int] = field(default_factory=dict)
  def run(self, _var_vals:dict[str, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
    var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    bufs = [cast(Buffer, x) for x in self.bufs] if jit else [cast(Buffer, x).ensure_allocated() for x in self.bufs]
    if PROFILE: cpu_events.append(ProfilePointEvent(self.prg.device, "exec", self.prg.display_name, {"metadata":self.metadata, "var_vals":var_vals}))
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.estimates.ops, var_vals))
      GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.estimates.mem, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        lds_est = sym_infer(self.prg.estimates.lds, var_vals)
        mem_est = min(mem_est, lds_est)   # there can't be more memory accessed than loads/stores. remove this when symbolic is fixed
        header_color = 'magenta' if jit else ('green' if self.prg.first_run else None)
        ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
        flops, membw, ldsbw = op_est/(et or 1e-20), mem_est/(et or 1e-20), lds_est/(et or 1e-20)
        flops_str = f"{flops*1e-9:9.2f} GFLOPS" if flops < 1e14 else colored(f"{flops*1e-12:9.2f} TFLOPS", 'green')
        mem_str = f"{membw*1e-9:6.1f}|{ldsbw*1e-9:<7.1f} GB/s" if membw < 1e13 else colored(f"{membw*1e-12:6.1f}|{ldsbw*1e-12:<7.1f} TB/s", 'green')
        print(f"{colored(f'*** {self.prg.device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
          f" {self.prg.display_name+' '*(44-ansilen(self.prg.display_name))} arg {len(bufs):2d} mem {GlobalCounters.mem_used/1e9:5.2f} GB"+
          ("" if et is None else f" tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({flops_str} {mem_str})")+
          f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in self.metadata] if self.metadata else ''}")
      self.prg.first_run = False
    return et

# NOTE: ctx is the buffers
si_lowerer = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), lambda ctx,sink: (runner:=get_runner(ctx[0].device, sink), [ctx[x] for x in runner.p.globals])),
  (UPat(Ops.BUFFER_VIEW), lambda ctx: (ViewOp(ctx[0]), list(ctx))),
  (UPat(Ops.COPY, name="copy"), lambda ctx,copy: ((BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) \
      if hasattr(Device[ctx[0].device].allocator, '_transfer') and all_same([x.device.split(":")[0] for x in ctx]) \
      else BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device)), list(ctx))),
])
def lower_schedule_item(si:ScheduleItem) -> ExecItem:
  return ExecItem(*cast(tuple[Runner,list], si_lowerer.rewrite(si.ast, si.bufs)), si.metadata, si.fixedvars)

def lower_schedule(schedule:list[ScheduleItem]) -> Generator[tuple[ScheduleItem, ExecItem], None, None]:
  while len(schedule):
    si = schedule.pop(0)
    try: yield (si, lower_schedule_item(si))
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {si.ast.op}")
        print("tensor operations:")
        pprint.pprint(si.metadata, indent=2)
      raise e

# **************** main run function ****************

capturing: list = []  # put classes with an add method in here

def run_schedule(schedule:list[ScheduleItem], var_vals:dict[str, int]|None=None, do_update_stats=True):
  for si, ei in lower_schedule(schedule):
    if len(capturing) and CAPTURING: capturing[0].add(ei)
    if VALIDATE_WITH_CPU and si.ast.op is Ops.SINK:
      # copy in allocated buffers from the GPU
      nb: tuple[Buffer, ...] = tuple(Buffer("CPU", b.size, b.dtype) for b in si.bufs)
      for cpu_b, gpu_b in zip(nb, si.bufs):
        if gpu_b.is_allocated(): cpu_b.ensure_allocated().copyin(gpu_b.as_buffer())

      # run on GPU
      ei.run(var_vals, do_update_stats=do_update_stats)

      # validate the output buffers match (NOTE: this is assuming the output is buffer 0)
      with Context(BEAM=0): lower_schedule_item(ScheduleItem(si.ast, nb, si.metadata, si.fixedvars)).run(var_vals, do_update_stats=do_update_stats)
      import numpy as np
      np.testing.assert_allclose(si.bufs[0].numpy(), nb[0].numpy(), rtol=1e-3, atol=1e-3)
    else:
      ei.run(var_vals, do_update_stats=do_update_stats)
class FastSumRunner(Runner):
  def __init__(self, device:str, info:dict, ast:UOp):
    super().__init__("fast_sum", device, Estimates())
    self.info = info
    self.vdsp = _get_vdsp()
    self.dtype_size = 4
    self.src_ptr_type = ctypes.POINTER(ctypes.c_float)
    self.dest_ptr_type = ctypes.POINTER(ctypes.c_float)
    self.size_c = ctypes.c_size_t(info["reduce_size"])
    self.p = ProgramSpec(f"fast_sum_{info['dest_id']}_{info['src_id']}", "", device, ast, None,
                         globals=[info['dest_id'], info['src_id']], outs=[info['dest_id']], ins=[info['src_id']])

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    dest_buf, src_buf = rawbufs
    dest = dest_buf.ensure_allocated()._buf
    src = src_buf.ensure_allocated()._buf
    src_base = src.va_addr
    dest_base = dest.va_addr

    reduce_size = self.info["reduce_size"]
    strides = self.info["strides"]
    dest_shape = self.info["dest_shape"]
    dest_strides = self.info["dest_strides"]
    reduce_axes = self.info["reduce_axes"]
    outer_shape = self.info["outer_shape"]

    st = time.perf_counter() if wait else None

    outer_elems = math.prod(outer_shape) if outer_shape else 1

    if not outer_shape:
      if self.vdsp is not None:
        ptr = ctypes.cast(src_base, self.src_ptr_type)
        dest_ptr = ctypes.cast(dest_base, self.dest_ptr_type)
        self.vdsp.vDSP_sve(ptr, 1, dest_ptr, self.size_c)
        GlobalCounters.global_ops += reduce_size
        GlobalCounters.global_mem += reduce_size * self.dtype_size + self.dtype_size
      elif np is not None:
        src_arr = np.ctypeslib.as_array((ctypes.c_float * src_buf.size).from_address(src_base))
        total = float(src_arr.sum(dtype=np.float64))
        ctypes.cast(dest_base, self.dest_ptr_type)[0] = total
        GlobalCounters.global_ops += reduce_size
        GlobalCounters.global_mem += reduce_size * self.dtype_size + self.dtype_size
      else:
        raise RuntimeError("no backend available for CPU fast sum")
    else:
      if np is None:
        raise RuntimeError("numpy is required for strided CPU fast sums")
      src_arr = np.ctypeslib.as_array((ctypes.c_float * src_buf.size).from_address(src_base))
      view = np.lib.stride_tricks.as_strided(src_arr, shape=self.info["shape"],
                                            strides=tuple(s*self.dtype_size for s in strides))
      summed = view.sum(axis=reduce_axes, keepdims=True)
      dest_arr = np.ctypeslib.as_array((ctypes.c_float * dest_buf.size).from_address(dest_base))
      dest_view = np.lib.stride_tricks.as_strided(dest_arr, shape=dest_shape,
                                                 strides=tuple(s*self.dtype_size for s in dest_strides))
      dest_view[...] = summed.astype(np.float32)
      GlobalCounters.global_ops += reduce_size * outer_elems
      GlobalCounters.global_mem += (reduce_size * outer_elems + outer_elems) * self.dtype_size

    if wait and st is not None: return (time.perf_counter() - st) * 1000.0
    return None
