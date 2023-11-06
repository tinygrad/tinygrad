from typing import List, cast, Dict, Callable, Any, Tuple
import functools, time, itertools, random
import numpy as np
from tinygrad.ops import ScheduleItem, LazyOp, LoadOps, Device, BufferOps, Interpreted, Compiled, TernaryOps, ReduceOps, BinaryOps, MovementOps, UnaryOps, InterpretedFlopCounter, FlopCounter
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG, prod, all_int, getenv, IMAGE, GlobalCounters, colored, ansilen

from tinygrad.runtime.lib import RawBuffer
from tinygrad.features.image import fix_schedule_for_images
from tinygrad.shape.symbolic import sym_infer

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

@functools.lru_cache(None)
def get_lazyop_info(ast:LazyOp) -> FlopCounter: return interpret_ast(InterpretedFlopCounter, ast)(None)

@functools.lru_cache(None)
def compile_ast(device:Compiled, ast:LazyOp) -> Tuple[Callable, str, dict]:
  # get linearizer
  from tinygrad.codegen.linearizer import Linearizer
  lin = Linearizer(ast, device.linearizer_opts)

  # TODO: search optimizations
  lin.hand_coded_optimizations()

  # generate uops from the AST
  lin.linearize()

  # render the source code
  # TODO: move global_size and local_size to runtime_args
  src, runtime_args = device.renderer(lin.function_name, lin.uops)
  if DEBUG >= 4: print(src)

  # move this to renderer?
  if lin.global_size: runtime_args['global_size'] = lin.global_size
  if lin.local_size: runtime_args['local_size'] = lin.local_size

  # compile the source code. TODO: pass in device identifier
  lib: bytes = device.compiler(src)

  # get the function
  return device.runtime(lin.function_name, lib), lin.display_name, runtime_args

local_size_cache = {}
def optimize_local_size(prg:Callable, global_size:List[int], rawbufs:List[RawBuffer]) -> List[int]:
  test_rawbuffers = [type(rawbufs[0])(rawbufs[0].size, rawbufs[0].dtype), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = prg.max_work_group_size() if hasattr(prg, 'max_work_group_size') else 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try:
      return prg(*test_rawbuffers, global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)], local_size=local_size, wait=True)
    except Exception:
      return float('inf')
  return min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])[1]

# *** main schedule runner ***

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  # HACK: images can be not usable due to shape
  if IMAGE >= 2: schedule = fix_schedule_for_images(schedule)

  # NOTE: if you for loop the schedule it's slow because nothing frees
  while len(schedule):
    si = schedule.pop(0)
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"
    if DEBUG >= 3: print_tree(si.ast)
    device = Device[si.out.device]
    if si.ast.op in LoadOps:
      # confirm the LoadOps are contiguous and in order
      for i,s in enumerate(si.ast.src): assert isinstance(s, LazyOp) and s.op == BufferOps.MEM and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"
      LOAD_OPS_DISPATCHER[cast(LoadOps, si.ast.op)](si.out, *si.inputs)
    else:
      lra = {}
      rawbufs = [x.realized for x in si.inputs]
      if isinstance(device, Interpreted):
        fxn = interpret_ast(device, si.ast)
        st = time.perf_counter()
        si.out.realized = fxn(*rawbufs)
        et = time.perf_counter() - st
        # TODO: this shouldn't be needed
        if si.out.realized.dtype != si.out.dtype:
          si.out.realized = device.from_underlying(device.fxn_for_op[UnaryOps.CAST](device.to_underlying(si.out.realized), (si.out.dtype, False)))
        name = "<interpreted>"
      else:
        # compile the program
        prg, name, runtime_args = compile_ast(device, si.ast)
        lra = runtime_args.copy()
        if DEBUG >= 2: lra['wait'] = True

        # symbolic
        if 'global_size' in lra: lra['global_size'] = [sym_infer(sz, si.var_vals) for sz in lra['global_size']]
        if 'local_size' in lra: lra['local_size'] = [sym_infer(sz, si.var_vals) for sz in lra['local_size']]

        # allocate the memory
        si.out.realized = device.buffer(si.out.st.size(), si.out.dtype)
        rawbufs = [si.out.realized] + rawbufs

        # local opt (TODO: confirm it's not symbolic)
        if 'global_size' in lra and 'local_size' not in lra:
          ckey = (device, si.ast)
          if ckey not in local_size_cache: local_size_cache[ckey] = optimize_local_size(prg, lra['global_size'], rawbufs)
          lra['local_size'] = local_size_cache[ckey]

        # add this function to JIT
        from tinygrad.jit import CacheCollector
        CacheCollector.add(prg, rawbufs, si.var_vals)

        # run the program
        if et := prg(*rawbufs, **lra): GlobalCounters.time_sum_s += et

      # get ast info
      info = get_lazyop_info(si.ast)
      op_estimate = sym_infer(info.flops, si.var_vals)
      mem_estimate = sym_infer(info.mem_estimate, si.var_vals)
      if DEBUG >= 2:
        jit = False
        print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(name+' '*(37-ansilen(name)))} arg {len(rawbufs):3d} sz {str(lra.get('global_size')):18s} {str(lra.get('local_size')):12s} OPs {int(op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += op_estimate
      GlobalCounters.global_mem += mem_estimate
    del si.out.op
    for v in si.out.views: del v.op
    assert si.out.realized and isinstance(si.out.realized, Device[si.out.device].buffer), f"device mismatch on realized got {type(si.out.realized)} expected {si.out.device}"
    assert si.out.realized.dtype == si.out.dtype, f"realized dtype is incorrect, {si.out.realized.dtype} != {si.out.dtype}"

# *** zero op LoadOps ***

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***     empty {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***      rand {buffer.device}    seed {buffer.op.arg:<10d}  shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=prod(buffer.shape), dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args())

# *** one op LoadOps ***

from tinygrad.runtime.lib import RawBufferMapped, RawBufferTransfer
from tinygrad.runtime.ops_disk import RawDiskBuffer
def _realize_from(buffer: LazyBuffer, src: LazyBuffer) -> None:
  assert src.realized.size == buffer.st.size(), f"size mismatch on FROM {src.realized.size} != {buffer.st.size()}"
  assert src.st.contiguous and buffer.st.contiguous, "all must be contiguous for from"
  if DEBUG >= 2: print(f"***      copy {buffer.device} <- {src.device} size {src.realized.size:<16d} shape {str(buffer.shape):23s} dtype {src.realized.dtype}")
  # TODO: make this generic
  if isinstance(src.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):
    assert all_int(buffer.shape), "does not support symbolic shape"
    buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())
    src.realized.readinto(cast(RawBufferMapped, buffer.realized)._buffer())
  elif isinstance(src.realized, RawBufferTransfer) and issubclass(Device[buffer.device].buffer, RawBufferTransfer) and getenv("P2P", 0) >= 1:
    buffer.realized = cast(RawBufferTransfer, Device[buffer.device].buffer).transfer(src.realized, buffer.shape, buffer.dtype, **buffer._device_extra_args())
  else:
    # TODO: schedule this as FROM to go to CPU, and a FROM to go to device
    buffer.realized = Device[buffer.device].buffer.fromCPU(src.realized.toCPU(), **buffer._device_extra_args())

# *** n op LoadOps ***

def _realize_custom(buffer: LazyBuffer, *inputs: LazyBuffer) -> None:
  if DEBUG >= 2: print(f"***    custom {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = buffer.op.arg(buffer, *inputs)

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.EMPTY: _realize_empty,
  LoadOps.RAND: _realize_rand,
  LoadOps.FROM: _realize_from,
  LoadOps.CUSTOM: _realize_custom,
}
