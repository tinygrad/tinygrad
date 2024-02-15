from typing import Dict, List, cast, DefaultDict, Optional, Tuple, Callable
import itertools, functools, random, math, time, multiprocessing, traceback, signal
from tinygrad.device import Device, Compiled, Buffer, CompiledASTRunner, Compiler
from tinygrad.ops import MemBuffer, LazyOp
from tinygrad.helpers import prod, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, Context, colored, to_function_name
from tinygrad.dtype import ImageDType
from tinygrad.codegen.linearizer import Linearizer
from collections import defaultdict
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import sym_infer

from tinygrad.codegen.kernel import Opt, OptOps
actions = [Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,7] for axis in range(6)]
actions += [Opt(op=OptOps.UNROLL, axis=axis, amt=amt) for amt in [0,4] for axis in range(4)]
actions += [Opt(op=OptOps.LOCAL, axis=axis, amt=amt) for amt in [2,3,4,8,13,16,29] for axis in range(5)]
actions += [Opt(op=OptOps.GROUPTOP, axis=axis, amt=amt) for amt in [13,16,29,32,256] for axis in range(3)]
actions += [Opt(op=OptOps.PADTO, axis=axis, amt=amt) for amt in [32] for axis in range(7)]
actions += [Opt(op=OptOps.LOCAL, axis=0, amt=32), Opt(op=OptOps.UPCASTMID, axis=1, amt=4),
            Opt(op=OptOps.GROUP, axis=0, amt=4), Opt(op=OptOps.GROUP, axis=0, amt=8), Opt(op=OptOps.GROUP, axis=1, amt=8),]
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def _get_test_global_size(global_size, max_global_size, var_vals):
  test_global_size, factor = [sym_infer(sz, var_vals) for sz in global_size], 1
  while prod(test_global_size) > max_global_size:
    for j in range(len(global_size)-1,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        factor *= 2
        break
  return test_global_size, factor

def _time_program(ast:LazyOp, rdev:Compiled, lib:bytes, global_size, local_size, var_vals, rawbufs, early_stop=None, max_global_size=65536, clear_l2=False, cnt=3, name="test"):  # noqa: E501
  factor = 1
  if global_size is not None and max_global_size is not None:
    global_size, factor = _get_test_global_size(global_size, max_global_size, var_vals)
  try: car = CompiledASTRunner(ast, name, "", rdev, global_size, local_size, precompiled=lib)
  except AssertionError: return [math.inf] * cnt
  tms = []
  for _ in range(cnt):
    if clear_l2:
      with Context(DEBUG=0): Tensor.rand(1024,1024).realize()
    tms.append(cast(float, car(rawbufs, var_vals, wait=True, do_update_stats=False))*factor)
    if early_stop is not None and early_stop < tms[-1]: break
  return tms

def _compile_linearizer(compiler:Compiler, lin:Linearizer, name:Optional[str]=None) -> Tuple[bytes, Optional[List[int]], Optional[List[int]]]:
  lin.linearize()
  src = compiler.render(name if name is not None else to_function_name(lin.name), lin.uops)   # NOTE: these all have the same name for deduping
  return compiler.compile(src), lin.global_size, lin.local_size

def _try_compile_linearized_w_idx(x, compiler:Compiler):
  try: return (x[0], _compile_linearizer(compiler, x[1], "test"))
  except Exception:
    if DEBUG >= 4: traceback.print_exc()
    return (x[0], None)

# workers should ignore ctrl c
def _init_worker(): signal.signal(signal.SIGINT, signal.SIG_IGN)

# *** external API ***

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Linearizer) -> List[Buffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[Buffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    buf_size = prod(lx[0].dtype.shape) if isinstance(lx[0].dtype, ImageDType) else max(y.st.real_size() for y in lx)
    rawbufs[k] = Buffer(lin.opts.device, buf_size, lx[0].dtype)
  assert all(r is not None for r in rawbufs)
  return cast(List[Buffer], rawbufs)

# get dictionary of all possible actions
def get_linearizer_actions(lin:Linearizer, include_0=True) -> Dict[int, Linearizer]:
  acted_lins = {0:lin} if include_0 else {}
  for i,a in enumerate(actions):
    if a.axis is not None and a.axis >= lin.shape_len: continue
    if a.axis is not None and lin.full_shape[a.axis] == a.amt and Opt(a.op, a.axis, 0) in actions: continue
    lin2 = lin.copy()
    try:
      lin2.apply_opt(a)
      up, lcl = 1, 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        if c in {"cyan", "green", "white"}: lcl *= s
      if up > 256 or lcl > 256 or ("green" in lin2.colors() and up*lcl > 2 ** 15): continue
      acted_lins[i+1] = lin2
    except Exception:
      pass
  return acted_lins

beam_pool = None
def beam_search(lin:Linearizer, rawbufs, amt:int, allow_test_size=True) -> Linearizer:
  global beam_pool
  key = {"ast": str(lin.ast), "amt": amt, "allow_test_size": allow_test_size, "device": lin.opts.device}
  if (val:=diskcache_get("beam_search", key)) is not None and not getenv("IGNORE_BEAM_CACHE") and CACHELEVEL >= 1:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  beam: List[Tuple[Linearizer, float]] = []
  seen_libs = set()

  default_parallel = 1 if lin.opts.device in {"CUDA", "HIP"} else 0
  if beam_pool is None and getenv("PARALLEL", default_parallel): beam_pool = multiprocessing.Pool(multiprocessing.cpu_count(), _init_worker)

  try:
    var_vals = {k:(k.max+k.min)//2 for k in lin.ast.vars()}
    exiting, st = False, time.perf_counter()
    dev = Device[lin.opts.device]
    assert isinstance(dev, Compiled)
    while not exiting:
      acted_lins = flatten([get_linearizer_actions(lin, include_0=False).values() for lin,_ in beam]) if len(beam) else [lin]
      timed_lins: List[Tuple[Linearizer, float]] = []
      _compile_fn = functools.partial(_try_compile_linearized_w_idx, compiler=cast(Compiled, Device[lin.opts.device]).compiler)
      for i,proc in (map(_compile_fn, enumerate(acted_lins)) if beam_pool is None else beam_pool.imap_unordered(_compile_fn, enumerate(acted_lins))):
        if proc is None: continue
        lib, global_size, local_size = proc
        if lib in seen_libs: continue
        seen_libs.add(lib)
        tms = _time_program(lin.ast, dev, lib, global_size, local_size, var_vals, rawbufs, early_stop=beam[0][1]*3 if len(beam) else 1.0)
        timed_lins.append((acted_lins[i], min(tms)))
        if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s: {timed_lins[-1][1]*1e6:12.2f} us       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}\033[K", end="")  # noqa: E501

      # done
      opts = sorted(timed_lins, key=lambda x: x[1])
      exiting = len(opts) == 0 or (len(beam) > 0 and beam[0][1] <= opts[0][1])
      if not exiting: beam = opts[:amt]
      assert len(beam) > 0, "no BEAM items succeeded?!?"
      if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s:", colored(f"{beam[0][1]*1e6:12.2f} us", "green" if exiting else None), f"from {len(acted_lins):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())  # noqa: E501
  except KeyboardInterrupt as e:
    if beam_pool is not None: beam_pool.terminate()
    raise e

  if CACHELEVEL >= 1: diskcache_put("beam_search", key, beam[0][0].applied_opts)
  if DEBUG >= 3: print(beam[0][0].applied_opts)
  return beam[0][0]

def optimize_local_size(clprg:Callable, global_size:List[int], rawbufs:List[Buffer]) -> List[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try: return clprg(*[x._buf for x in test_rawbuffers], global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)], local_size=local_size, wait=True)  # noqa: E501
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

def time_linearizer(lin:Linearizer, rawbufs:List[Buffer], allow_test_size=True, max_global_size=65536, cnt=3, disable_cache=False, clear_l2=False) -> float:  # noqa: E501
  key = {"ast": str(lin.ast), "opts": str(lin.applied_opts), "allow_test_size": allow_test_size, "max_global_size": max_global_size, "clear_l2": clear_l2, "device": lin.opts.device}  # noqa: E501
  if not disable_cache and CACHELEVEL >= 2 and (val:=diskcache_get("time_linearizer", key)) is not None: return min(val)

  dev = Device[lin.opts.device]
  assert isinstance(dev, Compiled)

  var_vals = {k:(k.max+k.min)//2 for k in lin.ast.vars()}
  lib, global_size, local_size = _compile_linearizer(dev.compiler, lin)
  tms = _time_program(lin.ast, dev, lib, global_size, local_size, var_vals, rawbufs, max_global_size=max_global_size if allow_test_size else None, clear_l2=clear_l2, cnt=cnt, name=to_function_name(lin.name))  # noqa: E501

  if CACHELEVEL >= 2: diskcache_put("time_linearizer", key, tms)
  return min(tms)
