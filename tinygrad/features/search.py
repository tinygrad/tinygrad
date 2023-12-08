from typing import Dict, List, cast, DefaultDict, Optional, Tuple, Callable
import itertools, random, math, time, multiprocessing, traceback, signal
from tinygrad.device import Device, Compiled, Buffer
from tinygrad.ops import MemBuffer, vars_from_ast
from tinygrad.helpers import prod, ImageDType, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, Context, colored, to_function_name
from tinygrad.codegen.linearizer import Linearizer, UOp
from tinygrad.shape.symbolic import sym_infer
from collections import defaultdict
from tinygrad.tensor import Tensor

from tinygrad.codegen.kernel import Opt, OptOps
actions = flatten([[Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,7]] for axis in range(6)])
actions += flatten([[Opt(op=OptOps.UNROLL, axis=axis, amt=amt) for amt in [0,4]] for axis in range(4)])
actions += flatten([[Opt(op=OptOps.LOCAL, axis=axis, amt=amt) for amt in [2,3,4,8,13,16,29]] for axis in range(5)])
actions += flatten([[Opt(op=OptOps.GROUPTOP, axis=axis, amt=amt) for amt in [13,16,29,32,256]] for axis in range(3)])
actions += flatten([[Opt(op=OptOps.PADTO, axis=axis, amt=amt) for amt in [32]] for axis in range(7)])
actions += [
  Opt(op=OptOps.LOCAL, axis=0, amt=32),
  Opt(op=OptOps.GROUP, axis=0, amt=4), Opt(op=OptOps.GROUP, axis=0, amt=8), Opt(op=OptOps.GROUP, axis=1, amt=8),
  Opt(op=OptOps.UPCASTMID, axis=1, amt=4),
]
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def tuplize_uops(uops:List[UOp]) -> Tuple: return tuple([(x.uop, x.dtype, tuple(uops.index(x) for x in x.vin), x.arg) for x in uops])

def get_test_global_size(global_size, max_global_size):
  test_global_size = global_size[:]
  while prod(test_global_size) > max_global_size:
    for j in range(2,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        break
  factor = prod(global_size) / prod(test_global_size)
  return test_global_size, factor

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Linearizer) -> List[Buffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[Buffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    rawbufs[k] = Buffer(Device.DEFAULT, prod(lx[0].dtype.shape) if isinstance(lx[0].dtype, ImageDType) else max(y.st.size() for y in lx), lx[0].dtype)
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
      if up > 256 or lcl > 256: continue
      acted_lins[i+1] = lin2
    except Exception:
      pass
  return acted_lins

def try_compile_linearized_w_idx(x):
  try: return (x[0], compile_linearizer(Device.DEFAULT, x[1], "test"))
  except Exception:
    if DEBUG >= 4: traceback.print_exc()
    return (x[0], None)

def compile_linearizer(dev:str, lin:Linearizer, name:Optional[str]=None) -> Tuple[bytes, Optional[List[int]], Optional[List[int]]]:
  lin.linearize()
  rdev = Device[dev]
  assert isinstance(rdev, Compiled)
  src, _ = rdev.renderer(name if name is not None else to_function_name(lin.name), lin.uops)   # NOTE: these all have the same name for deduping
  return rdev.compiler(src), lin.global_size, lin.local_size

def time_program(dev:str, lib:bytes, global_size, local_size, var_vals, rawbufs, early_stop=None, max_global_size=65536, clear_l2=False, cnt=3, name="test"):
  rdev = Device[dev]
  assert isinstance(rdev, Compiled)
  clprg = rdev.runtime(name, lib)
  factor = 1
  if global_size is not None:
    global_size = [sym_infer(sz, var_vals) for sz in global_size] + [1]*(3-len(global_size))
    if local_size is None:
      local_size = optimize_local_size(clprg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    else:
      local_size = [sym_infer(sz, var_vals) for sz in local_size] + [1]*(3-len(local_size))
    if max_global_size is not None:
      global_size, factor = get_test_global_size(global_size, max_global_size=max_global_size)
  lra = {}
  if global_size: lra['global_size'] = global_size
  if local_size: lra['local_size'] = local_size
  tms = []
  for _ in range(cnt):
    if clear_l2:
      with Context(DEBUG=0): Tensor.rand(1024,1024).realize()
    tms.append(clprg(*[x._buf for x in rawbufs], **lra, vals=var_vals.values(), wait=True)*factor)
    if early_stop is not None and early_stop < tms[-1]: break
  return tms

# workers should ignore ctrl c
def init_worker(): signal.signal(signal.SIGINT, signal.SIG_IGN)

def beam_search(lin:Linearizer, rawbufs, amt:int, allow_test_size=True) -> Linearizer:
  key = {"ast": str(lin.ast), "amt": amt, "allow_test_size": allow_test_size, "device": Device.DEFAULT}
  if (val:=diskcache_get("beam_search", key)) is not None and not getenv("IGNORE_BEAM_CACHE") and CACHELEVEL >= 1:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  beam: List[Tuple[Linearizer, float]] = []
  seen_libs = set()

  default_parallel = 1 if Device.DEFAULT in {"CUDA", "HIP"} else 0
  pool = multiprocessing.Pool(multiprocessing.cpu_count(), init_worker) if getenv("PARALLEL", default_parallel) else None

  try:
    var_vals = {k:(k.max+k.min)//2 for k in vars_from_ast(lin.ast)}
    exiting, st = False, time.perf_counter()
    dev = Device[Device.DEFAULT]
    assert isinstance(dev, Compiled)
    while not exiting:
      acted_lins = flatten([get_linearizer_actions(lin, include_0=False).values() for lin,_ in beam]) if len(beam) else [lin]
      timed_lins: List[Tuple[Linearizer, float]] = []
      for i,proc in (pool.imap_unordered(try_compile_linearized_w_idx, enumerate(acted_lins)) if pool is not None else map(try_compile_linearized_w_idx, enumerate(acted_lins))):
        if proc is None: continue
        lib, global_size, local_size = proc
        if lib in seen_libs: continue
        seen_libs.add(lib)
        tms = time_program(Device.DEFAULT, lib, global_size, local_size, var_vals, rawbufs, early_stop=beam[0][1]*3 if len(beam) else 1.0)   # > 1 second, run one time
        timed_lins.append((acted_lins[i], min(tms)))
        if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s: {timed_lins[-1][1]*1e6:12.2f} us       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}\033[K", end="")

      # done
      opts = sorted(timed_lins, key=lambda x: x[1])
      exiting = len(opts) == 0 or (len(beam) > 0 and beam[0][1] <= opts[0][1])
      if not exiting: beam = opts[:amt]
      assert len(beam) > 0, "no BEAM items succeeded?!?"
      if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s:", colored(f"{beam[0][1]*1e6:12.2f} us", "green" if exiting else None), f"from {len(acted_lins):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())
    if pool is not None: pool.close()    # the pool is closed
  except KeyboardInterrupt as e:
    if pool is not None: pool.terminate()
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
    try:
      return clprg(*[x._buf for x in test_rawbuffers], global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)], local_size=local_size, wait=True)
    except Exception:
      return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

def time_linearizer(lin:Linearizer, rawbufs:List[Buffer], allow_test_size=True, max_global_size=65536, cnt=3, disable_cache=False, clear_l2=False) -> float:
  key = {"ast": str(lin.ast), "opts": str(lin.applied_opts), "allow_test_size": allow_test_size, "max_global_size": max_global_size, "clear_l2": clear_l2, "device": Device.DEFAULT}
  if not disable_cache and CACHELEVEL >= 2 and (val:=diskcache_get("time_linearizer", key)) is not None: return min(val)

  var_vals = {k:(k.max+k.min)//2 for k in vars_from_ast(lin.ast)}
  lib, global_size, local_size = compile_linearizer(Device.DEFAULT, lin)
  tms = time_program(Device.DEFAULT, lib, global_size, local_size, var_vals, rawbufs, max_global_size=max_global_size if allow_test_size else None, clear_l2=clear_l2, cnt=cnt, name=to_function_name(lin.name))

  if CACHELEVEL >= 2: diskcache_put("time_linearizer", key, tms)
  return min(tms)
