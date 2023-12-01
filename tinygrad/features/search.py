from typing import Dict, List, cast, DefaultDict, Optional, Tuple, Callable
import itertools, random, math, time
from tinygrad.lazy import vars_from_ast
from tinygrad.device import Device, Compiled, Buffer
from tinygrad.ops import MemBuffer
from tinygrad.helpers import prod, ImageDType, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, Context, all_int, colored, Timing
from tinygrad.codegen.linearizer import Linearizer, UOp
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

# returns time in seconds
def time_linearizer(lin:Linearizer, rawbufs:List[Buffer], allow_test_size=True, max_global_size=65536, cnt=3, disable_cache=False, clear_l2=False) -> float:
  key = {"ast": str(lin.ast), "opts": str(lin.applied_opts), "allow_test_size": allow_test_size, "max_global_size": max_global_size, "clear_l2": clear_l2, "device": Device.DEFAULT}
  if not disable_cache and CACHELEVEL >= 2 and (val:=diskcache_get("time_linearizer", key)) is not None: return min(val)

  # Set the midpoint value value for var_vals to optimize shapes.
  var_vals = {k:(k.max+k.min)//2 for k in vars_from_ast(lin.ast)}
  try:
    lin.linearize()
    prg = cast(Compiled, Device[Device.DEFAULT]).to_program(lin)
    real_global_size = prg.global_size
    if allow_test_size and prg.global_size and all_int(tuple(prg.global_size)):
      test_global_size = prg.global_size[:]
      while prod(test_global_size) > max_global_size:
        for j in range(2,-1,-1):
          if test_global_size[j] > 16:
            test_global_size[j] //= 2
            break
      factor = prod(prg.global_size) / prod(test_global_size)
      prg.global_size = test_global_size
      #print(real_global_size, test_global_size, factor)
    else:
      factor = 1

    # TODO: this is copied from prg.__call__
    global_size, local_size = prg.launch_dims(var_vals)
    prg.global_size = real_global_size
    if global_size is not None and prg.global_size is not None and local_size is None and all_int(tuple(prg.global_size)):
      local_size = optimize_local_size(prg.clprg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]

    lra = prg.runtime_args.copy()
    if global_size: lra['global_size'] = global_size
    if local_size: lra['local_size'] = local_size

    tms = []
    for _ in range(cnt):
      if clear_l2:
        # TODO: this is too small for many L2 caches
        with Context(DEBUG=0): Tensor.rand(1024,1024).realize()
      tms.append(prg.clprg(*[x._buf for x in rawbufs], *var_vals.values(), **lra, wait=True)*factor)
  except Exception:
    if DEBUG >= 4:
      import traceback
      traceback.print_exc()
      print("FAILED")
      print(lin.ast)
      print(lin.applied_opts)
    tms = [float('inf')]
  if CACHELEVEL >= 2: diskcache_put("time_linearizer", key, tms)
  return min(tms)

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

def tuplize_uops(uops:List[UOp]) -> Tuple: return tuple([(x.uop, x.dtype, tuple(uops.index(x) for x in x.vin), x.arg) for x in uops])

def beam_search(lin:Linearizer, rawbufs, amt:int, allow_test_size=True) -> Linearizer:
  key = {"ast": str(lin.ast), "amt": amt, "allow_test_size": allow_test_size, "device": Device.DEFAULT}
  if (val:=diskcache_get("beam_search", key)) is not None and not getenv("IGNORE_BEAM_CACHE") and CACHELEVEL >= 1:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  # init the BEAM with the base linearizer
  beam: List[Tuple[Linearizer, float]] = [(lin, time_linearizer(lin, rawbufs, allow_test_size=allow_test_size))]

  # NOTE: real uops use a weird compare method that's only valid inside a linearizer
  seen_uops = {tuplize_uops(lin.linearize().uops): tuple(lin.applied_opts)}

  exiting, st = False, time.perf_counter()
  while not exiting:
    with Timing("linearize:  ", enabled=DEBUG>=3):
      acted_lins = flatten([get_linearizer_actions(lin, include_0=False).values() for lin,_ in beam])

      # linearize all
      for x in acted_lins: x.linearize()

      # dedup with uops
      acted_lins_dedup = []
      for lin in acted_lins:
        tuops = tuplize_uops(lin.uops)
        if tuops in seen_uops: continue
        seen_uops[tuops] = tuple(lin.applied_opts)
        acted_lins_dedup.append(lin)

    with Timing("compile:    ",enabled=DEBUG>=3):
      # time linearizers
      timed_lins: List[Tuple[Linearizer, float]] = [(v,time_linearizer(v,rawbufs,allow_test_size=allow_test_size)) for v in acted_lins_dedup]
      opts = sorted(timed_lins, key=lambda x: x[1])

    # done
    exiting = len(opts) == 0 or beam[0][1] <= opts[0][1]
    if not exiting: beam = opts[:amt]
    if DEBUG >= 2: print(f"{time.perf_counter() - st:7.2f}s:", colored(f"{beam[0][1]*1e6:12.2f} us", "green" if exiting else None), f"from {len(acted_lins):3d} -> {len(opts):3d} actions", beam[0][0].colored_shape())

  if CACHELEVEL >= 1: diskcache_put("beam_search", key, beam[0][0].applied_opts)
  if DEBUG >= 3: print(beam[0][0].applied_opts)
  return beam[0][0]

def optimize_local_size(clprg:Callable, global_size:List[int], rawbufs:List[Buffer]) -> List[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = clprg.max_work_group_size() if hasattr(clprg, 'max_work_group_size') else 1024
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
