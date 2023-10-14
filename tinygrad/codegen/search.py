from typing import Dict, List, cast, DefaultDict, Optional
from copy import deepcopy
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Device, Compiled, MemBuffer
from tinygrad.helpers import prod, getenv
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.runtime.lib import RawBuffer
from collections import defaultdict

from tinygrad.codegen.optimizer import Opt, OptOps
"""
actions = [
  Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=3), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=5), Opt(op=OptOps.UPCAST, axis=0, amt=6), Opt(op=OptOps.UPCAST, axis=0, amt=7),
  Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=1, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=5), Opt(op=OptOps.UPCAST, axis=1, amt=6), Opt(op=OptOps.UPCAST, axis=1, amt=7),
  Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=3), Opt(op=OptOps.UPCAST, axis=2, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=5), Opt(op=OptOps.UPCAST, axis=2, amt=6), Opt(op=OptOps.UPCAST, axis=2, amt=7),
  Opt(op=OptOps.UPCAST, axis=3, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=3, amt=5), Opt(op=OptOps.UPCAST, axis=3, amt=6), Opt(op=OptOps.UPCAST, axis=3, amt=7),
  Opt(op=OptOps.UPCAST, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=3), Opt(op=OptOps.UPCAST, axis=4, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=5),
  Opt(op=OptOps.UPCAST, axis=5, amt=3), Opt(op=OptOps.UPCAST, axis=5, amt=4),
  Opt(op=OptOps.UNROLL, axis=0, amt=2), Opt(op=OptOps.UNROLL, axis=0, amt=3), Opt(op=OptOps.UNROLL, axis=0, amt=4), Opt(op=OptOps.UNROLL, axis=0, amt=5), Opt(op=OptOps.UNROLL, axis=0, amt=6), Opt(op=OptOps.UNROLL, axis=0, amt=7), Opt(op=OptOps.UNROLL, axis=0, amt=8), Opt(op=OptOps.UNROLL, axis=0, amt=9), Opt(op=OptOps.UNROLL, axis=0, amt=10), Opt(op=OptOps.UNROLL, axis=0, amt=11), Opt(op=OptOps.UNROLL, axis=0, amt=12), Opt(op=OptOps.UNROLL, axis=0, amt=13), Opt(op=OptOps.UNROLL, axis=0, amt=14), Opt(op=OptOps.UNROLL, axis=0, amt=15), Opt(op=OptOps.UNROLL, axis=0, amt=16), Opt(op=OptOps.UNROLL, axis=0, amt=17), Opt(op=OptOps.UNROLL, axis=0, amt=20), Opt(op=OptOps.UNROLL, axis=0, amt=21), Opt(op=OptOps.UNROLL, axis=0, amt=24), Opt(op=OptOps.UNROLL, axis=0, amt=25), Opt(op=OptOps.UNROLL, axis=0, amt=27), Opt(op=OptOps.UNROLL, axis=0, amt=28), Opt(op=OptOps.UNROLL, axis=0, amt=30), Opt(op=OptOps.UNROLL, axis=0, amt=31), Opt(op=OptOps.UNROLL, axis=0, amt=32),
  Opt(op=OptOps.UNROLL, axis=1, amt=2), Opt(op=OptOps.UNROLL, axis=1, amt=3), Opt(op=OptOps.UNROLL, axis=1, amt=4), Opt(op=OptOps.UNROLL, axis=1, amt=5), Opt(op=OptOps.UNROLL, axis=1, amt=6), Opt(op=OptOps.UNROLL, axis=1, amt=7), Opt(op=OptOps.UNROLL, axis=1, amt=8), Opt(op=OptOps.UNROLL, axis=1, amt=9), Opt(op=OptOps.UNROLL, axis=1, amt=10), Opt(op=OptOps.UNROLL, axis=1, amt=11), Opt(op=OptOps.UNROLL, axis=1, amt=16), Opt(op=OptOps.UNROLL, axis=1, amt=24), Opt(op=OptOps.UNROLL, axis=1, amt=26), Opt(op=OptOps.UNROLL, axis=1, amt=27), Opt(op=OptOps.UNROLL, axis=1, amt=28), Opt(op=OptOps.UNROLL, axis=1, amt=32),
  Opt(op=OptOps.UNROLL, axis=2, amt=2), Opt(op=OptOps.UNROLL, axis=2, amt=3), Opt(op=OptOps.UNROLL, axis=2, amt=4), Opt(op=OptOps.UNROLL, axis=2, amt=5), Opt(op=OptOps.UNROLL, axis=2, amt=6), Opt(op=OptOps.UNROLL, axis=2, amt=7), Opt(op=OptOps.UNROLL, axis=2, amt=8), Opt(op=OptOps.UNROLL, axis=2, amt=9), Opt(op=OptOps.UNROLL, axis=2, amt=11), Opt(op=OptOps.UNROLL, axis=2, amt=12), Opt(op=OptOps.UNROLL, axis=2, amt=13), Opt(op=OptOps.UNROLL, axis=2, amt=14), Opt(op=OptOps.UNROLL, axis=2, amt=16), Opt(op=OptOps.UNROLL, axis=2, amt=24), Opt(op=OptOps.UNROLL, axis=2, amt=26), Opt(op=OptOps.UNROLL, axis=2, amt=27), Opt(op=OptOps.UNROLL, axis=2, amt=28), Opt(op=OptOps.UNROLL, axis=2, amt=30), Opt(op=OptOps.UNROLL, axis=2, amt=31), Opt(op=OptOps.UNROLL, axis=2, amt=32),
  Opt(op=OptOps.UNROLL, axis=3, amt=2), Opt(op=OptOps.UNROLL, axis=3, amt=3), Opt(op=OptOps.UNROLL, axis=3, amt=11),
  Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=3), Opt(op=OptOps.LOCAL, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=8), Opt(op=OptOps.LOCAL, axis=0, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=32),
  Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.LOCAL, axis=1, amt=3), Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=1, amt=8), Opt(op=OptOps.LOCAL, axis=1, amt=16),
  Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.LOCAL, axis=2, amt=3), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=16),
  Opt(op=OptOps.LOCAL, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=3, amt=3), Opt(op=OptOps.LOCAL, axis=3, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=8), Opt(op=OptOps.LOCAL, axis=3, amt=16),
  Opt(op=OptOps.LOCAL, axis=4, amt=2), Opt(op=OptOps.LOCAL, axis=4, amt=3), Opt(op=OptOps.LOCAL, axis=4, amt=16),
  Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.GROUP, axis=2, amt=8),
  Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.GROUPTOP, axis=1, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.GROUPTOP, axis=2, amt=256)]
"""
actions = [
  Opt(op=OptOps.UPCAST, axis=0, amt=0), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=3), Opt(op=OptOps.UPCAST, axis=0, amt=4),
  Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=1, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=4),
  Opt(op=OptOps.UPCAST, axis=2, amt=0), Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=3), Opt(op=OptOps.UPCAST, axis=2, amt=4),
  Opt(op=OptOps.UPCAST, axis=3, amt=0), Opt(op=OptOps.UPCAST, axis=3, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=4),
  Opt(op=OptOps.UPCAST, axis=4, amt=0), Opt(op=OptOps.UPCAST, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=3), Opt(op=OptOps.UPCAST, axis=4, amt=4),
  Opt(op=OptOps.UPCAST, axis=5, amt=0), Opt(op=OptOps.UPCAST, axis=5, amt=2), Opt(op=OptOps.UPCAST, axis=5, amt=3), Opt(op=OptOps.UPCAST, axis=5, amt=4),
  Opt(op=OptOps.UNROLL, axis=0, amt=0), Opt(op=OptOps.UNROLL, axis=0, amt=4),
  Opt(op=OptOps.UNROLL, axis=1, amt=0), Opt(op=OptOps.UNROLL, axis=1, amt=4),
  Opt(op=OptOps.UNROLL, axis=2, amt=0), Opt(op=OptOps.UNROLL, axis=2, amt=4),
  Opt(op=OptOps.UNROLL, axis=3, amt=0),
  Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=3), Opt(op=OptOps.LOCAL, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=8), Opt(op=OptOps.LOCAL, axis=0, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=32),
  Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.LOCAL, axis=1, amt=3), Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=1, amt=8), Opt(op=OptOps.LOCAL, axis=1, amt=16),
  Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.LOCAL, axis=2, amt=3), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=16),
  Opt(op=OptOps.LOCAL, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=3, amt=3), Opt(op=OptOps.LOCAL, axis=3, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=8), Opt(op=OptOps.LOCAL, axis=3, amt=16),
  Opt(op=OptOps.LOCAL, axis=4, amt=2), Opt(op=OptOps.LOCAL, axis=4, amt=3), Opt(op=OptOps.LOCAL, axis=4, amt=4), Opt(op=OptOps.LOCAL, axis=4, amt=8), Opt(op=OptOps.LOCAL, axis=4, amt=16),
  Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.GROUP, axis=2, amt=8),
  Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.GROUPTOP, axis=1, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.GROUPTOP, axis=2, amt=256)
]
device:Compiled = cast(Compiled, Device[Device.DEFAULT])

# returns time in seconds
logtm = open(getenv("LOGTM", ""),"a") if getenv("LOGTM", "") else None
def time_linearizer(lin:Linearizer, rawbufs:List[RawBuffer], allow_test_size=True, max_global_size=65536, cnt=3, should_copy=True) -> float:
  if should_copy: lin = deepcopy(lin)  # TODO: remove the need for this
  var_vals = {k:k.min for k in vars_from_ast(lin.ast)}
  try:
    lin.linearize()
    prg = device.to_program(lin)
    real_global_size = prg.global_size[:]
    if allow_test_size:
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
    tms = [prg(rawbufs, var_vals, force_wait=True)*factor for _ in range(cnt)]
    prg.global_size = real_global_size
  except Exception:
    #print("FAILED")
    #print(lin.ast)
    #print(lin.applied_opts)
    tms = [float('inf')]
  if logtm: logtm.write(str((lin.ast, lin.applied_opts, tms))+"\n")
  return min(tms)

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Linearizer) -> List[RawBuffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[RawBuffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    rawbufs[k] = device.buffer(max(y.st.size() for y in lx), lx[0].dtype)
  assert all(r is not None for r in rawbufs)
  return cast(List[RawBuffer], rawbufs)

# get dictionary of all possible actions
def get_linearizer_actions(lin:Linearizer) -> Dict[int, Linearizer]:
  acted_lins = {0:deepcopy(lin)}
  for i,a in enumerate(actions):
    lin2 = deepcopy(lin)
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
