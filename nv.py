from os import getenv
from typing import List, cast
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import prod
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Compiled, Device, LoadOps
from tinygrad.runtime.lib import RawBuffer
from tinygrad.tensor import Tensor
import os

os.environ["TRITON"] = "0"
os.environ["CUDA"] = "1"

import shelve
logtm = shelve.open(getenv("LOGTM", "")) if getenv("LOGTM", "") else None
def time_linearizer(lin:Linearizer, rawbufs:List[RawBuffer], allow_test_size=True, max_global_size=65536, cnt=3, should_copy=True, disable_cache=False) -> float:
  print("timing for lin", lin, rawbufs)
  key = str((lin.ast, lin.applied_opts))
  if should_copy and not disable_cache and logtm is not None and key in logtm: return min(logtm[key])  # pylint: disable=E1135 # NOTE: we check should_copy since this may have side effects
  if should_copy: lin = lin.copy() # TODO: remove the need for this
  var_vals = {k:k.min for k in vars_from_ast(lin.ast)}
  print("all vars", var_vals)
  try:
    lin.linearize()
    prg = cast(Compiled, Device[Device.DEFAULT]).to_program(lin)
    print("prg is", prg.prg)
    real_global_size = prg.global_size
    print("real_global_size", real_global_size)
    if allow_test_size and prg.global_size:
      test_global_size = prg.global_size[:]
      while prod(test_global_size) > max_global_size:
        for j in range(2,-1,-1):
          if test_global_size[j] > 16:
            test_global_size[j] //= 2
            break
      print("test_global_size", test_global_size)
      factor = prod(prg.global_size) / prod(test_global_size)
      prg.global_size = test_global_size
      print("factor", factor)
      #print(real_global_size, test_global_size, factor)
    else:
      print("-- setting factor to 1")
      factor = 1
    # TODO: this is super broken for var_vals
    # TODO: this is copied from prg.__call__
    global_size, local_size = prg.launch_dims(var_vals)
    print("launch with", global_size, local_size)
    if global_size is not None and local_size is None:
      local_size = prg.optimize_local_size(global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      print("final sizes", global_size, local_size)
    tms = [prg.clprg(global_size, local_size, *rawbufs, *var_vals.values(), wait=True)*factor for _ in range(cnt)]
    print("all times", tms)
    prg.global_size = real_global_size
    print(prg.global_size)
  except Exception:
    import traceback; traceback.print_exc()
    print("==========FAILED")
    print(lin.ast)
    print(lin.applied_opts)
    tms = [float('inf')]
  if logtm is not None: logtm[key] = tms
  return min(tms)

print(Device.DEFAULT)
si = [si for si in Tensor([1,2,3,4]).add(1).lazydata.schedule() if si.ast.op not in LoadOps][0]
rawbufs = [Device[Device.DEFAULT].buffer(si.out.st.size(), si.out.dtype)] + [Device[Device.DEFAULT].buffer(x.st.size(), x.dtype) for x in si.inputs]
tm = time_linearizer(Linearizer(si.ast), rawbufs, allow_test_size=False, cnt=10, should_copy=False)
print(tm)

