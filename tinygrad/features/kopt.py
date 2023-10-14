from typing import Callable
import time
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.optimizer import Opt, OptOps
from tinygrad.helpers import DEBUG, prod, getenv
from tinygrad.lazy import vars_from_ast

def get_divisors(n, min_div = 1, max_div = 512):
  if min_div > 1: yield 1
  for d in range(min_div, min(max_div, n//2) + 1):
    if n % d == 0: yield d

def kernel_optimize_opts(k:Linearizer):
  import nevergrad as ng
  opts = []
  for i in range(k.first_reduce):
    # TODO: the upcast always happen first, you might want to reverse this?
    # TODO: the order of the locals might improve things too
    opts.append(ng.p.TransitionChoice([Opt(OptOps.UPCAST,i,s) for s in get_divisors(k.full_shape[i], max_div=8)]))
    opts.append(ng.p.TransitionChoice([Opt(OptOps.LOCAL,i,s) for s in get_divisors(k.full_shape[i], min_div=4)]))
  for i in range(k.shape_len-k.first_reduce):
    opts.append(ng.p.TransitionChoice([Opt(OptOps.UNROLL,i,s) for s in get_divisors(k.full_shape[k.first_reduce+i], max_div=8)]))
  return opts

def kernel_optimize_search(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, baseline, bufs, var_vals):
  import nevergrad as ng
  def opt(x):
    try:
      k = create_k()
      for o in x: k.apply_opt(o)
      prg = to_prg(k)
      first_tm = prg.exec(bufs, var_vals, force_wait=True, optimizing=True)
      if baseline*5 < first_tm*1000: return first_tm*1000  # very slow
      tm = min([first_tm]+[prg.exec(bufs, var_vals, force_wait=True, optimizing=True) for _ in range(2)])*1000
      return tm
    except Exception:
      if DEBUG >= 3:
        import traceback
        traceback.print_exc()
      return 10000_000   # 10000 seconds is infinity
  opts = kernel_optimize_opts(k)
  if not opts: return "BASELINE"
  search_space = prod([len(x.choices) for x in opts])
  st = time.perf_counter()
  budget = getenv("BUDGET", 200)
  optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Tuple(*opts), budget=min(search_space, budget))
  recommendation = optimizer.minimize(opt)
  et = time.perf_counter() - st
  if DEBUG >= 1: print(f"optimizer({et:6.2f} s to search) space {search_space:8d} with tm {recommendation.loss:5.2f} ms vs baseline {baseline:5.2f} ms, a {baseline/recommendation.loss:5.2f}x gain : {k.colored_shape()}")
  return recommendation.value if recommendation.loss < baseline else "BASELINE"

# optimization
global_db = None
def kernel_optimize(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, bufs, key):
  global global_db

  skey = str(key)

  if getenv("KOPT") == 2 and global_db is None:
    import shelve
    global_db = shelve.open("/tmp/kopt_cache")

  if global_db is not None and skey in global_db:
    choice = global_db[skey]
  elif k.has_variable_shape():
    # don't optimize variable shapes
    choice = "BASELINE"
  else:
    var_vals = {k:k.min for k in vars_from_ast(k.ast)}
    # get baseline
    def get_baseline():
      k = create_k()
      k.hand_coded_optimizations()
      prg = to_prg(k)
      return min([prg.exec(bufs, var_vals, force_wait=True, optimizing=True) for _ in range(5)])*1000
    choice = kernel_optimize_search(k, create_k, to_prg, get_baseline(), bufs, var_vals)
    if global_db is not None:
      global_db[skey] = choice
      global_db.sync()

  if choice == "BASELINE":
    k.hand_coded_optimizations()
  else:
    for o in choice: k.apply_opt(o)