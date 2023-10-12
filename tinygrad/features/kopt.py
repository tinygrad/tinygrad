from typing import Callable, Dict, Tuple
import time
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.optimizer import OptOps, Opt
from tinygrad.helpers import DEBUG, prod, getenv
from tinygrad.lazy import vars_from_ast

def get_divisors(n, min_div = 1, max_div = 512, extra=None):
  if min_div > 1: yield 1
  if extra is not None and extra < min_div and extra != 1 and n % extra == 0: yield extra
  for d in range(min_div, max_div + 1):
    if n % d == 0: yield d
  if extra is not None and extra > max_div and extra != 1 and n % extra == 0: yield extra

def kernel_optimize_opts(k:Linearizer, suggestion):
  import nevergrad as ng
  suggestion = suggestion_to_dict(suggestion)
  opts = []
  for i in range(k.first_reduce):
    # TODO: the upcast always happen first, you might want to reverse this?
    # TODO: the order of the locals might improve things too
    opts.append([Opt(OptOps.UPCAST, i, s) for s in get_divisors(k.full_shape[i], max_div=8, extra=suggestion.get((OptOps.UPCAST, i)))])
    opts.append([Opt(OptOps.LOCAL, i, s) for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get((OptOps.LOCAL, i)))])
  for i in range(k.first_reduce, k.shape_len):
    opts.append([Opt(OptOps.UPCAST, i, s) for s in get_divisors(k.full_shape[i], max_div=8, extra=suggestion.get((OptOps.UPCAST, i)))])
    opts.append([Opt(OptOps.GROUP, i, s) for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get((OptOps.GROUP, i))) if all(st.shape[i] % s == 0 or st.shape[i] == 1 for st in k.sts)])
    opts.append([Opt(OptOps.GROUPTOP, i, s) for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get((OptOps.GROUPTOP, i))) if all(st.shape[i] % s == 0 or st.shape[i] == 1 for st in k.sts)])
  # nevergrad parameters, default parameter choice
  return [ng.p.TransitionChoice(opt) for opt in opts], [opt[0] for opt in opts]

def suggestion_to_dict(suggestion):
  result: Dict[Tuple[OptOps, int], int] = {}
  # this is lossy
  for op in (suggestion or []): result[(op.op, op.axis)] = result.get((op.op, op.axis), 1) * op.amt
  return result

def normalize_suggestion(default_opt, suggestion):
  return tuple([Opt(op, axis, amt) for (op, axis), amt in ({(op, axis): amt for op, axis, amt in default_opt} | suggestion_to_dict(suggestion)).items()])

def kernel_optimize_search(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, baseline, bufs, var_vals, suggestion):
  import nevergrad as ng
  def opt(x):
    try:
      k = create_k()
      apply_ng_opt(k, x)
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
  opts, default_opt = kernel_optimize_opts(k, suggestion)
  if not opts: return "BASELINE"
  search_space = prod([len(x.choices) for x in opts])
  st = time.perf_counter()
  budget = getenv("BUDGET", 200)
  optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Tuple(*opts), budget=min(search_space, budget))
  optimizer.suggest(normalize_suggestion(default_opt, suggestion))
  recommendation = optimizer.minimize(opt)
  et = time.perf_counter() - st
  if DEBUG >= 1: print(f"optimizer({et:6.2f} s to search) space {search_space:8d} with tm {recommendation.loss:5.2f} ms vs baseline {baseline:5.2f} ms, a {baseline/recommendation.loss:5.2f}x gain : {k.colored_shape()}")
  return apply_ng_opt(create_k(), recommendation.value) if recommendation.loss < baseline else "BASELINE"

def fix_opt_axes(create_k, opts):
  k = create_k()
  axis_idxs = list(range(len(k.full_shape)))
  fixed_ops = []
  for opt in opts:
    axis = opt.axis
    if axis >= k.first_reduce: axis -= k.local_dims + len(k.group_for_reduce)
    fixed_ops.append(Opt(opt.op, axis_idxs[axis], opt.amt))

    if opt.amt == k.full_shape[axis]: axis_idxs.pop(axis)

    k.apply_opt(opt)

  return fixed_ops

def apply_ng_opt(k, x):
  axis_idxs = list(range(len(k.full_shape)))
  orig_first_reduce = k.first_reduce
  for op, axis, amt in x:
    if amt == 1: continue

    pre_axis = axis = axis_idxs.index(axis)
    if axis >= orig_first_reduce: axis += k.local_dims + len(k.group_for_reduce)

    if amt == k.full_shape[axis]: axis_idxs.pop(pre_axis)

    k.apply_opt(Opt(op, axis, amt))

  return k.applied_opts

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
      return min([prg.exec(bufs, var_vals, force_wait=True, optimizing=True) for _ in range(5)])*1000, k.applied_opts
    baseline, opt = get_baseline()
    choice = kernel_optimize_search(k, create_k, to_prg, baseline, bufs, var_vals, fix_opt_axes(create_k, opt))
    if global_db is not None:
      global_db[skey] = choice
      global_db.sync()

  if choice == "BASELINE": k.hand_coded_optimizations()
  else: [k.apply_opt(x) for x in choice]