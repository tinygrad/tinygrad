import time, functools, multiprocessing as mp, traceback
from typing import Callable

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import DEBUG, prod, getenv, GlobalCounters, ansilen
from tinygrad.lazy import var_vals_from_ast

def get_divisors(n, min_div = 1, max_div = 512, extra=None):
  if min_div > 1: yield 1
  if extra is not None and extra < min_div and extra != 1 and n % extra == 0: yield extra
  for d in range(min_div, max_div + 1):
    if n % d == 0: yield d
  if extra is not None and extra > max_div and extra != 1 and n % extra == 0: yield extra

def kernel_optimize_opts(k:Linearizer, suggestion):
  import nevergrad as ng
  if suggestion is None: suggestion = {}
  opts = []
  for i in range(k.first_reduce):
    # TODO: the upcast always happen first, you might want to reverse this?
    # TODO: the order of the locals might improve things too
    opts.append([(i,s,"U") for s in get_divisors(k.full_shape[i], max_div=8, extra=suggestion.get(("U", i)))])
    opts.append([(i,s,"L") for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get(("L", i)))])
  for i in range(k.first_reduce, k.shape_len):
    opts.append([(i,s,"R") for s in get_divisors(k.full_shape[i], max_div=8, extra=suggestion.get(("R", i)))])
    opts.append([(i,s,"G") for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get(("G", i))) if all(st.shape[i] % s == 0 or st.shape[i] == 1 for st in k.sts)])
  # nevergrad parameters, default parameter choice
  return [ng.p.TransitionChoice(opt) for opt in opts], [opt[0] for opt in opts]

def normalize_suggestion(default_opt, suggestion):
  return tuple([(i, s, typ) for (typ, i), s in ({(typ, i): s for i,s,typ in default_opt} | suggestion).items()])  # maintain ordering from default_opt

def catch_exception(f, on_fail=None):
  @functools.wraps(f)
  def inner(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except Exception:
      if DEBUG >= 3: traceback.print_exc()
      return on_fail
  return inner

def compile_kernel(x, create_k, to_prg):
  k = create_k()
  if DEBUG >= 2: print(f"Shape: {k.full_shape}; Applying opt: {list(y for y in x if y[1] != 1)}")
  k.apply_auto_opt(x)
  k.linearize()
  assert len(k.uops) < 2 ** 12, f"too many uops: {len(k.uops)}"  # device target compiler will take significantly longer than Linearizer
  prg = to_prg(k)
  return k.display_name, prg

def run_and_time(prg, baseline, bufs, var_vals):
  first_tm = prg.exec(bufs, var_vals, force_wait=True, optimizing=True)
  if baseline*5 < first_tm*1000: return first_tm*1000  # very slow
  return min([first_tm]+[prg.exec(bufs, var_vals, force_wait=True, optimizing=True) for _ in range(10)])*1000

def kernel_optimize_search(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, baseline, bufs, var_vals, suggestion):
  import nevergrad as ng

  def cheap(x): return catch_exception(compile_kernel)(x, create_k, to_prg) is not None

  opts, default_opt = kernel_optimize_opts(k, suggestion)
  if not opts: return "BASELINE"
  search_space = prod([len(x.choices) for x in opts])
  st = time.perf_counter()
  budget, num_workers = getenv("BUDGET", 200), getenv("KOPT_WORKERS", 16)
  optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Tuple(*opts), budget=min(search_space, budget), num_workers=num_workers)
  optimizer.register_callback("tell", (bar := ng.callbacks.ProgressBar()))
  if suggestion is not None: optimizer.suggest(normalize_suggestion(default_opt, suggestion))

  if num_workers == 0:
    optimizer.parametrization.register_cheap_constraint(cheap)
    recommendation = optimizer.minimize(catch_exception(lambda x: run_and_time(compile_kernel(x, create_k, to_prg)[1], baseline, bufs, var_vals), 10_000))
  else:
    from extra.helpers import _CloudpickleFunctionWrapper
    best, best_ran, best_name = 10_000, 0, ""
    q, ran = [], 0
    with mp.Pool(num_workers) as pool:
      while optimizer.num_tell < optimizer.budget:
        while len(q) < num_workers and optimizer.num_ask < optimizer.budget:
          ask = optimizer.ask()
          q.append((ask, pool.apply_async(compile_kernel, (ask.value, _CloudpickleFunctionWrapper(create_k), _CloudpickleFunctionWrapper(to_prg)))))
        while len(q) > num_workers-1 or (optimizer.num_ask == optimizer.budget and q):
          ask, prg = q.pop(0)
          try:
            name, prg = prg.get(timeout=5)
            tm = run_and_time(prg, baseline, bufs, var_vals)
          except Exception:
            if DEBUG >= 3: traceback.print_exc()
            optimizer.tell(ask, 10_000, constraint_violation=1.0)
          else:
            optimizer.tell(ask, tm)
            ran += 1
            if tm < best: best, best_ran, best_name = tm, ran, name
            bar._progress_bar.set_description(f"{baseline:7.3f}/{best:7.3f} ({baseline / best * 100:4.0f}%) @ {best_ran:4}/{ran:4} - {best_name + ' ' * (37 - ansilen(best_name))}")
    recommendation = optimizer.provide_recommendation()
    if DEBUG >= 1 and ran == 0: print(f"WARNING: no kernels ran! Shape: {k.full_shape}; suggestion: {[(i, s, typ) for (typ, i), s in suggestion.items()] if suggestion is not None else None}")

  et = time.perf_counter() - st
  del bar, optimizer
  if DEBUG >= 1: print(f"optimizer({et:6.2f} s to search) space {search_space:8d} with tm {recommendation.loss:5.2f} ms vs baseline {baseline:5.2f} ms, a {baseline/recommendation.loss:5.2f}x gain : {k.colored_shape()}")
  return recommendation.value if recommendation.loss < baseline else "BASELINE"

# optimization
global_db = None
def kernel_optimize(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, bufs, key):
  global global_db

  skey = str(key)

  if getenv("KOPT") in [2, 3] and global_db is None:
    import shelve
    KOPT_CACHE_PATH = getenv("KOPT_CACHE", "/tmp/kopt_cache")
    global_db = shelve.open(KOPT_CACHE_PATH)

  if global_db is not None and skey in global_db:
    choice = global_db[skey]
    if DEBUG >= 3: print(f"Shape: {k.full_shape}; from kopt_cache: {list(y for y in choice if y[1] != 1) if not isinstance(choice, str) else choice}")
  elif k.has_variable_shape() or getenv("KOPT") == 3:
    # don't optimize variable shapes or if KOPT=3
    choice = "BASELINE"
  else:
    orig_kernel_count = GlobalCounters.kernel_count
    var_vals = {k:k.min for k in var_vals_from_ast(k.ast)}
    # get baseline
    def get_baseline():
      k = create_k()
      suggestion = k.hand_coded_optimizations()
      prg = to_prg(k)
      return min([prg.exec(bufs, var_vals, force_wait=True, optimizing=True) for _ in range(5)])*1000, suggestion
    baseline, suggestion = get_baseline()
    if DEBUG >= 2: print(f"Shape: {k.full_shape}; suggestion: {[(i, s, typ) for (typ, i), s in suggestion.items()] if suggestion is not None else None}")
    KOPT_THRESH = getenv("KOPT_THRESH")  # us
    if baseline >= KOPT_THRESH / 1000:
      choice = kernel_optimize_search(k, create_k, to_prg, baseline, bufs, suggestion, var_vals)
      if global_db is not None:
        global_db[skey] = choice
        global_db.sync()
      if DEBUG >= 2: print(f"Shape: {k.full_shape}; KOPT choice: {choice}")
    else:
      choice = "BASELINE"

    GlobalCounters.kernel_count = orig_kernel_count

  if choice == "BASELINE": k.hand_coded_optimizations()
  else: k.apply_auto_opt(choice)