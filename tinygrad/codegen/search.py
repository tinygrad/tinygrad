from typing import Callable
import time
import functools
import multiprocessing as mp
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import DEBUG, prod, getenv, GlobalCounters

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


@functools.lru_cache
def compile_kernel(x, create_k, to_prg):
  try:
    k = create_k()
    k.process()
    if DEBUG >= 2: print(f"Shape: {k.full_shape}; Applying opt: {list(y for y in x if y[1] != 1)}")
    k.apply_auto_opt(x)
    k.linearize()
    assert len(k.uops) < 2 ** 12, f"too many uops: {len(k.uops)}"  # device target compiler will take significantly longer than Linearizer
    prg = to_prg(k)
    return prg
  except Exception:
    if DEBUG >= 3:
      import traceback
      traceback.print_exc()
    return None


def kernel_optimize_search(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, baseline, bufs, suggestion):
  import nevergrad as ng

  def cheap(x):
    try:
      compile_kernel(x, create_k, to_prg)
      return True
    except Exception:
      if DEBUG >= 3:
        import traceback
        traceback.print_exc()
      return False
  def run_and_time(prg):
    if prg is None:
      return None
    try:
      first_tm = prg.exec(bufs, force_wait=True, optimizing=True)
      if baseline*5 < first_tm*1000: return first_tm*1000  # very slow
      tm = min([first_tm]+[prg.exec(bufs, force_wait=True, optimizing=True) for _ in range(10)])*1000
      return tm
    except Exception:
      if DEBUG >= 3:
        import traceback
        traceback.print_exc()
      return None
  opts, default_opt = kernel_optimize_opts(k, suggestion)
  if not opts: return "BASELINE"
  search_space = prod([len(x.choices) for x in opts])
  st = time.perf_counter()
  budget = getenv("BUDGET", 200)
  num_workers = getenv("KOPT_WORKERS", 16)
  optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Tuple(*opts), budget=min(search_space, budget), num_workers=num_workers)
  bar = ng.callbacks.ProgressBar()
  optimizer.register_callback("tell", bar)
  if suggestion is not None: optimizer.suggest(tuple([(i, s, typ) for (typ, i), s in ({(typ, i): s for i,s,typ in default_opt} | suggestion).items()]))  # maintain ordering from default_opt

  if num_workers == 1:
    optimizer.parametrization.register_cheap_constraint(cheap)
    recommendation = optimizer.minimize(lambda x: run_and_time(compile_kernel(x, create_k, to_prg)) or 10_000)
  else:
    from extra.helpers import _CloudpickleFunctionWrapper
    best = 10_000
    with mp.Pool(num_workers) as pool:
      q = []
      while optimizer.num_tell < optimizer.budget:
        while len(q) < num_workers * 4 and optimizer.num_ask < optimizer.budget:
          ask = optimizer.ask()
          q.append((ask, pool.apply_async(compile_kernel, (ask.value, _CloudpickleFunctionWrapper(create_k), _CloudpickleFunctionWrapper(to_prg)))))
        while len(q) > num_workers * 2 or (optimizer.num_ask == optimizer.budget and q):
          ask, prg = q.pop(0)
          prg = prg.get()
          if prg is None: optimizer.tell(ask, 10_000, constraint_violation=1.0)
          else:
            tm = run_and_time(prg)
            if tm is None: optimizer.tell(ask, 10_000, constraint_violation=1.0)
            else:
              optimizer.tell(ask, tm)
              best = min(best, tm)
              bar._progress_bar.set_description(str(best))
    recommendation = optimizer.provide_recommendation()

  et = time.perf_counter() - st
  del bar
  if DEBUG >= 1: print(f"optimizer({et:6.2f} s to search) space {search_space:8d} with tm {recommendation.loss:5.2f} ms vs baseline {baseline:5.2f} ms, a {baseline/recommendation.loss:5.2f}x gain : {k.colored_shape()}")
  return recommendation.value if recommendation.loss < baseline else "BASELINE"

# optimization
global_db = None
def kernel_optimize(k:Linearizer, create_k:Callable[[], Linearizer], to_prg, bufs):
  global global_db

  k.process()
  skey = str(k.key)

  if getenv("KOPT") in [2, 3] and global_db is None:
    import shelve
    global_db = shelve.open("/tmp/kopt_cache")

  if global_db is not None and skey in global_db:
    choice = global_db[skey]
    if DEBUG >= 1: print(f"from kopt_cache: {list(y for y in choice if y[1] != 1) if not isinstance(choice, str) else choice}")
  elif k.has_variable_shape() or getenv("KOPT") == 3:
    # don't optimize variable shapes or if KOPT=3
    choice = "BASELINE"
  else:
    orig_kernel_count = GlobalCounters.kernel_count
    # get baseline
    def get_baseline():
      k = create_k()
      suggestion = k.hand_coded_optimizations()
      prg = to_prg(k)
      return min([prg.exec(bufs, force_wait=True, optimizing=True) for _ in range(5)])*1000, suggestion
    baseline, suggestion = get_baseline()
    if DEBUG >= 2: print(f"suggestion: {suggestion}")
    KOPT_THRESH = getenv("KOPT_THRESH")  # us
    if baseline >= KOPT_THRESH / 1000:
      choice = kernel_optimize_search(k, create_k, to_prg, baseline, bufs, suggestion)
      if global_db is not None:
        global_db[skey] = choice
        global_db.sync()
    else:
      choice = "BASELINE"

    GlobalCounters.kernel_count = orig_kernel_count

  if choice == "BASELINE": k.hand_coded_optimizations()
  else: k.apply_auto_opt(choice)