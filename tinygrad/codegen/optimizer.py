import nevergrad as ng
from tinygrad.helpers import DEBUG, prod

def apply_opt(k, x):
  for axis, amt, typ in x:
    if axis is None or amt == 1: continue
    if typ == "R":
      typ = "U"
      axis += k.first_reduce
    assert k.full_shape[axis] % amt == 0, "no longer valid shift"
    if typ == "U":
      k.shift_to(axis, amt)
      k.upcast()
    elif typ == "L":
      k.shift_to(axis, amt, insert_before=k.first_reduce)
      k.local_dims += 1
  k.simplify_ones()

UPCASTS = [1,2,3,4,5,6,7,8]
LOCALS = [1,2,3,4,5,6,7,8,16,24,32]

# optimization
def kernel_optimize(k, create_k, runtime):
  def opt(x):
    try:
      k = create_k()
      k.process()
      apply_opt(k, x)
      prg = k.codegen().build(runtime)
      tm = min([prg.exec(k.bufs, force_wait=True) for _ in range(3)])*1000
      return tm
    except Exception:
      if DEBUG >= 3:
        import traceback
        traceback.print_exc()
      return 10000_000   # 10000 seconds is infinity
  k.process()
  opts = []
  for i in range(k.first_reduce):
    # TODO: the upcast always happen first, you might want to reverse this?
    # TODO: the order of the locals might improve things too
    opts.append(ng.p.TransitionChoice([(i,s,"U") for s in UPCASTS if k.full_shape[i]%s == 0]))
    opts.append(ng.p.TransitionChoice([(i,s,"L") for s in LOCALS if k.full_shape[i]%s == 0]))
  for i in range(k.shape_len-k.first_reduce):
    opts.append(ng.p.TransitionChoice([(i,s,"R") for s in UPCASTS if k.full_shape[k.first_reduce+i]%s == 0]))
  if len(opts) == 0: return
  search_space = prod([len(x.choices) for x in opts])
  optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Tuple(*opts), budget=min(search_space, 200))
  recommendation = optimizer.minimize(opt)
  apply_opt(k, recommendation.value)
  if DEBUG >= 1: print("optimizer hit", k.colored_shape(), "in search space", search_space)
