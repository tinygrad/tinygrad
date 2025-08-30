from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, KernelInfo, graph_rewrite, _substitute, AxisType
from tinygrad.uop.symbolic import symbolic
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import colored, BEAM, getenv
from tinygrad.codegen.opt.kernel import axis_colors, Opt, OptOps, KernelOptError
from tinygrad.renderer import Renderer

def flatten_range(r:UOp):
  off = 2 if r.op is Ops.STORE else 1
  rngs = r.src[off:]
  if not len(rngs): return None
  new_rngs = [x for x in UOp.sink(*rngs).toposort() if x.op is Ops.RANGE]
  return r.replace(src=r.src[:off]+tuple(new_rngs))

pm_flatten_range = PatternMatcher([
  # real ranges only
  (UPat((Ops.REDUCE, Ops.STORE), name="r"), flatten_range),
])

def count_divmod(x:UOp): return len([u for u in x.toposort() if u.op in {Ops.IDIV, Ops.MOD}])

class SimpleKernel:
  def __init__(self, ast:UOp, opts:Renderer):
    self.ast, self.opts = ast, opts
    self.applied_opts = list(self.ast.arg.applied_opts) if self.ast.arg is not None else []

  @property
  def rngs(self): return sorted([u for u in self.ast.parents if u.op is Ops.RANGE and u.vmax > 0], key=lambda x: x.arg)
  @property
  def maxarg(self): return max([x.arg[0] for x in self.rngs], default=0)

  @property
  def termination(self):
     # NOTE: this one is better than the one in kernel.py, which is kind of a problem
    terminators = [u for u in self.ast.parents if u.op in {Ops.REDUCE, Ops.STORE}]
    termination = {}
    for t in terminators:
      for u in t.src[1 if t.op is Ops.REDUCE else 2:]: termination[u] = t
    return termination

  def copy(self): return SimpleKernel(self.get_optimized_ast(), self.opts)

  def get_optimized_ast(self):
    name = "k" + colored('_', 'BLACK').join(['']+[colored(x.src[0].render(), axis_colors[x.arg[-1]]) for x in self.rngs])
    return self.ast.replace(arg=KernelInfo(name=name, applied_opts=tuple(self.applied_opts)))

  def convert_loop_to_global(self):
    if not self.opts.has_local: return None
    store_rngs = self.ast.src[0].src[2:]

    # filter any not in local stores
    local_store_rngs = [x.ranges for x in self.ast.toposort() if (x.op is Ops.STORE and x.src[0].dtype.addrspace == AddrSpace.LOCAL) \
                        or (x.op is Ops.BUFFERIZE and x.arg == AddrSpace.LOCAL)]
    for ls in local_store_rngs: store_rngs = [x for x in store_rngs if x in ls]

    store_rng = [x for x in UOp.sink(*store_rngs).toposort() if x.op is Ops.RANGE] if store_rngs else []
    rng = [x.replace(arg=(x.arg[0], AxisType.GLOBAL)) if x.arg[1] == AxisType.LOOP and x in store_rng else x for x in self.rngs]

    self.ast = self.ast.substitute(dict(zip(self.rngs, rng)))

  def simplify_merge_adjacent(self):
    i = 0
    while i < len(self.rngs)-1:
      r0, r1 = self.rngs[i], self.rngs[i+1]
      # same axistype and same termination
      if r0.arg[1] == r1.arg[1] and self.termination[r0] == self.termination[r1]:
        s0, s1 = r0.src[0], r1.src[0]
        new_range = r0.replace(src=(s0*s1,)).simplify()
        # this checks the legality of a merge
        oidx = self.ast.simplify()
        nidx = graph_rewrite(oidx, _substitute+symbolic+pm_flatten_range, ctx={r0:new_range//s1, r1:new_range%s1}, name=f"check_merge_{i}_{i+1}")
        # it simplifies
        if count_divmod(nidx) <= count_divmod(oidx):
          # it is correct
          midx = graph_rewrite(nidx, _substitute+symbolic+pm_flatten_range, ctx={new_range:r0*s1+r1}, name=f"correct_merge_{i}_{i+1}")
          if oidx is midx:
            self.ast = nidx
            continue
      i += 1

  def colored_shape(self, pad:int|None=None, dense=False) -> str:
    return ' '.join([colored(x.src[0].render(), axis_colors[x.arg[-1]]) for x in self.rngs])

  def shift_to(self, axis:int, amount:int, new_type:AxisType, top:bool=False):
    try:
      rng = self.rngs[axis]
    except IndexError:
      raise KernelOptError(f"bad axis {axis}")

    if amount == 0:
      amount = rng.src[0].arg
      old_sz = 1
    else:
      old_sz = rng.src[0].arg // amount
      assert old_sz > 0, f"bad old_sz on {axis} {amount} {rng}"

    new_rng = UOp.range(amount, self.maxarg+1, new_type)

    if old_sz == 1:
      self.ast.substitute({rng:new_rng})
    else:
      replaced_rng = rng.replace(src=(UOp.const(dtypes.int, old_sz),))
      sub_axis = (new_rng * old_sz + replaced_rng) if top else (replaced_rng * amount + new_rng)
      self.ast.substitute({rng:sub_axis})
    return new_rng

  def apply_opt(self, opt:Opt, append_opt:bool=True) -> UOp|None:
    if opt.op is OptOps.LOCAL:
      self.shift_to(opt.axis, opt.arg, AxisType.LOCAL)
    elif opt.op is OptOps.UPCAST:
      self.shift_to(opt.axis, opt.arg, AxisType.UPCAST)
    elif opt.op is OptOps.UNROLL:
      try:
        r_axis = [x for x in self.rngs if x.arg[1] is AxisType.REDUCE][opt.axis]
      except IndexError:
        raise KernelOptError(f"bad reduce axis {opt.axis}")
      self.shift_to(self.rngs.index(r_axis), opt.arg, AxisType.UNROLL)
    else:
      raise KernelOptError(f"unsupported opt {opt.op}")
    if append_opt:
      self.applied_opts.append(opt)

def bufs_from_ast(ast:UOp, dname:str) -> list[Buffer]:
  glbls = sorted([x for x in ast.parents if x.op is Ops.DEFINE_GLOBAL], key=lambda x: x.arg)
  return [Buffer(dname, x.dtype.size, x.dtype.base) for x in glbls]

def apply_opts(ctx:Renderer, ast:UOp):
  if ast.tag is not None: return None
  k = SimpleKernel(ast, ctx)
  k.convert_loop_to_global()
  k.simplify_merge_adjacent()
  if BEAM >= 1:
    from tinygrad.codegen.opt.search import beam_search
    rawbufs = bufs_from_ast(ast, ctx.device)
    k = beam_search(k, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  return k.get_optimized_ast().replace(tag=1)

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.SINK, name="ast"), apply_opts),
])
