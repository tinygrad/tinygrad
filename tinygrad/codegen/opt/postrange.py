import math, itertools
from collections import defaultdict
from typing import cast, Final
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, KernelInfo, graph_rewrite, _substitute, AxisType
from tinygrad.uop.symbolic import symbolic
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import colored, POSTBEAM, getenv, DEBUG, to_function_name
from tinygrad.codegen.opt.kernel import axis_colors, Opt, OptOps, KernelOptError, check, axis_letters
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

class Scheduler:
  def __init__(self, ast:UOp, opts:Renderer):
    self.ast, self.opts = ast, opts
    self.applied_opts = list(self.ast.arg.applied_opts) if self.ast.arg is not None else []

  @property
  def rngs(self): return sorted([u for u in self.ast.parents if u.op is Ops.RANGE and u.vmax > 0], key=lambda x: x.arg)
  @property
  def full_shape(self): return [x.vmax+1 for x in self.rngs]
  @property
  def axis_types(self): return [x.arg[-1] for x in self.rngs]
  @property
  def maxarg(self): return max([x.arg[0] for x in self.rngs], default=0)

  # strings like ['g0', 'g1', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'R0', 'r0', 'r1', 'r2', 'u0', 'u1', 'u2']
  def shape_str(self) -> list[str]:
    ret: list[str] = []
    cnt: dict[AxisType, int] = {}
    for x in self.axis_types:
      cnt[x] = (cnt[x] + 1) if x in cnt else 0
      ret.append(f"{axis_letters[x]}{cnt[x]}")
    return ret
  def shape_str_to_axis(self, nms:list[str]) -> tuple[int, ...]: return tuple([self.shape_str().index(x) for x in nms])

  @property
  def termination(self):
    terminators = [u for u in self.ast.parents if u.op in {Ops.REDUCE, Ops.STORE}]
    termination = {}
    for t in terminators:
      # works without pm_flatten_range
      for u in UOp.sink(*t.src[1 if t.op is Ops.REDUCE else 2:]).parents:
        if u.op is Ops.RANGE: termination[u] = t
    return termination

  def copy(self): return Scheduler(self.get_optimized_ast(), self.opts)

  kernel_cnt: Final[defaultdict[str, int]] = defaultdict(int)
  def get_optimized_ast(self, name_override:str|None=None):
    if name_override is not None: name = name_override
    else:
      name = "k" + colored('_', 'BLACK').join(['']+[colored(x.src[0].render(), axis_colors[x.arg[-1]]) for x in self.rngs])
      Scheduler.kernel_cnt[(function_name := to_function_name(name))] += 1
      num = f"n{Scheduler.kernel_cnt[function_name]-1}" if Scheduler.kernel_cnt[function_name] > 1 else ""
      name += colored(num, 'BLACK')
    self.ast = graph_rewrite(self.ast, pm_flatten_range, "flatten range")
    return self.ast.replace(arg=KernelInfo(name=name, applied_opts=tuple(self.applied_opts)), tag=1)

  def convert_loop_to_global(self):
    if not self.opts.has_local: return None
    store_rngs = self.ast.src[0].src[2:]

    # filter any not in local stores
    local_store_rngs = [x.ranges for x in self.ast.toposort() if (x.op is Ops.STORE and x.src[0].dtype.addrspace == AddrSpace.LOCAL) \
                        or (x.op is Ops.BUFFERIZE and x.arg == AddrSpace.LOCAL)]
    for ls in local_store_rngs: store_rngs = tuple([x for x in store_rngs if x in ls])

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

  def colored_shape(self) -> str:
    return ' '.join([colored(f'{x.src[0].render():2s}', axis_colors[x.arg[-1]]) for x in self.rngs])

  def shift_to(self, rng:UOp, amount:int, new_type:AxisType, top:bool=False):
    if rng.src[0].divides(amount) is None: raise KernelOptError("can't divide that")
    old_sz = rng.src[0].arg // amount
    assert old_sz > 0, f"bad old_sz on {amount} {rng}"

    new_rng = UOp.range(amount, self.maxarg+1, new_type)
    replaced_rng = rng.replace(src=(UOp.const(dtypes.int, old_sz),))
    sub_axis = (new_rng * old_sz + replaced_rng) if top else (replaced_rng * amount + new_rng)
    self.ast = self.ast.substitute({rng:sub_axis}, name=f"shift {rng.arg[0]} {amount}")
    return replaced_rng, new_rng

  def apply_opt(self, opt:Opt, append_opt:bool=True):
    if opt.op in {OptOps.LOCAL, OptOps.GROUP, OptOps.GROUPTOP}:
      check(self.opts.has_local, "locals needed for opt")

    try:
      if opt.op in {OptOps.UNROLL, OptOps.GROUP, OptOps.GROUPTOP}:
        check(opt.axis is not None)
        rng = [x for x in self.rngs if x.arg[-1] is AxisType.REDUCE][opt.axis]
        check(rng.arg[-1] in {AxisType.REDUCE}, "can only unroll/upcast reduce")
      else:
        rng = self.rngs[opt.axis]
        check(rng.arg[-1] in {AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP})
    except IndexError:
      raise KernelOptError(f"bad opt {opt} on axis")

    opt_to_at = {
      OptOps.LOCAL: AxisType.LOCAL, OptOps.UPCAST: AxisType.UPCAST,
      OptOps.UNROLL: AxisType.UNROLL, OptOps.GROUP: AxisType.GROUP_REDUCE,
      OptOps.GROUPTOP: AxisType.GROUP_REDUCE}

    if opt.op in opt_to_at:
      amt = rng.src[0].arg if opt.arg == 0 else opt.arg
      if opt.op is OptOps.UNROLL: check(amt <= 32, "don't unroll more than 32")
      if opt.op is OptOps.UPCAST: check((self.opts is not None and self.opts.device == "DSP") or amt <= 16, "don't upcast more than 16")
      self.shift_to(rng, amt, opt_to_at[opt.op], top=opt.op==OptOps.GROUPTOP)
    elif opt.op is OptOps.TC:
      check(len(self.applied_opts) == 0, "tensor core opts must be first") # TODO: remove the need for this by having warps
      check(opt.axis is not None, "tensor core opts must have an axis")
      check(opt.arg is not None and isinstance(opt.arg, tuple) and len(opt.arg) == 3, "tensor core opts must have valid arg")
      check(-1 <= (tc_select:=cast(tuple, opt.arg)[0]) < len(self.opts.tensor_cores), "tensor core opts must have valid tc_select")
      check(0 <= (tc_opt:=cast(tuple, opt.arg)[1]) <= 2, "tensor core opts must have valid tc_opt")
      check(0 < (use_tensor_cores:=cast(tuple, opt.arg)[2]) <= 2, "use_tensor_cores value is not valid")
      check(self._apply_tc_opt(use_tensor_cores, cast(int, opt.axis), tc_select, tc_opt), "no tensor core available")
    elif opt.op is OptOps.SWAP:
      raise RuntimeError("broken, this can form a loop")
      altrng = self.rngs[opt.arg]
      self.ast = self.ast.substitute({rng:rng.replace(arg=(*altrng.arg[0:-1], rng.arg[-1])),
                                      altrng:altrng.replace(arg=(*rng.arg[0:-1], altrng.arg[-1]))})
    else:
      raise KernelOptError(f"unsupported opt {opt.op}")
    if append_opt:
      self.applied_opts.append(opt)

  def _apply_tc_opt(self, use_tensor_cores:int, axis:int, tc_select:int, opt_level:int) -> bool:
    reduceops = [x for x in self.ast.toposort() if x.op is Ops.REDUCE]
    if not len(reduceops): raise KernelOptError("no reduce ops for TensorCore")
    reduceop = reduceops[0]
    if use_tensor_cores and reduceop is not None and reduceop.arg is Ops.ADD:
      mul = reduceop.src[0] if reduceop.src[0].op is not Ops.CAST else reduceop.src[0].src[0]
      if mul.op is not Ops.MUL: return False
      in0, in1 = mul.src
      try:
        tensor_cores = self.opts.tensor_cores if tc_select == -1 else [self.opts.tensor_cores[tc_select]]
      except IndexError:
        raise KernelOptError(f"invalid tensor core choice {tc_select}")
      for tc in tensor_cores:
        if tc.dtype_in == in0.dtype.scalar() and tc.dtype_in == in1.dtype.scalar() and tc.dtype_out == reduceop.dtype.scalar():
          # tensor cores have three ranges. X, Y, and REDUCE
          in0_ranges = sorted([u for u in in0.ranges if u not in in1.ranges], key=lambda x: x.arg[0])
          in1_ranges = sorted([u for u in in1.ranges if u not in in0.ranges], key=lambda x: x.arg[0])
          red_ranges = sorted(reduceop.src[1:], key=lambda x: x.arg[0])
          if DEBUG >= 3:
            print(f"TC({axis}): {[(x.arg[0],x.vmax+1) for x in in0_ranges]}",
                              f"{[(x.arg[0],x.vmax+1) for x in in1_ranges]} {[(x.arg[0],x.vmax+1) for x in red_ranges]}")
          if not len(in0_ranges) or not len(in1_ranges) or not len(red_ranges): return None

          # pick ranges
          # NOTE: why are in1 and in0 switched?
          axis_choices = list(itertools.product(in1_ranges, in0_ranges, red_ranges))
          if not (axis < len(axis_choices)): return None
          axes = list(axis_choices[axis])

          # do optimizations and save the ranges
          try:
            for i,a in enumerate(axes):
              check(a.src[0].divides(tc.dims[i]) is not None, "doesn't divide evenly")
              #self.apply_opt(Opt(OptOps.PADTO, self.rng.index(a), tc.dims[i]), append_opt=False) # PADTO might fail
          except KernelOptError: continue

          ne: list[UOp] = []
          for opt in tc.opts:
            axes[int(opt[1])], new_range = self.shift_to(axes[int(opt[1])], 2, {"u":AxisType.UPCAST, "l":AxisType.LOCAL}[opt[0]])
            ne.append(new_range)
          for _, amt in tc.get_reduce_axes():
            axes[2], new_range = self.shift_to(axes[2], amt, AxisType.UNROLL)
            ne.append(new_range)

          if use_tensor_cores != 2:
            # fix the srcs
            reduceop = [x for x in self.ast.toposort() if x.op is Ops.REDUCE][0]
            tne = [x.replace(tag=1) for x in ne]
            ret = reduceop.substitute(dict(zip(ne, tne)))
            srcs = list((ret.src[0] if ret.src[0].op is not Ops.CAST else ret.src[0].src[0]).src)
            srcs = [x.substitute(dict(zip(tne, [ne[i] for i in p]))) for x,p in zip(srcs, tc.permutes_for_shape_str(tc.base_shape_str()))]

            # get reduce/upcast axes for the tensor cores
            tc_reduce_axes = self.shape_str_to_axis([f"r{i}" for i in range(len(tc.get_reduce_axes()))])
            base_upcast_axes = tuple([(s,2) for s in self.shape_str_to_axis(tc.base_upcast_axes())])
            tc_upcast_axes = tuple([base_upcast_axes[:int(math.log2(tc.elements_per_thread[i]))] for i in range(3)])

            # axes to range number (was done in lowerer)
            tc_upcast_axes = tuple([tuple([(self.rngs[a].arg[0], sz) for a,sz in v]) for v in tc_upcast_axes])
            tc_reduce_axes = tuple([self.rngs[a].arg[0] for a in tc_reduce_axes])

            # construct the op
            # TODO: remove tc_upcast_axes from the arg
            # do the reduce_axes always disappear? i think they don't
            # they need to be moved into the WMMA srcs
            wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.opts.device, tc.threads, tc_upcast_axes, ()) #tc_reduce_axes)
            wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
              UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0], tag=1),
              UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1], tag=1),
              UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg, tag=1)
            tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2], tag=1)

            # preserve extra reduces
            reduce_ranges = [x for x in UOp.sink(*reduceop.src[1:]).toposort() if x.op is Ops.RANGE and x.arg[0] not in tc_reduce_axes]
            if len(reduce_ranges): tc_uop = UOp(Ops.REDUCE, tc_uop.dtype, (tc_uop,)+tuple(reduce_ranges), Ops.ADD)
            self.ast = self.ast.substitute({reduceop: tc_uop})
          return True
    return False

def bufs_from_ast(ast:UOp, dname:str) -> list[Buffer]:
  glbls = sorted([x for x in ast.parents if x.op is Ops.DEFINE_GLOBAL], key=lambda x: x.arg)
  return [Buffer(dname, x.dtype.size, x.dtype.base) for x in glbls]

def apply_opts(ctx:Renderer, ast:UOp):
  if ast.tag is not None: return None
  k = Scheduler(ast, ctx)
  k.convert_loop_to_global()
  if POSTBEAM >= 1:
    k.simplify_merge_adjacent()
    from tinygrad.codegen.opt.search import beam_search
    rawbufs = bufs_from_ast(ast, ctx.device)
    k = beam_search(k, rawbufs, POSTBEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  else:
    if ast.arg is not None and ast.arg.opts_to_apply is not None:
      for opt in ast.arg.opts_to_apply: k.apply_opt(opt)
  return k.get_optimized_ast(name_override=ast.arg.name if ast.arg is not None and ast.arg.name != "test" else None)

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.SINK, name="ast"), apply_opts),
])
