import math, itertools
from collections import defaultdict
from typing import cast, Final
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, KernelInfo, graph_rewrite, _substitute, AxisType
from tinygrad.uop.symbolic import symbolic
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import colored, BEAM, getenv, DEBUG, to_function_name, NOOPT, argsort, round_up
from tinygrad.codegen.opt import axis_colors, Opt, OptOps, KernelOptError, check, axis_letters
from tinygrad.renderer import Renderer
from tinygrad.schedule.rangeify import remove_tags

# NOTE: LOCAL and GROUP_REDUCE have the same priority. the order here matters
axis_to_pos = {AxisType.LOOP: -1, AxisType.GLOBAL: 0, AxisType.LOCAL: 1, AxisType.UPCAST: 2,
               AxisType.GROUP_REDUCE: 1, AxisType.REDUCE: 3, AxisType.UNROLL: 4}

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
    self.dont_use_locals = self.ast.arg.dont_use_locals if self.ast.arg is not None else False
    self.applied_opts = list(self.ast.arg.applied_opts) if self.ast.arg is not None else []

  @property
  def rngs(self):
    # always in order by axistype
    return sorted([u for u in self.ast.parents if u.op is Ops.RANGE and u.vmax > 0], key=lambda x: (axis_to_pos[x.arg[-1]],) + x.arg[0:-1])
  @property
  def shape_len(self): return len(self.rngs)
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
      name = "k" + colored('_', 'BLACK').join(['']+[colored(x.src[0].render(), color) for x,color in zip(self.rngs, self.colors())])
      Scheduler.kernel_cnt[(function_name := to_function_name(name))] += 1
      num = f"n{Scheduler.kernel_cnt[function_name]-1}" if Scheduler.kernel_cnt[function_name] > 1 else ""
      name += colored(num, 'BLACK')
    self.ast = graph_rewrite(self.ast, pm_flatten_range, name="flatten range")
    return self.ast.replace(arg=KernelInfo(name=name, applied_opts=tuple(self.applied_opts), dont_use_locals=self.dont_use_locals), tag=1)

  def convert_loop_to_global(self):
    if not self.opts.has_local: return None
    store_rngs = self.ast.src[0].src[2:]

    # filter any not in local stores
    local_store_rngs = [x.ranges for x in self.ast.toposort() if (x.op is Ops.STORE and x.src[0].ptrdtype.addrspace == AddrSpace.LOCAL) \
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
      termination = self.termination
      if r0.arg[1] == r1.arg[1] and r0 in termination and r1 in termination and termination[r0] == termination[r1]:
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

  def colors(self) -> list[str]: return [axis_colors[x] if not self.dont_use_locals or not x == AxisType.GLOBAL else "BLUE" for x in self.axis_types]
  def colored_shape(self) -> str: return ' '.join([colored(f'{x.src[0].render():4s}', color) for x,color in zip(self.rngs, self.colors())])

  def shift_to(self, rng:UOp, amount:int, new_type:AxisType, top:bool=False):
    if (old_sz:=rng.src[0].divides(amount)) is None:
      raise KernelOptError(f"{amount} can't divide {rng.src[0]} in {self.colored_shape()}")
    new_rng = UOp.range(amount, self.maxarg+1, new_type)
    replaced_rng = rng.replace(src=(UOp.const(dtypes.int, old_sz),))
    sub_axis = (new_rng * old_sz + replaced_rng) if top else (replaced_rng * amount + new_rng)
    self.ast = self.ast.substitute({rng:sub_axis}, name=f"shift {rng.arg[0]} {amount}")
    return replaced_rng, new_rng

  def ranges_of(self, *axis_type:AxisType) -> list[UOp]: return [r for r in self.rngs if r.arg[-1] in axis_type]
  def axes_of(self, *axis_type:AxisType) -> list[int]: return [i for i,t in enumerate(self.axis_types) if t in axis_type]
  @property
  def upcastable_dims(self): return self.axes_of(AxisType.GLOBAL, AxisType.LOCAL)
  @property
  def unrollable_dims(self): return self.axes_of(AxisType.REDUCE, AxisType.GROUP_REDUCE)

  def real_axis(self, op:OptOps, axis:int|None):
    try:
      if axis is None: return -1
      if op is OptOps.UNROLL: return self.unrollable_dims[axis]
      if op in {OptOps.GROUP, OptOps.GROUPTOP}: return self.axes_of(AxisType.REDUCE)[axis]
      check(axis < self.shape_len, f"invalid axis on {axis=} {op=} {self.shape_len=}")
      return axis
    except IndexError as e: raise KernelOptError from e

  def apply_opt(self, opt:Opt, append_opt:bool=True):
    if opt.op is OptOps.NOLOCALS:
      check(all(x not in {AxisType.LOCAL, AxisType.GROUP_REDUCE} for x in self.axis_types), "no locals can't have locals")
      self.dont_use_locals = True
      self.applied_opts.append(opt)
      return

    if opt.op in {OptOps.LOCAL, OptOps.GROUP, OptOps.GROUPTOP}:
      check(self.opts.has_local, "locals needed for opt")

    rng = self.rngs[self.real_axis(opt.op, opt.axis)]

    opt_to_at = {
      OptOps.LOCAL: AxisType.LOCAL, OptOps.UPCAST: AxisType.UPCAST,
      OptOps.UNROLL: AxisType.UNROLL, OptOps.GROUP: AxisType.GROUP_REDUCE,
      OptOps.GROUPTOP: AxisType.GROUP_REDUCE}

    if opt.op in opt_to_at:
      amt:int = (rng.vmax+1) if opt.arg == 0 else cast(int, opt.arg)
      if opt.op is OptOps.UNROLL:
        check(amt <= 32, "don't unroll more than 32")
        check(rng.arg[-1] in {AxisType.GROUP_REDUCE, AxisType.REDUCE}, "unroll is for GROUP_REDUCE/REDUCE")
      if opt.op is OptOps.UPCAST:
        check((self.opts is not None and self.opts.device == "DSP") or amt <= 16, "don't upcast more than 16")
        check(rng.arg[-1] in {AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP}, "upcast is for GLOBAL/LOCAL/LOOP")
      if opt.op is OptOps.LOCAL:
        check(not self.dont_use_locals, "can't use locals")
        check(rng.arg[-1] in {AxisType.GLOBAL, AxisType.LOOP}, "local is for globals")
      if opt.op in {OptOps.GROUP, OptOps.GROUPTOP}:
        check(not self.dont_use_locals, "can't use locals")
        check(rng.arg[-1] == AxisType.REDUCE, "group is for reduce")
      self.shift_to(rng, amt, opt_to_at[opt.op], top=opt.op==OptOps.GROUPTOP)
    elif opt.op is OptOps.TC:
      check(len(self.applied_opts) == 0, "tensor core opts must be first") # TODO: remove the need for this by having warps
      check(opt.axis is not None, "tensor core opts must have an axis")
      check(opt.arg is not None and isinstance(opt.arg, tuple) and len(opt.arg) == 3, "tensor core opts must have valid arg")
      check(-1 <= (tc_select:=cast(tuple, opt.arg)[0]) < len(self.opts.tensor_cores), "tensor core opts must have valid tc_select")
      check(0 <= (tc_opt:=cast(tuple, opt.arg)[1]) <= 2, "tensor core opts must have valid tc_opt")
      check(0 < (use_tensor_cores:=cast(tuple, opt.arg)[2]) <= 2, "use_tensor_cores value is not valid")
      check(self._apply_tc_opt(use_tensor_cores, cast(int, opt.axis), tc_select, tc_opt), "no tensor core available")
    elif opt.op is OptOps.PADTO:
      check(rng.src[0].op is Ops.CONST, "only pad const")
      replaced_rng = UOp.range(round_up(rng.vmax+1, cast(int, opt.arg)), *rng.arg)
      replaces = {rng:replaced_rng}
      for b in self.bufs:
        if rng in b.src[1].sparents:
          valid = replaced_rng < rng.vmax+1
          if len(b.src) > 2: valid = b.src[2] & valid
          replaces[b] = b.replace(src=b.src[0:2]+(valid,))
      self.ast = self.ast.substitute(replaces, f"padto {rng.arg[:-1]} {opt.arg}")
    elif opt.op is OptOps.SWAP:
      try:
        altrng = self.rngs[opt.arg]
      except IndexError:
        raise KernelOptError
      check(rng.arg[-1] == AxisType.GLOBAL and altrng.arg[-1] == AxisType.GLOBAL, "swap only for globals")
      self.ast = self.ast.substitute({rng:rng.replace(arg=(*altrng.arg[0:-1], rng.arg[-1]), tag=1),
                                      altrng:altrng.replace(arg=(*rng.arg[0:-1], altrng.arg[-1]), tag=1)})
      self.ast = graph_rewrite(self.ast, remove_tags)
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
          if not len(in0_ranges) or not len(in1_ranges) or not len(red_ranges): continue

          # pick ranges
          # NOTE: why are in1 and in0 switched?
          axis_choices = list(itertools.product(in1_ranges, in0_ranges, red_ranges))
          if not (axis < len(axis_choices)): continue
          axes = list(axis_choices[axis])

          # do optimizations and save the ranges
          try:
            for i,a in enumerate(axes):
              # apply_opt should return the updated range?
              idx = self.rngs.index(a)
              self.apply_opt(Opt(OptOps.PADTO, idx, tc.dims[i]), append_opt=False) # PADTO might fail
              axes[i] = self.rngs[idx]
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
            srcs = [x.substitute(dict(zip(tne, [ne[i] for i in argsort(p)]))) for x,p in zip(srcs, tc.permutes_for_shape_str(tc.base_shape_str()))]

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
            wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.opts.device, tc.threads, tc_upcast_axes, ()) #, tc_reduce_axes)
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

  # helpers for hand_coded_optimizations
  @property
  def reduceop(self) -> UOp|None:
    red = [x for x in self.ast.parents if x.op is Ops.REDUCE]
    if not len(red): return None
    return UOp(Ops.REDUCE_AXIS, red[0].dtype, red[0].src, (red[0].arg, ()))
  @property
  def bufs(self) -> list[UOp]: return [x for x in self.ast.toposort() if x.op is Ops.INDEX][::-1]
  @property
  def output_shape(self):
    return [s if at not in {AxisType.REDUCE, AxisType.UNROLL, AxisType.GROUP_REDUCE} else 1 for s,at in zip(self.full_shape, self.axis_types)]
  @property
  def upcasted(self) -> int: return len(self.axes_of(AxisType.UPCAST, AxisType.UNROLL))
  @property
  def group_for_reduces(self) -> int: return len(self.axes_of(AxisType.GROUP_REDUCE))

def bufs_from_ast(ast:UOp, dname:str) -> list[Buffer]:
  glbls = sorted([x for x in ast.parents if x.op is Ops.DEFINE_GLOBAL], key=lambda x: x.arg)
  return [Buffer(dname, x.ptrdtype.size, x.dtype.base) for x in glbls]

def apply_opts(ctx:Renderer, ast:UOp):
  if ast.tag is not None: return None
  k = Scheduler(ast, ctx)
  k.convert_loop_to_global()
  if BEAM >= 1:
    k.simplify_merge_adjacent()
    from tinygrad.codegen.opt.search import beam_search
    rawbufs = bufs_from_ast(ast, ctx.device)
    k = beam_search(k, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  elif ast.arg is not None and ast.arg.opts_to_apply is not None:
    for opt in ast.arg.opts_to_apply: k.apply_opt(opt)
  elif not NOOPT:
    k.simplify_merge_adjacent()
    from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
    # NOTE: hand_coded_optimizations doesn't support multiblock opts yet
    if all(len(u.src) == 1 for u in ast.parents if u.op is Ops.LOAD):
      for opt in hand_coded_optimizations(k): k.apply_opt(opt)
  return k.get_optimized_ast(name_override=ast.arg.name if ast.arg is not None and ast.arg.name != "test" else None)

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.SINK, name="ast"), apply_opts),
])
