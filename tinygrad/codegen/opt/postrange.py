from typing import cast
from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, AxisType, sint, ssimplify, KernelInfo
from tinygrad.uop.ops import Variable, GroupOp
from tinygrad.helpers import colored, argfix, prod
from tinygrad.codegen.opt.kernel import Opt, OptOps, axis_colors, KernelOptError, check
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.device import Device

# this is Kernel now
class RangeManip:
  def __init__(self, ast:UOp, opts:Renderer|None=None):
    self.ast = ast
    self.opts = opts if opts is not None else Device[Device.DEFAULT].renderer

    self.vars: list[Variable] = self.ast.variables()
    # NOTE: this requires a specific order with the [::-1], this is likely a bug
    self.bufs: list[UOp] = [x for x in self.ast.toposort() if x.op in {Ops.LOAD, Ops.STORE}][::-1]

    self.replaces = {}
    self.rng = sorted([u for u in ast.toposort() if u.op is Ops.RANGE and u.vmax > 0], key=lambda x: x.arg)
    self.tensor_core = None

    # convert LOOP to GLOBAL
    if self.opts.has_local:
      rng = [x.replace(arg=(x.arg[0], AxisType.GLOBAL)) if x.arg[1] == AxisType.LOOP else x for x in self.rng]
      self.replaces.update(dict(zip(self.rng, rng)))
      self.rng = rng

    self.reduceops = [x for x in self.ast.toposort() if x.op is Ops.REDUCE]

    self.applied_opts: list[Opt] = []

  # TODO: need this?
  def copy(self):
    return RangeManip(self.ast, self.opts)

  @property
  def reduceop(self) -> UOp|None: return self.reduceops[0] if len(self.reduceops) > 0 else None

  @property
  def name(self): return "k"+colored('_', 'BLACK').join(['']+[colored(s.src[0].render(), axis_colors[s.arg[1]]) for s in self.rng])

  @property
  def shape_len(self): return len(self.rng)
  @property
  def output_shape(self) -> tuple[sint, ...]:
    return tuple([ssimplify(x.src[0]) for x in self.ast.src[0].src[2:]])
  @property
  def full_shape(self) -> tuple[sint, ...]: return tuple([ssimplify(x.src[0]) for x in self.rng])
  @property
  def axis_types(self) -> list[AxisType]: return [x.arg[1] for x in self.rng]

  def axes_of(self, *axis_type:AxisType) -> list[int]: return [i for i,t in enumerate(self.axis_types) if t in argfix(axis_type)]
  @property
  def upcasted(self) -> int: return len(self.axes_of(AxisType.UPCAST, AxisType.UNROLL))
  @property
  def group_for_reduces(self) -> int: return len(self.axes_of(AxisType.GROUP_REDUCE))

  # heuristic helpers
  @property
  def upcastable_dims(self) -> list[int]: return [i for i in self.axes_of(AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP) \
                                                  if isinstance(s:=self.full_shape[i], int) and s > 1]
  @property
  def unrollable_dims(self) -> list[int]: return [i for i in self.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE) \
                                                  if isinstance(s:=self.full_shape[i], int) and s > 1]

  def shift_to(self, axis:int, amount:int, new_type:AxisType, top:bool=False, insert_at:int|None=None):
    old_sz = self.rng[axis].src[0].arg // amount
    assert old_sz > 0, f"bad old_sz on {axis} {amount} {self.rng[axis]}"

    maxarg = max([x.arg[0] for x in self.rng])
    new_rng = UOp.range(dtypes.int, amount, maxarg+1, new_type)

    if old_sz == 1:
      self.replaces[self.rng[axis]] = new_rng
      self.rng.insert(insert_at if insert_at is not None else len(self.rng), new_rng)
      del self.rng[axis]
    else:
      replaced_rng = self.rng[axis].replace(src=(UOp.const(dtypes.int, old_sz),))
      self.replaces[self.rng[axis]] = replaced_rng * amount + new_rng
      self.rng[axis] = replaced_rng
      self.rng.insert(insert_at if insert_at is not None else len(self.rng), new_rng)

  def renumber(self):
    # renumber in the order of the self.rng array
    for i,r in enumerate(self.rng):
      if r.arg[0] != i:
        rng = r.replace(arg=(i, r.arg[1]))
        self.replaces[r] = rng
        self.rng[i] = rng

  def real_axis(self, op:OptOps, axis:int|None):
    try:
      if axis is None: return -1
      if op is OptOps.UNROLL: return self.unrollable_dims[axis]
      if op in {OptOps.GROUP, OptOps.GROUPTOP}: return self.axes_of(AxisType.REDUCE)[axis]
      check(axis < self.shape_len, "invalid axis")
      return axis
    except IndexError as e: raise KernelOptError from e

  def apply_opt(self, opt:Opt):
    axis = self.real_axis(opt.op, opt.axis)
    amt = arg if (arg:=cast(int, opt.arg)) != 0 else self.full_shape[axis]

    if opt.op is OptOps.LOCAL:    # cyan
      # NOTE: LLVM/CPU can use locals too, but they are treated the same as globals (still helpful for L1 cache)
      # it's disabled for now since it makes BEAM slow for little gain
      check(self.opts.has_local, "target does not support local")
      check(self.axis_types[axis] is AxisType.GLOBAL, "local is for globals")
      self.shift_to(axis, amt, AxisType.LOCAL, insert_at=max(self.axes_of(AxisType.GLOBAL, AxisType.LOCAL))+1)
    elif opt.op is OptOps.UNROLL:                     # purple
      check(self.axis_types[axis] not in (AxisType.UPCAST, AxisType.UNROLL), "can't upcasted already upcasted")
      check(amt <= 32, "don't unroll more than 32")
      self.shift_to(axis, amt, AxisType.UNROLL, insert_at=None)
      #self.shift_to(axis, amt, AxisType.LOOP, insert_at=None)
    elif opt.op is OptOps.UPCAST:                     # yellow
      check(axis in self.upcastable_dims, f"{axis=} not in {self.upcastable_dims=}")
      # NOTE: assume the first get_local_axes() LOCAL are for TC
      check(not (self.tensor_core and axis in self.axes_of(AxisType.LOCAL)[:len(self.tensor_core.get_local_axes())]), "can't upcast TC locals")
      check((self.opts is not None and self.opts.device == "DSP") or amt <= 16, "don't upcast more than 16")
      self.shift_to(axis, amt, AxisType.UPCAST, insert_at=max(self.axes_of(AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP, AxisType.UPCAST))+1)
    else:
      raise RuntimeError(f"{opt.op} not supported")

    self.applied_opts.append(opt)

def add_name(ctx:Renderer, s:UOp):
  if s.tag == 1: return None
  manip = RangeManip(s, ctx)
  arg = s.arg if s.arg is not None else KernelInfo()
  if arg.opts_to_apply:
    for opt in arg.opts_to_apply: manip.apply_opt(opt)
  else:
    pass

  manip.renumber()
  s = s.substitute(manip.replaces)
  return s.replace(arg=replace(arg, name=manip.name, opts_to_apply=None), tag=1)

def flatten_range(r:UOp):
  off = 2 if r.op is Ops.STORE else 1
  rngs = r.src[off:]
  if not len(rngs): return None
  new_rngs = [x for x in UOp.sink(*rngs).toposort() if x.op is Ops.RANGE]
  return r.replace(src=r.src[:off]+tuple(new_rngs))

pm_postrange_opt = PatternMatcher([
  (UPat((Ops.REDUCE, Ops.STORE), name="r"), flatten_range),
  (UPat(Ops.SINK, name="s"), add_name),
])

# local optimizer (late)

def load_to_locals(l:UOp):
  # if already processed or not GLOBAL, skip
  if l.tag == 1 or l.src[0].dtype.addrspace != AddrSpace.GLOBAL: return None

  # use the global buffer index as the index to create the new load ranges
  load_index = l.src[0].src[0].arg

  # get all non GLOBAL ranges in the scope of the load
  rngs = [x for x in l.ranges if x.arg[1] not in {AxisType.GLOBAL, AxisType.REDUCE}]

  # create new ranges for the GLOBAL -> LOCAL copy
  # NOTE: these don't have to have the same AxisType
  #new_rngs = [UOp.range(dtypes.int, x.vmax+1, load_index*10000+x.arg[0], x.arg[1]) for x in rngs]
  new_rngs = [UOp.range(dtypes.int, x.vmax+1, load_index*10000+x.arg[0], AxisType.LOCAL if x.vmax+1 != 4 else AxisType.LOOP) for x in rngs]
  ls = prod([x.vmax+1 for x in new_rngs if x.arg[1] == AxisType.LOCAL])
  assert ls == 256, f"local size is {ls}"

  # update the global load to use the new ranges
  new_global_load = l.replace(tag=1).substitute(dict(zip(rngs, new_rngs)))

  # NOTE: new_rngs/rngs can be permuted as desired if you permute them together. this puts LOCAL at the end
  rngs_ordered = sorted(enumerate(rngs), key=lambda x: -10000 if x[1].arg[1] in (AxisType.UNROLL, AxisType.LOOP) else 0)
  rngs = [x for _,x in rngs_ordered]
  new_rngs = [new_rngs[i] for i,_ in rngs_ordered]

  # BUFFERIZE+INDEX to create a new buffer, store to it, and load from it
  return UOp(Ops.BUFFERIZE, l.dtype, (new_global_load,)+tuple(new_rngs), arg=AddrSpace.LOCAL).index(*rngs)

from tinygrad.schedule.rangeify import pm_add_buffers, rangeify_codegen

pm_postrange_opt_2 = PatternMatcher([
  (UPat(Ops.LOAD, name="l"), load_to_locals),
])+pm_add_buffers+rangeify_codegen

