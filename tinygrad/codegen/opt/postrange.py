from typing import cast
from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, AxisType, sint, ssimplify, KernelInfo
from tinygrad.helpers import colored, argfix
from tinygrad.codegen.opt.kernel import Opt, OptOps, axis_colors, KernelOptError, check
from tinygrad.dtype import dtypes
from tinygrad.renderer import Renderer
from tinygrad.device import Device

# this is Kernel now
class RangeManip:
  def __init__(self, ast:UOp, opts:Renderer|None=None):
    self.opts = opts if opts is not None else Device[Device.DEFAULT].renderer

    self.replaces = {}
    self.rng = sorted([u for u in ast.toposort() if u.op is Ops.RANGE], key=lambda x: x.arg)

    # convert LOOP to GLOBAL
    if self.opts.has_local:
      rng = [x.replace(arg=(x.arg[0], AxisType.GLOBAL)) if x.arg[1] == AxisType.LOOP else x for x in self.rng]
      self.replaces.update(dict(zip(self.rng, rng)))
      self.rng = rng

  @property
  def name(self): return "k"+colored('_', 'BLACK').join(['']+[colored(s.src[0].render(), axis_colors[s.arg[1]]) for s in self.rng])

  @property
  def shape_len(self): return len(self.rng)
  @property
  def full_shape(self) -> tuple[sint, ...]: return tuple([ssimplify(x.src[0]) for x in self.rng])
  @property
  def axis_types(self) -> list[AxisType]: return [x.arg[1] for x in self.rng]

  def axes_of(self, *axis_type:AxisType) -> list[int]: return [i for i,t in enumerate(self.axis_types) if t in argfix(axis_type)]

  # heuristic helpers
  @property
  def upcastable_dims(self) -> list[int]: return [i for i in self.axes_of(AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP) \
                                                  if isinstance(s:=self.full_shape[i], int) and s > 1]
  @property
  def unrollable_dims(self) -> list[int]: return [i for i in self.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE) \
                                                  if isinstance(s:=self.full_shape[i], int) and s > 1]

  def shift_to(self, axis:int, amount:int, new_type:AxisType, top:bool=False, insert_at:int|None=None):
    replaced_rng = self.rng[axis].replace(src=(UOp.const(dtypes.int, self.rng[axis].src[0].arg // amount),))

    maxarg = max([x.arg[0] for x in self.rng])
    new_rng = UOp.range(dtypes.int, amount, maxarg+1, AxisType.UPCAST)

    self.replaces[self.rng[axis]] = replaced_rng * amount + new_rng

    self.rng[axis] = replaced_rng
    self.rng.append(new_rng)

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

    if opt.op is OptOps.UPCAST:                     # yellow
      check(self.axis_types[axis] not in (AxisType.UPCAST, AxisType.UNROLL), "can't upcasted already upcasted")
      check(amt <= 32, "don't unroll more than 32")
      self.shift_to(axis, amt, AxisType.UNROLL, insert_at=None)

def add_name(ctx:Renderer, s:UOp):
  manip = RangeManip(s, ctx)
  arg = s.arg if s.arg is not None else KernelInfo()
  if arg.opts_to_apply:
    for opt in arg.opts_to_apply:
      manip.apply_opt(opt)
  s = s.substitute(manip.replaces)
  return s.replace(arg=replace(arg, name=manip.name, opts_to_apply=None))

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.SINK, name="s"), add_name),
])