from dataclasses import replace
from tinygrad.uop.ops import UOp, Ops, sint, ssimplify, AxisType
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.renderer import Renderer
from tinygrad.dtype import dtypes

class RKernel(Kernel):
  def __init__(self, ast:UOp, opts:Renderer|None=None):
    self.rng = sorted([u for u in ast.toposort() if u.op is Ops.RANGE and u.vmax > 0], key=lambda x: x.arg)
    super().__init__(ast, opts)
    self.sts.clear()

    # convert LOOP to GLOBAL
    self.replaces = {}
    if self.opts.has_local:
      rng = [x.replace(arg=(x.arg[0], AxisType.GLOBAL)) if x.arg[1] == AxisType.LOOP else x for x in self.rng]
      self.replaces.update(dict(zip(self.rng, rng)))
      self.rng = rng

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

  @property
  def axis_types(self) -> list[AxisType]: return [x.arg[1] for x in self.rng]
  @property
  def shape_len(self): return len(self.rng)

  @property
  def full_shape(self) -> tuple[sint, ...]: return tuple([ssimplify(x.src[0]) for x in self.rng])
  @property
  def output_shape(self) -> tuple[sint, ...]: return tuple([ssimplify(x.src[0]) for x in self.ast.src[0].src[2:]])

  def get_optimized_ast(self, name_override:str|None=None) -> UOp:
    return self.ast.substitute(self.replaces).replace(arg=replace(self.ast.arg, name=self.name, opts_to_apply=None))

  # does nothing
  @axis_types.setter
  def axis_types(self, value): pass
