import math
from tinygrad.uop.ops import UOp, Ops, sint, ssimplify, AxisType, KernelInfo
from tinygrad.codegen.opt.kernel import Kernel, Opt, OptOps
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
      store_rng = self.ast.src[0].src[2:]
      rng = [x.replace(arg=(x.arg[0], AxisType.GLOBAL)) if x.arg[1] == AxisType.LOOP and x in store_rng else x for x in self.rng]
      self.replaces.update(dict(zip(self.rng, rng)))
      self.rng = rng

  def _apply_tc_opt(self, use_tensor_cores:int, axis:int, tc_select:int, opt_level:int) -> bool:
    reduceop = [x for x in self.ast.toposort() if x.op is Ops.REDUCE][0]
    if use_tensor_cores and reduceop is not None and reduceop.arg is Ops.ADD:
      tensor_cores = self.opts.tensor_cores if tc_select == -1 else [self.opts.tensor_cores[tc_select]]
      for tc in tensor_cores:
        if tc.dtype_in == dtypes.float and tc.dtype_out == dtypes.float:
          axes = [1,0]

          # do optimizations and save the ranges
          ne: list[UOp] = []
          for opt in tc.opts:
            ne.append(self.apply_opt(Opt({"u":OptOps.UPCAST, "l":OptOps.LOCAL}[opt[0]], axes[int(opt[1])], 2), append_opt=False))
          for _, amt in tc.get_reduce_axes():
            ne.append(self.apply_opt(Opt(OptOps.UNROLL, 0, amt), append_opt=False)) # TODO: this should be the reduce, not 0

          # early realize for TC
          self.ast = self.ast.substitute(self.replaces)
          self.replaces = {}

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
          tc_upcast_axes = tuple([tuple([(self.rng[a].arg[0], sz) for a,sz in v]) for v in tc_upcast_axes])
          tc_reduce_axes = tuple([self.rng[a].arg[0] for a in tc_reduce_axes])

          # construct the op
          # TODO: remove tc_upcast_axes from the arg
          wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.opts.device, tc.threads, tc_upcast_axes, tc_reduce_axes)
          wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
            UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0]),
            UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1]),
            UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg)
          tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2])

          # preserve extra reduces
          reduce_ranges = [x for x in UOp.sink(*reduceop.src[1:]).toposort() if x.op is Ops.RANGE and x.arg[0] not in tc_reduce_axes]
          if len(reduce_ranges): tc_uop = UOp(Ops.REDUCE, tc_uop.dtype, (tc_uop,)+tuple(reduce_ranges), Ops.ADD)
          self.ast = self.ast.substitute({reduceop: tc_uop})
          return True
    return False

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
      self.replaces[self.rng[axis]] = (new_rng * old_sz + replaced_rng) if top else (replaced_rng * amount + new_rng)
      self.rng[axis] = replaced_rng
      self.rng.insert(insert_at if insert_at is not None else len(self.rng), new_rng)
    return new_rng

  @property
  def axis_types(self) -> list[AxisType]: return [x.arg[1] for x in self.rng]
  @property
  def shape_len(self): return len(self.rng)

  @property
  def full_shape(self) -> tuple[sint, ...]: return tuple([ssimplify(x.src[0]) for x in self.rng])
  @property
  def output_shape(self) -> tuple[sint, ...]: return tuple([ssimplify(x.src[0]) for x in self.ast.src[0].src[2:]])

  def get_optimized_ast(self, name_override:str|None=None) -> UOp:
    ret = self.ast
    kernel_name = ret.arg.name if ret.arg is not None and ret.arg.name != "test" else self.name if name_override is None else name_override
    rarg = KernelInfo(kernel_name, tuple(self.axis_types), self.dont_use_locals, tuple(self.applied_opts))
    return ret.substitute(self.replaces).replace(arg=rarg)

  # does nothing
  @axis_types.setter
  def axis_types(self, value): pass
