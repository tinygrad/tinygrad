from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen.opt.kernel import Opt, OptOps
from tinygrad.engine.realize import get_program, CompiledRunner
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.codegen.opt.postrange import bufs_from_ast

if __name__ == "__main__":
  c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1179648), arg=0, src=())
  c1 = UOp.range(UOp.const(dtypes.int, 512), 0, AxisType.GLOBAL)
  c2 = UOp.range(UOp.const(dtypes.int, 64), 1, AxisType.GLOBAL)
  c3 = UOp.range(UOp.const(dtypes.int, 6), 2, AxisType.GLOBAL)
  c4 = UOp.range(UOp.const(dtypes.int, 6), 3, AxisType.GLOBAL)
  c5 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(2097152), arg=1, src=())
  c6 = UOp.range(UOp.const(dtypes.int, 64), 1004, AxisType.REDUCE)
  c7 = UOp.range(UOp.const(dtypes.int, 3), 1005, AxisType.REDUCE)
  c8 = UOp.range(UOp.const(dtypes.int, 3), 1006, AxisType.REDUCE)
  c9 = c5.index(((((((c1*UOp.const(dtypes.int, 4096))+(c3*UOp.const(dtypes.int, 8)))+c4)+(c6*UOp.const(dtypes.int, 64)))+(c7*UOp.const(dtypes.int, 8)))+c8), UOp.const(dtypes.bool, True)).load()
  c10 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(36864), arg=2, src=())
  c11 = c10.index(((((c2*UOp.const(dtypes.int, 576))+(c6*UOp.const(dtypes.int, 9)))+(c7*UOp.const(dtypes.int, 3)))+c8), UOp.const(dtypes.bool, True)).load()
  c12 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64), arg=3, src=())
  c13 = c12.index(c2, UOp.const(dtypes.bool, True)).load()
  c14 = ((c9*c11).reduce(c6, c7, c8, arg=Ops.ADD)+c13)
  c15 = c0.index(((((c1*UOp.const(dtypes.int, 2304))+(c2*UOp.const(dtypes.int, 36)))+(c3*UOp.const(dtypes.int, 6)))+c4), UOp.const(dtypes.bool, True)).store(c14, c1, c2, c3, c4)

  opts = [Opt(op=OptOps.TC, axis=0, arg=(0, 0, 1)), Opt(op=OptOps.UPCAST, axis=3, arg=0), Opt(op=OptOps.UPCAST, axis=2, arg=2),
          Opt(op=OptOps.LOCAL, axis=1, arg=8), Opt(op=OptOps.GROUP, axis=1, arg=0)]
  ast = c15.sink(arg=KernelInfo(name='test', axis_types=(), dont_use_locals=False, applied_opts=(), opts_to_apply=tuple(opts)))
  bufs = bufs_from_ast(ast, Device.DEFAULT)
  prg = get_program(ast)
  print(prg.src)
  for i in range(10):
    print(f"try {i}")
    try: CompiledRunner(prg)(*bufs)
    except RuntimeError: pass
