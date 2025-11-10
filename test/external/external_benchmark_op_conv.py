# ruff: noqa: E501 E712
from tinygrad import dtypes, Device
from tinygrad.uop.ops import UOp, AxisType, Ops, KernelInfo
from tinygrad.codegen import full_rewrite
# from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import dedup, getenv
from tinygrad.device import Buffer
from tinygrad.dtype import ImageDType, Invalid

# PYTHONPATH="." DEV=QCOM FLOAT16=1 IMAGE=2 NOLOCALS=1 taskset -c 4-7 python3 examples/openpilot/compile3.py https://github.com/commaai/openpilot/raw/720392c9a5b986981fdbed1bb8c47a6c5573a50e/selfdrive/modeld/models/driving_vision.onnx

def vision_conv_143():
  c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((16, 1024, 4)), (), 0)
  c2 = UOp.range(32, 3, AxisType.LOOP)
  c5 = UOp.range(128, 4, AxisType.LOOP)
  c8 = UOp.range(16, 2, AxisType.LOOP)
  c16 = UOp.range(7, 0, AxisType.REDUCE)
  c17 = c8*2+c16
  c24 = ((c17<3)!=True)&(c17<35)
  c26 = UOp.range(7, 1, AxisType.REDUCE)
  c27 = c2*2+c26
  c32 = ((c27<3)!=True)&(c27<67)
  c34 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((32, 1024, 4)), (), 1)
  c38 = c5//2
  c45 = (c32&c24).where((c27*64+c38+c17*4096+-12480), UOp.const(dtypes.index, Invalid))
  c48 = (c24&c32).where(c34.index(c45), UOp.const(dtypes.float, 0.0))
  c49 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((64, 49, 4)), (), 2)
  c61 = c48*c49.index((c26*4+c5%2+c16*28+c38*196))
  c63 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(128), (), 3)
  c65 = c61.reduce(c16, c26, arg=Ops.ADD)+c63.index(c5)
  c67 = c0.index((c2*128+c5+c8*4096), ptr=True).store(c65).end(c8, c2, c5)

  opts = None
  return c67.sink(arg=KernelInfo(name="conv", opts_to_apply=opts))

def vision_conv_153():
  c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((8, 1024, 4)), (), 0)
  c2 = UOp.range(16, 3, AxisType.LOOP)
  c5 = UOp.range(256, 4, AxisType.LOOP)
  c8 = UOp.range(8, 2, AxisType.LOOP)
  c16 = UOp.range(7, 0, AxisType.REDUCE)
  c17 = c8*2+c16
  c24 = ((c17<3)!=True)&(c17<19)
  c26 = UOp.range(7, 1, AxisType.REDUCE)
  c27 = c2*2+c26
  c32 = ((c27<3)!=True)&(c27<35)
  c34 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((16, 1024, 4)), (), 1)
  c38 = c5//2
  c45 = (c32&c24).where((c27*128+c38+c17*4096+-12672), UOp.const(dtypes.index, Invalid))
  c48 = (c24&c32).where(c34.index(c45), UOp.const(dtypes.float, 0.0))
  c49 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((128, 49, 4)), (), 2)
  c61 = c48*c49.index((c26*4+c5%2+c16*28+c38*196))
  c63 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(256), (), 3)
  c65 = c61.reduce(c16, c26, arg=Ops.ADD)+c63.index(c5)
  c67 = c0.index((c2*256+c5+c8*4096), ptr=True).store(c65).end(c8, c2, c5)

  opts = None
  return c67.sink(arg=KernelInfo(name="conv", opts_to_apply=opts))

ast = vision_conv_143() if getenv("NUM", 143) == 143 else vision_conv_153()

compiler = Device.default.compiler
renderer = Device.default.renderer
allocator = Device.default.allocator

uops = full_rewrite(ast, renderer)
src = renderer.render(uops)

lib = compiler.compile(src)
ps = ProgramSpec("conv", src, Device.DEFAULT, ast, uops)
cr = CompiledRunner(ps, precompiled=lib)

gs = sorted(dedup([u for u in ast.toposort() if u.op is Ops.DEFINE_GLOBAL]), key=lambda u: u.arg)
# print(len(gs))
# print([g.dtype for g in gs])
bufs = [Buffer(ps.device, g.size, g.dtype if isinstance(g.dtype, ImageDType) else g.dtype._base).ensure_allocated() for g in gs]

t = cr(bufs, wait=True)
print(f"{t*1e6:.2f} us")