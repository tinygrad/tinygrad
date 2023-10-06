from tinygrad.ops import LazyOp, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import Timing, dtypes, prod
from extra.utils import print_tree
from tinygrad.runtime.ops_gpu import renderer, CLBuffer, CLProgram

if __name__ == "__main__":
  # kernel 11 has big regression
  # from 124.01 GFLOPS -> 87.82 GFLOPS
  # old: re_S32_16_6_36       with [6, 16, 32]     [6, 4, 16]  new style [1,4,2]   [6,4,16]
  # new: r_16_2_2_16_3_36_4_4_4                arg   5 sz [2, 16, 1]         [3, 16, 2]

  op = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.imageh((32, 2304, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 36, 4, 1, 1), strides=(0, 9216, 144, 0, 0, 0, 0, 4, 1, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.imageh((6, 144, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 36, 4, 1, 1), strides=(0, 0, 0, 0, 0, 576, 1, 16, 4, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(dtypes.float, False)),), arg=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1)),), arg=(dtypes.imageh((32, 384, 4)), False)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=4, dtype=dtypes.imageh((32, 384, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), strides=(0, 1536, 24, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None)
  print_tree(op)

  with Timing("linearize: "):
    lin = Linearizer(op)
    lin.hand_coded_optimizations()
    lin.linearize()

  print(lin.display_name)
  with Timing("renderer: "):
    fxn = renderer(lin.function_name, lin.uops)

  bufs = {0: CLBuffer(prod(lin.output_shape), lin.info.dtype)}
  for x in op.get_lazyops():
    if x.op != BufferOps.MEM: continue
    bufs[x.arg.idx] = CLBuffer(x.arg.st.size(), x.arg.dtype)

  prg = CLProgram(lin.function_name, fxn)
  tm = min([prg(lin.global_size, lin.local_size, *[bufs[i] for i in range(len(bufs))], wait=True) for _ in range(10)])
  print(f"{lin.info.flops*1e-9/tm:8.2f} GFLOPS")


