from tinygrad.ops import LazyOp, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import Timing, dtypes, prod, flatten, getenv, ansilen, colored
from extra.utils import print_tree
from tinygrad.runtime.ops_gpu import renderer, CLBuffer, CLProgram

if __name__ == "__main__":
  # kernel 9
  # 139.79 -> 114.37 (145.32 GFLOPS standalone)
  #op = LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BinaryOps.MAX, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.imageh((32, 384, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 6, 4, 1, 1), strides=(0, 1536, 24, 0, 0, 0, 0, 4, 1, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.imageh((36, 24, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 6, 4, 1, 1), strides=(0, 0, 0, 0, 0, 96, 1, 16, 4, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(dtypes.float, False)),), arg=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1)),), arg=(dtypes.imageh((32, 2304, 4)), False)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BinaryOps.MAX, src=(LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=UnaryOps.EXP2, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.imageh((32, 384, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 6, 4, 1, 1), strides=(0, 1536, 24, 0, 0, 0, 0, 4, 1, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.imageh((36, 24, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 6, 4, 1, 1), strides=(0, 0, 0, 0, 0, 96, 1, 16, 4, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(dtypes.float, False)),), arg=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1)),), arg=(dtypes.imageh((32, 2304, 4)), False)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.4426950408889634, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None)
  # search find 154.91 GFLOPS!

  # kernel 10
  # 63.12 -> 32.62 (60 GFLOPS standalone)
  #op = LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BinaryOps.MAX, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.imageh((32, 2304, 4)), st=ShapeTracker(views=(View(shape=(1, 144, 1, 1, 4, 34, 4, 66), strides=(0, 1, 0, 0, 0, 9216, 0, 144), offset=-9360, mask=((0, 1), (0, 144), (0, 1), (0, 1), (0, 4), (1, 33), (0, 4), (1, 65)), contiguous=False), View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 3, 3), strides=(0, 264, 1, 143616, 35904, 0, 0, 0, 0, 9240, 67), offset=0, mask=None, contiguous=False))))), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.imageh((36, 9, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 3, 3), strides=(0, 0, 0, 36, 1, 0, 0, 0, 0, 12, 4), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(dtypes.float, False)),), arg=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1)),), arg=(dtypes.imageh((32, 2304, 4)), False)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1), strides=(0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BinaryOps.MAX, src=(LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=UnaryOps.EXP2, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.imageh((32, 2304, 4)), st=ShapeTracker(views=(View(shape=(1, 144, 1, 1, 4, 34, 4, 66), strides=(0, 1, 0, 0, 0, 9216, 0, 144), offset=-9360, mask=((0, 1), (0, 144), (0, 1), (0, 1), (0, 4), (1, 33), (0, 4), (1, 65)), contiguous=False), View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 3, 3), strides=(0, 264, 1, 143616, 35904, 0, 0, 0, 0, 9240, 67), offset=0, mask=None, contiguous=False))))), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.imageh((36, 9, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 3, 3), strides=(0, 0, 0, 36, 1, 0, 0, 0, 0, 12, 4), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(dtypes.float, False)),), arg=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1)),), arg=(dtypes.imageh((32, 2304, 4)), False)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1), strides=(0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.4426950408889634, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=None)), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 32, 64, 36, 4, 1, 1, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None)

  # kernel 11 has big regression
  # from 124.01 GFLOPS -> 87.82 GFLOPS (95.67 GFLOPS standalone)
  # old: re_S32_16_6_36       with [6, 16, 32]     [6, 4, 16]  new style [1,4,2]   [6,4,16]
  # new: r_16_2_2_16_3_36_4_4_4                arg   5 sz [2, 16, 1]         [3, 16, 2]
  op = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.imageh((32, 2304, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 36, 4, 1, 1), strides=(0, 9216, 144, 0, 0, 0, 0, 4, 1, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.imageh((6, 144, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 36, 4, 1, 1), strides=(0, 0, 0, 0, 0, 576, 1, 16, 4, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(dtypes.float, False)),), arg=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1)),), arg=(dtypes.imageh((32, 384, 4)), False)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=4, dtype=dtypes.imageh((32, 384, 4)), st=ShapeTracker(views=(View(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), strides=(0, 1536, 24, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None)
  # search finds 143.88 GFLOPS!

  # print the kernel tree
  print_tree(op)

  def test(real_local, upcasts=[]):
    lin = Linearizer(op)
    print(lin.colored_shape())
    if not real_local:
      lin.hand_coded_optimizations()
    else:
      lin.required_optimizations()
      lin.simplify_ones()
      # global reshape
      base_shape = lin.info.dtype.shape
      lin.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
      # upcast 4
      for axis,amount in upcasts:
        lin.shift_to(axis, amount)
        lin.upcast()
      # local reshape
      #real_local = [8 if CLProgram.max_work_group_size() == 256 else 16,4,6]
      #real_local = [1,1,1]
      lin.reshape_and_permute(lambda x: flatten([(x[i]//l,l) for i,l in enumerate(real_local)])+list(x[3:]), [0,2,4,1,3,5] + [x+3 for x in range(3, lin.shape_len)])
      lin.local_dims += 3
    lin.linearize()

    fxn = renderer(lin.function_name, lin.uops)
    #fxn = fxn.replace("<= 35", "< 36")
    #print(fxn)

    bufs = {0: CLBuffer(prod(lin.output_shape), lin.info.dtype)}
    #bufs[4] = bufs[0]   # buffer reuse on accumulate
    for x in op.get_lazyops():
      if x.op != BufferOps.MEM or x.arg.idx in bufs: continue
      bufs[x.arg.idx] = CLBuffer(x.arg.st.size(), x.arg.dtype)

    prg = CLProgram(lin.function_name, fxn)
    tm = min([prg(lin.global_size, lin.local_size, *[bufs[i] for i in range(len(bufs))], wait=True) for _ in range(20)])
    gflops = lin.info.flops*1e-9/tm
    print(lin.display_name+' '*(37-ansilen(lin.display_name)), upcasts, lin.global_size, lin.local_size,
          "old", [g*l for g,l in zip(lin.global_size, lin.local_size)],
          "output type", lin.info.dtype,
          f"{tm*1e6:10.2f} us",
          f"{gflops:8.2f} GFLOPS")
    return gflops

  if getenv("BASELINE"):
    test(None)
    exit(0)

  test([32, 1, 6], [(1, 4)])

  """
  max_gflops = 0
  for axis in [0,1,2]:
    for amt in [2,3,4,6,8]:
      for l0 in [1,2,4,8,16,32]:
        for l1 in [1,2,4,8,16]:
          for l2 in [1,2,3,6]:
            try:
              gflops = test([l0,l1,l2], [(axis,amt)])
              if gflops > max_gflops:
                max_gflops = gflops
                print(colored(f"new max GFLOPS: {max_gflops:8.2f}", "red"))
            except Exception:
              continue
  print(f"max GFLOPS: {max_gflops:8.2f}")
  """

