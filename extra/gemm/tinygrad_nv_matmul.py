import functools
from tinygrad import Tensor, dtypes, Device
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem
from tinygrad.ops import LazyOp, BufferOps, MemBuffer, ShapeTracker

class HackedKernel(Kernel):
  # add locals
  def get_optimized_ast(self) -> LazyOp:
    ast = super().get_optimized_ast()
    @functools.lru_cache(None)
    def fixup_ast(op:LazyOp) -> LazyOp:
      if op.op is BufferOps.LOAD and op.arg.idx in [1]:
        #local_shape = (1,) * self.global_dims + self.full_shape[self.global_dims:self.global_dims+self.local_dims+self.group_for_reduces] + \
        #  (1,) * (self.shape_len - self.upcasted - self.group_for_reduces - self.first_reduce) + tuple([x[0] for x in self.upcasted_axis(0)])
        shape = list(op.arg.st.views[0].shape)
        strides = op.arg.st.views[0].strides
        shape[:self.global_dims] = [1] * self.global_dims
        shape[self.first_reduce] = 1
        start_shape = tuple(1 if st == 0 else s for s,st in zip(shape, strides))
        new_st = ShapeTracker.from_shape(start_shape)
        order = list(range(len(shape)))
        #if op.arg.idx == 1: order[3], order[5] = order[5], order[3]
        if op.arg.idx == 1: order[5], order[9] = order[9], order[5]
        store_st = new_st.permute(tuple(order))
        load_st = new_st.expand(tuple(shape)) #.permute(tuple(order))
        main_load = LazyOp(BufferOps.LOAD, arg=MemBuffer(op.arg.idx, op.arg.dtype, op.arg.st.permute(tuple(order))))
        local_store = LazyOp(BufferOps.STORE, (main_load,), MemBuffer(-op.arg.idx, op.dtype, store_st))
        local_load = LazyOp(BufferOps.LOAD, (local_store,), MemBuffer(-op.arg.idx, op.dtype, load_st))
        return local_load
      return LazyOp(op.op, tuple(fixup_ast(x) for x in op.src), op.arg)
    ast = fixup_ast(ast)
    print(ast)
    return ast

N = 4096
if __name__ == "__main__":
  A, B = Tensor.empty(N, N, dtype=dtypes.float16), Tensor.empty(N, N, dtype=dtypes.float16)
  C = A.matmul(B)
  si = C.schedule()[-1]
  ast = si.ast
  if Device.DEFAULT in {"AMD", "HIP"}:
    k = HackedKernel(ast, opts=Device[Device.DEFAULT].renderer)
    opts = [
      Opt(op=OptOps.TC, axis=0, amt=0),
      Opt(op=OptOps.UPCAST, axis=0, amt=4),
      Opt(op=OptOps.UPCAST, axis=1, amt=4),
      #Opt(op=OptOps.LOCAL, axis=0, amt=4),
      Opt(op=OptOps.LOCAL, axis=1, amt=4),
    ]
  else:
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    opts = [
      Opt(op=OptOps.TC, axis=0, amt=0),
      Opt(op=OptOps.UPCAST, axis=0, amt=4),
      Opt(op=OptOps.UPCAST, axis=1, amt=8),
      Opt(op=OptOps.LOCAL, axis=0, amt=2),
      Opt(op=OptOps.LOCAL, axis=1, amt=2),
      Opt(op=OptOps.LOCAL, axis=0, amt=2),
    ]
  for opt in opts: k.apply_opt(opt)
  prg = k.to_program()
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  tflops = []
  for i in range(5):
    tm = ei.run(wait=True)
    tflops.append((2*N*N*N/tm)*1e-12)
  print(f"TFLOPS: {max(tflops):.2f}")
