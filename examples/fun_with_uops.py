from tinygrad import UOp, dtypes, Device
from tinygrad.uop.ops import Ops
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.schedule.rangeify import get_rangeify_map
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.linearizer import linearize
#from tinygrad.engine.schedule import create_schedule_with_vars

ren = Device["CPU"].renderer

def my_matrix_mul(A:UOp, B:UOp, C:UOp) -> UOp:
  """
  for i in UOp.range(10):
    for j in UOp.range(10):
      C[i,j] = 0.0
      for k in UOp.range(10):
        C[i, j] += A[i, k] * B[k, j]
  """

  # TODO: make string ranges work
  i = UOp.range(10, 0) #"i")
  j = UOp.range(10, 1) #"j")
  #C = C.after(C[i, j].store(UOp.const(dtypes.float, 0.0)))
  # TOOD: support multiindex
  C = C.after(C[i*10+j].store(UOp.const(dtypes.float, 0.0)))
  k = UOp.range(10, 2) #"k")
  # TODO: remove loads and ptrs in general pre rendering
  #store = C[i, j].store(C[i, j].load() + (A[i, k].load() * B[k, j].load()))
  # TODO: make reshape work
  store = C[i*10+j].store(C.after(k)[i*10+j].load() + (A[i*10+k].load() * B[k*10+j].load()))
  return store.end(i,j,k).sink()

#a = UOp.new_buffer("CPU", 10*10, dtypes.float).reshape((10,10))
#b = UOp.new_buffer("CPU", 10*10, dtypes.float).reshape((10,10))
#c = UOp.new_buffer("CPU", 10*10, dtypes.float).reshape((10,10))
a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(100), arg=0) #.reshape((10,10))
b = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(100), arg=1) #.reshape((10,10))
c = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(100), arg=2) #.reshape((10,10))
out = my_matrix_mul(a, b, c)
print(out.pyrender())

print(ren.render(linearize(full_rewrite_to_sink(out, optimize=False))))

#ss = get_rangeify_map(out)[c]
#print("\n\n",ss.pyrender())

exit(0)


# store a 10 by 10 grid of 0s
x,y = UOp.range(10, "x"), UOp.range(10, "y")
base = UOp.const(dtypes.float, 0, device="CPU").contiguous()
buf = UOp.bufferize(base, x, y, arg=BufferizeOpts("CPU"))
print(buf.pyrender())

# add 1
x,y = UOp.range(10, "x2"), UOp.range(10, "y2")
buf2 = UOp.bufferize(buf[x,y]+1, x, y, arg=BufferizeOpts("CPU"))

# TODO: pyrender should be print
print("\n\n")
print(buf2.pyrender())

print("\n\n")
sink = buf2.sink()
# TODO: have a flag here to disable opt
rr = get_rangeify_map(sink)[buf2]
print(rr.pyrender())

# schedule
#for si in create_schedule_with_vars(rr): print(si)

