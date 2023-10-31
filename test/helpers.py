from tinygrad.ops import LazyOp, LoadOps
from tinygrad.nn.state import get_parameters

# for speed
def derandomize(x):
  if isinstance(x, LazyOp):
    new_op = LoadOps.EMPTY if x.op == LoadOps.RAND else x.op
    return LazyOp(new_op, tuple([derandomize(s) for s in x.src]), x.arg)
  x.op = derandomize(x.op)
  return x

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = derandomize(p.lazydata)
    p.realize()
