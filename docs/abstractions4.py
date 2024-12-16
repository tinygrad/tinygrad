# UOps are used everywhere, but they aren't always the same UOps
from tinygrad import Tensor

print("**** some math ****")
a = Tensor([1.,2,3])
b = Tensor([4.,5,6])
c: Tensor = (a+b)*2
print(c.lazydata)

print("\n**** gradient ****")
da = c.sum().gradient(a)[0]
print(da.lazydata)

#print("\n**** schedule ****")
# in the schedule, we have two COPY and one kernel
#for si in c.schedule(): print(si)

from examples.beautiful_mnist import Model
m = Model()
from tinygrad.nn.state import get_parameters
for p in get_parameters(m): p.realize()
c = m(Tensor.rand(1, 1, 28, 28))

from tinygrad.engine.schedule import remove_movement_ops
from tinygrad.ops import merge_views
from tinygrad.ops import graph_rewrite, PatternMatcher, UOp, track_rewrites, UPat, Ops

view_before_const = PatternMatcher([
  (UPat(Ops.VIEW, src=(UPat(Ops.BUFFER, name="b"), UPat(Ops.CONST, name="c")), name="v"),
    lambda b,c,v: UOp(Ops.CONST, c.dtype, (UOp(Ops.VIEW, v.dtype, (UOp(Ops.DEVICE, arg=b.arg[1]),), v.arg),), c.arg)),
])

@track_rewrites(named=True)
def test_rewrite(x): graph_rewrite(UOp.sink(x), PatternMatcher([]))
g = graph_rewrite(c.lazydata, remove_movement_ops+merge_views)
#g = graph_rewrite(g, view_before_const)
test_rewrite(g)


#c.realize()
exit(0)

# open questions:

# 1. What Tensors should get Buffers?
#    only the Tensors specified in the realize gain buffers
#    make a best effort approach for others
#    give buffers to all Tensors in scope (can break a lot of folding)

# we can do a substitute to add TENSOR UOps to all UOps we want to potentially get a Buffer
# however, you need a concept of "strong" and "weak" Tensors if you want to fold.
# "strong" Tensors are Tensors that are explicitly specified in the realize
# "weak" Tensors are Tensors that are in scope but not in the realize

# without weak tensors, it becomes easy to recompute the whole model. example of bad case:
def model(x): return x.sum()   # much bigger in practice
x = Tensor([1.,2.,3])
loss = model(x)
loss.backward()
x.grad.realize()   # optim.step
# so loss (Tensor + UOp) is still in scope here. it should not require going back to the source
print(loss.item())  # will this rerun the forward pass? will it even rerun with weak tensors?
# with UOp mutability and weak tensors, this will become either realized or a short graph. i think that's a requirement

# 2. What should the scope policy of Buffers be?
#    Kept around only if the Tensor is around (can make things unrealizable)
#    Kept around if any unrealized child Tensor is around
#    Kept around if any child Tensor (even realized) is around (stuff isn't freed)

# it's acceptable for `backward` to only work if the gradient path hasn't been realized
# that choice allows `no_grad` to be meaningless. if backward worked on realized graphs, you wouldn't be able to free memory

# "Kept around if any unrealized child Tensor is around" is probably the only choice that makes sense

# 3. Is UOp mutability okay? Are there any alternatives that work?

# without UOp mutability, there's no clean solution to this
a = Tensor([1., 2, 3])
b = Tensor([4., 5, 6])
c = a+b
d = c*2
c.realize()
# d has a UOp graph keeping a and b alive, but not if the c UOp is mutable
# Q: is this case uncommon enough that we can ignore it?
# alternatively, we could do rewrites of all the in-scope Tensors to add the new buffers

# 4. How do we represent a CONST Tensor? Currently it uses a fake BUFFER
#      UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
#        UOp(Ops.BUFFER, dtypes.float.ptr(), arg=(-1, 'METAL', 1), src=()),
#        UOp(Ops.CONST, dtypes.float, arg=2.0, src=()),)),)),)),))

# **** scheduler flow ****
# 0. As input, you have a List of UOps
# 1. The realized UOps have to become buffers. We can either make the buffer the base or the UOp itself. Probably the base.
# 2. Many other Tensors and intermediates may become linked to Buffers. During any rewrites, you have to track this.
