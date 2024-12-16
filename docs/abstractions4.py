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

print("\n**** schedule ****")
# in the schedule, we have two COPY and one kernel
for si in c.schedule(): print(si)

# open questions:

# 1. What Tensors should get Buffers?
#    only the Tensors specified in the realize gain buffers
#    make a best effort approach for others
#    give buffers to all Tensors in scope (can break a lot of folding)

# we can do a substitute to add TENSOR UOps to all UOps we want to potentially get a Buffer
# however, you need a concept of "strong" and "weak" Tensors if you want to fold.
# "strong" Tensors are Tensors that are explicitly specified in the realize
# "weak" Tensors are Tensors that are in scope but not in the realize

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

# 4. How do we represent a CONST Tensor? Currently it uses a fake BUFFER
#      UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
#        UOp(Ops.BUFFER, dtypes.float.ptr(), arg=(-1, 'METAL', 1), src=()),
#        UOp(Ops.CONST, dtypes.float, arg=2.0, src=()),)),)),)),))
