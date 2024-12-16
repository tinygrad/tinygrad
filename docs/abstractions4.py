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

# 2. What should the scope policy of Buffers be?
#    Kept around only if the Tensor is around (can make things unrealizable)
#    Kept around if any unrealized child Tensor is around
#    Kept around if any child Tensor (even realized) is around (stuff isn't freed)

# 3. Is UOp mutability okay?
