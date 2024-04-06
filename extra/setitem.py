from tinygrad import Tensor, TinyJit, Variable

# # this works
# t = Tensor.zeros(6, 6).contiguous().realize()
# t[2:4, 3:5] = Tensor.ones(2, 2)
# print(t.numpy())

# # this is assign in jit
# @TinyJit
# def f(t, a):
#   t.assign(a + 1).realize()

# for i in range(4):
#   t = Tensor.zeros(6, 6).contiguous().realize()
#   a = Tensor.full((6, 6), fill_value=float(i)).contiguous()
#   f(t, a)
#   print(t.numpy())

# # this is partial assign in jit
# @TinyJit
# def f(t, a):
#   t[2:4, 3:5] = a
#   t.realize()

# for i in range(4):
#   t = Tensor.zeros(6, 6).contiguous().realize()
#   a = Tensor.full((2, 2), fill_value=i+3.0).contiguous()
#   f(t, a)
#   print(t.numpy())

# # variable from st, not from any of the input buffers
# t = Tensor.zeros(6, 6).contiguous().realize()
# v = Variable("v", 1, 6).bind(3)
# t.shrink(((v,v+1),(0, 6))).assign(Tensor.ones(6).reshape(1, 6)).realize()
# print(t.numpy())

@TinyJit
def f(t, v):
  t.shrink(((v,v+1),(0, 6))).assign(Tensor.rand(6).reshape(1, 6)).realize()

t = Tensor.zeros(6, 6).contiguous().realize()
for i in range(4):
  v = Variable("v", 0, 6).bind(i)
  f(t, v)
  print(t.numpy())