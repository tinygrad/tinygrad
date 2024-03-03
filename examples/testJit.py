from tinygrad import Tensor, TinyJit, nn

from typing import Tuple
import torch

t1 = torch.arange(16).reshape(4,4).float()
t2 = Tensor.arange(16).reshape(4,4).float()

o1 = t1.softmax(dim=-1)#.argmax()
o2 = t2.softmax(axis=-1)#.argmax()

# o1 = t1.argmax()
# o2 = t2.argmax()
print(o1.shape, o2.shape)
print(o1, o2.numpy())

# class ActorCritic:
#   def __init__(self, in_features, out_features, hidden_state=32):
#     self.l1 = nn.Linear(in_features, hidden_state)
#     self.l2 = nn.Linear(hidden_state, out_features)

#     self.c1 = nn.Linear(in_features, hidden_state)
#     self.c2 = nn.Linear(hidden_state, 1)

#   def __call__(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
#     x = self.l1(obs).tanh()
#     act = self.l2(x).log_softmax()
#     x = self.c1(obs).relu()
#     return act, self.c2(x)
  

# @TinyJit
# def foo(x:Tensor) -> Tensor:
#   return x.exp().multinomial()
#   x.abs()
#   return (x*x*x).multinomial()


# # print(foo.cnt)
# for i in range(10):
#   temp = foo(Tensor([i,i+1,i+2, i]))
#   print( i, temp.numpy(),temp)
# temp = foo(Tensor(1.0))
# print(temp, foo.cnt)
# temp = foo(Tensor(1.0))
# print(temp, foo.cnt)
# temp = foo(Tensor(1.0))
# print(temp, foo.cnt)
# temp = foo(Tensor(1.0))
# print(temp, foo.cnt)


# print(foo.cnt)