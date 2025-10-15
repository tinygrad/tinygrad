import sys
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters

w_data = [
  [ 1,  1, -1],
  [ 1,  1, -1],
  [ 1,  1, -1],
]
w = Tensor(w_data, dtype=dtypes.float32).reshape(1,1,3,3)

B_data = [
  [ 1,  1, -1],
  [ 1,  1, -1],
  [0, 0, 0]
]

t = Tensor([[ 1,  2,  3,  4,  5,  6],
              [ 7,  8,  9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18],
              [19, 20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35, 36]], dtype=dtypes.float32)
Bt = Tensor([                          # (6,6)
  [4,  0, -5,  0, 1, 0],
  [0, -4, -4,  1, 1, 0],
  [0,  4, -4, -1, 1, 0],
  [0, -2, -1,  2, 1, 0],
  [0,  2, -1, -2, 1, 0],
  [0,  4,  0, -5, 0, 1],
], dtype=dtypes.float32)

B = Tensor(B_data, dtype=dtypes.float32).reshape(1,1,3,3)

out = Bt@t@Bt.transpose()
out = t@Bt

GlobalCounters.reset()
print(out.numpy())
# out2 = Bt.transpose()@t@Bt
# print("Output Tensor - matmul:\n", out2.numpy())
print("ops", GlobalCounters.global_ops)
print("mem", GlobalCounters.global_mem)

#print(out.numpy())