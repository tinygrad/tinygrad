from copy import deepcopy

from numpy.testing import assert_allclose
from tinygrad import Context
import sys
import time
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters

# ---- 8x8 input, written out explicitly ----
inp_data = [
  [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
  [  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
  [ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
  [ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
  [ 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
  [ 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
  [ 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
  [ 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
  [ 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88],
  [ 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
  [ 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
  [ 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
  [ 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
  [ 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
  [ 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136],
  [ 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144],
]

# inp_data = [
#   [  1,  2,  3,  4],
#   [  9, 10, 11, 12],
#   [ 17, 18, 19, 20],
#   [ 25, 26, 27, 28],
# ]

w_8x8_data = [  # 8x8 example weight tensor
  [ 1,  2,  3,  4,  5,  6,  7,  8],
  [ 9, 10, 11, 12, 13, 14, 15, 16],
  [17, 18, 19, 20, 21, 22, 23, 24],
  [25, 26, 27, 28, 29, 30, 31, 32],
  [33, 34, 35, 36, 37, 38, 39, 40],
  [41, 42, 43, 44, 45, 46, 47, 48],
  [49, 50, 51, 52, 53, 54, 55, 56],
  [57, 58, 59, 60, 61, 62, 63, 64],
]

# ---- 3x3 weight, written out explicitly (one in/out channel) ----
# example: vertical edge detector (like a simple Sobel-ish kernel)
w_data = [
  [ 1,  1, -1],
  [ 1,  1, -1],
  [ 1,  1, -1],
]

# tinygrad expects NCHW for inputs and OIHW for weights
# cast to float32 to keep backends happy
if "--arange" in sys.argv:
  x = Tensor.arange(1*1*4*4).cast(dtypes.float32).reshape(1, 1, 4,4)
  w = Tensor.arange(1*1*3*3).cast(dtypes.float32).reshape(1, 1, 3, 3)
else:
  x = Tensor(w_8x8_data, dtype=dtypes.float32).reshape(1,1,8,8)
  w = Tensor(w_data,  dtype=dtypes.float32).reshape(1,1,3,3)

if "--arange3" in sys.argv:
  x = Tensor.arange(1*1*256*256).cast(dtypes.float32).reshape(1, 1,256,256)
  w = Tensor.arange(1*1*3*3).cast(dtypes.float32).reshape(1, 1, 3, 3)
  #x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
  
out = Tensor.conv2d(x,w)
print(out.numpy())