#!/usr/bin/env python3
from tinygrad import Tensor
from tinygrad.dtype import dtypes

# ---------- explicit 8x8 input ----------
inp_data = [
  [  1,  2,  3,  4,  5,  6,  7,  8],
  [  9, 10, 11, 12, 13, 14, 15, 16],
  [ 17, 18, 19, 20, 21, 22, 23, 24],
  [ 25, 26, 27, 28, 29, 30, 31, 32],
  [ 33, 34, 35, 36, 37, 38, 39, 40],
  [ 41, 42, 43, 44, 45, 46, 47, 48],
  [ 49, 50, 51, 52, 53, 54, 55, 56],
  [ 57, 58, 59, 60, 61, 62, 63, 64],
]

# ---------- explicit 3x3 weight ----------
w_data = [
  [ 1,  0, -1],
  [ 1,  0, -1],
  [ 1,  0, -1],
]

# ---------- Winograd F(4,3): alpha = 6 ----------
G = Tensor([
  [ 1/4,   0.0,   0.0],
  [-1/6, -1/6, -1/6],
  [-1/6,  1/6, -1/6],
  [ 1/24, 1/12, 1/6],
  [ 1/24,-1/12, 1/6],
  [ 0.0,   0.0,  1.0],
], dtype=dtypes.float32)               # (6,3)

Bt = Tensor([                          # (6,6)
  [4,  0, -5,  0, 1, 0],
  [0, -4, -4,  1, 1, 0],
  [0,  4, -4, -1, 1, 0],
  [0, -2, -1,  2, 1, 0],
  [0,  2, -1, -2, 1, 0],
  [0,  4,  0, -5, 0, 1],
], dtype=dtypes.float32)

At = Tensor([                          # (4,6)
  [1, 1,  1,  1,  1, 0],
  [0, 1, -1,  2, -2, 0],
  [0, 1,  1,  4,  4, 0],
  [0, 1, -1,  8, -8, 1],
], dtype=dtypes.float32)
A  = At.transpose()                    # (6,4)
B  = Bt.transpose()                    # (6,6)

def mm(a,b): return a @ b

def transform_input_tile(d6x6):       # Bᵗ d B
  return mm(mm(Bt, d6x6), B)

def transform_weight(g3x3):           # G g Gᵗ
  return mm(mm(G, g3x3), G.transpose())

def inverse_output_tile(m6x6):        # ✅ At M A (not A M)
  return mm(mm(At, m6x6), A)

# tensors
x = Tensor(inp_data, dtype=dtypes.float32)     # (8,8)
g = Tensor(w_data,  dtype=dtypes.float32)      # (3,3)
GgGt = transform_weight(g)                     # (6,6)

# valid conv output is 6x6
out = [[0.0]*6 for _ in range(6)]
m, r, alpha = 4, 3, 6
tiles_y = (6 + m - 1) // m     # 2
tiles_x = (6 + m - 1) // m     # 2

for ty in range(tiles_y):
  for tx in range(tiles_x):
    oy0, ox0 = ty*m, tx*m
    iy0, ix0 = oy0,  ox0               # stride = m for F(4,3)
    d = x[iy0:iy0+alpha, ix0:ix0+alpha]        # (6,6)
    d_hat = transform_input_tile(d)            # (6,6)
    m_hat = d_hat * GgGt                       # Hadamard (6,6)
    y_tile = inverse_output_tile(m_hat)        # (4,4)
    h = min(m, 6 - oy0); w = min(m, 6 - ox0)   # stitch, crop edges
    yt = y_tile[:h, :w].numpy()
    for i in range(h):
      out[oy0+i][ox0:ox0+w] = yt[i].tolist()

Y_winograd = Tensor(out, dtype=dtypes.float32)

# sanity: direct conv for comparison
x4d = Tensor(inp_data, dtype=dtypes.float32).reshape(1,1,8,8)
w4d = Tensor(w_data,  dtype=dtypes.float32).reshape(1,1,3,3)
Y_direct = x4d.conv2d(w4d, stride=1, padding=0).reshape(6,6)

print("Winograd F(4,3) 6x6:\n", Y_winograd.numpy())
print("Direct conv 6x6:\n",     Y_direct.numpy())
print("Max abs diff:", float((Y_winograd - Y_direct).abs().max().numpy()))