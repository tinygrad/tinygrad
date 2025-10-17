#!/usr/bin/env python3
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
# with Context(WINO=1):
#   t = time.time()
#   GlobalCounters.reset()
#   out = x.conv2d(w)
#   t2 = time.time()
#   tnew = (t2 - t)*1000
#   new_result = out.numpy()
#   print(new_result)
#   new_ops = deepcopy(GlobalCounters.global_ops)
#   print("ops", GlobalCounters.global_ops)
#   print("mem", GlobalCounters.global_mem)
  

# with Context(WINO_OLD=1):
#   t = time.time()
#   GlobalCounters.reset()
#   out = x.conv2d(w)
#   t2 = time.time()
#   told = (t2 - t)*1000
#   old_result = out.numpy()
#   old_ops = deepcopy(GlobalCounters.global_ops)
#   print("ops", GlobalCounters.global_ops)
#   print("mem", GlobalCounters.global_mem)
#   print("ops_ratio", new_ops/old_ops)
#   print("timedifference (new - old) ms", tnew-told)

# with Context(WINO=0):
#   t = time.time()
#   GlobalCounters.reset()
#   out = x.conv2d(w)
#   t2 = time.time()
#   tnormal = (t2 - t)*1000
#   print(out.numpy())
#   conv_ops = deepcopy(GlobalCounters.global_ops)
#   #assert_allclose(old_result, out.numpy(), atol=1e-5)
#   assert_allclose(new_result, out.numpy(), atol=1e-1)
#   print("normal conv ops", GlobalCounters.global_ops)
#   print("normal conv mem", GlobalCounters.global_mem)
#   print("ops-new to conv_ops ratio", new_ops/conv_ops)
#   print("ops-old to conv_ops ratio", old_ops/conv_ops)
#   print("timedifference (new - normal) ms", tnew-tnormal)
#   print("timedifference (old - normal) ms", told-tnormal)
#   #print("time_new", tnew, "ms", "time_old", told, "ms")
# #print("w",w.numpy())

# #w = Tensor(w_data,  dtype=dtypes.float32).reshape(1,1,3,3)
# #w = Tensor.arange(1*1*3*3*3).reshape(1, 1, 3, 3, 3)
# # t = time.time()
# # #print(t)
# # out = x.conv2d(w)
# # print(out.numpy())
# # t2 = time.time()
# # print((t2 - t)*1000, "ms")
# # #print(out.numpy())
# # print("ops", GlobalCounters.global_ops)
# # print("mem", GlobalCounters.global_mem)


# # [[[[ 24.  27.  30.  33.  36.  39.]
# #    [ 48.  51.  54.  57.  60.  63.]
# #    [ 72.  75.  78.  81.  84.  87.]
# #    [ 96.  99. 102. 105. 108. 111.]
# #    [120. 123. 126. 129. 132. 135.]
# #    [144. 147. 150. 153. 156. 159.]]]]

# # [[[[ 24.  27.  30.  33.  36.  39.]
# #    [ 48.  51.  54.  57.  60.  63.]
# #    [ 72.  75.  78.  81.  84.  87.]
# #    [ 96.  99. 102. 105. 108. 111.]
# #    [120. 123. 126. 129. 132. 135.]
# #    [144. 147. 150. 153. 156. 159.]]]]


# # def pratice_rewrite(ctx: IndexingContext, redu: UOp):
# #   if redu.src[0].op is not Ops.MUL: return None
# #   mul = redu.src[0]
# #   #print("Choosing first")
# #   idx = mul.src[0]
# #   idx2 = mul.src[1]
# #   ky = idx.src[1]
# #   kx = idx.src[2]
# #   buf = idx.src[0]
# #   print(idx2.src[2])
# #   #print(f"ky: {ky}, kx: {kx}, buf: {buf}")
# #   # The where(invalid) pattern gets optimized in symbolic.py line 477 based on validity analysis
# #   cond = kx.eq(UOp.const(dtypes.index, 3))

# #   rp1, rp2, rp3 = [buf.index(UOp.const(dtypes.index, i), kx) for i in range(3)]
# #   # Guard the first index parameter: when kx==3 use 0, otherwise use invalid (gets optimized)
# #   guarded_first_idx = cond.where(UOp.const(dtypes.index, 0), UOp.invalid())
# #   return (buf.index(guarded_first_idx, ky) * idx2.replace(src=(idx2.src[0], UOp(Ops.CONST, dtypes.index, arg=0), idx2.src[2])))



# # Bt =[                          # (6,6)
# #   [4,  0, -5,  0, 1, 0],
# #   [0, -4, -4,  1, 1, 0],
# #   [0,  4, -4, -1, 1, 0],
# #   [0, -2, -1,  2, 1, 0],
# #   [0,  2, -1, -2, 1, 0],
# #   [0,  4,  0, -5, 0, 1],
# # ]
# # def pratice_rewrite(ctx: IndexingContext, redu: UOp):
# #   # Expect MUL with activation-like INDEX on the left
# #   if redu.src[0].op is not Ops.MUL: return None
# #   mul = redu.src[0]
# #   act_like = mul.src[1]
# #   w_like = mul.src[0]
# #   if act_like.op is not Ops.INDEX or len(act_like.src) < 3: return None

# #   #print(f"act_like: {act_like}, w_like: {w_like}")
# #   buf = w_like.src[0]
# #   ky  = w_like.src[1]   # output row loop (iy)
# #   kx  = act_like.src[2]   # output col loop (ix)
# #   #print(f"ky: {ky}, kx: {kx}, buf: {buf}")
# #   # print(f"ky: {ky}, kx: {kx}, buf: {buf}")
# #   # # --- helpers ---
# #   # def consti(i: int) -> UOp: return UOp.const(dtypes.index, i)
# #   # def constf(x: float) -> UOp: return UOp.const(dtypes.float, x)

# #   # one, zero, neg1 = constf(1.0), constf(0.0), constf(-1.0)

# #   # # ***** 1) READ D WITH ONLY CONST INDICES (no SUBSTITUTE, no NOOP) *****
# #   # # 3x3 toy -> nine scalar INDEX nodes
# #   # D00 = buf.index(consti(0), consti(0))
# #   # D01 = buf.index(consti(0), consti(1))
# #   # D02 = buf.index(consti(0), consti(2))
# #   # D10 = buf.index(consti(1), consti(0))
# #   # D11 = buf.index(consti(1), consti(1))
# #   # D12 = buf.index(consti(1), consti(2))
# #   # D20 = buf.index(consti(2), consti(0))
# #   # D21 = buf.index(consti(2), consti(1))
# #   # D22 = buf.index(consti(2), consti(2))

# #   # # ***** 2) B COEFFICIENTS AS LITERALS (no indexing B) *****
# #   # # Your B_data (from the file) for rows is:
# #   # #   rows 0,1: [1, 1, -1], row 2: [0,0,0]
# #   # # We'll express these per-i (ky) using tiny masks (no tables).
# #   # is_i01 = UOp(Ops.CMPLT, dtypes.bool, src=(ky, consti(2)))     # ky in {0,1}
# #   # row_mask = UOp(Ops.WHERE, dtypes.float, src=(is_i01, one, zero))
# #   # # row coeffs: for i∈{0,1} → [1,1,-1], for i=2 → [0,0,0]
# #   # b_ip0 = row_mask * one
# #   # b_ip1 = row_mask * one
# #   # b_ip2 = row_mask * neg1

# #   # # For columns, reuse the same pattern: j∈{0,1} → [1,1,-1], j=2 → [0,0,0]
# #   # is_j01 = UOp(Ops.CMPLT, dtypes.bool, src=(kx, consti(2)))     # kx in {0,1}
# #   # col_mask = UOp(Ops.WHERE, dtypes.float, src=(is_j01, one, zero))
# #   # b_jq0 = col_mask * one
# #   # b_jq1 = col_mask * one
# #   # b_jq2 = col_mask * neg1

# #   # # ***** 3) FORM C_q(i) = Σ_p B[i,p]*D[p,q] *****
# #   # C0 = b_ip0*D00 + b_ip1*D10 + b_ip2*D20
# #   # C1 = b_ip0*D01 + b_ip1*D11 + b_ip2*D21
# #   # C2 = b_ip0*D02 + b_ip1*D12 + b_ip2*D22

# #   # # ***** 4) FORM Y(i,j) = Σ_q C_q(i) * B[j,q] *****
# #   # out = C0*b_jq0 + C1*b_jq1 + C2*b_jq2

# #   return nmode_kron(buf, [ky, kx], Bt, [])

# # practice = PatternMatcher([
# #  (UPat(Ops.REDUCE, name="redu"), lambda ctx, redu: pratice_rewrite(ctx, redu))
# #  ])

# #!/usr/bin/env python3
# from tinygrad import Tensor

# # Create input tensor (batch_size=1, channels=3, height=8, width=8)
# input_tensor = Tensor.randn(1, 3, 8, 8)
# print("Input tensor shape:", input_tensor.shape)

# # Create weight tensor for conv2d (out_channels=16, in_channels=3, kernel_height=3, kernel_width=3)
# weight_tensor = Tensor.randn(16, 3, 3, 3)
# print("Weight tensor shape:", weight_tensor.shape)

# # Perform conv2d with kernel size 3x3, stride 1, dilation 1
# output = input_tensor.conv2d(weight_tensor, stride=1, dilation=1)
# print("Output tensor shape:", output.shape)
# print("Output tensor:")
# print(output.numpy())


# # from tinygrad import Tensor
# # X = Tensor.rand(1, 2, 4, 4).realize()        # N=1, Cin=2, H=W=6 (enables a 4x4 Winograd tile)
# # W = Tensor.rand(2, 2, 3, 3).realize()        # Cout=2, Cin=2, KH=KW=3
# # Y = X.conv2d(W)                              
# # print("X,W,Y shapes:", X.shape, W.shape, Y.shape)
# # print(Y.numpy())
# # #print(out.numpy())

# from tinygrad import Tensor
# import numpy as np

# # Shapes
# N, Cin, Cout, H, W, KH, KW = 1, 2, 2, 4, 4, 3, 3

# # X: channel 0 = 1's, channel 1 = 10's
# X_np = np.zeros((N, Cin, H, W), dtype=np.float32)
# X_np[:, 0] = 1.0
# X_np[:, 1] = 10.0

# # W: 3x3 all-ones per (cout, cin) scaled by constants
# scales = np.array([[1.0, 2.0],
#                    [4.0, 8.0]], dtype=np.float32)  # shape (Cout, Cin)
# W_np = np.ones((Cout, Cin, KH, KW), dtype=np.float32)
# for co in range(Cout):
#   for ci in range(Cin):
#     W_np[co, ci] *= scales[co, ci]

# # Tinygrad tensors
# X = Tensor(X_np).realize()
# W = Tensor(W_np).realize()

# # Conv (valid, stride=1)
# Y = X.conv2d(W)
# print("X,W,Y shapes:", X.shape, W.shape, Y.shape)

# # Expected numbers (every position identical in this setup)
# s = 9.0  # sum of a 3x3 ones patch
# exp0 = (1.0*1.0 + 2.0*10.0) * s   # = 189
# exp1 = (4.0*1.0 + 8.0*10.0) * s   # = 756
# print("Expected per-pixel Y[0]=189, Y[1]=756")
# print("X =\n", X.numpy())
# print("W =\n", W.numpy())
# np.set_printoptions(precision=2, suppress=True)
# print("\nY =\n", Y.numpy())

# #WINO, WINO_OLD, CAPTURING, TRACEMETA = ContextVar("WINO", 0), ContextVar("WINO_OLD", 0), ContextVar("CAPTURING", 1), ContextVar("TRACEMETA", 1)