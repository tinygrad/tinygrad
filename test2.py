#!/usr/bin/env python3
import sys
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
  x = Tensor.arange(4*2*9*9).cast(dtypes.float32).reshape(4, 2,9,9)
  w = Tensor.arange(4*2*3*3).cast(dtypes.float32).reshape(4, 2, 3, 3)
  x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
  
#out = Tensor.conv2d(x,w)
GlobalCounters.reset()
print("x",x.numpy())
print("w",w.numpy())

#w = Tensor(w_data,  dtype=dtypes.float32).reshape(1,1,3,3)
#w = Tensor.arange(1*1*3*3*3).reshape(1, 1, 3, 3, 3)
out = x.conv2d(w)
print(out.numpy())
print("ops", GlobalCounters.global_ops)
print("mem", GlobalCounters.global_mem)


# [[[[ 24.  27.  30.  33.  36.  39.]
#    [ 48.  51.  54.  57.  60.  63.]
#    [ 72.  75.  78.  81.  84.  87.]
#    [ 96.  99. 102. 105. 108. 111.]
#    [120. 123. 126. 129. 132. 135.]
#    [144. 147. 150. 153. 156. 159.]]]]

# [[[[ 24.  27.  30.  33.  36.  39.]
#    [ 48.  51.  54.  57.  60.  63.]
#    [ 72.  75.  78.  81.  84.  87.]
#    [ 96.  99. 102. 105. 108. 111.]
#    [120. 123. 126. 129. 132. 135.]
#    [144. 147. 150. 153. 156. 159.]]]]


# def pratice_rewrite(ctx: IndexingContext, redu: UOp):
#   if redu.src[0].op is not Ops.MUL: return None
#   mul = redu.src[0]
#   #print("Choosing first")
#   idx = mul.src[0]
#   idx2 = mul.src[1]
#   ky = idx.src[1]
#   kx = idx.src[2]
#   buf = idx.src[0]
#   print(idx2.src[2])
#   #print(f"ky: {ky}, kx: {kx}, buf: {buf}")
#   # The where(invalid) pattern gets optimized in symbolic.py line 477 based on validity analysis
#   cond = kx.eq(UOp.const(dtypes.index, 3))

#   rp1, rp2, rp3 = [buf.index(UOp.const(dtypes.index, i), kx) for i in range(3)]
#   # Guard the first index parameter: when kx==3 use 0, otherwise use invalid (gets optimized)
#   guarded_first_idx = cond.where(UOp.const(dtypes.index, 0), UOp.invalid())
#   return (buf.index(guarded_first_idx, ky) * idx2.replace(src=(idx2.src[0], UOp(Ops.CONST, dtypes.index, arg=0), idx2.src[2])))



# Bt =[                          # (6,6)
#   [4,  0, -5,  0, 1, 0],
#   [0, -4, -4,  1, 1, 0],
#   [0,  4, -4, -1, 1, 0],
#   [0, -2, -1,  2, 1, 0],
#   [0,  2, -1, -2, 1, 0],
#   [0,  4,  0, -5, 0, 1],
# ]
# def pratice_rewrite(ctx: IndexingContext, redu: UOp):
#   # Expect MUL with activation-like INDEX on the left
#   if redu.src[0].op is not Ops.MUL: return None
#   mul = redu.src[0]
#   act_like = mul.src[1]
#   w_like = mul.src[0]
#   if act_like.op is not Ops.INDEX or len(act_like.src) < 3: return None

#   #print(f"act_like: {act_like}, w_like: {w_like}")
#   buf = w_like.src[0]
#   ky  = w_like.src[1]   # output row loop (iy)
#   kx  = act_like.src[2]   # output col loop (ix)
#   #print(f"ky: {ky}, kx: {kx}, buf: {buf}")
#   # print(f"ky: {ky}, kx: {kx}, buf: {buf}")
#   # # --- helpers ---
#   # def consti(i: int) -> UOp: return UOp.const(dtypes.index, i)
#   # def constf(x: float) -> UOp: return UOp.const(dtypes.float, x)

#   # one, zero, neg1 = constf(1.0), constf(0.0), constf(-1.0)

#   # # ***** 1) READ D WITH ONLY CONST INDICES (no SUBSTITUTE, no NOOP) *****
#   # # 3x3 toy -> nine scalar INDEX nodes
#   # D00 = buf.index(consti(0), consti(0))
#   # D01 = buf.index(consti(0), consti(1))
#   # D02 = buf.index(consti(0), consti(2))
#   # D10 = buf.index(consti(1), consti(0))
#   # D11 = buf.index(consti(1), consti(1))
#   # D12 = buf.index(consti(1), consti(2))
#   # D20 = buf.index(consti(2), consti(0))
#   # D21 = buf.index(consti(2), consti(1))
#   # D22 = buf.index(consti(2), consti(2))

#   # # ***** 2) B COEFFICIENTS AS LITERALS (no indexing B) *****
#   # # Your B_data (from the file) for rows is:
#   # #   rows 0,1: [1, 1, -1], row 2: [0,0,0]
#   # # We'll express these per-i (ky) using tiny masks (no tables).
#   # is_i01 = UOp(Ops.CMPLT, dtypes.bool, src=(ky, consti(2)))     # ky in {0,1}
#   # row_mask = UOp(Ops.WHERE, dtypes.float, src=(is_i01, one, zero))
#   # # row coeffs: for i∈{0,1} → [1,1,-1], for i=2 → [0,0,0]
#   # b_ip0 = row_mask * one
#   # b_ip1 = row_mask * one
#   # b_ip2 = row_mask * neg1

#   # # For columns, reuse the same pattern: j∈{0,1} → [1,1,-1], j=2 → [0,0,0]
#   # is_j01 = UOp(Ops.CMPLT, dtypes.bool, src=(kx, consti(2)))     # kx in {0,1}
#   # col_mask = UOp(Ops.WHERE, dtypes.float, src=(is_j01, one, zero))
#   # b_jq0 = col_mask * one
#   # b_jq1 = col_mask * one
#   # b_jq2 = col_mask * neg1

#   # # ***** 3) FORM C_q(i) = Σ_p B[i,p]*D[p,q] *****
#   # C0 = b_ip0*D00 + b_ip1*D10 + b_ip2*D20
#   # C1 = b_ip0*D01 + b_ip1*D11 + b_ip2*D21
#   # C2 = b_ip0*D02 + b_ip1*D12 + b_ip2*D22

#   # # ***** 4) FORM Y(i,j) = Σ_q C_q(i) * B[j,q] *****
#   # out = C0*b_jq0 + C1*b_jq1 + C2*b_jq2

#   return nmode_kron(buf, [ky, kx], Bt, [])

# practice = PatternMatcher([
#  (UPat(Ops.REDUCE, name="redu"), lambda ctx, redu: pratice_rewrite(ctx, redu))
#  ])