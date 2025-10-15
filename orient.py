# test_wino_toggle.py
import math
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, GlobalCounters

def run_conv(label, x, w, *, WINO=0, WINO_OLD=0, padding=0, stride=1, dilation=1):
  # make sure only one is set at a time
  with Context(WINO=int(bool(WINO)), WINO_OLD=int(bool(WINO_OLD))):
    GlobalCounters.reset()
    y = x.conv2d(w, padding=padding, stride=stride, dilation=dilation)
    ynp = y.numpy()
    ops, mem = GlobalCounters.global_ops, GlobalCounters.global_mem
    print(f"\n[{label}] out:\n{ynp}")
    print(f"[{label}] ops={ops:,}  mem={mem:,}")
    return y

def max_abs(a, b):
  return float((a - b).abs().max().numpy())

if __name__ == "__main__":
  # 1×1×6×6 input with easy-to-see values
  X = Tensor([
      [ [ [ 1,  2,  3,  4,  5,  6],
          [ 7,  8,  9, 10, 11, 12],
          [13, 14, 15, 16, 17, 18],
          [19, 20, 21, 22, 23, 24],
          [25, 26, 27, 28, 29, 30],
          [31, 32, 33, 34, 35, 36], ] ]],
      dtype=dtypes.float32)

  # a few kernels to catch orientation mistakes
  K_delta = Tensor([[[[0,0,0],[0,1,0],[0,0,0]]]], dtype=dtypes.float32)     # should reproduce top-left valid window
  K_ones  = Tensor([[[[1,1,1],[1,1,1],[1,1,1]]]], dtype=dtypes.float32)     # box filter
  K_asym  = Tensor([[[[ 1,  1, -1],
                      [ 1,  1, -1],
                      [ 1,  1, -1]]]], dtype=dtypes.float32)                # asymmetric to expose transposes

  def test_kernel(name, W):
    print("\n" + "="*80)
    print(f"Kernel: {name}")

    # Baseline path: force non-Winograd
   # y_ref = run_conv("BASELINE", X, W, WINO=0, WINO_OLD=0)

    # Old Winograd
    #y_old = run_conv("WINO_OLD", X, W, WINO=0, WINO_OLD=1)

    # New Winograd
    y_new = run_conv("WINO_NEW", X, W, WINO=1, WINO_OLD=0)

    # Compare numerically
    # e_old = max_abs(y_old, y_ref)
    # e_new = max_abs(y_new, y_ref)
    #e_new_vs_old = max_abs(y_new, y_old)

    # print(f"\nmax|WINO_OLD - BASELINE| = {e_old:.6g}")
    # print(f"max|WINO_NEW - BASELINE| = {e_new:.6g}")
    # print(f"max|WINO_NEW - WINO_OLD| = {e_new_vs_old:.6g}")

  # Valid 3x3 conv on a 6x6 tile -> 4x4 output; no padding/stride
  test_kernel("delta", K_delta)
  # test_kernel("all_ones", K_ones)
  # test_kernel("asymmetric", K_asym)