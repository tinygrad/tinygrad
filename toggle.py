# test_wino_toggle.py
import math
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, GlobalCounters

def run_conv(label, x, w, *, WINO=0, WINO_OLD=0, padding=0, stride=1, dilation=1):
  with Context(WINO=int(bool(WINO)), WINO_OLD=int(bool(WINO_OLD))):
    GlobalCounters.reset()
    y = x.conv2d(w, padding=padding, stride=stride, dilation=dilation)
    ynp = y.numpy()
    ops, mem = GlobalCounters.global_ops, GlobalCounters.global_mem
    print(f"\n[{label}] out shape={tuple(ynp.shape)}:\n{ynp}")
    print(f"[{label}] ops={ops:,}  mem={mem:,}")
    return y

def max_abs(a, b): return float((a - b).abs().max().numpy())

def run_case(name, X, kernels, *, padding=0, stride=1, dilation=1):
  print("\n" + "="*80)
  print(f"CASE: {name}  X.shape={tuple(X.shape)}  padding={padding} stride={stride} dilation={dilation}")

  for kname, W in kernels:
    print("\n" + "-"*40)
    print(f"Kernel: {kname}")

    y_ref = run_conv("BASELINE", X, W, WINO=0, WINO_OLD=0, padding=padding, stride=stride, dilation=dilation)
    y_old = run_conv("WINO_OLD", X, W, WINO=0, WINO_OLD=1, padding=padding, stride=stride, dilation=dilation)
    y_new = run_conv("WINO_NEW", X, W, WINO=1, WINO_OLD=0, padding=padding, stride=stride, dilation=dilation)

    e_old = max_abs(y_old, y_ref)
    e_new = max_abs(y_new, y_ref)
    e_new_vs_old = max_abs(y_new, y_old)
    print(f"\nmax|WINO_OLD - BASELINE| = {e_old:.6g}")
    print(f"max|WINO_NEW - BASELINE| = {e_new:.6g}")
    print(f"max|WINO_NEW - WINO_OLD| = {e_new_vs_old:.6g}")

def grid(h, w, start=1):
  # deterministic ramp to eyeball correctness
  vals = [[start + r*w + c for c in range(w)] for r in range(h)]
  return Tensor([[vals]], dtype=dtypes.float32)  # (1,1,H,W)

def make_kernels_1x1():
  K_delta = Tensor([[[[0,0,0],[0,1,0],[0,0,0]]]], dtype=dtypes.float32)
  K_ones  = Tensor([[[[1,1,1],[1,1,1],[1,1,1]]]], dtype=dtypes.float32)
  K_asym  = Tensor([[[[ 1,  1, -1],
                      [ 1,  1, -1],
                      [ 1,  1, -1]]]], dtype=dtypes.float32)
  return [("delta", K_delta), ("all_ones", K_ones), ("asymmetric", K_asym)]

def make_kernels_cin2_cout3():
  # Cin=2 -> Cout=3, each Cout uses both input channels (good Cin reduction test)
  # weights are simple patterns so mistakes are obvious
  import numpy as np
  W = np.zeros((3,2,3,3), dtype="float32")
  # out0: channel 0 = delta, channel 1 = ones
  W[0,0,1,1] = 1.0     # delta on in0
  W[0,1,:,:] = 1.0     # ones on in1
  # out1: channel 0 = asymmetric, channel 1 = delta
  W[1,0,:,:] = [[ 1,  1, -1],
                [ 1,  1, -1],
                [ 1,  1, -1]]
  W[1,1,1,1] = 1.0
  # out2: channel 0 = ones, channel 1 = asymmetric
  W[2,0,:,:] = 1.0
  W[2,1,:,:] = [[ 1,  1, -1],
                [ 1,  1, -1],
                [ 1,  1, -1]]
  return [("mixed_Cout3_Cin2", Tensor(W))]

def make_input_cin2(h, w):
  # two channels: channel0 = ramp, channel1 = ramp * 10 (to spot mixing)
  import numpy as np
  ch0 = np.array([[r*w+c+1 for c in range(w)] for r in range(h)], dtype="float32")
  ch1 = 10.0 * ch0
  x = np.stack([ch0, ch1], axis=0)[None, ...]   # (1,2,H,W)
  return Tensor(x, dtype=dtypes.float32)

if __name__ == "__main__":
  # 1) VALID on odd sizes (non-4n outputs): 7x5 -> (7-3+1)x(5-3+1) = 5x3
  X1 = grid(7, 5)
  run_case("VALID 7x5", X1, make_kernels_1x1(), padding=0, stride=1, dilation=1)

  # 2) SAME padding on non-4n sizes: 8x7 with padding=1 -> output 8x7
  X2 = grid(8, 7)
  run_case("SAME 8x7 (pad=1)", X2, make_kernels_1x1(), padding=1, stride=1, dilation=1)

  # 3) STRIDE 2 on odd width/height: 9x8, stride=2 (output dims not multiples of 4)
  X3 = grid(9, 8)
  run_case("STRIDE2 9x8", X3, make_kernels_1x1(), padding=0, stride=2, dilation=1)

  # 4) DILATION 2 on awkward size: 10x9 with pad=2 keeps more borders; still non-4n
  X4 = grid(10, 9)
  run_case("DILATION2 + PAD2 10x9", X4, make_kernels_1x1(), padding=2, stride=1, dilation=2)

  # 5) Multi-channel / multi-output to stress Cin reduction & mixing
  X5 = make_input_cin2(7, 6)  # (1,2,7,6)
  run_case("Cin=2, Cout=3 (7x6)", X5, make_kernels_cin2_cout3(), padding=0, stride=1, dilation=1)

  # 6) Batch>1 to ensure tiling math doesn’t assume N=1
  X6a = grid(6, 6)            # (1,1,6,6)
  X6b = grid(5, 7, start=100) # (1,1,5,7)
  X6  = Tensor.cat((X6a, X6b), dim=0)  # (2,1,H?,W?) -> cat requires same H,W; so make both 6x6:
  X6  = Tensor.cat((grid(6,6), grid(6,6, start=100)), dim=0)  # (2,1,6,6)
  run_case("Batch=2", X6, make_kernels_1x1(), padding=0, stride=1, dilation=1)
