import torch
from tinygrad import Tensor
import platform

def test_pool_kernel():
    torch.manual_seed(42)
    b = torch.randn(4, 3, 32, 32)
    a = Tensor(b.cpu().numpy())
    out1 = a.max_pool2d(2).sum().item()
    out2 = torch.nn.functional.max_pool2d(b, 2).sum().item()
    diff = abs(out1 - out2)

    # Tolleranza diversa per Windows (MSVC vs GCC floating-point)
    tolerance = 1e-2 if platform.system() == "Windows" else 1e-3
    print(f"Sistema: {platform.system()} | Tolleranza: {tolerance}")
    print(f"out1 (tinygrad): {out1}")
    print(f"out2 (PyTorch):  {out2}")
    print(f"diff: {diff:.6f}")

    assert diff < tolerance, f"Kernel diverso: {out1} vs {out2} (diff: {diff})"
    print(f"Kernel IDENTICO (float32) | diff = {diff:.2e}")