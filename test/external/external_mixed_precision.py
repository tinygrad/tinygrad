from tinygrad import Tensor, Device, dtypes
import torch, unittest
import torch.nn.functional as F
import numpy as np

# https://docs.pytorch.org/docs/stable/amp#cuda-op-specific-behavior
@unittest.skipUnless(Device.DEFAULT in {"CUDA", "NV"} and torch.cuda.is_available(), "This compares torch cuda behavior to tinygrad")
class CompareTorchCUDAMixedPrecision(unittest.TestCase):
  def setUp(self):
    # disable tensorfloat32 in torch, so that torch uses normal f32, for clean comparison with tinygrad
    self.old_cuda_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    self.old_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    self.old_f32_matmul_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

  def tearDown(self):
    torch.backends.cuda.matmul.allow_tf32 = self.old_cuda_matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = self.old_cudnn_allow_tf32
    torch.set_float32_matmul_precision(self.old_f32_matmul_precision)

  def test_torch_amp_softmax_bf16_input(self):
    # torch softmax on CUDA, with automatic mixed precision enabled, will cast bf16 inputs to f32, run softmax in f32, and return f32 tensor
    x = torch.randn(128, 10, device="cuda")
    y = Tensor(x.cpu().numpy())
    x, y = x.to(torch.bfloat16), y.cast(dtypes.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      assert x.dtype is torch.bfloat16
      torch_f32 = F.softmax(x, dim=-1)
    # notice the dtype changed from bfloat16 to float32 without us specifying that explicitly
    assert torch_f32.dtype is torch.float32

    tiny_bf16 = y.softmax(-1)
    assert tiny_bf16.dtype is dtypes.bfloat16

    tiny_f32 = y.cast(dtypes.float32).softmax(-1)
    tiny_f32_exp = y.softmax(-1, dtype=dtypes.float32)
    assert y.dtype is dtypes.bfloat16
    assert tiny_f32.dtype is dtypes.float32
    assert tiny_f32_exp.dtype is dtypes.float32

    torch_f32 = torch_f32.cpu().numpy()
    # here tinygrad casts bf16 to f32 for numpy compatibility, but the underlying numbers are unchanged
    tiny_bf16 = tiny_bf16.numpy()
    tiny_f32 = tiny_f32.numpy()
    tiny_f32_exp = tiny_f32_exp.numpy()

    np.testing.assert_allclose(torch_f32, tiny_f32, rtol=1e-6, atol=1e-8)
    # notice how different the softmax results are for bf16 versus f32
    self.assertRaises(AssertionError, np.testing.assert_allclose, torch_f32, tiny_bf16, rtol=1e-3, atol=1e-4)
    # if allow the initial max/difference steps of softmax occur in bf16, the results are very different than f32
    self.assertRaises(AssertionError, np.testing.assert_allclose, torch_f32, tiny_f32_exp, rtol=1e-3, atol=1e-4)
    self.assertRaises(AssertionError, np.testing.assert_allclose, tiny_f32, tiny_f32_exp, rtol=1e-3, atol=1e-4)

if __name__=="__main__":
  unittest.main()