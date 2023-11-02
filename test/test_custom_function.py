# this is an example of how you can write terrible DSP compute breaking ops like warpPerspective
# here we use a CUSTOM op to write atan2

import unittest
import numpy as np
from typing import Optional, Tuple
from tinygrad.helpers import prod, dtypes

# *** first, we implement the atan2 op at the lowest level ***
# `atan2_gpu` for GPUBuffers and `atan2_cpu` for CPUBuffers
from tinygrad.lazy import LazyBuffer, create_lazybuffer
from tinygrad.ops import ASTRunner, Device
from tinygrad.shape.shapetracker import ShapeTracker
import pytest

pytestmark = pytest.mark.webgpu

# we don't always have GPU support, so the type signature is the abstract CompiledBuffer instead of GPUBuffer
def atan2_gpu(ret:LazyBuffer, a:LazyBuffer, b:LazyBuffer):
  assert a.device == "GPU" and b.device == "GPU", "gpu function requires GPUBuffers"
  assert a.dtype == b.dtype and a.dtype == dtypes.float32, "gpu function only supports float32"
  ret.realized = Device[ret.device].buffer(prod(ret.shape), ret.dtype)
  ASTRunner("atan2_gpu", """
    __kernel void atan2_gpu(global float *c, global float *a, global float *b) {
      int idx = get_global_id(0);
      c[idx] = atan2(a[idx], b[idx]);
    }""", global_size=[prod(ret.shape)]).build(Device[ret.device].compiler, Device[ret.device].runtime).exec([ret.realized, a.realized, b.realized])
  return ret.realized

def atan2_cpu(ret:LazyBuffer, a:LazyBuffer, b:LazyBuffer):
  return Device[ret.device].from_underlying(np.arctan2(a.realized._buf, b.realized._buf))

# *** second, we write the ATan2 mlop ***
# NOTE: The derivative of atan2 doesn't need a custom op! https://www.liquisearch.com/atan2/derivative
# In general, it is also optional to write a backward function, just your backward pass won't work without it

from tinygrad.ops import LazyOp, LoadOps, BinaryOps, UnaryOps
from tinygrad.lazy import LazyBuffer
from tinygrad.tensor import Function

class ATan2(Function):
  def forward(self, a:LazyBuffer, b:LazyBuffer) -> LazyBuffer:
    assert prod(a.shape) == prod(b.shape) and a.device == b.device, "shape or device mismatch"
    self.a, self.b = a, b
    ast = LazyOp(LoadOps.CUSTOM, (a.contiguous(), b.contiguous()), {"GPU": atan2_gpu, "CPU": atan2_cpu}[a.device])
    return create_lazybuffer(a.device, ShapeTracker.from_shape(a.shape), LoadOps, ast, max(a.dtype, b.dtype))
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    denom = (self.a.e(BinaryOps.MUL, self.a)).e(BinaryOps.ADD, self.b.e(BinaryOps.MUL, self.b))
    return grad_output.e(BinaryOps.MUL, self.b.e(BinaryOps.DIV, denom)) if self.needs_input_grad[0] else None, \
           grad_output.e(BinaryOps.MUL, self.a.const(0).e(BinaryOps.SUB, self.a).e(BinaryOps.DIV, denom)) if self.needs_input_grad[1] else None

# *** third, we use our lovely new mlop in some tests ***

from tinygrad.tensor import Tensor

@unittest.skipUnless(Device.DEFAULT in ["CPU", "GPU"], "atan2 is only implemented for CPU and GPU")
class TestCustomFunction(unittest.TestCase):
  def test_atan2_forward(self):
    # create some random Tensors, permute them just because we can
    a = Tensor.randn(4,4,requires_grad=True).permute(1,0)
    b = Tensor.randn(4,4,requires_grad=True).permute(1,0)

    # run the forward pass. note: up until the .numpy(), it's all lazy
    c = ATan2.apply(a, b)
    print(c.numpy())

    # check the forward pass (in numpy)
    np.testing.assert_allclose(c.numpy(), np.arctan2(a.numpy(), b.numpy()), atol=1e-5)

  # fun fact, this never actually calls forward, so it works in all the backends
  def test_atan2_backward(self):
    # have to go forward before we can go backward
    a = Tensor.randn(4,4,requires_grad=True).permute(1,0)
    b = Tensor.randn(4,4,requires_grad=True).permute(1,0)
    c = ATan2.apply(a, b)

    # run the backward pass
    c.mean().backward()
    assert a.grad is not None and b.grad is not None, "tinygrad didn't compute gradients"
    print(a.grad.numpy())
    print(b.grad.numpy())

    # check the backward pass (in torch)
    import torch
    ta, tb = torch.tensor(a.numpy(), requires_grad=True), torch.tensor(b.numpy(), requires_grad=True)
    tc = torch.atan2(ta, tb)
    tc.mean().backward()
    assert ta.grad is not None and tb.grad is not None, "torch didn't compute gradients"
    np.testing.assert_allclose(a.grad.numpy(), ta.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), tb.grad.numpy(), atol=1e-5)

  def test_atan2_jit(self):
    # custom ops even work in the JIT!
    from tinygrad.jit import TinyJit

    @TinyJit
    def jitted_atan2(a:Tensor, b:Tensor) -> Tensor:
      return ATan2.apply(a, b).realize()

    for _ in range(5):
      a = Tensor.randn(4,4,requires_grad=True).permute(1,0)
      b = Tensor.randn(4,4,requires_grad=True).permute(1,0)
      c = jitted_atan2(a, b)
      np.testing.assert_allclose(c.numpy(), np.arctan2(a.numpy(), b.numpy()), atol=1e-5)

if __name__ == "__main__":
  unittest.main()
