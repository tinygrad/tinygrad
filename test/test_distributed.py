import unittest, copy
import numpy as np
from tinygrad import Tensor, Device, nn
from tinygrad.nn.distributed import FSDP
from test.helpers import needs_second_gpu
from os import getenv

# ** Define a simple network to do tests on **
class Net:
  def __init__(self):
    self.l1 = nn.Linear(5, 8)
    self.l2 = nn.Linear(8, 4)

  def __call__(self, x:Tensor) -> Tensor:
    return self.l2(self.l1(x).relu())


class TestFSDP(unittest.TestCase):
  @needs_second_gpu
  def setUp(self):
    self.net = Net()
    self.devices =tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2)))

    # Save original state for comparison (copying tensors to avoid sharing memory)
    self.original_state = {k: v.numpy() for k, v in nn.state.get_state_dict(self.net).items()}

    self.fsdp = FSDP(self.net, self.devices)

  def test_sharding(self):
    for name, param in nn.state.get_state_dict(self.net).items():
      self.assertEqual(param.device, self.devices)
      self.assertEqual(param.uop.axis, 0)
      # Check if sharded correctly
      for i, lb in enumerate(param.uop.src):
        expected_shape = list(self.original_state[name].shape)
        expected_shape[0] //= 2
        self.assertEqual(lb.shape, tuple(expected_shape))

  def test_logical_shapes(self):
    for name, shape in self.fsdp.logical_shapes.items():
      self.assertEqual(shape, self.original_state[name].shape)

  def test_unit_collection(self):
    # Net has l1 and l2 as leaf modules
    units = self.fsdp.units
    self.assertEqual(len(units), 2)
    self.assertIn(self.net.l1, units)
    self.assertIn(self.net.l2, units)

  def test_gathering(self):
    for name, param in nn.state.get_state_dict(self.net).items():
      shape = self.fsdp.logical_shapes[name]
      gathered = self.fsdp.gather_param(shape, param)
      np.testing.assert_allclose(gathered.numpy(), self.original_state[name], atol=1e-5, rtol=1e-5)

  def test_forward(self):
    # We need a new net with the same weights to compare
    ref_net = Net()
    nn.state.load_state_dict(ref_net, {k: Tensor(v) for k, v in self.original_state.items()})

    x = Tensor.randn(4, 5)
    out_ref = ref_net(x)
    out_fsdp = self.fsdp(x)

    np.testing.assert_allclose(out_fsdp.numpy(), out_ref.numpy(), atol=1e-5, rtol=1e-5)

  def test_backward(self):
      import numpy as np
      x = Tensor.randn(4, 5)
    
      # 1. Reference pass
      ref_net = self.net.__class__()
      nn.state.load_state_dict(ref_net, {k: Tensor(v) for k, v in self.original_state.items()})
    
      # Ensure reference params track gradients
      for param in nn.state.get_state_dict(ref_net).values():
          param.requires_grad = True
    
      out_ref = ref_net(x)
      out_ref.mean().backward()
    
      ref_grads = {k: v.grad.numpy() for k, v in nn.state.get_state_dict(ref_net).items()}
    
      out_fsdp = self.fsdp(x)
      out_fsdp.mean().backward()
      
      self.fsdp.sync_grad()
    
      for name, param in nn.state.get_state_dict(self.net).items():
        sharded_grad = param.grad
        self.assertIsNotNone(sharded_grad, f"Gradient for {name} is None")
    
        # 1. Gather to CPU (creates a Single-Device tensor)
        # This performs an All-Gather on the sharded gradients to reconstruct the full padded tensor
        grad_cpu = sharded_grad.to(self.devices[0])
    
        # 2. Remove padding (using logical shape)
        shape = self.fsdp.logical_shapes[name]
        slices = tuple(slice(0, s) for s in shape)
        grad_cpu = grad_cpu[slices]
    
        # 3. Compare
        np.testing.assert_allclose(grad_cpu.numpy(), ref_grads[name], atol=1e-5, rtol=1e-5)
