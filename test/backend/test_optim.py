import numpy as np
import torch
import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.optim import Adam, SGD, AdamW, Muon, LAMB
from tinygrad.helpers import Context
from test.helpers import needs_second_gpu, slow

np.random.seed(1337)
x_init = np.random.randn(1,4).astype(np.float32)
W_init = np.random.randn(4,4).astype(np.float32)
m_init = np.random.randn(1,4).astype(np.float32)

def _param(tensor, val):
  return tensor(val, requires_grad=True) if tensor is torch.tensor else tensor(val)

class TeenyNet:
  def __init__(self, tensor):
    self.x = _param(tensor, x_init.copy())
    self.W = _param(tensor, W_init.copy())
  def forward(self):
    return (self.x * self.W).sum()

class TinyNet:
  def __init__(self, tensor):
    self.x = _param(tensor, x_init.copy())
    self.W = _param(tensor, W_init.copy())
    self.m = tensor(m_init.copy())

  def forward(self):
    out = self.x.matmul(self.W).relu()
    # print(out.detach().numpy())
    out = out.log_softmax(1)
    out = out.mul(self.m).add(self.m).sum()
    return out

def step(tensor, optim, steps=1, teeny=False, **kwargs):
  net = TeenyNet(tensor) if teeny else TinyNet(tensor)
  optim = optim([net.x, net.W], **kwargs)
  for _ in range(steps):
    out = net.forward()
    optim.zero_grad()
    out.backward()
    optim.step()
  return net.x.detach().numpy(), net.W.detach().numpy()

class TestMLPerfAdamWKernel(unittest.TestCase):
  def test_replicated_clip_grads(self):
    from examples.mlperf.optim import clip_grads
    rng, devices = np.random.default_rng(11), tuple(f"CPU:{i}" for i in range(4))
    arrays = [rng.standard_normal((16, 7), dtype=np.float32), rng.standard_normal((3,), dtype=np.float32)]
    grads = [Tensor(x, dtype=dtypes.bfloat16).shard(devices).realize() for x in arrays]
    norm, _ = clip_grads(grads, grad_acc=2, clip_norm=1.0)
    rounded = [Tensor(x, dtype=dtypes.bfloat16).numpy().astype(np.float32) / 2 for x in arrays]
    expected = np.sqrt(sum(np.square(x).sum(dtype=np.float32) for x in rounded))
    np.testing.assert_allclose(norm.numpy(), expected, rtol=2e-4, atol=2e-4)

  def test_master_weight_transition(self):
    from examples.mlperf.optim import _adamw_master_step
    rng = np.random.default_rng(7)
    param = Tensor(rng.standard_normal(256, dtype=np.float32), dtype=dtypes.bfloat16).realize()
    grad = Tensor(rng.standard_normal(256, dtype=np.float32), dtype=dtypes.bfloat16).realize()
    m, v = Tensor.randn(256, dtype=dtypes.bfloat16).realize(), Tensor.rand(256, dtype=dtypes.bfloat16).realize()
    master = param.float().contiguous().realize()
    m0, v0, w0 = m.clone().realize(), v.clone().realize(), master.clone().realize()
    lr, b1_t, b2_t = Tensor([1e-3]).realize(), Tensor([0.9]).realize(), Tensor([0.95]).realize()
    expected_m = (0.9 * m0.float() + 0.1 * grad.float()).cast(dtypes.bfloat16)
    expected_v = (0.95 * v0.float() + 0.05 * grad.float().square()).cast(dtypes.bfloat16)
    calc_m, calc_v = 0.9 * m0.float() + 0.1 * grad.float(), 0.95 * v0.float() + 0.05 * grad.float().square()
    expected_w = w0 - lr * ((calc_m / (1.0-b1_t)) / ((calc_v / (1.0-b2_t)).sqrt()+1e-5) + 0.1*w0)
    Tensor.realize(expected_m, expected_v, expected_w)

    _adamw_master_step(param, grad, m, v, master, lr, b1_t, b2_t, b1=0.9, b2=0.95, eps=1e-5, wd=0.1)
    Tensor.realize(param, m, v, master)
    np.testing.assert_array_equal(m.numpy(), expected_m.numpy())
    np.testing.assert_array_equal(v.numpy(), expected_v.numpy())
    np.testing.assert_allclose(master.numpy(), expected_w.numpy(), rtol=2e-7, atol=2e-7)
    np.testing.assert_array_equal(param.numpy(), expected_w.cast(dtypes.bfloat16).numpy())

@slow
class TestOptim(unittest.TestCase):
  def setUp(self): self.enterContext(Context(TRAINING=1))

  def _test_optim(self, tinygrad_optim, torch_optim, steps, opts, atol, rtol):
    for x,y in zip(step(Tensor, tinygrad_optim, steps, **opts),
                   step(torch.tensor, torch_optim, steps, **opts)):
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

  def _test_sgd(self, steps, opts, atol, rtol): self._test_optim(SGD, torch.optim.SGD, steps, opts, atol, rtol)
  def _test_adam(self, steps, opts, atol, rtol): self._test_optim(Adam, torch.optim.Adam, steps, opts, atol, rtol)
  def _test_adamw(self, steps, opts, atol, rtol): self._test_optim(AdamW, torch.optim.AdamW, steps, opts, atol, rtol)
  def _test_muon(self, steps, opts, atol, rtol): self._test_optim(Muon, torch.optim.Muon, steps, opts, atol, rtol)

  def test_multistep_sgd_high_lr_teeny(self): self._test_sgd(2, {'lr': 1.1, 'teeny': True}, 1e-6, 1e-5)
  def test_multistep_adam_high_lr_teeny(self): self._test_adam(2, {'lr': 1.1, 'teeny': True}, 2e-4, 5e-4)
  def test_multistep_muon_high_lr_teeny(self): self._test_muon(2, {'lr': 1.1, 'teeny': True}, 1e-2, 5e-4)

  def test_sgd(self): self._test_sgd(1, {'lr': 0.001}, 1e-6, 0)
  def test_sgd_high_lr(self): self._test_sgd(1, {'lr': 10}, 1e-6, 1e-5)
  def test_sgd_wd(self): self._test_sgd(1, {'lr': 0.001, 'weight_decay': 0.1}, 1e-6, 0)
  def test_sgd_high_lr_wd(self): self._test_sgd(1, {'lr': 10, 'weight_decay': 0.1}, 1e-6, 1e-5)

  def test_multistep_sgd(self): self._test_sgd(10, {'lr': 0.001}, 1e-6, 0)
  def test_multistep_sgd_high_lr(self): self._test_sgd(10, {'lr': 10}, 1e-6, 3e-4)
  def test_multistep_sgd_wd(self): self._test_sgd(10, {'lr': 0.001, 'weight_decay': 0.1}, 1e-6, 0)
  def test_multistep_sgd_high_lr_wd(self): self._test_sgd(10, {'lr': 9, 'weight_decay': 0.1}, 1e-6, 3e-4)

  def test_multistep_sgd_momentum(self): self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9}, 1e-6, 0)
  def test_multistep_sgd_high_lr_momentum(self): self._test_sgd(10, {'lr': 10, 'momentum': 0.9}, 1e-5, 3e-4)
  def test_multistep_sgd_momentum_wd(self): self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.1}, 1e-6, 0)
  def test_multistep_sgd_high_lr_momentum_wd(self): self._test_sgd(10, {'lr': 10, 'momentum': 0.9, 'weight_decay': 0.1}, 1e-5, 3e-4)

  def test_multistep_sgd_nesterov_momentum(self): self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9, 'nesterov': True}, 1e-5, 0)
  def test_multistep_sgd_high_lr_nesterov_momentum(self): self._test_sgd(10, {'lr': 10, 'momentum': 0.9, 'nesterov': True}, 1e-5, 3e-4)
  def test_multistep_sgd_nesterov_momentum_wd(self):
    self._test_sgd(10, {'lr': 0.001, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.1}, 1e-5, 0)
  def test_multistep_sgd_high_lr_nesterov_momentum_wd(self):
    self._test_sgd(10, {'lr': 9, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.1}, 1e-5, 3e-4)

  def test_muon(self): self._test_muon(1, {'lr': 0.001}, 1e-3, 0)
  # TODO: disabled due to big atol
  # def test_muon_high_lr(self): self._test_muon(1, {'lr': 10}, 1e-6, 3e-4)
  # NOTE: big weight_decay so a missing wd would be way over atol
  def test_muon_wd(self): self._test_muon(1, {'lr': 0.001, 'weight_decay': 10}, 1e-3, 3e-4)
  # TODO: disabled due to big atol
  # def test_muon_high_lr_wd(self): self._test_muon(1, {'lr': 10, 'weight_decay': 0.01}, 1e-6, 5e-4)

  # NOTE: momentum set to 0.95 by default, nesterov set to True by default
  def test_multistep_muon_momentum_wd(self): self._test_muon(10, {'lr': 0.001, 'weight_decay': 0.01}, 3e-3, 0)
  # ns defaults are numerically unstable, but it is tolerable in real training (see nsteps/nparam tests)
  # TODO: disabled due to big atol
  # def test_multistep_muon_high_lr_momentum_wd(self): self._test_muon(10, {'lr': 10, 'weight_decay': 0.01}, 1e-1, 3e-4)
  def test_multistep_muon_no_nesterov_momentum(self): self._test_muon(10, {'lr': 0.001, 'nesterov': False}, 1e-3, 0)
  # TODO: disabled due to big atol
  # def test_multistep_muon_high_lr_no_nesterov_momentum(self): self._test_muon(10, {'lr': 10, 'nesterov': False}, 5e-2, 1e-1)

  def test_muon_ns_steps(self): self._test_muon(1, {'lr': 0.001, 'ns_steps': 3}, 1e-4, 0)
  # TODO: disabled due to big atol
  # def test_muon_high_lr_ns_steps(self): self._test_muon(1, {'lr': 10, 'ns_steps': 3}, 1e-5, 3e-4)
  def test_muon_ns_coefficients(self): self._test_muon(1, {'lr': 0.001,'ns_coefficients': (2.0,-1.5,0.5)}, 1e-5, 3e-4)
  # TODO: disabled due to big atol
  # def test_muon_high_lr_ns_coefficients(self): self._test_muon(1, {'lr': 10,'ns_coefficients': (2.0,-1.5,0.5)}, 1e-5, 3e-4)

  def test_muon_momentum_wd_ns_steps_ns_coefficients(self):
    self._test_muon(10, {'lr': 0.001, 'momentum': 0.90, 'weight_decay': 0.01, 'ns_steps': 3, 'ns_coefficients': (2.0,-1.5,0.5)}, 1e-4, 0)
  # TODO: disabled due to big atol
  # def test_multistep_muon_high_lr_momentum_wd_ns_steps_ns_coefficients(self):
  #   self._test_muon(10, {'lr': 10, 'momentum': 0.90, 'weight_decay': 0.01, 'ns_steps': 3, 'ns_coefficients': (2.0,-1.5,0.5)}, 1e-5, 3e-4)

  def test_adam(self): self._test_adam(1, {'lr': 0.001}, 1e-5, 0)
  def test_adam_high_lr(self): self._test_adam(1, {'lr': 10}, 1e-4, 1e-4)
  def test_adamw(self): self._test_adamw(1, {'lr': 0.001}, 1e-5, 0)
  def test_adamw_high_lr(self): self._test_adamw(1, {'lr': 10}, 1e-4, 1e-4)

  def test_multistep_adam(self): self._test_adam(10, {'lr': 0.001}, 1e-5, 0)
  def test_multistep_adam_high_lr(self): self._test_adam(10, {'lr': 10}, 2e-3, 5e-4)

  def test_multistep_adamw(self): self._test_adamw(10, {'lr': 0.001}, 1e-5, 0)
  def test_multistep_adamw_high_lr(self): self._test_adamw(10, {'lr': 10}, 5e-4, 2e-3)

  def test_duped_weights(self):
    for Opt in [Adam, AdamW, SGD]:
      losses = []
      for i in range(2):
        w = Tensor(x_init.copy())
        opt = Opt([w], lr=0.1) if i == 0 else Opt([w, w], lr=0.1)

        loss = None
        for _ in range(3):
          loss = w.sum()
          opt.zero_grad()
          loss.backward()
          opt.step()
        losses.append(loss.numpy())

      np.testing.assert_allclose(losses[0], losses[1], atol=1e-4, rtol=0)

  @unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
  def test_mixed_precision(self):
    self.enterContext(Context(DEFAULT_FLOAT=dtypes.half))
    # weight update would overflow without upcasting
    self._test_sgd(10, {'lr': 1e10}, 1e-6, 3e-4)
    self._test_adam(1, {'lr': 1e10}, 1e-4, 1e-4)
    self._test_adamw(1, {'lr': 1e10}, 1e-4, 1e-4)

  def test_assert_tensor_train(self):
    t = Tensor.ones((1,1))
    optimizer = Adam([t])
    optimizer.zero_grad()
    t.sum().backward()
    with Context(TRAINING=0):
      self.assertRaises(RuntimeError, optimizer.step)
    with Context(TRAINING=1):
      optimizer.step()

  def test_lamb_cpu_offload(self):
    # test that LAMB works when optimizer params (m, v, b1_t, b2_t) are moved to CPU
    t = Tensor(x_init.copy())
    opt = LAMB([t])
    # move optimizer state to CPU
    for p in opt.m + opt.v + [opt.b1_t, opt.b2_t]: p.to_("CPU")
    # run a step
    t.sum().backward()
    opt.step()
    self.assertEqual(t.device, Device.DEFAULT)
    self.assertEqual(opt.m[0].device, "CPU")

  @needs_second_gpu
  def test_lamb_cpu_offload_multi(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    t = Tensor(x_init.copy()).shard(ds, axis=1)
    ds = t.device
    opt = LAMB([t])
    # move optimizer state to CPU
    for p in opt.m + opt.v + [opt.b1_t, opt.b2_t]: p.to_("CPU")
    # run a step
    t.sum().backward()
    opt.step()
    self.assertEqual(t.device, ds)
    self.assertEqual(opt.m[0].device, "CPU")

if __name__ == '__main__':
  unittest.main()
