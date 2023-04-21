import numpy as np
import torch
import unittest
from tinygrad.nn.optim import SGD
from extra.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

np.random.seed(1337)

def current_lr(optim): return optim.param_groups[0]['lr'] if hasattr(optim, 'param_groups') else optim.lr
def get_lrs(optim, sched, epochs, steps=1, accs=None):
  lrs = [current_lr(optim)]
  for e in range(epochs):
    for _ in range(steps):
      optim.step()
    sched.step() if accs is None else sched.step(accs[e])
    lrs.append(current_lr(optim))
  return lrs

class TestLrScheduler(unittest.TestCase):
  def _test_lr_scheduler(self, tinygrad_sched, torch_sched, epochs, opts, atol, rtol):
    accs = opts.pop('accs', None)
    tinygrad_optim, torch_optim = SGD([], lr=0.01), torch.optim.SGD([torch.tensor([0.], requires_grad=True)], lr=0.01)
    tinygrad_sched, torch_sched = tinygrad_sched(tinygrad_optim, **opts), torch_sched(torch_optim, **opts)

    tinygrad_lrs = get_lrs(tinygrad_optim, tinygrad_sched, epochs, accs=accs)
    torch_lrs = get_lrs(torch_optim, torch_sched, epochs, accs=accs)

    np.testing.assert_allclose(tinygrad_lrs, torch_lrs, atol=atol, rtol=rtol)

  def _test_multisteplr(self, epochs, opts, atol, rtol): 
    self._test_lr_scheduler(MultiStepLR, torch.optim.lr_scheduler.MultiStepLR, epochs, opts, atol, rtol)
  def _test_reducelronplateau(self, epochs, opts, atol, rtol): 
    opts['accs'] = np.random.randn(epochs)
    self._test_lr_scheduler(ReduceLROnPlateau, torch.optim.lr_scheduler.ReduceLROnPlateau, epochs, opts, atol, rtol)
  def _test_cosineannealinglr(self, epochs, opts, atol, rtol): 
    opts['T_max'] = epochs
    self._test_lr_scheduler(CosineAnnealingLR, torch.optim.lr_scheduler.CosineAnnealingLR, epochs, opts, atol, rtol)
  
  def test_multisteplr(self): self._test_multisteplr(10, {'milestones': [1, 2, 7]}, 1e-6, 1e-6)
  def test_multisteplr_gamma(self): self._test_multisteplr(10, {'milestones': [1, 2, 7], 'gamma': 0.1337}, 1e-6, 1e-6)

  def test_reducelronplateau(self): self._test_reducelronplateau(100, {}, 1e-6, 1e-6)
  def test_reducelronplateau_max(self): self._test_reducelronplateau(100, {'mode': 'max'}, 1e-6, 1e-6)
  def test_reducelronplateau_factor(self): self._test_reducelronplateau(100, {'factor': 0.1337}, 1e-6, 1e-6)
  def test_reducelronplateau_patience(self): self._test_reducelronplateau(100, {'patience': 3}, 1e-6, 1e-6)
  def test_reducelronplateau_threshold(self): self._test_reducelronplateau(100, {'threshold': 1e-6}, 1e-6, 1e-6)
  def test_reducelronplateau_threshold_mode(self): self._test_reducelronplateau(100, {'threshold_mode': 'abs'}, 1e-6, 1e-6)

  def test_cosineannealinglr(self): self._test_cosineannealinglr(100, {}, 1e-6, 1e-6)
  def test_cosineannealinglr_eta_min(self): self._test_cosineannealinglr(100, {'eta_min': 0.001}, 1e-6, 1e-6)

if __name__ == '__main__':
  unittest.main()
