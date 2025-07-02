import unittest
from tinygrad import Tensor, nn, Variable

# outerworld range should support three things
# 1. full optimizer steps
# 2. gradient accumulation
# 3. stacked linear layers

class Model:
  def __init__(self): self.w = nn.Linear(64, 8, bias=False)
  def __call__(self, x:Tensor) -> Tensor: return self.w(x)

def get_model_and_opt():
  Tensor.manual_seed(1337)
  m = Model()
  opt = nn.optim.SGD(nn.state.get_parameters(m), lr=0.1, weight_decay=0)
  return m, opt

class TestOuterworldRange(unittest.TestCase):
  STEPS = 10
  BS = 20

  @classmethod
  def setUpClass(cls):
    Tensor.manual_seed(1338)
    # it learns to compute mean
    cls.X = Tensor.randn(cls.STEPS, cls.BS, 64).contiguous().realize()
    cls.Y = cls.X.reshape(cls.STEPS, cls.BS, 8, 8).mean(axis=-1).contiguous().realize()
    cls.losses = cls._get_model_baseline()

  @classmethod
  @Tensor.train()
  def _get_model_baseline(self):
    m, opt = get_model_and_opt()
    losses = []
    for i in range(self.STEPS):
      opt.zero_grad()
      loss = (m(self.X[i]) - self.Y[i]).square().mean()
      loss.backward()
      loss.realize(*opt.schedule_step())
      losses.append(loss.item())
    return losses

  def _compare(self, losses):
    for x,y in zip(self.losses, losses): self.assertAlmostEqual(x, y, places=5)

  @Tensor.train()
  def test_model_variable(self):
    m, opt = get_model_and_opt()
    losses = []
    vi = Variable('i', 0, self.STEPS-1)
    for i in range(self.STEPS):
      vib = vi.bind(i)
      opt.zero_grad()
      loss = (m(self.X[vib]) - self.Y[vib]).square().mean()
      loss.backward()
      loss.realize(*opt.schedule_step())
      losses.append(loss.item())
    self._compare(losses)

  @Tensor.train()
  def test_model_scheduled(self):
    m, opt = get_model_and_opt()
    losses = []
    for i in range(self.STEPS):
      opt.zero_grad()
      loss = (m(self.X[i]) - self.Y[i]).square().mean()
      loss.backward()
      opt.schedule_step()
      losses.append(loss)
    self._compare([x.item() for x in losses])

  @unittest.expectedFailure
  @Tensor.train()
  def test_model_scheduled_variable(self):
    m, opt = get_model_and_opt()
    losses = []
    vi = Variable('i', 0, self.STEPS-1)
    for i in range(self.STEPS):
      vib = vi.bind(i)
      opt.zero_grad()
      loss = (m(self.X[vib]) - self.Y[vib]).square().mean()
      loss.backward()
      opt.schedule_step()
      losses.append(loss)
    self._compare([x.item() for x in losses])

if __name__ == "__main__":
  unittest.main()
