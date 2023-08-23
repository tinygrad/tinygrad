import unittest, time
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE
from tinygrad.ops import GlobalCounters, LazyOp, LoadOps
from tinygrad.lazy import Device
from tinygrad.helpers import CI, dtypes

from examples.hlb_cifar10 import SpeedyResNet
from examples.llama import Transformer, MODEL_PARAMS
from examples.stable_diffusion import UNetModel

def helper_test(nm, gen, train, max_memory_allowed, max_kernels_allowed):
  tms = []
  for _ in range(4):
    GlobalCounters.reset()
    Device[Device.DEFAULT].synchronize()
    st = time.perf_counter_ns()
    train(*gen())
    Device[Device.DEFAULT].synchronize()
    tms.append(time.perf_counter_ns() - st)

  kernels_used = len(train.jit_cache) if hasattr(train, "jit_cache") else None
  print(f"{nm}: used {GlobalCounters.mem_used/1e9:.2f} GB and {kernels_used} kernels in {min(tms)/1e6:.2f} ms")
  assert GlobalCounters.mem_used/1e9 < max_memory_allowed, f"{nm} used more than {max_memory_allowed:.2f} GB"
  assert not kernels_used or kernels_used <= max_kernels_allowed, f"{nm} used more than {max_kernels_allowed} kernels"

# for speed
def derandomize(x):
  if isinstance(x, LazyOp):
    if x.op == LoadOps.RAND: x.op = LoadOps.EMPTY
    x.src = [derandomize(s) for s in x.src]
  elif hasattr(x, "op"):
    x.op = derandomize(x.op)
  return x

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = derandomize(p.lazydata)
    p.realize()

class TestRealWorld(unittest.TestCase):
  @unittest.skipUnless(not CI, "too big for CI")
  def test_stable_diffusion(self):
    model = UNetModel()
    derandomize_model(model)
    @TinyJit
    def test(t, t2): return model(t, 801, t2).realize()
    helper_test("test_sd", lambda: (Tensor.randn(1, 4, 64, 64),Tensor.randn(1, 77, 768)), test, 18.0, 967)

  @unittest.skipUnless(Device.DEFAULT in JIT_SUPPORTED_DEVICE, "needs JIT")
  def test_llama(self):
    old_type = Tensor.default_type
    Tensor.default_type = dtypes.float16

    args_tiny = {"dim": 1024, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    model = Transformer(**(args_tiny if CI else MODEL_PARAMS[1]["7B"]["args"]))
    derandomize_model(model)
    @TinyJit
    def test(t): return model(t, 0).realize()
    helper_test("test_llama", lambda: (Tensor([[1,]]),), test, 0.22 if CI else 13.5, 126 if CI else 486)

    Tensor.default_type = old_type

  @unittest.skipUnless(Device.DEFAULT in JIT_SUPPORTED_DEVICE, "needs JIT")
  def test_train_cifar(self):
    # TODO: with default device
    #old_default = Device.DEFAULT
    #Device.DEFAULT = "FAKE"
    #Device['fake'].codegen = Device[old_default].codegen

    # TODO: with train
    old_training = Tensor.training
    Tensor.training = True
    model = SpeedyResNet(Tensor.ones((12,3,2,2)))
    optimizer = optim.SGD(get_parameters(model), lr=0.01, momentum=0.8, nesterov=True, weight_decay=0.15)

    BS = 32 if CI else 512

    @TinyJit
    def train(X):
      out = model(X)
      loss = out.mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    helper_test("train_cifar", lambda: (Tensor.randn(BS, 3, 32, 32),), train, (1.0/48)*BS, 153)

    # reset device
    Tensor.training = old_training
    #Device.DEFAULT = old_default

if __name__ == '__main__':
  unittest.main()
