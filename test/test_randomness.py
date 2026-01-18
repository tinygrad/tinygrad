import unittest, math
from functools import partial

from tinygrad import nn, dtypes, Tensor, Device, TinyJit, Variable
from tinygrad.helpers import getenv, CI, OSX
from tinygrad.device import is_dtype_supported
from tinygrad.engine.realize import CompiledRunner
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.nir import NIRRenderer
from test.helpers import not_support_multi_device, needs_second_gpu

import numpy as np
import torch
from hypothesis import given, settings, strategies as strat

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

# https://gist.github.com/devries/11405101
def ksprob(a):
  fac, total, termbf = 2.0, 0.0, 0.0
  a2 = -2.0 * a * a
  for j in range(1, 101):
    term = fac * math.exp(a2 * j * j)
    total += term
    if math.fabs(term) <= 0.001 * termbf or math.fabs(term) <= 1e-8 * total:
      return total
    fac = -fac
    termbf = math.fabs(term)
  return 1.0

def kstest(l1, l2):
  n1, n2 = len(l1), len(l2)
  l1.sort()
  l2.sort()
  j1, j2, d, fn1, fn2 = 0, 0, 0.0, 0.0, 0.0
  while j1 < n1 and j2 < n2:
    d1, d2 = l1[j1], l2[j2]
    if d1 <= d2:
      fn1 = (float(j1) + 1.0) / float(n1)
      j1 += 1
    if d2 <= d1:
      fn2 = (float(j2) + 1.0) / float(n2)
      j2 += 1
    dtemp = math.fabs(fn2 - fn1)
    if dtemp > d:
      d = dtemp
  ne = float(n1 * n2) / float(n1 + n2)
  nesq = math.sqrt(ne)
  prob = ksprob((nesq + 0.12 + 0.11 / nesq) * d)
  return prob

def equal_distribution(tiny_func, torch_func=None, numpy_func=None, shape=(40, 43), alpha=0.04):
  Tensor.manual_seed(1337)
  torch.manual_seed(1337)
  np.random.seed(1337)
  assert not (torch_func is None and numpy_func is None), "no function to compare with"
  x1 = tiny_func(*shape).numpy().flatten()
  x2 = tiny_func(shape).numpy().flatten()
  if numpy_func is not None: y = numpy_func(shape).flatten()
  if torch_func is not None: z = torch_func(shape).numpy().flatten()
  return (numpy_func is None or (kstest(x1, y) >= alpha and kstest(x2, y) >= alpha)) and \
    (torch_func is None or (kstest(x1, z) >= alpha and kstest(x2, z) >= alpha))

def normal_test(func, shape=(20, 23), alpha=0.05): return equal_distribution(func, numpy_func=lambda x: np.random.randn(*x), shape=shape, alpha=alpha)

class TestRandomness(unittest.TestCase):
  def test_rand(self):
    self.assertFalse(normal_test(Tensor.rand))
    self.assertTrue(equal_distribution(Tensor.rand, torch.rand, lambda x: np.random.rand(*x)))

  @unittest.skipUnless(is_dtype_supported(dtypes.float16) and is_dtype_supported(dtypes.ulong), "need float16 and ulong support")
  def test_rand_float16(self):
    N = 128
    x = Tensor.rand((2, N, N), dtype=dtypes.float16)
    assert x.dtype == dtypes.float16
    nx = x.numpy()
    # seed dependant, check output range is [0, 1)
    assert nx[nx == 1].size == 0
    assert nx[nx == 0].size > 0
    equal_distribution(lambda *x: Tensor.rand(*x, dtype=dtypes.float16), torch.rand, lambda x: np.random.rand(*x), shape=(2, N, N))

  @unittest.skipIf(CI and Device.DEFAULT in {"NV", "CUDA"}, "gpuocelot doesn't support certain ops needed for threefry")
  def test_threefry_against_reference(self):
    Tensor.manual_seed(1337)

    # reference generated using
    """
    key0 = 1337
    key1 = 0
    values = jax.extend.random.threefry_2x32((np.uint32(key1), np.uint32(key0)), np.tile(np.arange(10, dtype=np.uint32), 2))
    print(f"[{', '.join(f'{v}' for v in values)}]")
    """
    # JAX reference implementation
    jr = np.array([96694167, 3899677701, 3777760592, 361541278, 4247778752, 3205134549, 1911899812,
                    3457752739, 1813072390, 3423281881, 2019547449, 3238527978, 4081885405, 2759498550,
                    585929100, 37685650, 1493162330, 3176705340, 139929675, 3743710624])

    counts0 = counts1 = Tensor.arange(10, dtype=dtypes.uint32)
    r = Tensor._threefry_random_bits(Tensor([0, 1337], dtype='uint32'), counts0, counts1).numpy()

    np.testing.assert_allclose(jr, r)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, (NIRRenderer, PTXRenderer)), "PTX and NIR use pointer arithmetic")
  def test_threefry_doesnt_use_long(self):
    sched = Tensor.rand(20).schedule()
    for si in sched:
      si.lower()
      if isinstance(si.prg, CompiledRunner):
        for u in si.prg.p.uops:
          self.assertNotIn(u.dtype, {dtypes.long, dtypes.ulong}, msg=f"long found in {si.prg.p.name}")

  def test_threefry_against_reference_full(self):
    Tensor.manual_seed(1337)
    # reference generated using
    """
    key0 = 1337
    key1 = int.from_bytes(hashlib.sha256(int(0).to_bytes(4)).digest(), "big") & 0xffffffff
    values = jax.extend.random.threefry_2x32((np.uint32(key1), np.uint32(key0)), np.tile(np.arange(10, dtype=np.uint32), 2))
    values = (values >> (32 - 23)) | np.array(1, dtype=np.float32).view(np.uint32)
    values =  values.view(np.float32) - 1
    print(f"[{', '.join(f'{v}' for v in values)}]")
    """
    jr = np.array([0.64602804, 0.92046547, 0.7401037, 0.3942032, 0.40671802, 0.6436621, 0.5206623, 0.22375143,
                   0.70807624, 0.38364744, 0.41685486, 0.47665608, 0.009511948, 0.65394187, 0.99575675, 0.9577522,
                   0.09252262, 0.71196556, 0.6976292, 0.27724016], dtype=np.float32)
    r = Tensor.rand(20).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

    # next 20, np.tile(np.arange(20, 30, dtype=np.uint32), 2)
    jr = np.array([0.021026373, 0.1256187, 0.7586163, 0.28140187, 0.706741, 0.7084174, 0.8895695, 0.8290298,
                   0.6767577, 0.67285323, 0.3406446, 0.61420345, 0.17236173, 0.46465623, 0.5711199, 0.6435076,
                   0.124486566, 0.23862779, 0.20885861, 0.16646779], dtype=np.float32)
    r = Tensor.rand(20).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

    # next 10, np.tile(np.arange(40, 45, dtype=np.uint32), 2)
    jr = np.array([0.96615326, 0.3364507, 0.39027202, 0.24160278, 0.10455513, 0.75493455, 0.72733414, 0.97796345,
                   0.8045577, 0.21859896], dtype=np.float32)

    r = Tensor.rand(10).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

  @needs_second_gpu
  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_threefry_tensors_cnt(self):
    Tensor.manual_seed(1337)

    Tensor.rand(20).realize()

    assert len(Tensor._device_rng_counters) == 1
    assert len(Tensor._device_seeds) == 1

    Tensor.rand(20, device=f"{Device.DEFAULT}:1").realize()

    assert len(Tensor._device_rng_counters) == 2
    assert len(Tensor._device_seeds) == 2

    Tensor.manual_seed(2)

    assert len(Tensor._device_rng_counters) == 0
    assert len(Tensor._device_seeds) == 0

  @needs_second_gpu
  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_threefry_same_kernels(self):
    Tensor.manual_seed(0)

    Tensor.rand(1).realize()

    s = Tensor.rand(20).schedule()
    s2 = Tensor.rand(20).schedule()

    assert len(s) == len(s2), f"{len(s)} != {len(s2)}"
    for x,y in zip(s, s2):
      if not (x.ast == y.ast):
        print(f"{x.ast} != {y.ast}")

    Tensor.rand(1, device=f"{Device.DEFAULT}:1").realize()

    s3 = Tensor.rand(20, device=f"{Device.DEFAULT}:1").schedule()
    s4 = Tensor.rand(20, device=f"{Device.DEFAULT}:1").schedule()

    assert len(s3) == len(s4), f"{len(s3)} != {len(s4)}"
    assert len(s2) == len(s4), f"{len(s)} != {len(s3)}"
    for x,y in zip(s3, s4):
      if not (x.ast == y.ast):
        print(f"{x.ast} != {y.ast}")

  @unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "need bfloat16 support")
  def test_rand_bfloat16(self):
    N = 128
    x = Tensor.rand((2, N, N), dtype=dtypes.bfloat16)
    assert x.dtype == dtypes.bfloat16
    nx = x.numpy()
    assert nx[nx == 1].size == 0
    assert nx[nx == 0].size > 0
    equal_distribution(lambda *x: Tensor.rand(*x, dtype=dtypes.bfloat16).float(), torch.rand, lambda x: np.random.rand(*x), shape=(2, N, N))

  def test_rand_like(self):
    empty = Tensor.empty((80, 44))
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_randn_like(self):
    empty = Tensor.empty((80, 44))
    rand = Tensor.randn_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_zero_shape(self):
    empty = Tensor.empty(0, 20)
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_more_dims(self):
    empty = Tensor.empty((1, 2, 3, 4, 5, 6))
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_dtype(self):
    empty = Tensor.empty((80, 44), dtype=dtypes.float16)
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

    empty = Tensor.empty((80, 44))
    rand = Tensor.rand_like(empty, dtype=dtypes.float16)
    assert rand.shape == empty.shape
    assert rand.dtype == dtypes.float16
    assert rand.device == empty.device

  def test_randn_like_dtype(self):
    empty = Tensor.empty((80, 44), dtype=dtypes.float16)
    rand = Tensor.randn_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

    empty = Tensor.empty((80, 44))
    rand = Tensor.randn_like(empty, dtype=dtypes.float16)
    assert rand.shape == empty.shape
    assert rand.dtype == dtypes.float16
    assert rand.device == empty.device

  def test_randn(self):
    self.assertEqual(Tensor.randn(3,3,dtype=dtypes.half).dtype, dtypes.half)
    self.assertTrue(normal_test(Tensor.randn))
    self.assertTrue(equal_distribution(Tensor.randn, torch.randn, lambda x: np.random.randn(*x)))

  def test_randn_device(self):
    self.assertEqual(Tensor.randn(3,3,device="CPU").device, "CPU")

  @given(strat.sampled_from([dtypes.float, dtypes.float16, dtypes.bfloat16]))
  @unittest.skipIf(Device.DEFAULT in ["HSA", "AMD"], "bfloat16 local buffer broken in HSA")
  def test_randn_finite(self, default_float):
    if not is_dtype_supported(default_float): return
    old_default_float = dtypes.default_float
    # low precision can result in inf from randn
    dtypes.default_float = default_float
    t = Tensor.randn(256, 256)
    mx = t.max().numpy().item()
    mn = t.min().numpy().item()
    print(f"testing with {default_float=}")
    assert math.isfinite(mx), mx
    assert math.isfinite(mn), mn
    dtypes.default_float = old_default_float

  def test_randint(self):
    self.assertFalse(normal_test(Tensor.randint))
    self.assertTrue(equal_distribution(partial(Tensor.randint, low=-2, high=5),
                                       numpy_func=lambda x: np.random.randint(low=-2, high=5, size=x)))
    self.assertTrue(equal_distribution(partial(Tensor.randint, low=-2, high=5, dtype="int32"),
                                       numpy_func=lambda x: np.random.randint(low=-2, high=5, size=x)))
    self.assertTrue(Tensor.randint(1, device="CPU").device=="CPU")
    # check types of args
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=0.1, high=3)
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=0, high=3.5)
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=1, high=3, dtype="float")
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=0, high=3, dtype=dtypes.float32)

  def test_normal(self):
    self.assertTrue(normal_test(Tensor.normal))
    self.assertTrue(equal_distribution(Tensor.normal, lambda x: torch.nn.init.normal_(torch.empty(x), mean=0, std=1),
                                                      lambda x: np.random.normal(loc=0, scale=1, size=x)))

  def test_uniform(self):
    self.assertFalse(normal_test(Tensor.uniform))
    self.assertTrue(equal_distribution(Tensor.uniform, lambda x: torch.nn.init.uniform_(torch.empty(x)), lambda x: np.random.uniform(size=x)))
    self.assertTrue(equal_distribution(partial(Tensor.uniform, low=-100, high=100, dtype=dtypes.int32),
                                       numpy_func=lambda x: np.random.randint(low=-100, high=100, size=x)))

  def test_scaled_uniform(self):
    self.assertFalse(normal_test(Tensor.scaled_uniform))
    self.assertTrue(equal_distribution(Tensor.scaled_uniform, lambda x: torch.nn.init.uniform_(torch.empty(x), a=-1, b=1) / math.sqrt(math.prod(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) / math.sqrt(math.prod(x))))

  def test_glorot_uniform(self):
    self.assertFalse(normal_test(Tensor.glorot_uniform))
    self.assertTrue(equal_distribution(Tensor.glorot_uniform, lambda x: torch.nn.init.xavier_uniform_(torch.empty(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) * math.sqrt(6 / (x[0] + math.prod(x[1:])))))

  def test_kaiming_uniform(self):
    for shape in [(32, 128, 3, 3), (80, 44), (3, 55, 35)]:
      self.assertTrue(equal_distribution(Tensor.kaiming_uniform, lambda x: torch.nn.init.kaiming_uniform_(torch.empty(x)), shape=shape))

  def test_kaiming_normal(self):
    for shape in [(32, 128, 3, 3), (80, 44), (3, 55, 35)]:
      self.assertTrue(equal_distribution(Tensor.kaiming_normal, lambda x: torch.nn.init.kaiming_normal_(torch.empty(x)), shape=shape))

  def test_multinomial(self):
    self.assertRaises(AssertionError, lambda: Tensor(2).multinomial(1, replacement=False))
    self.assertRaises(AssertionError, lambda: Tensor([1, 9]).multinomial(0, replacement=False))
    def _check_with_torch(w, num_samples, replacement):
      tiny_res = Tensor(w).multinomial(num_samples, replacement=replacement)
      torch_res = torch.tensor(w).multinomial(num_samples, replacement=replacement)
      self.assertEqual(tiny_res.shape, torch_res.shape)
      if torch_res.ndim == 1:
        tiny_res = tiny_res.unsqueeze(0)
        torch_res = torch_res.unsqueeze(0)
      for i in range(torch_res.shape[0]):
        self.assertTrue(equal_distribution(lambda *_: tiny_res[i], lambda _: torch_res[i]))
    _check_with_torch(w=[0.231, 0., 1., 0.5], num_samples=300, replacement=True)
    _check_with_torch(w=[[0.2, 0.8]], num_samples=300, replacement=True)  # 2D but only 1 row
    _check_with_torch(w=[[0.453, 0., 1., 0.81], [0.1, 0.8, 0., 0.1]], num_samples=300, replacement=True)
    # no-replacement isn't supported, unless taking only one sample
    w = [0.1, 0.9]
    self.assertRaises(AssertionError, lambda: Tensor(w).multinomial(100, replacement=False))

    @TinyJit
    def sample_one(): return Tensor(w).multinomial(1, replacement=False).realize()

    tiny_samples = [sample_one().item() for _ in range(1000)]
    torch_samples = [torch.tensor(w).multinomial(1, replacement=False).item() for _ in range(1000)]
    self.assertTrue(equal_distribution(lambda *_: Tensor(tiny_samples), lambda _: torch.tensor(torch_samples)))

  def test_multinomial_counterexample(self):
    tiny_res = Tensor([0.3, 0.6, 0.1]).multinomial(4000, replacement=True)
    torch_res = torch.tensor([0.3, 0.6, 0.1]).multinomial(4000, replacement=True)
    self.assertTrue(equal_distribution(lambda *_: tiny_res, lambda _: torch_res))
    torch_res = torch.tensor([0.2, 0.7, 0.1]).multinomial(4000, replacement=True)
    self.assertFalse(equal_distribution(lambda *_: tiny_res, lambda _: torch_res))

  def test_conv2d_init(self):
    params = (128, 256, (3,3))
    assert equal_distribution(lambda *_: nn.Conv2d(*params).weight, lambda _: torch.nn.Conv2d(*params).weight.detach())
    assert equal_distribution(lambda *_: nn.Conv2d(*params).bias, lambda _: torch.nn.Conv2d(*params).bias.detach())

  def test_linear_init(self):
    params = (64, 256)
    assert equal_distribution(lambda *_: nn.Linear(*params).weight, lambda _: torch.nn.Linear(*params).weight.detach())
    assert equal_distribution(lambda *_: nn.Linear(*params).bias, lambda _: torch.nn.Linear(*params).bias.detach())

  def test_bn_init(self):
    params = (64,)
    assert equal_distribution(lambda *_: nn.BatchNorm2d(*params).weight, lambda _: torch.nn.BatchNorm2d(*params).weight.detach())
    assert equal_distribution(lambda *_: nn.BatchNorm2d(*params).bias, lambda _: torch.nn.BatchNorm2d(*params).bias.detach())

  def test_rand_chain(self):
    # NOTE: this fails if property propagates deeper than stack limit
    for _ in range(833): Tensor.rand(1)
    Tensor.rand(1).realize()

# TODO: still fails with MAX_KERNEL_BUFFERS
@unittest.skipIf(Device.DEFAULT == "WEBGPU" and not OSX, "WEBGPU Vulkan can only run kernels with up to 10 buffers")
class TestSample(unittest.TestCase):
  def test_sample(self):
    X = Tensor.rand(10000, 50).realize()
    BS = 16
    idxs = np.random.randint(0, X.shape[0], size=(BS))
    # this uncovered a bug with arg sort order
    batch = [Variable(f'idx{i}', 0, X.shape[0]-1).bind(s) for i,s in enumerate(idxs.tolist())]
    x = Tensor.cat(*[X.shrink(((batch[i], batch[i]+1), None)) for i in range(BS)])
    print(idxs)
    ret = x.numpy()
    base = X.numpy()[idxs]
    np.testing.assert_equal(ret, base)

if __name__ == "__main__":
  unittest.main()
