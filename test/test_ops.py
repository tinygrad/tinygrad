import time, math, unittest, functools
import numpy as np
from typing import List, Callable, Optional, Sequence
import torch
import warnings
from tinygrad.helpers import getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL
from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import is_dtype_supported
from tinygrad.dtype import DTypeLike

if getenv("TINY_BACKEND"):
  import extra.torch_backend.backend # noqa: F401 # pylint: disable=unused-import
  torch.set_default_device("tiny")

if CI:
  warnings.filterwarnings("ignore", message="Non-empty compiler output encountered")

FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)

def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-2, high=2):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op(low, high, shps, vals, forward_only)

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  # move inputs to a different device, test the device of intermediate tensors are correct
  if mt:=getenv("MOVE_TENSOR", ""):
    for t in tst: t.to_(mt)

  st = time.monotonic()
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, tinygrad_output, torch_output, atol, rtol):
    if PRINT_TENSORS: print(s, tinygrad_output, torch_output)
    try:
      assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={tinygrad_output.shape} | torch={torch_output.shape}"
      assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={tinygrad_output.dtype} | torch={torch_output.dtype}"
      if np.issubdtype(tinygrad_output.dtype, np.floating):
        np.testing.assert_allclose(tinygrad_output, torch_output, atol=atol, rtol=rtol)
      else:
        np.testing.assert_equal(tinygrad_output, torch_output)
    except Exception as e:
      raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")

  if DEBUG >= 6:
    np.set_printoptions(linewidth=200, suppress=True)
    print(ret.numpy())
    print(out.detach().cpu().numpy())
  compare("forward pass", ret.numpy(), out.detach().cpu().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY and ts and tst:
    st = time.monotonic()
    torch_grads = torch.autograd.grad(torch_fxn(*ts).sum(), ts)
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    # NOTE: we now have to recompute the forward pass since we realized it
    tiny_grads = tinygrad_fxn(*tst).sum().gradient(*tst)
    Tensor.realize(*tiny_grads)
    tinygrad_fbp = time.monotonic() - st

    for i, (t, torch_grad) in enumerate(zip(tiny_grads, torch_grads)):
      compare(f"backward pass tensor {i}", t.numpy(), torch_grad.detach().cpu().numpy(), atol=grad_atol, rtol=grad_rtol)

  if not CI:
    print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
          (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(low, high, shps, vals, forward_only=False):
  if shps is None:
    ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else:
    np.random.seed(0)
    np_data = [np.random.uniform(low=low, high=high, size=size).astype(_to_np_dtype(dtypes.default_float)) for size in shps]
    ts = [torch.tensor(data, requires_grad=(not forward_only)) for data in np_data]
  for i in range(len(ts)):
    # NOTE: torch default int64 for python ints input
    if ts[i].dtype == torch.int64: ts[i] = ts[i].type(torch.int32)
  tst = [Tensor(x.detach().cpu().numpy(), requires_grad=(not forward_only and not FORWARD_ONLY)) for x in ts]
  return ts, tst

class TestOps(unittest.TestCase):

  def helper_test_exception(self, shps, torch_fxn, tinygrad_fxn, expected, forward_only=False, exact=False, vals=None, low=-1.5, high=1.5):
    if getenv("MOCKGPU") and Device.DEFAULT == "NV": self.skipTest('helper_test_exception fails in CI CUDA')
    ts, tst = prepare_test_op(low, high, shps, vals, forward_only)
    with self.assertRaises(expected) as torch_cm:
      torch_fxn(*ts)
    with self.assertRaises(expected) as tinygrad_cm:
      tinygrad_fxn(*tst)
    if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))
    if not CI: print("\ntesting %40r   torch/tinygrad exception: %s / %s" % (shps, torch_cm.exception, tinygrad_cm.exception), end="")

  def test_full_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

  def test_full(self):
    helper_test_op([], lambda: torch.full((45,65), 4, dtype=torch.int32), lambda: Tensor.full((45,65), 4), forward_only=True)

  def test_negative_dims(self):
    creation_methods: List[Callable[..., Tensor]] = [
      Tensor.empty,
      Tensor.rand,
      Tensor.zeros,
      Tensor.ones,
      Tensor.randn,
      Tensor.randint,
      Tensor.normal,
      Tensor.uniform,
      Tensor.scaled_uniform,
      Tensor.glorot_uniform
    ]

    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 2)
      with self.assertRaises(ValueError): method((2, -3))
      with self.assertRaises(ValueError): method((2, -3, 0))

  def test_negative_dims_full(self):
    with self.assertRaises(ValueError): Tensor.full((-3,), 2)
    with self.assertRaises(ValueError): Tensor.full((2, -3), 4)
    with self.assertRaises(ValueError): Tensor.full((2, -3, 0), 4)

  def test_negative_dims_eye(self):
    with self.assertRaises(ValueError): Tensor.eye(-3, 3)
    with self.assertRaises(ValueError): Tensor.eye(3, -3)
    with self.assertRaises(ValueError): Tensor.eye(-3, -3)

  def test_negative_dims_kaiming(self):
    creation_methods = [Tensor.kaiming_uniform, Tensor.kaiming_normal]
    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 3)
      with self.assertRaises(ValueError): method((-3, 3), 3)
      with self.assertRaises(ValueError): method((-3, -3), 3)

  def test_zeros(self):
    helper_test_op([], lambda: torch.zeros(45,65), lambda: Tensor.zeros(45,65), forward_only=True)
    helper_test_op([], lambda: torch.zeros([45,65]), lambda: Tensor.zeros([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.zeros([]), lambda: Tensor.zeros([]), forward_only=True)

  def test_zeros_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.zeros_like(b), lambda: Tensor.zeros_like(a), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.zeros_like(b), lambda: Tensor.zeros_like(a), forward_only=True)

  def test_empty_0(self):
    helper_test_op([], lambda: torch.empty(45,65)*0/0, lambda: Tensor.empty(45,65)*0/0, forward_only=True)

  def test_ones(self):
    helper_test_op([], lambda: torch.ones(45,65), lambda: Tensor.ones(45,65), forward_only=True)
    helper_test_op([], lambda: torch.ones([45,65]), lambda: Tensor.ones([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.ones([]), lambda: Tensor.ones([]), forward_only=True)

  def test_ones_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.ones_like(b), lambda: Tensor.ones_like(a), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.ones_like(b), lambda: Tensor.ones_like(a), forward_only=True)

  def test_eye(self):
    helper_test_op([], lambda: torch.eye(10), lambda: Tensor.eye(10), forward_only=True)
    helper_test_op([], lambda: torch.eye(3, 5), lambda: Tensor.eye(3, 5), forward_only=True)
    helper_test_op([], lambda: torch.eye(5, 3), lambda: Tensor.eye(5, 3), forward_only=True)
    helper_test_op([], lambda: torch.eye(1), lambda: Tensor.eye(1), forward_only=True)
    helper_test_op([], lambda: torch.eye(0), lambda: Tensor.eye(0), forward_only=True)

  def test_split(self):
    def tensor(s): return torch.arange(math.prod(s), dtype=torch.int32).reshape(s), Tensor.arange(math.prod(s)).reshape(s)
    test_cases = [
      (tensor((10,)),       5, {}),
      (tensor((10,)), [1,4,5], {}),
      (tensor((10,)),       3, {}),
      (tensor((3,4,)),      1, {}),
      (tensor((3,4,)),      1, {'dim':1}),
      (tensor((4,4,)),  [2,2], {}),
      (tensor((4,4,)),  [2,2], {'dim':1}),
      (tensor((10000,)), 2500, {}),
    ]

    for (tor, ten), sizes, args in test_cases:
      tor_splits, ten_splits = tor.split(sizes, **args), ten.split(sizes, **args)
      assert len(tor_splits) == len(ten_splits)
      for tor_chunk, ten_chunk in zip(tor_splits, ten_splits):
        helper_test_op([], lambda: tor_chunk, lambda: ten_chunk, forward_only=True)

  def test_chunk(self):
    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(6, 1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(6, 0)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 0)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(3, -1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(3, -1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 3, 3).chunk(3, -2)
    ten = Tensor.arange(13).repeat((8, 3, 3)).chunk(3, -2)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

  def test_meshgrid(self):
    x, xt = torch.tensor([0.,1.,2.], requires_grad=True), Tensor([0.,1.,2.], requires_grad=True)
    y, yt = torch.tensor([3.,4.,5.,6.], requires_grad=True), Tensor([3.,4.,5.,6.], requires_grad=True)
    z, zt = torch.tensor([7.,8.,9.], requires_grad=True), Tensor([7.,8.,9.], requires_grad=True)
    for indexing in ("ij", "xy"):
      tor = torch.meshgrid(x, indexing=indexing)
      ten = xt.meshgrid(indexing=indexing)
      self.assertEqual(len(tor), len(ten))
      for tor_i, ten_i in zip(tor, ten):
        helper_test_op([], lambda: tor_i, lambda: ten_i)
      tor = torch.meshgrid(x, y, indexing=indexing)
      ten = xt.meshgrid(yt, indexing=indexing)
      self.assertEqual(len(tor), len(ten))
      for tor_i, ten_i in zip(tor, ten):
        helper_test_op([], lambda: tor_i, lambda: ten_i)
      tor = torch.meshgrid(x, torch.tensor(10., requires_grad=True), y, z, indexing=indexing)
      ten = xt.meshgrid(Tensor(10., requires_grad=True), yt, zt, indexing=indexing)
      self.assertEqual(len(tor), len(ten))
      for tor_i, ten_i in zip(tor, ten):
        helper_test_op([], lambda: tor_i, lambda: ten_i)

    self.helper_test_exception([], lambda: torch.meshgrid(x, indexing="bad"), lambda: xt.meshgrid(indexing="bad"), expected=RuntimeError)

  def test_arange(self):
    helper_test_op([], lambda: torch.arange(10, dtype=torch.int32), lambda: Tensor.arange(10), forward_only=True)
    helper_test_op([], lambda: torch.arange(36, dtype=torch.int32), lambda: Tensor.arange(36), forward_only=True)
    helper_test_op([], lambda: torch.arange(5, 10, 3, dtype=torch.int32), lambda: Tensor.arange(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.arange(10, 5, -3, dtype=torch.int32), lambda: Tensor.arange(10, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(11, 5, -3, dtype=torch.int32), lambda: Tensor.arange(11, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(1, 78, 2, dtype=torch.int32), lambda: Tensor.arange(1, 78, 2), forward_only=True)
    helper_test_op([], lambda: torch.arange(5.5, 175.5, 2.5), lambda: Tensor.arange(5.5, 175.5, 2.5), forward_only=True)
    helper_test_op([], lambda: torch.arange(-30.2, -0.3, 0.75), lambda: Tensor.arange(-30.2, -0.3, 0.75), forward_only=True)
    helper_test_op([], lambda: torch.arange(-50.3, -380.2, -2.25), lambda: Tensor.arange(-50.3, -380.2, -2.25), forward_only=True)

  def test_arange_big(self):
    helper_test_op([], lambda: torch.arange(256, dtype=torch.int32), lambda: Tensor.arange(256), forward_only=True)

  def test_arange_4096(self):
    helper_test_op([], lambda: torch.arange(4096, dtype=torch.int32), lambda: Tensor.arange(4096), forward_only=True)

  def test_linspace(self):
    helper_test_op([], lambda: torch.linspace(5, 10, 3), lambda: Tensor.linspace(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 1), lambda: Tensor.linspace(5, 10, 1), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 0), lambda: Tensor.linspace(5, 10, 0), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 30), lambda: Tensor.linspace(5, 10, 30), forward_only=True)
    helper_test_op([], lambda: torch.linspace(-5.5, 5.5, 10), lambda: Tensor.linspace(-5.5, 5.5, 10), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5.5, -5.5, 10), lambda: Tensor.linspace(5.5, -5.5, 10), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 3, dtype=torch.int32), lambda: Tensor.linspace(5, 10, 3, dtype="int32"), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 20, dtype=torch.int32), lambda: Tensor.linspace(5, 10, 20, dtype="int32"), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, -5, 20, dtype=torch.int32), lambda: Tensor.linspace(5, -5, 20, dtype="int32"), forward_only=True)
    self.helper_test_exception([], lambda: torch.linspace(5, 10, 3, dtype=torch.bool), lambda: Tensor.linspace(5, 10, 3, dtype="bool"),
                               expected=(RuntimeError, ValueError))
    self.helper_test_exception([], lambda: torch.linspace(1, 2, -1), lambda: Tensor.linspace(1, 2, -1), expected=(RuntimeError, ValueError))

  def test_sum_fake(self):
    helper_test_op([(256, 1)], lambda x: x.sum(axis=1))

  def test_sum_collapse(self):
    helper_test_op([], lambda: torch.ones(256,256).sum(axis=1), lambda: Tensor.ones(256,256).sum(axis=1), forward_only=True)

  def test_sum_collapse_neg(self):
    helper_test_op([], lambda: (-torch.ones(3,3)).sum(axis=1), lambda: (-Tensor.ones(3,3)).sum(axis=1), forward_only=True)

  def test_sum_pad_collapse(self):
    helper_test_op([], lambda: torch.nn.functional.pad(torch.ones(256,256), pad=(0,64,0,0)).sum(axis=1),
                       lambda: Tensor.ones(256,256).pad(((0,0), (0,64))).sum(axis=1), forward_only=True)

  # this is more complex and won't fold for a while
  def test_sum_cat_collapse(self):
    helper_test_op([], lambda: torch.cat([torch.ones(256,256), torch.zeros(256,64)], dim=1).sum(axis=1),
                       lambda: Tensor.cat(Tensor.ones(256,256), Tensor.zeros(256,64), dim=1).sum(axis=1), forward_only=True)

  def test_max_dont_collapse(self):
    helper_test_op([], lambda: torch.ones(256,256).max(1)[0], lambda: Tensor.ones(256,256).max(1), forward_only=True)

  def test_where(self):
    helper_test_op(
      [(100,)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32),
      lambda x: (x > 0.5).where(4, 2), forward_only=True)

    for shps in [[(8,),(1,),(1,)], [(10,10),(10,),(10,)], [(100,)]*3, [(10,10)]*3]:
      helper_test_op(
        shps,
        lambda x, a, b: torch.where(x > 0.5, a, b),
        lambda x, a, b: (x > 0.5).where(a, b), forward_only=True)

  def test_where_permute(self):
    helper_test_op(
      [(5, 5)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32).permute((1, 0)),
      lambda x: (x > 0.5).where(4, 2).permute((1, 0)), forward_only=True)

  def _test_cmp(self, fxn, reverse=True):
    # test different dtypes
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0,1,2], [2,1,0]])
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[True, True, False], [False,True,False]])
    # test broadcasting
    for shps in [[(3, 4, 5), (3, 4, 5)], [(3, 4, 5), (5,)], [(5,), (3, 4, 5)]]:
      helper_test_op(shps, fxn, fxn, forward_only=True)
    # test cmp with const
    helper_test_op(None, lambda x,y: fxn(x,2), lambda x,y: fxn(x,2), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    if reverse: helper_test_op(None, lambda x,y: fxn(2,y), lambda x,y: fxn(2,y), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    # test special floats  # TODO: fix nan
    specials = [0.0, 1.0, -1.0, math.inf, -math.inf]#, math.nan]
    for s0 in specials:
      for s1 in specials:
        helper_test_op(None, fxn, fxn, forward_only=True, vals=[[s0], [s1]])

  def test_cmp_eq(self): self._test_cmp(lambda x,y: x==y, reverse=False)
  def test_cmp_gt(self): self._test_cmp(lambda x,y: x>y)
  def test_cmp_ge(self): self._test_cmp(lambda x,y: x>=y)
  def test_cmp_lt(self): self._test_cmp(lambda x,y: x<y)
  def test_cmp_le(self): self._test_cmp(lambda x,y: x<=y)

  def test_cmp_ne_backwards(self):
    # new grad zeroes these out
    """
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 != t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 != tt2).sum().backward)
    """
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt != 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t != 0)).sum().backward()
    np.testing.assert_allclose(t.grad.numpy(), tt.grad.numpy(), rtol=1e-5)

  def test_cmp_lt_backwards(self):
    # new grad zeroes these out
    """
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 < t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 < tt2).sum().backward)
    """
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt < 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t < 0)).sum().backward()
    np.testing.assert_allclose(t.grad.numpy(), tt.grad.numpy(), rtol=1e-5)

  # TODO: fix backward of these functions
  def test_trunc(self):
    helper_test_op([()], lambda x: x.trunc(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.trunc(), forward_only=True)
    helper_test_op(None, lambda x: x.trunc(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
  def test_floor(self):
    helper_test_op([()], lambda x: x.floor(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.floor(), forward_only=True)
    helper_test_op(None, lambda x: x.floor(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
  def test_ceil(self):
    helper_test_op([()], lambda x: x.ceil(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.ceil(), forward_only=True)
    helper_test_op(None, lambda x: x.ceil(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
  def test_round(self):
    helper_test_op([()], lambda x: x.round(), forward_only=True)
    helper_test_op([(45,35)], lambda x: x.round(), forward_only=True)
    helper_test_op(None, lambda x: x.round(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
    helper_test_op(None, lambda x: x.round(), vals=[[2.5, -1.5]], forward_only=True)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU" and CI, "isinf check of 'nan' fails on CI software-based vulkan")
  def test_isinf(self):
    val = [float('-inf'), 0., float('inf'), float('nan'), 1.1]
    helper_test_op(None, torch.isinf, Tensor.isinf, vals=[val], forward_only=True)
    np.testing.assert_equal(Tensor(val).isinf(detect_positive=True, detect_negative=False).numpy(), [False, False, True, False, False])
    np.testing.assert_equal(Tensor(val).isinf(detect_positive=False, detect_negative=True).numpy(), [True, False, False, False, False])

  def test_isnan(self):
    helper_test_op(None, torch.isnan, Tensor.isnan, vals=[[float('-inf'), 0., float('inf'), float('nan'), 1.1]], forward_only=True)

  def test_lerp(self):
    helper_test_op([(45,35), (45,35), (45,35)], lambda x,y,z: x.lerp(y,z))
    helper_test_op(None, lambda x,y,z: x.lerp(y,z), vals=[[1.,2.,3.], [4.,5.,6.], 0.5])

  @unittest.skipIf(Device.DEFAULT == "QCOM", "OpenCL fails to compile this (both on GPU(qcom)/QCOM backends)")
  def test_tril(self):
    helper_test_op([(3,3)], lambda x: x.tril())
    helper_test_op([(3,3)], lambda x: x.tril(1))
    helper_test_op([(3,3)], lambda x: x.tril(2))
    helper_test_op([(3,3)], lambda x: x.tril(-1))
    helper_test_op([(3,3)], lambda x: x.tril(-2))
    helper_test_op([(4,5)], lambda x: x.tril(4))
    helper_test_op([(4,5)], lambda x: x.tril(5))
    helper_test_op([(4,5)], lambda x: x.tril(6))
    helper_test_op([(4,5)], lambda x: x.tril(-4))
    helper_test_op([(4,5)], lambda x: x.tril(-5))
    helper_test_op([(4,5)], lambda x: x.tril(-6))
    helper_test_op([(5,3,3)], lambda x: x.tril())
    helper_test_op([(5,0,3)], lambda x: x.tril())
    helper_test_op([(5,3,3)], lambda x: x.tril(1))
    helper_test_op(None, lambda x: x.tril(), vals=[[[True] * 3] * 3], forward_only=True)

  @unittest.skipIf(Device.DEFAULT == "QCOM", "OpenCL fails to compile this (both on GPU(qcom)/QCOM backends)")
  def test_triu(self):
    helper_test_op([(3,3)], lambda x: x.triu())
    helper_test_op([(3,3)], lambda x: x.triu(1))
    helper_test_op([(3,3)], lambda x: x.triu(2))
    helper_test_op([(3,3)], lambda x: x.triu(-1))
    helper_test_op([(3,3)], lambda x: x.triu(-2))
    helper_test_op([(4,5)], lambda x: x.triu(4))
    helper_test_op([(4,5)], lambda x: x.triu(5))
    helper_test_op([(4,5)], lambda x: x.triu(6))
    helper_test_op([(4,5)], lambda x: x.triu(-4))
    helper_test_op([(4,5)], lambda x: x.triu(-5))
    helper_test_op([(4,5)], lambda x: x.triu(-6))
    helper_test_op([(5,3,3)], lambda x: x.triu())
    helper_test_op([(5,0,3)], lambda x: x.triu())
    helper_test_op([(5,3,3)], lambda x: x.triu(1))
    helper_test_op(None, lambda x: x.triu(), vals=[[[True] * 3] * 3], forward_only=True)

  def test_maximum(self):
    helper_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
    helper_test_op([(), ()], torch.maximum, Tensor.maximum)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., -4.], 3.])
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    helper_test_op(None, torch.maximum, Tensor.maximum,
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.max(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum,
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.min(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 3], forward_only=True)

  def test_minimum(self):
    helper_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
    helper_test_op([(), ()], torch.minimum, Tensor.minimum)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], 3.])
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.max(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.max(dtypes.int), dtypes.min(dtypes.int)], dtypes.min(dtypes.int)], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 3], forward_only=True)

  def test_tiny_add(self):
    helper_test_op([(3), (3)], lambda x,y: x+y, Tensor.add, forward_only=True)
  def test_tiny_mul(self):
    helper_test_op([(64), (64)], lambda x,y: x*y, Tensor.mul, forward_only=True)

  def test_add(self):
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y)
    helper_test_op([(), ()], lambda x,y: x+y)
  def test_add3(self):
    helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: x+y+z)
  def test_broadcasted_add(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x+y)
    helper_test_op([(45,65), ()], lambda x,y: x+y)
  def test_broadcasted_add_2(self):
    helper_test_op([(45,65), (65,)], lambda x,y: x+y)

  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub)
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y)
    helper_test_op([(), ()], lambda x,y: x-y)
  def test_scalar_sub(self):
    helper_test_op([(45,65)], lambda x: x-2)
    helper_test_op([()], lambda x: x-2)
  def test_scalar_rsub(self):
    helper_test_op([(45,65)], lambda x: 2-x)
    helper_test_op([()], lambda x: 2-x)

  def test_neg(self):
    helper_test_op([(45,65)], lambda x: -x)
    helper_test_op([(45,65)], lambda x: x.neg())
    helper_test_op([()], lambda x: x.neg())
  def test_logical_not(self):
    helper_test_op(None, torch.logical_not, Tensor.logical_not, vals=[[True, False, True]], forward_only=True)
    helper_test_op(None, torch.logical_not, Tensor.logical_not, vals=[[1.,2.,0.,0.5]], forward_only=True)

  def test_mul(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y, Tensor.mul)
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y)
    helper_test_op([(), ()], lambda x,y: x*y)
  def test_scalar_mul(self):
    helper_test_op([(45,65)], lambda x: x*2)
    helper_test_op([(45,65)], lambda x: x*-1)
    helper_test_op([(45,65)], lambda x: 255*x)
    helper_test_op([(45,65)], lambda x: 2*x)
    helper_test_op([()], lambda x: x*2)
    helper_test_op([()], lambda x: 2*x)

  def test_div(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y)
    helper_test_op([(), ()], lambda x,y: x/y)

  def test_div_rounding_mode(self):
    for denominator in [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]:
      # int numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      # float numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])

    for denominator in [-10.0, -5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0, 10.0]:
      # int numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      # float numerator
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])

    self.helper_test_exception(None, lambda x,y: x.div(y, rounding_mode="typo"), lambda x,y: x.div(y, rounding_mode="typo"), forward_only=True,
                               vals=[[5], [0]], expected=RuntimeError)

  def test_div_int(self):
    helper_test_op(None, lambda x,y: x/y, Tensor.div, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x,y: x//y, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x: x/2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, lambda x: x//2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, functools.partial(torch.div, rounding_mode="trunc"), Tensor.idiv, forward_only=True,
                   vals=[[-4, 7, 5, 4, -7, 8], [2, -3, 8, -2, 3, 5]])
    if is_dtype_supported(dtypes.uint64):
      x = Tensor(2**64 - 1, dtype=dtypes.uint64).idiv(1)
      np.testing.assert_equal(x.numpy(), 2**64 - 1)
    # 1 // 0 is device dependent, but it should not raise
    Tensor([1]).idiv(1).realize()
    if not (CI and (Device.DEFAULT=="LLVM" or getenv("PTX"))):  # TODO: crashed in CI
      # ... because if might be in a where branch that the output is well defined
      t = Tensor([-1, 0, 1, 2])
      np.testing.assert_equal((t > 0).where(1//t, t).numpy(), [-1, 0, 1, 0])

  def test_scalar_div(self):
    helper_test_op([(45,65)], lambda x: x/255)
    helper_test_op([(45,65)], lambda x: x/1)
    helper_test_op([(45,65)], lambda x: 1/x)
    helper_test_op([(45,65)], lambda x: x/2)
    helper_test_op([(45,65)], lambda x: 2/x)
    helper_test_op([()], lambda x: x/2)
    helper_test_op([()], lambda x: 2/x)

  def test_mod(self):
    a = [-4, 7, 5, 4, -7, 8]
    b = [2, -3, 8, -2, 3, 5]
    for float_a in [True, False]:
      for float_b in [True, False]:
        va = [float(ai) for ai in a] if float_a else a
        vb = [float(bi) for bi in b] if float_b else b
        helper_test_op(None, lambda x,y: x%y, Tensor.mod, forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x,y: x%y, forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x: x%2, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x%3, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x%3.5, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100%x, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100.5%x, forward_only=True, vals=[va])

  def test_mul_naninf(self):
    helper_test_op([(45,65)], lambda x: x*math.inf)
    helper_test_op([(45,65)], lambda x: x*-math.inf)
    helper_test_op([(45,65)], lambda x: x*math.nan)
  def test_div_naninf(self):
    helper_test_op([(45,65)], lambda x: x/math.inf)
    helper_test_op([(45,65)], lambda x: x/-math.inf)
    helper_test_op([(45,65)], lambda x: x/math.nan)
    helper_test_op([(45,65)], lambda x: math.inf/x)
    helper_test_op([(45,65)], lambda x: (-math.inf)/x)
    helper_test_op([(45,65)], lambda x: math.nan/x)

  def test_pow_full(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y)
    helper_test_op([(45,65), (45,65)], lambda x,y: x.pow(y))

  def test_pow(self):
    helper_test_op([(45,65)], lambda x: x**0)
    helper_test_op([(45,65)], lambda x: x**1)
    helper_test_op([(45,65)], lambda x: x**2)
    helper_test_op([(45,65)], lambda x: x**3)
    helper_test_op([(45,65)], lambda x: x**-2)
    helper_test_op([()], lambda x: x**2)
    helper_test_op([()], lambda x: x**-2)
    # Regression tests for https://github.com/tinygrad/tinygrad/issues/1151
    helper_test_op([(45,65)], lambda x: x**3, low=-30, high=-27)
    helper_test_op([()], lambda x: x**3, low=-30, high=-27)
    # Regression tests for https://github.com/tinygrad/tinygrad/issues/1251
    helper_test_op([(45,65)], lambda x: x**0.2, low=-30, high=-27)
    helper_test_op([(45,65)], lambda x: x**1.2, low=-30, high=-27)
    helper_test_op([()], lambda x: x**0.2, low=-30, high=-27)
    helper_test_op([()], lambda x: x**1.2, low=-30, high=-27)
    a, b = Tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
    helper_test_op([], lambda: b**1.1, lambda: a**1.1)

  def test_pow_const(self):
    helper_test_op([(45,65)], lambda x: x**0.0)
    helper_test_op([(45,65)], lambda x: x**1.0)
    helper_test_op([(45,65)], lambda x: x**-1.0)
    helper_test_op([(45,65)], lambda x: x**8.0)
    helper_test_op([(45,65)], lambda x: x**5.5)
    helper_test_op([(45,65)], lambda x: x**-5.5)
    # helper_test_op([(45,65)], lambda x: x**-8.0)  # TODO: fix this
    helper_test_op([(45,65)], lambda x: 1.0**x)
    helper_test_op([(45,65)], lambda x: 5.5**x)
    helper_test_op([(45,65)], lambda x: (-5.5)**x)
    helper_test_op([(45,65)], lambda x: 8.0**x)
    helper_test_op([(45,65)], lambda x: x**2.0)
    helper_test_op([(45,65)], lambda x: 2.0**x)
    helper_test_op([()], lambda x: x**2.0)
    helper_test_op([()], lambda x: 2.0**x)
    helper_test_op(None, lambda x: 0**x, vals=[[-2.,-1,0,1,2,3]])
    helper_test_op(None, lambda x: (-2)**x, vals=[[-2.,-1,0,1,2,3]])

  def test_pow_const_direct(self):
    # x ** c
    def get_tiny_gradient(x, c):
      t = Tensor([x], dtype=dtypes.float)
      return (t ** c)[0].gradient(t)[0].item()
    def get_torch_gradient(x, c):
      t = torch.tensor([x], dtype=torch.float, requires_grad=True)
      return torch.autograd.grad(t ** c, t)[0].item()
    for x in [-math.inf, 0, 1, math.inf]:
      for c in [-1, 0, 0.3, 1, 2]:
        tiny_out = get_tiny_gradient(x, c)
        torch_out = get_torch_gradient(x, c)
        if math.isnan(tiny_out):
          assert math.isnan(torch_out)
        else:
          self.assertAlmostEqual(tiny_out, torch_out, msg=f"{x}, {c}")

  def test_pow_zero_tensor(self):
    helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.0]])
    # TODO: fix WEBGPU
    if Device.DEFAULT != "WEBGPU":
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.3]])
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [-0.3]])
  def test_pow_zero_const(self):
    helper_test_op(None, lambda x: x**0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**0.0, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-1.0, vals=[[-1.0, 0.0, 1.0]])

  @unittest.skip("not supported")
  def test_pow_int(self):
    def _test(base, exponent): helper_test_op(None, lambda x,y: x**y, vals=[base, exponent], forward_only=True)

    for base in ([1, 2, 3], [-1, -2, -3]):
      for exponent in ([2, 3, 4], [-2, -3, -4]):
        _test(base, exponent)
    # NOTE: torch 0 ** -1 is 0
    _test([0, 0, 0], [0, 1, 2])

    np.testing.assert_equal((Tensor(11) ** Tensor(7)).item(), 11 ** 7)
    np.testing.assert_equal((Tensor([11]) ** Tensor(7)).item(), 11 ** 7)
    # TODO: fix non-precise int pow
    with self.assertRaises(AssertionError): np.testing.assert_equal((Tensor(11) ** Tensor([7])).item(), 11 ** 7)
    with self.assertRaises(AssertionError): np.testing.assert_equal((Tensor([11]) ** Tensor([7])).item(), 11 ** 7)

    # pow to a const int
    helper_test_op([], lambda: torch.tensor([2], dtype=torch.int) ** torch.tensor(-2, dtype=torch.int),
                       lambda: Tensor([2]) ** Tensor(-2), forward_only=True)

  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt())
    helper_test_op(None, lambda x: x.sqrt(), vals=[[0.0]])
    helper_test_op([()], lambda x: x.sqrt())
  def test_rsqrt(self):
    helper_test_op([(45,65)], lambda x: x.rsqrt())
    helper_test_op(None, lambda x: x.rsqrt(), vals=[[0.0]])
    helper_test_op([()], lambda x: x.rsqrt())

  def test_xor(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor^tor, lambda: ten^ten, forward_only=True)
    helper_test_op([], lambda: tor^0x1337, lambda: ten^0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337^tor, lambda: 0x1337^ten, forward_only=True)

    self.helper_test_exception([(4), (4)], torch.bitwise_xor, Tensor.bitwise_xor, expected=RuntimeError)

  def test_and(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor&tor, lambda: ten&ten, forward_only=True)
    helper_test_op([], lambda: tor&0x1337, lambda: ten&0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337&tor, lambda: 0x1337&ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0&tor1, lambda: ten0&ten1, forward_only=True)

    helper_test_op(None, lambda x: (1 < x) & (x < 2), forward_only=True, vals=[[1.2, 1.2, 1.2, 3.2]])

    self.helper_test_exception([(4), (4)], torch.bitwise_and, Tensor.bitwise_and, expected=RuntimeError)

  def test_or(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor|tor, lambda: ten|ten, forward_only=True)
    helper_test_op([], lambda: tor|0x1337, lambda: ten|0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337|tor, lambda: 0x1337|ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0|tor1, lambda: ten0|ten1, forward_only=True)

    self.helper_test_exception([(4), (4)], torch.bitwise_or, Tensor.bitwise_or, expected=RuntimeError)

  def test_bitwise_not(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor.bitwise_not(), lambda: ten.bitwise_not(), forward_only=True)
    helper_test_op([], lambda: ~tor, lambda: ~ten, forward_only=True)

    data = [[True, False]]
    tor = torch.tensor(data, dtype=torch.bool)
    ten = Tensor(data, dtype=dtypes.bool)
    helper_test_op([], lambda: tor.bitwise_not(), lambda: ten.bitwise_not(), forward_only=True)
    helper_test_op([], lambda: ~tor, lambda: ~ten, forward_only=True)

    self.helper_test_exception([(4)], torch.bitwise_not, Tensor.bitwise_not, expected=RuntimeError)

  def test_lshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    # cast to int32 because torch does not support uint32
    helper_test_op([], lambda: tor << 0, lambda: (ten << 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 2, lambda: (ten << 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 31, lambda: (ten << 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.__lshift__(2), lambda: ten.__lshift__(2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_left_shift(2), lambda: ten.lshift(2).cast(dtypes.int32), forward_only=True)

  def test_rshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    # cast to int32 because torch does not support uint32
    helper_test_op([], lambda: tor >> 0, lambda: (ten >> 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 2, lambda: (ten >> 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 31, lambda: (ten >> 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.__rshift__(2), lambda: ten.__rshift__(2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_right_shift(2), lambda: ten.rshift(2).cast(dtypes.int32), forward_only=True)

  def test_idiv_shift_rewrite_negative(self):
    a = Tensor(-5).idiv(2).item()
    b = Tensor(-5).contiguous().idiv(2).item()
    self.assertEqual(a, b)
    self.assertEqual(Tensor(-1).contiguous().idiv(4).item(), 0)  # NOTE this is trunc-div behaviour

  def test_sin(self):
    helper_test_op([(45,65)], lambda x: x.sin())
    helper_test_op([()], lambda x: x.sin())
    # works on real CUDA but not CI
    if not ((getenv("MOCKGPU") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.sin(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.sin(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)
  def test_cos(self):
    helper_test_op([(45,65)], lambda x: x.cos())
    helper_test_op([()], lambda x: x.cos())
    if not ((getenv("MOCKGPU") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.sin(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.cos(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)
  def test_tan(self):
    # NOTE: backward has much higher diff with input close to pi/2 and -pi/2
    helper_test_op([(45,65)], lambda x: x.tan(), low=-1.5, high=1.5)
    helper_test_op([(45,65)], lambda x: x.tan(), low=-5, high=5)
    helper_test_op([()], lambda x: x.tan())
    if not ((getenv("MOCKGPU") and Device.DEFAULT == "NV") or Device.DEFAULT == "WEBGPU"):
      helper_test_op(None, lambda x: x.sin(), vals=[[math.nan, math.inf, -math.inf, 0.0]])
      helper_test_op(None, lambda x: x.cos(), vals=[[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6]],
                    atol=3e-3, rtol=3e-3, grad_atol=3e-3, grad_rtol=3e-3)

  def test_asin(self):
    helper_test_op([(45,65)], lambda x: x.asin(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.asin(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.asin(), low=300, high=303)
  def test_acos(self):
    # high grad atol
    helper_test_op([(45,65)], lambda x: x.acos(), low=-1, high=1)
    helper_test_op([(45,65)], lambda x: x.acos(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acos(), low=300, high=303)
  def test_atan(self):
    helper_test_op([(45,65)], lambda x: x.atan())
    helper_test_op([(45,65)], lambda x: x.atan(), low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.atan(), low=300, high=303)

  def test_relu(self):
    helper_test_op([(64,64)], lambda x: x.relu())
    helper_test_op([()], lambda x: x.relu())
  def test_relu_exact(self):
    helper_test_op(None, lambda x: x.relu(), vals=[[-1.,0,1]])
  def test_relu_maximum_exact(self):
    helper_test_op(None, lambda x: torch.maximum(x, torch.zeros_like(x, requires_grad=False)), lambda x: Tensor.maximum(x, 0), vals=[[-1.,0,1]])
  def test_leaky_relu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leaky_relu)
    helper_test_op([()], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leaky_relu)
  def test_celu(self):
    for val in range(1, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
      helper_test_op([()], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
  def test_selu(self):
    helper_test_op([(45,65)], torch.nn.functional.selu, Tensor.selu)
    helper_test_op([()], torch.nn.functional.selu, Tensor.selu)
  def test_silu(self):
    helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.silu)
    helper_test_op([()], torch.nn.functional.silu, Tensor.silu)
  def test_swish(self):
    helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.swish)
    helper_test_op([()], torch.nn.functional.silu, Tensor.swish)

  def test_abs(self):
    helper_test_op([(45,65)], torch.abs, Tensor.abs)
    helper_test_op([()], torch.abs, Tensor.abs)
  def test_abs_exact(self):
    helper_test_op(None, torch.abs, Tensor.abs, vals=[[-1.,0,1]])

  @unittest.skipIf(TRANSCENDENTAL and Device.DEFAULT=="AMD", "TODO: remu crashes")
  def test_log(self):
    helper_test_op([(45,65)], torch.log, Tensor.log)
    helper_test_op(None, torch.log, Tensor.log, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.log, Tensor.log)
  def test_log2(self):
    helper_test_op([(45,65)], torch.log2, Tensor.log2)
    helper_test_op(None, torch.log2, Tensor.log2, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.log2, Tensor.log2)

  @unittest.skipIf(TRANSCENDENTAL and Device.DEFAULT=="AMD", "TODO: remu crashes")
  def test_exp(self):
    helper_test_op([(45,65)], torch.exp, Tensor.exp)
    helper_test_op(None, torch.exp, Tensor.exp, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.exp, Tensor.exp)
  def test_exp2(self):
    helper_test_op([(45,65)], torch.exp2, Tensor.exp2)
    helper_test_op(None, torch.exp2, Tensor.exp2, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.exp2, Tensor.exp2)

  def test_sign(self):
    helper_test_op([(45,65)], torch.sign, Tensor.sign)
    helper_test_op([()], torch.sign, Tensor.sign)
  def test_sign_exact(self):
    helper_test_op(None, torch.sign, Tensor.sign, vals=[[-1.,0,1]])

  def test_softsign(self):
    helper_test_op([(45,65)], torch.nn.functional.softsign, Tensor.softsign)
    helper_test_op([()], torch.nn.functional.softsign, Tensor.softsign)
  def test_softsign_exact(self):
    helper_test_op(None, torch.nn.functional.softsign, Tensor.softsign, vals=[[-1.,0,1]])

  def test_sigmoid(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid)
    helper_test_op([()], torch.sigmoid, Tensor.sigmoid)
  def test_sigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)
  def test_sigmoid_alt_extreme(self):
    def sigmoid(x:Tensor): return x.exp() / (1 + x.exp())
    x = Tensor([300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)
    x = Tensor([-300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)
  def test_hardsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
    helper_test_op([()], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
  def test_hardsigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)
  def test_softplus(self):
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=3), lambda t: Tensor.softplus(t, beta=3), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda t: torch.nn.functional.softplus(t, beta=1/3), lambda t: Tensor.softplus(t, beta=1/3), grad_atol=1e-6)
    # # TODO: support threshold and enable this
    # helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=300, high=400)
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6, low=-400, high=-300)
    helper_test_op([()], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)

  def test_erf(self):
    helper_test_op([(45,65)], torch.erf, Tensor.erf)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=300, high=400)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=-400, high=-300)
    helper_test_op([()], torch.erf, Tensor.erf)

  def test_gelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu)
  def test_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, low=300, high=400)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, low=-400, high=-300)
  def test_quick_gelu(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
    helper_test_op([()], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
  def test_quick_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=300, high=400)
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=-400, high=-300)

  def test_elu(self):
    helper_test_op([(45,65)], torch.nn.functional.elu, Tensor.elu)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.elu(x, alpha=0.1), lambda x: Tensor.elu(x, alpha=0.1))
    helper_test_op([()], torch.nn.functional.elu, Tensor.elu)
  def test_relu6(self):
    helper_test_op([(45,65)], torch.nn.functional.relu6, Tensor.relu6)
    helper_test_op([()], torch.nn.functional.relu6, Tensor.relu6)
  def test_hardswish(self):
    helper_test_op([(45,65)], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)
    helper_test_op([()], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)
  def test_mish(self):
    helper_test_op([(45,65)], torch.nn.functional.mish, Tensor.mish)
    helper_test_op([()], torch.nn.functional.mish, Tensor.mish)

  def test_topk(self):
    # test with k=2, dim=1, largest=True (default)
    helper_test_op([(5,5)], lambda x: torch.topk(x, k=2, dim=1)[0], lambda x: x.topk(k=2, dim=1)[0], forward_only=True)
    helper_test_op([(5,5)], lambda x: torch.topk(x, k=2, dim=1)[1].type(torch.int32), lambda x: x.topk(k=2, dim=1)[1], forward_only=True)
    
    # test with k=2, dim=1, largest=False
    helper_test_op([(5,5)], lambda x: torch.topk(x, k=2, dim=1, largest=False)[0], lambda x: x.topk(k=2, dim=1, largest=False)[0], forward_only=True)
    helper_test_op([(5,5)], lambda x: torch.topk(x, k=2, dim=1, largest=False)[1].type(torch.int32), lambda x: x.topk(k=2, dim=1, largest=False)[1], forward_only=True)
    
    # test with specific values
    vals = Tensor([[1, 3, 2], [5, 4, 0]])
    ret_vals, ret_idxs = vals.topk(k=2, dim=1)
    assert ret_vals.numpy().tolist() == [[3, 2], [5, 4]]
    assert ret_idxs.numpy().tolist() == [[1, 2], [0, 1]]
