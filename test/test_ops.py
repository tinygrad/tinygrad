import torch
import time
import math
import numpy as np
import unittest
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, IMAGE, DEBUG, CI, dtypes, Context, NOOPT
from tinygrad.ops import Device

if CI:
  import warnings
  warnings.filterwarnings("ignore", message="Non-empty compiler output encountered")

FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, forward_only=False, vals=None, a=-0.5, b=3):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op(a, b, shps, vals, forward_only)

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  st = time.monotonic()
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, x,y,atol,rtol):
    if PRINT_TENSORS: print(s, x, y)
    assert x.shape == y.shape, f"shape mismatch: tinygrad={x.shape} | torch={y.shape}"
    try:
      np.testing.assert_allclose(x,y, atol=atol, rtol=rtol)
    except Exception:
      raise Exception(f"{s} failed shape {x.shape}")

  if DEBUG >= 6:
    np.set_printoptions(linewidth=200, suppress=True)
    print(ret.numpy())
    print(out.detach().numpy())
  compare("forward pass", ret.numpy(), out.detach().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY:
    st = time.monotonic()
    (out+1).square().mean().backward()
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    (ret+1).square().mean().backward()
    for tt in tst: tt.grad.realize()
    tinygrad_fbp = time.monotonic() - st

    for i, (t, tt) in enumerate(zip(ts, tst)):
      compare(f"backward pass tensor {i}", tt.grad.numpy(), t.grad.detach().numpy(), atol=grad_atol, rtol=grad_rtol)

  if not CI: print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(a, b, shps, vals, forward_only=False):
  torch.manual_seed(0)
  np.random.seed(0)
  if shps is None: ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else: ts = [torch.tensor((np.random.random(size=x) + a) * b, requires_grad=(not forward_only), dtype=torch.float32) for x in shps]
  tst = [Tensor(x.detach().numpy(), requires_grad=(not forward_only and not FORWARD_ONLY)) for x in ts]
  return ts, tst

class TestOps(unittest.TestCase):

  def helper_test_exception(self, shps, torch_fxn, tinygrad_fxn, expected, exact=False, vals=None, a=-0.5, b=3):
    ts, tst = prepare_test_op(a, b, shps, vals)
    with self.assertRaises(expected) as torch_cm:
      torch_fxn(*ts)
    with self.assertRaises(expected) as tinygrad_cm:
      tinygrad_fxn(*tst)
    if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))
    if not CI: print("\ntesting %40r   torch/tinygrad exception: %s / %s" % (shps, torch_cm.exception, tinygrad_cm.exception), end="")

  def test_full_like(self):
    a = Tensor([[1,2,3],[4,5,6]])
    b = torch.tensor([[1,2,3],[4,5,6]])
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)
  def test_full(self):
    helper_test_op([], lambda: torch.full((45,65), 4), lambda: Tensor.full((45,65), 4), forward_only=True)
  def test_zeros(self):
    helper_test_op([], lambda: torch.zeros(45,65), lambda: Tensor.zeros(45,65), forward_only=True)
    helper_test_op([], lambda: torch.zeros([45,65]), lambda: Tensor.zeros([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.zeros([]), lambda: Tensor.zeros([]), forward_only=True)
  def test_zeros_like(self):
    a = Tensor([[1,2,3],[4,5,6]])
    b = torch.tensor([[1,2,3],[4,5,6]])
    helper_test_op([], lambda: torch.zeros_like(b), lambda: Tensor.zeros_like(a), forward_only=True)
  def test_empty_0(self):
    helper_test_op([], lambda: torch.empty(45,65)*0/0, lambda: Tensor.empty(45,65)*0/0, forward_only=True)
  def test_ones(self):
    helper_test_op([], lambda: torch.ones(45,65), lambda: Tensor.ones(45,65), forward_only=True)
    helper_test_op([], lambda: torch.ones([45,65]), lambda: Tensor.ones([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.ones([]), lambda: Tensor.ones([]), forward_only=True)
  def test_ones_like(self):
    a = Tensor([[1,2,3],[4,5,6]])
    b = torch.tensor([[1,2,3],[4,5,6]])
    helper_test_op([], lambda: torch.ones_like(b), lambda: Tensor.ones_like(a), forward_only=True)
  def test_eye(self):
    helper_test_op([], lambda: torch.eye(10), lambda: Tensor.eye(10), forward_only=True)
    helper_test_op([], lambda: torch.eye(1), lambda: Tensor.eye(1), forward_only=True)

  def test_chunk(self):
    tor = torch.arange(13).repeat(8, 1).chunk(6, 1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13).repeat(8, 1).chunk(6, 0)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 0)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13).repeat(8, 1).chunk(3, -1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(3, -1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13).repeat(8, 3, 3).chunk(3, -2)
    ten = Tensor.arange(13).repeat((8, 3, 3)).chunk(3, -2)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

  def test_arange(self):
    helper_test_op([], lambda: torch.arange(10), lambda: Tensor.arange(10), forward_only=True)
    helper_test_op([], lambda: torch.arange(5, 10, 3), lambda: Tensor.arange(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.arange(10, 5, -3), lambda: Tensor.arange(10, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(11, 5, -3), lambda: Tensor.arange(11, 5, -3), forward_only=True)
  def test_arange_simple(self):
    helper_test_op([], lambda: torch.arange(10), lambda: Tensor.arange(10), forward_only=True)
  def test_arange_big(self):
    helper_test_op([], lambda: torch.arange(256), lambda: Tensor.arange(256), forward_only=True)

  def test_where(self):
    helper_test_op(
      [(100,)],
      lambda x: torch.where(x > 0.5, 4, 2),
      lambda x: (x > 0.5).where(4, 2), forward_only=True)

    for shps in [[(8,),(1,),(1,)], [(10,10),(10,),(10,)], [(100,)]*3, [(10,10)]*3]:
      helper_test_op(
        shps,
        lambda x, a, b: torch.where(x > 0.5, a, b),
        lambda x, a, b: (x > 0.5).where(a, b), forward_only=True)

  def test_where_permute(self):
    helper_test_op(
      [(5, 5)],
      lambda x: torch.where(x > 0.5, 4, 2).permute((1, 0)),
      lambda x: (x > 0.5).where(4, 2).permute((1, 0)), forward_only=True)

  def _test_cmp(self, fxn, reverse=True):
    for shps in [[(3, 4, 5), (3, 4, 5)], [(3, 4, 5), (5,)], [(5,), (3, 4, 5)]]:
      helper_test_op(shps, fxn, fxn, forward_only=True)
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    helper_test_op(None, lambda x,y: fxn(x,2), lambda x,y: fxn(x,2), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[True, True, False], [False,True,False]])
    if reverse: helper_test_op(None, lambda x,y: fxn(2,y), lambda x,y: fxn(2,y), forward_only=True, vals=[[0.,1,2], [2.,1,0]])

  def test_cmp_eq(self): self._test_cmp(lambda x,y: x==y, reverse=False)
  def test_cmp_gt(self): self._test_cmp(lambda x,y: x>y)
  def test_cmp_ge(self): self._test_cmp(lambda x,y: x>=y)
  def test_cmp_lt(self): self._test_cmp(lambda x,y: x<y)
  def test_cmp_le(self): self._test_cmp(lambda x,y: x<=y)

  def test_cmp_eq_backwards(self):
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 == t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 == tt2).sum().backward)

  def test_cmp_lt_backwards(self):
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 < t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 < tt2).sum().backward)

  #@unittest.skip("this is broken with contiguous")
  def test_trunc(self):
    helper_test_op([(45,65)], lambda x: torch.trunc(x), lambda x: x.trunc(), forward_only=True)
    a, b = Tensor([1.0, 2.1, 0.0, -5.0, -2.5]), torch.tensor([1.0, 2.1, 0.0, -5.0, -2.5])
    helper_test_op([], lambda: torch.trunc(b), lambda: Tensor.trunc(a), forward_only=True)
  #@unittest.skip("this is broken with contiguous")
  def test_floor(self):
    helper_test_op([(45,65)], lambda x: torch.floor(x), lambda x: x.floor(), forward_only=True)
    a, b = Tensor([1.0, 2.1, 0.0, -5.0, -2.5]), torch.tensor([1.0, 2.1, 0.0, -5.0, -2.5])
    helper_test_op([], lambda: torch.floor(b), lambda: Tensor.floor(a), forward_only=True)
  #@unittest.skip("this is broken with contiguous")
  def test_ceil(self):
    helper_test_op([(45,65)], lambda x: torch.ceil(x), lambda x: x.ceil(), forward_only=True)
    a, b = Tensor([1.0, 2.1, 0.0, -5.0, -2.5]), torch.tensor([1.0, 2.1, 0.0, -5.0, -2.5])
    helper_test_op([], lambda: torch.ceil(b), lambda: Tensor.ceil(a), forward_only=True)
  def test_tril(self):
    helper_test_op([(3,3)], lambda x: x.tril(), lambda x: x.tril())
    helper_test_op([(3,3)], lambda x: x.tril(1), lambda x: x.tril(1))
    helper_test_op([(3,3)], lambda x: x.tril(-1), lambda x: x.tril(-1))
    helper_test_op([(5,3,3)], lambda x: x.tril(), lambda x: x.tril())
    helper_test_op([(5,3,3)], lambda x: x.tril(1), lambda x: x.tril(1))
  def test_triu(self):
    helper_test_op([(3,3)], lambda x: x.triu(), lambda x: x.triu())
    helper_test_op([(3,3)], lambda x: x.triu(1), lambda x: x.triu(1))
    helper_test_op([(3,3)], lambda x: x.triu(-1), lambda x: x.triu(-1))
    helper_test_op([(5,3,3)], lambda x: x.triu(), lambda x: x.triu())
    helper_test_op([(5,3,3)], lambda x: x.triu(1), lambda x: x.triu(1))
  def test_maximum(self):
    helper_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
    helper_test_op([(), ()], torch.maximum, Tensor.maximum)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., 4.], [1., 2., 3., 0.]])
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1, 0, 3, 4], [1, 2, 3, 0]], forward_only=True)
  def test_minimum(self):
    helper_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
    helper_test_op([(), ()], torch.minimum, Tensor.minimum)
  def test_add(self):
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
  def test_add_number(self):
    helper_test_op([(), ()], lambda x,y: x+y, Tensor.add)
  def test_add3(self):
    helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: x+y+z)
  def test_add_simple(self):
    helper_test_op([(256), (256)], lambda x,y: x+y, Tensor.add, forward_only=True)
  def test_broadcasted_add(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x+y, lambda x,y: x+y)
    helper_test_op([(45,65), ()], lambda x,y: x+y, lambda x,y: x+y)
  def test_broadcasted_add_2(self):
    helper_test_op([(45,65), (65,)], lambda x,y: x+y, lambda x,y: x+y)
  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub)
    helper_test_op([(), ()], lambda x,y: x-y, Tensor.sub)
  def test_neg(self):
    helper_test_op([(45,65)], lambda x: -x)
    helper_test_op([()], lambda x: -x)
  def test_mul(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y, Tensor.mul)
  def test_mul_number(self):
    helper_test_op([(), ()], lambda x,y: x*y, Tensor.mul)
  def test_mul_const(self):
    helper_test_op([(45,65)], lambda x: x*2, lambda x: x*2)
    helper_test_op([(45,65)], lambda x: x*-1, lambda x: x*-1)
    helper_test_op([(45,65)], lambda x: 255*x, lambda x: 255*x)
  def test_div(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
    helper_test_op([(), ()], lambda x,y: x/y, Tensor.div)
    helper_test_op(None, lambda x,y: x/y, Tensor.div, forward_only=True, vals=[[5],[1]])
    helper_test_op(None, lambda x: (x/2).to(torch.int), lambda x: x/2, forward_only=True, vals=[[3]])
  def test_div_const(self):
    helper_test_op([(45,65)], lambda x: x/255, lambda x: x/255)
    helper_test_op([(45,65)], lambda x: x/1, lambda x: x/1)
    helper_test_op([(45,65)], lambda x: 1/x, lambda x: 1/x)
    helper_test_op([(45,65)], lambda x: x/2, lambda x: x/2)
    helper_test_op([(45,65)], lambda x: 2/x, lambda x: 2/x)
    helper_test_op([()], lambda x: x/2, lambda x: x/2)
    helper_test_op([()], lambda x: 2/x, lambda x: 2/x)
  @unittest.skipIf(Device.DEFAULT in ["METAL", "WEBGPU"], "WEBGPU does not have support for inf/nan, METAL has issues with -inf")
  def test_mul_const_naninf(self):
    helper_test_op([(45,65)], lambda x: x*float("inf"), lambda x: x*float("inf"))
    helper_test_op([(45,65)], lambda x: x*-float("inf"), lambda x: x*-float("inf"))
    helper_test_op([(45,65)], lambda x: x*float("nan"), lambda x: x*float("nan"))
  @unittest.skipIf(Device.DEFAULT in ["METAL", "WEBGPU"], "WEBGPU does not have support for inf/nan, METAL has issues with -inf")
  def test_div_const_naninf(self):
    helper_test_op([(45,65)], lambda x: x/float("inf"), lambda x: x/float("inf"))
    helper_test_op([(45,65)], lambda x: x/-float("inf"), lambda x: x/-float("inf"))
    helper_test_op([(45,65)], lambda x: x/float("nan"), lambda x: x/float("nan"))
    helper_test_op([(45,65)], lambda x: float("inf")/x, lambda x: float("inf")/x)
    helper_test_op([(45,65)], lambda x: (-float("inf"))/x, lambda x: (-float("inf"))/x)
    helper_test_op([(45,65)], lambda x: float("nan")/x, lambda x: float("nan")/x)
  def test_pow_full(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y, Tensor.pow, a=0)
  def test_pow(self):
    # TODO: why is a=0 for these tests?
    helper_test_op([(45,65)], lambda x: x**2, lambda x: Tensor.pow(x,2), a=0)
    helper_test_op([(45,65)], lambda x: x**3, lambda x: Tensor.pow(x,3), a=0)
    helper_test_op([(45,65)], lambda x: x**-2, lambda x: Tensor.pow(x,-2), a=0)
    helper_test_op([()], lambda x: x**2, lambda x: Tensor.pow(x,2), a=0)
    helper_test_op([()], lambda x: x**-2, lambda x: Tensor.pow(x,-2), a=0)
    # Regression tests for https://github.com/tinygrad/tinygrad/issues/1151
    helper_test_op([(45,65)], lambda x: x**3, lambda x: Tensor.pow(x,3), a=-10)
    helper_test_op([()], lambda x: x**3, lambda x: Tensor.pow(x,3), a=-10)
    # Regression tests for https://github.com/tinygrad/tinygrad/issues/1251
    helper_test_op([(45,65)], lambda x: x**0.2, lambda x: Tensor.pow(x,0.2), a=-10)
    helper_test_op([(45,65)], lambda x: x**1.2, lambda x: Tensor.pow(x,1.2), a=-10)
    helper_test_op([()], lambda x: x**0.2, lambda x: Tensor.pow(x,0.2), a=-10)
    helper_test_op([()], lambda x: x**1.2, lambda x: Tensor.pow(x,1.2), a=-10)
    a, b = Tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
    helper_test_op([], lambda: b**1.1, lambda: a**1.1, )
  def test_pow_const(self):
    helper_test_op([(45,65)], lambda x: x**1.0, lambda x: x**1.0)
    helper_test_op([(45,65)], lambda x: x**-1.0, lambda x: x**-1.0)
    helper_test_op([(45,65)], lambda x: 1.0**x, lambda x: 1.0**x)
    helper_test_op([(45,65)], lambda x: x**2.0, lambda x: x**2.0)
    helper_test_op([(45,65)], lambda x: 2.0**x, lambda x: 2.0**x)
    helper_test_op([()], lambda x: x**2.0, lambda x: x**2.0)
    helper_test_op([()], lambda x: 2.0**x, lambda x: 2.0**x)
  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt(), Tensor.sqrt, a=0)
    helper_test_op([()], lambda x: x.sqrt(), Tensor.sqrt, a=0)
  def test_rsqrt(self):
    helper_test_op([(45,65)], lambda x: torch.rsqrt(x), Tensor.rsqrt, a=0)
    helper_test_op([()], lambda x: torch.rsqrt(x), Tensor.rsqrt, a=0)

  def test_sin(self):
    helper_test_op([(45,65)], lambda x: x.sin(), Tensor.sin, a=0)
  def test_cos(self):
    helper_test_op([(45,65)], lambda x: x.cos(), Tensor.cos, a=0)
  def test_tan(self):
    helper_test_op([(45,65)], lambda x: x.tan(), Tensor.tan, a=0)

  def test_relu(self):
    helper_test_op([(64,64)], lambda x: x.relu(), Tensor.relu)
    helper_test_op([()], lambda x: x.relu(), Tensor.relu)
  def test_relu_exact(self):
    helper_test_op(None, lambda x: x.relu(), Tensor.relu, vals=[[-1.,0,1]])
  def test_relu_maximum_exact(self):
    helper_test_op(None, lambda x: torch.maximum(x, torch.zeros_like(x, requires_grad=False)), lambda x: Tensor.maximum(x, 0), vals=[[-1.,0,1]])
  def test_leakyrelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leakyrelu)
    helper_test_op([()], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leakyrelu)
  def test_celu(self):
    for val in range(1, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
      helper_test_op([()], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
  def test_abs(self):
    helper_test_op([(45,65)], lambda x: torch.abs(x), Tensor.abs)
    helper_test_op([()], lambda x: torch.abs(x), Tensor.abs)
  def test_log(self):
    helper_test_op([(45,65)], lambda x: torch.log(x), Tensor.log)
    helper_test_op([()], lambda x: torch.log(x), Tensor.log)
  def test_log2(self):
    helper_test_op([(45,65)], lambda x: torch.log2(x), Tensor.log2)
    helper_test_op([()], lambda x: torch.log2(x), Tensor.log2)
  def test_exp(self):
    helper_test_op([(45,65)], lambda x: torch.exp(x), Tensor.exp)
    helper_test_op([()], lambda x: torch.exp(x), Tensor.exp)
  def test_exp2(self):
    helper_test_op([(45,65)], lambda x: torch.exp2(x), Tensor.exp2)
    helper_test_op([()], lambda x: torch.exp2(x), Tensor.exp2)
  def test_sign(self):
    helper_test_op([(45,65)], lambda x: torch.sign(x), Tensor.sign)
    helper_test_op([()], lambda x: torch.sign(x), Tensor.sign)
  def test_softsign(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.softsign(x), Tensor.softsign)
    helper_test_op([()], lambda x: torch.nn.functional.softsign(x), Tensor.softsign)
  def test_sigmoid(self):
    helper_test_op([(45,65)], lambda x: x.sigmoid(), Tensor.sigmoid)
    helper_test_op([(45,65)], lambda x: x.sigmoid(), Tensor.sigmoid, a=100)
    helper_test_op([(45,65)], lambda x: x.sigmoid(), Tensor.sigmoid, a=-100)
    helper_test_op([()], lambda x: x.sigmoid(), Tensor.sigmoid, forward_only=True)
  def test_softplus(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.softplus(x), Tensor.softplus, atol=1e-6, grad_atol=1e-6)
    helper_test_op([()], lambda x: torch.nn.functional.softplus(x), Tensor.softplus, atol=1e-6, grad_atol=1e-6)
  def test_gelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu)
    #helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, a=100)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, a=-100)
  def test_quick_gelu(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, a=100)
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, a=-100)
    helper_test_op([()], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
  def test_elu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.elu(x), Tensor.elu)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.elu(x, alpha=0.1), lambda x: Tensor.elu(x, alpha=0.1))
    helper_test_op([()], lambda x: torch.nn.functional.elu(x), Tensor.elu)
  def test_relu6(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.relu6(x), Tensor.relu6)
    helper_test_op([()], lambda x: torch.nn.functional.relu6(x), Tensor.relu6)
  def test_hardswish(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.hardswish(x), Tensor.hardswish, atol=1e-6, grad_atol=1e-6)
    helper_test_op([()], lambda x: torch.nn.functional.hardswish(x), Tensor.hardswish, atol=1e-6, grad_atol=1e-6)
  def test_mish(self):
    def _mish_pytorch(x):
      return x*torch.tanh(torch.nn.functional.softplus(x))
    helper_test_op([(45,65)], _mish_pytorch, Tensor.mish, atol=1e-4)
    helper_test_op([()], _mish_pytorch, Tensor.mish, atol=1e-4)
  @unittest.skipIf(IMAGE>0, "no 1d dot for images")
  def test_dot_1d(self):
    helper_test_op([(65), (65)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
    helper_test_op([(65), (65,45)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
    helper_test_op([(45,65), (65)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
    helper_test_op([(32,45,65), (65)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
    helper_test_op([(65), (32,65,45)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
    self.helper_test_exception([(4), (1,2)], lambda x, y: x.matmul(y), Tensor.dot, expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,1), (4)], lambda x, y: x.matmul(y), Tensor.dot, expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(1), (4)], lambda x, y: x.matmul(y), Tensor.dot, expected=(RuntimeError, AssertionError))
  def test_dot(self):
    helper_test_op([(45,65), (65,100)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
    helper_test_op([(32,45,65), (32,65,100)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
    self.helper_test_exception([(2, 4), (1, 3)], lambda x, y: x.matmul(y), Tensor.dot, expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2, 1), (4, 3)], lambda x, y: x.matmul(y), Tensor.dot, expected=(RuntimeError, AssertionError))
    with self.assertRaises(AssertionError):
      a = Tensor(3.14)
      a.matmul(a)
  def test_simple_cumsum(self):
    helper_test_op([(1024)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0), atol=1e-6)
  def test_cumsum(self):
    helper_test_op([(20)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0), atol=1e-6)
    helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0), atol=1e-6)
    helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=1), lambda x: Tensor.cumsum(x, axis=1), atol=1e-6)
    helper_test_op([(20,30,40)], lambda x: torch.cumsum(x, dim=2), lambda x: Tensor.cumsum(x, axis=2), atol=1e-6)
    helper_test_op([(20,30,40)], lambda x: torch.cumsum(x, dim=-1), lambda x: Tensor.cumsum(x, axis=-1), atol=1e-6)

  def test_argmax(self):
    self.assertEqual(torch.Tensor([2,2]).argmax().numpy(), Tensor([2,2]).argmax().numpy()) # check if returns first index for same max
    helper_test_op([(10,20)], lambda x: x.argmax(), lambda x: x.argmax(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(0, False), lambda x: x.argmax(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, False), lambda x: x.argmax(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, True), lambda x: x.argmax(1, True), forward_only=True)
  def test_argmin(self):
    self.assertEqual(torch.Tensor([2, 2]).argmin().numpy(), Tensor([2, 2]).argmin().numpy())
    helper_test_op([(10,20)], lambda x: x.argmin(), lambda x: x.argmin(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(0, False), lambda x: x.argmin(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, False), lambda x: x.argmin(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, True), lambda x: x.argmin(1, True), forward_only=True)

  def test_matmul_simple(self):
    helper_test_op([(4), (4,4)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
  def test_matmul(self):
    helper_test_op([(64), (64,99)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)

  @unittest.skipIf(IMAGE>0, "no batched matmul on images")
  def test_matmul_batched(self):
    helper_test_op([(3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)

  @unittest.skipIf(IMAGE>0, "no batched matmul on images")
  def test_matmul_batched_vector(self):
    helper_test_op([(4,3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
  def test_small_gemm(self):
    helper_test_op([(8,8), (8,8)], lambda x,y: x.matmul(y), lambda x,y: x@y, atol=1e-3)
  def test_small_gemm_eye(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y, atol=1e-3, vals=[np.eye(8).astype(np.float32), np.eye(8).astype(np.float32)])
  def test_gemm(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-3)
  def test_big_gemm(self):
    helper_test_op([(256,256), (256,256)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-3)
  def test_broadcastdot(self):
    helper_test_op([(10,45,65), (65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
    with self.assertRaises(AssertionError):
      a = Tensor(3.14)
      b = Tensor.ones(3,3)
      a @ b
  def test_multidot(self):
    helper_test_op([(10,45,65), (10,65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
    helper_test_op([(3,3,45,65), (3,3,65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
  def test_sum_simple(self):
    helper_test_op(None, lambda x: x.sum(), Tensor.sum, vals=[[1.,1.]])
  def test_sum_full(self):
    helper_test_op([(16384)], lambda x: x.sum(), lambda x: x.sum())
  def test_sum_small_full(self):
    helper_test_op([(45,5)], lambda x: x.sum(), Tensor.sum)
  def test_sum_relu(self):
    helper_test_op([(3,4,5)], lambda x: x.relu().sum().relu(), lambda x: x.relu().sum().relu())
  def test_sum(self):
    helper_test_op([(45,3)], lambda x: x.sum(), Tensor.sum)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=3), lambda x: Tensor.sum(x, axis=3))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,3)), lambda x: Tensor.sum(x, axis=(1,3)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(0,2)), lambda x: Tensor.sum(x, axis=(0,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,2)), lambda x: Tensor.sum(x, axis=(1,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1), lambda x: Tensor.sum(x, axis=1))
    helper_test_op([()], lambda x: x.sum(), Tensor.sum)
  def test_min(self):
    helper_test_op([(3,3)], lambda x: x.min(), Tensor.min)
    helper_test_op([(45,3)], lambda x: x.min(), Tensor.min)
    helper_test_op([(45,3)], lambda x: x.min().mul(0.5), lambda x: Tensor.min(x).mul(0.5))
    helper_test_op([()], lambda x: x.min(), Tensor.min)
  def test_max(self):
    helper_test_op([(45,3)], lambda x: x.max(), Tensor.max)
    helper_test_op([(45,3)], lambda x: x.max().mul(0.5), lambda x: Tensor.max(x).mul(0.5))
    helper_test_op(None, lambda x: x.max().mul(0.5), lambda x: Tensor.max(x).mul(0.5),
            vals=[
                [[1.0,1.0,0.0,1.0]],
                ])
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: Tensor.max(x, axis=1))
    helper_test_op([()], lambda x: x.max(), Tensor.max)
  def test_mean(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean())
    helper_test_op([()], lambda x: x.mean())
  def test_mean_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean(axis=(1,2)), lambda x: Tensor.mean(x, axis=(1,2)))
  def test_std(self):
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x), lambda x: Tensor.std(x))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=None, correction=0), lambda x: Tensor.std(x, correction=0))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=None, correction=5), lambda x: Tensor.std(x, correction=5))
  def test_std_axis(self):
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=0), lambda x: Tensor.std(x, axis=0))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=2), lambda x: Tensor.std(x, axis=2))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=[1, 2]), lambda x: Tensor.std(x, axis=[1, 2]))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=None), lambda x: Tensor.std(x, axis=None))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, correction=0, dim=0), lambda x: Tensor.std(x, axis=0, correction=0))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, correction=0, dim=2), lambda x: Tensor.std(x, axis=2, correction=0))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, correction=0, dim=[1, 2]), lambda x: Tensor.std(x, axis=[1, 2], correction=0))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, correction=0, dim=None), lambda x: Tensor.std(x, axis=None, correction=0))
  def test_std_keepdim(self):
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=None, keepdim=True), lambda x: Tensor.std(x, keepdim=True))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, dim=0, keepdim=True, correction=0), lambda x: Tensor.std(x, keepdim=True, correction=0, axis=0))
  def test_log_softmax(self):
    helper_test_op([(45,65)], lambda x: torch.nn.LogSoftmax(dim=1)(x), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.nn.LogSoftmax(dim=0)(x), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
  def test_log_softmax_other_axis(self):
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(0), lambda x: x.log_softmax(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(1), lambda x: x.log_softmax(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(2), lambda x: x.log_softmax(2), atol=1e-7, grad_atol=1e-7)
  def test_tanh(self):
    helper_test_op([(45,65)], lambda x: x.tanh(), Tensor.tanh, atol=1e-6, grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.tanh(), Tensor.tanh, atol=1e-6, grad_atol=1e-6, a=-100)
    helper_test_op([()], lambda x: x.tanh(), Tensor.tanh, atol=1e-6, grad_atol=1e-6)
  def test_hardtanh(self):
    for val in range(10, 30, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.hardtanh(x,-val, val), lambda x: x.hardtanh(-val, val), atol=1e-6, grad_atol=1e-6)
      helper_test_op([()], lambda x: torch.nn.functional.hardtanh(x,-val, val), lambda x: x.hardtanh(-val, val), atol=1e-6, grad_atol=1e-6)
  def test_topo_sort(self):
    helper_test_op([(45,65)], lambda x: (x+x)*x, lambda x: x.add(x).mul(x), atol=1e-6, grad_atol=1e-6)
    helper_test_op([()], lambda x: (x+x)*x, lambda x: x.add(x).mul(x), atol=1e-6, grad_atol=1e-6)

  def test_scalar_mul(self):
    helper_test_op([(45,65)], lambda x: x*2, lambda x: x*2)
    helper_test_op([()], lambda x: x*2, lambda x: x*2)
  def test_scalar_rmul(self):
    helper_test_op([(45,65)], lambda x: 2*x, lambda x: 2*x)
    helper_test_op([()], lambda x: 2*x, lambda x: 2*x)
  def test_scalar_sub(self):
    helper_test_op([(45,65)], lambda x: x-2, lambda x: x-2)
    helper_test_op([()], lambda x: x-2, lambda x: x-2)
  def test_scalar_rsub(self):
    helper_test_op([(45,65)], lambda x: 2-x, lambda x: 2-x)
    helper_test_op([()], lambda x: 2-x, lambda x: 2-x)
  def test_flip_eye_crash(self):
    helper_test_op([], lambda: (torch.eye(10)@torch.eye(10).flip(0)),
                       lambda: (Tensor.eye(10)@Tensor.eye(10).flip(0)), forward_only=True)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "this test uses more than 8 bufs passing the WEBGPU limit") #TODO: remove after #1461
  def test_broadcast_full(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div)]: #, (torch.pow, Tensor.pow)]:
      for shapes in [((5,13,24,16), (5,1,24,1)), ((1,3,1,7,1), (2,1,5,1,8))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          helper_test_op(shapes, torch_op, tinygrad_op, a=-0.5 if tinygrad_op != Tensor.pow else 0.0)

  def test_broadcast_simple(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x/y, lambda x,y: x/y)
    helper_test_op([(45,65), ()], lambda x,y: x/y, lambda x,y: x/y)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "this test uses more than 8 bufs passing the WEBGPU limit") #TODO: remove after #1461
  def test_broadcast_partial(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div)]: #, (torch.pow, Tensor.pow)]:
      for shapes in [((1,32,32,32), (1,32,1,1)), ((5,13,24,16,2), (1,13,24,1,1)),
                     ((4,1), (4,5)), ((1,4), (5,4))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          # NOTE: ANE backwards?
          helper_test_op(shapes, torch_op, tinygrad_op, a=-0.5 if tinygrad_op != Tensor.pow else 0.0)

  def test_slice_in_bounds_1dim(self):
    helper_test_op([(3)], lambda x: x[1:3], lambda x: x[1:3])
    helper_test_op([(3)], lambda x: x[0:2], lambda x: x[0:2])
    helper_test_op([(3)], lambda x: x[-2:2], lambda x: x[-2:2])

  def test_slice_on_0dim_tensor(self):
    helper_test_op([()], lambda x: x[None], lambda x: x[None])

    with self.assertRaises(IndexError):
      a = Tensor(3.14)
      a[0]

  def test_slice_int_indexing(self):
    helper_test_op([(3)], lambda x: x[1], lambda x: x[1])
    helper_test_op([(3)], lambda x: x[-2], lambda x: x[-2])
    helper_test_op([(10,10)], lambda x: x[1], lambda x: x[1])
    helper_test_op([(3,3,3)], lambda x: x[1,1,1], lambda x: x[1,1,1])

  def test_slice_in_bounds_multidim(self):
    helper_test_op([(3,3,3)], lambda x: x[1:2], lambda x: x[1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 2], lambda x: x[1:2, 2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2], lambda x: x[1:2, 1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2, 0:-1], lambda x: x[1:2, 1:2, 0:-1])

  def test_slice_with_none(self):
    helper_test_op([(3,3,3)], lambda x: x[None], lambda x: x[None])
    helper_test_op([(3,3,3)], lambda x: x[1:2, None], lambda x: x[1:2, None])
    helper_test_op([(3,3,3)], lambda x: x[1:2, None, 1:2], lambda x: x[1:2, None, 1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2, None, -1], lambda x: x[1:2, 1:2, None, -1])

  def test_slice_one_endpoint_out_of_bounds(self):
    helper_test_op([(3,3,3)], lambda x: x[0:4], lambda x: x[0:4])
    helper_test_op([(3,3,3)], lambda x: x[-6:4], lambda x: x[-6:4])
    helper_test_op([(3,3,3)], lambda x: x[1:50], lambda x: x[1:50])
    helper_test_op([(3,3,3)], lambda x: x[1:50, 1:2, -1], lambda x: x[1:50, 1:2, -1])

  def test_slice_stride_gt_one(self):
    helper_test_op([(7,5,10)], lambda x: x[::2, ::3, ::4], lambda x: x[::2, ::3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, ::3, ::4], lambda x: x[1:5:2, ::3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, 3, ::4], lambda x: x[1:5:2, 3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, None, None, 3, None, ::4], lambda x: x[1:5:2, None, None, 3, None, ::4])

  def test_slice_negative_strides(self):
    # Torch doesn't support slicing with negative steps
    a = np.random.randn(10, 10, 10).astype(np.float32)
    t = Tensor(a)
    np.testing.assert_allclose(a[::-1], t[::-1].numpy())
    np.testing.assert_allclose(a[::-2], t[::-2].numpy())
    np.testing.assert_allclose(a[:, 2:0:-1], t[:, 2:0:-1].numpy())
    np.testing.assert_allclose(a[:, 2:0:-1, 3:1:-2], t[:, 2:0:-1, 3:1:-2].numpy())
    np.testing.assert_allclose(a[4:0:-3, 2:0:-1, -1:-5:-2], t[4:0:-3, 2:0:-1, -1:-5:-2].numpy())

  @unittest.skip("No suppport for tensors with 0s in shape")
  def test_slice_both_endpoints_out_of_bounds(self):
    helper_test_op([(3,3,3)], lambda x: x[5:10], lambda x: x[5:10], forward_only=True)
    helper_test_op([(3,3,3)], lambda x: x[-15:-7], lambda x: x[-15:-7], forward_only=True)

  @unittest.skip("No suppport for tensors with 0s in shape")
  def test_slice_start_gt_end(self):
    helper_test_op([(3,3,3)], lambda x: x[-2:2], lambda x: x[-2:2], forward_only=True)
    helper_test_op([(3,3,3)], lambda x: x[-2:-5], lambda x: x[-2:-5], forward_only=True)

  @unittest.skip("No suppport for tensors with 0s in shape")
  def test_slice_empty(self):
    helper_test_op([(10,10)], lambda x: x[1:1], lambda x: x[1:1], forward_only=True)

  @unittest.skip("No suppport for tensors with 0s in shape")
  def test_slice_zero_in_shape(self):
    helper_test_op([(10,10)], lambda x: x[1:1], lambda x: x[1:1])  # x.shape = (0, 10)
    helper_test_op([(3,3,3)], lambda x: x[-2:-5], lambda x: x[-2:-5])  # x.shape = (0, 3, 3)

  def test_slice_errors(self):
    a = Tensor.ones(4, 3)
    with self.assertRaises(IndexError):
      a[1, 77, 77, 77]  # IndexError: (finds too many indices before the out of bounds)
      a[1, 77]  # IndexError: (out of bounds).
      a[0, -77]
      a[..., ...] # IndexError: only single ellipsis

  def test_slice_ellipsis(self):
    helper_test_op([(3,3,3,3)], lambda x: x[..., 0], lambda x: x[..., 0])
    helper_test_op([(3,3,3,3)], lambda x: x[0, ...], lambda x: x[0, ...])
    helper_test_op([(3,3,3,3)], lambda x: x[0, ..., 0], lambda x: x[0, ..., 0])
    helper_test_op([(3,3,3,3)], lambda x: x[0:3, ..., 2:3], lambda x: x[0:3, ..., 2:3])
    helper_test_op([(3,3,3,3)], lambda x: x[None, 0:3, ..., 0, None], lambda x: x[None, 0:3, ..., 0, None])

  def test_pad2d(self):
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)), lambda x: x.pad2d(padding=(1,2,3,4)))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4)), lambda x: x.pad2d(padding=(-1,2,-3,4)))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=5), lambda x: x.pad2d(padding=(1,2,3,4),value=5))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4), value=5), lambda x: x.pad2d(padding=(-1,2,-3,4),value=5))
  def test_pad(self):
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)),lambda x: x.pad(((3,4),(1,2))))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=5), lambda x: x.pad(((3,4), (1,2)), value=5))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=float("inf")), lambda x: x.pad(((3,4), (1,2)), value=float("inf")))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=float("-inf")), lambda x: x.pad(((3,4), (1,2)), value=float("-inf")))

  def test_transpose(self):
    helper_test_op([(3,3,3)], lambda x: x.transpose(1,2), lambda x: x.transpose(1,2))
    helper_test_op([(3,3,3)], lambda x: x.transpose(0,2), lambda x: x.transpose(0,2))
    helper_test_op([(1,2,3,4)], lambda x: x.movedim((3,0,2,1),(0,1,2,3)), lambda x: x.permute(order=(3,0,2,1)))
    helper_test_op([(3,4,5,6)], lambda x: x.movedim((3,2,1,0),(0,1,2,3)), lambda x: x.permute(order=(3,2,1,0)))
    helper_test_op([()], lambda x: x.permute(()), lambda x: x.permute(()))

  def test_reshape(self):
    helper_test_op([(4,3,6,6)], lambda x: torch.reshape(x, (-1,3,6,6)), lambda x: x.reshape(shape=(-1,3,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: torch.reshape(x, (-1,1,6,6)), lambda x: x.reshape(shape=(-1,1,6,6)))
    helper_test_op([()], lambda x: torch.reshape(x, []), lambda x: x.reshape([]))
    helper_test_op([(1,)], lambda x: torch.reshape(x, []), lambda x: x.reshape([]))
    helper_test_op([()], lambda x: torch.reshape(x, [1]), lambda x: x.reshape([1]))

    with self.assertRaises(AssertionError):
      x = Tensor.ones((4,3,6,6))
      x.reshape([])

  def test_flip(self):
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,)), lambda x: x.flip(axis=(0,)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,1)), lambda x: x.flip(axis=(0,1)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,1,3)), lambda x: x.flip(axis=(0,1,3)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (3,)), lambda x: x.flip(axis=(3,)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,1,3)).flip((0,)), lambda x: x.flip(axis=(0,1,3)).flip(0))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (3,)), lambda x: x.flip(axis=(-1,)))
    helper_test_op([()], lambda x: torch.flip(x, ()), lambda x: x.flip(axis=()))
    helper_test_op([(1,)], lambda x: torch.flip(x, ()), lambda x: x.flip(axis=()))
    helper_test_op([(4, 3, 6, 6)], lambda x: torch.flip(x, ()), lambda x: x.flip(axis=()))

  def test_squeeze(self):
    helper_test_op([(1,3,6,6)], lambda x: torch.squeeze(x, 0), lambda x: x.squeeze(dim=0))
    helper_test_op([(4,3,1,6)], lambda x: torch.squeeze(x, 1), lambda x: x.squeeze(dim=1))
    helper_test_op([(4,3,6,6)], lambda x: torch.squeeze(x, 3), lambda x: x.squeeze(dim=3))
    self.helper_test_exception([(4,3,6,6)], lambda x: torch.squeeze(x, 50), lambda x: x.squeeze(dim=50), expected=IndexError, exact=True)
    self.helper_test_exception([(4,3,6,6)], lambda x: torch.squeeze(x, -50), lambda x: x.squeeze(dim=-50), expected=IndexError, exact=True)
    helper_test_op([(4,3,6,1)], lambda x: torch.squeeze(x, -1), lambda x: x.squeeze(dim=-1))
    helper_test_op([(4,3,6,6)], lambda x: torch.squeeze(x), lambda x: x.squeeze())
    helper_test_op([(1,3,6,6)], lambda x: torch.squeeze(x), lambda x: x.squeeze())
    helper_test_op([(2,3,1)], lambda x: torch.squeeze(x), lambda x: x.squeeze())
    helper_test_op([()], lambda x: torch.squeeze(x, -1), lambda x: x.squeeze(dim=-1))
    helper_test_op([()], lambda x: torch.squeeze(x, 0), lambda x: x.squeeze(dim=0))
    self.helper_test_exception([()], lambda x: torch.squeeze(x, 10), lambda x: x.squeeze(dim=10), expected=IndexError, exact=True)
    helper_test_op([()], lambda x: torch.squeeze(x), lambda x: x.squeeze())

  def test_unsqueeze(self):
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, 0), lambda x: x.unsqueeze(dim=0))
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, 4), lambda x: x.unsqueeze(dim=4))
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, -1), lambda x: x.unsqueeze(dim=-1))
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, -3), lambda x: x.unsqueeze(dim=-3))
    helper_test_op([()], lambda x: torch.unsqueeze(x, 0), lambda x: x.unsqueeze(dim=0))

  def test_flatten(self):
    for axis in range(3):
      helper_test_op([(4,3,6,6)], lambda x: torch.flatten(x, start_dim=axis), lambda x: x.flatten(axis))
    helper_test_op([()], lambda x: x.flatten(), lambda x: x.flatten())
    helper_test_op([(1,)], lambda x: x.flatten(), lambda x: x.flatten())

  def test_detach(self):
    helper_test_op([(4,3,6,6)], lambda x: x.detach(), lambda x: x.detach(), forward_only=True)
    helper_test_op([()], lambda x: x.detach(), lambda x: x.detach(), forward_only=True)

  def test_expand(self):
    arg = (4,3,2,6)
    helper_test_op([(4,3,1,6)], lambda x: x.expand(arg), lambda x: x.expand(shape=arg))
    helper_test_op([()], lambda x: x.expand([]), lambda x: x.expand(shape=[]))

  @unittest.skip("very slow")
  def test_sd_big_conv(self):
    # internal shape (1, 1, 512, 62, 62, 512, 3, 3) overflows a int
    helper_test_op([(1,256,64,64), (512,256,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-2)

  @unittest.skip("slow")
  def test_large_bs_conv(self):
    # large batch size can cause OpenCL image to exceed max image height on macOS
    # (or cause the conv kernel to overflow short sampling coords)
    helper_test_op([(4096,3,3,3), (1,3,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-4, rtol=1e-2)

  @unittest.skip("slow")
  def test_large_ic_conv(self):
    # large input channel count can cause OpenCL image to exceed max image width on macOS
    helper_test_op([(1,2048,3,3), (1,2048,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-4)

  def test_biased_conv2d(self):
    C = 8
    helper_test_op([(1,C,5,5), (C,C,1,1), (C,)],
      lambda x,w,b: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w,b).relu(),w,b),
      lambda x,w,b: Tensor.conv2d(x,w,b).relu().conv2d(w,b), atol=1e-4)

  def test_simple_conv2d(self):
    helper_test_op([(1,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_noopt(self):
    # useful with IMAGE enabled
    with Context(NOOPT=1):
      self.test_simple_conv2d()

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv3d(self):
    helper_test_op([(1,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_padded_conv3d(self):
    helper_test_op([(1,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w,padding=1).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=[1,1,1,1,1,1]).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_m4(self):
    helper_test_op([(1,16,18,18), (16,16,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_1x1(self):
    helper_test_op([(1,4,9,9), (4,4,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_1x1_m4(self):
    helper_test_op([(1,16,32,32), (16,16,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_nested_conv2d(self):
    helper_test_op([(1,32,9,9), (32,32,3,3), (32,32,3,3)],
      lambda x,w1,w2: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w1).relu(), w2).relu(),
      lambda x,w1,w2: x.conv2d(w1).relu().conv2d(w2).relu(), atol=1e-4, grad_rtol=1e-5)

  # expect reduce nodes == 3
  def test_simple_conv2d_nhwc(self):
    # weights (from tf): filter_height x filter_width x in_channels x out_channels
    helper_test_op([(2,9,9,10), (3,3,10,20)],
      lambda x,w: torch.nn.functional.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)).relu(),
      lambda x,w: Tensor.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_batched(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  # conv transpose

  def test_simple_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w).relu(),
      lambda x,w: Tensor.conv_transpose2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_bias_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b).relu(),
      lambda x,w,b: Tensor.conv_transpose2d(x,w,b).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_grouped_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w,groups=2).relu(),
      lambda x,w: Tensor.conv_transpose2d(x,w,groups=2).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_padded_conv_transpose2d(self):
    for padding in [(1,2), (2,1), 2, 1, 0]:
      helper_test_op([(2,4,9,9), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w,padding=padding).relu(),
        lambda x,w: Tensor.conv_transpose2d(x,w,padding=padding).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_dilated_conv_transpose2d(self):
    for dilation in [(1,2), (2,1), 2, 1]:
      helper_test_op([(2,4,9,9), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w,dilation=dilation).relu(),
        lambda x,w: Tensor.conv_transpose2d(x,w,dilation=dilation).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_strided_conv_transpose2d(self):
    for stride in [(2,1), (1,2), 1]:
      helper_test_op([(2,4,4,5), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w, stride=stride).relu(),
        lambda x,w: Tensor.conv_transpose2d(x,w,stride=stride).relu(), atol=1e-4, grad_rtol=1e-5)

  @unittest.skipIf(Device.DEFAULT == "METAL" and CI, "broken in METAL CI")
  def test_output_padded_conv_transpose2d(self):
    for output_padding, stride in [((1,1), (2,3)), ((2,1), (3,2))]:
      helper_test_op([(2,4,6,5), (4,4,3,3),(4,)],
        lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride).relu(),
        lambda x,w,b: Tensor.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride).relu(), atol=1e-4, grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv_transpose3d(self):
    helper_test_op([(2,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose3d(x,w).relu(),
      lambda x,w: Tensor.conv_transpose2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  @unittest.skipIf((IMAGE>0), "no conv1d on images")
  def test_conv1d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for H in [1,2,5]:
          for groups in [1,3] if cin == 3 and H == 5 else [1]:
            with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H):
              helper_test_op([(bs,cin,11), (6,cin//groups,H)],
                lambda x,w: torch.nn.functional.conv1d(x,w,groups=groups).relu(),
                lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_simple_padding_conv1d(self):
    bs = 6
    cin = 2
    groups = 1
    H = 5
    p = (1,1)
    helper_test_op([(bs,cin,11), (6,cin//groups,H)],
      lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4)

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_strided_conv1d_simple(self):
    bs, H = 2, 3
    helper_test_op([(bs,1,5), (1,1,H)],
      lambda x,w: torch.nn.functional.conv1d(x,w,stride=2).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=2).relu(), atol=1e-4)

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_asymmetric_padding_conv1d(self):
    for p in [(0,1), (2,1), (2,0)]:
      with self.subTest(padding := p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n), (1,1,k)],
              lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4)
            helper_test_op([(1,1,n), (1,1,k)],
              lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4)

  def _test_conv2d(self, bs=1, cin=1):
    for H in [1,2,3]:
      for W in [1,2,3,5]:
        for groups in [1,3] if cin == 3 and H == 3 and W == 3 else [1]:
          with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H, width=W):
            helper_test_op([(bs,cin,11,7), (6,cin//groups,H,W)],
              lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
              lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)
  def test_conv2d(self): self._test_conv2d(bs=1, cin=3)
  def test_conv2d_bs_4_cin_3(self): self._test_conv2d(bs=4, cin=3)
  def test_conv2d_bs_1_cin_1(self): self._test_conv2d(bs=1, cin=1)
  def test_conv2d_bs_4_cin_1(self): self._test_conv2d(bs=4, cin=1)

  def test_large_input_conv2d(self):
    bs = 4
    cin = 16
    groups = 1
    H = 5
    W = 2
    helper_test_op([(bs,cin,64,64), (6,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      # needed to relax tolerance on NVIDIA
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-3, grad_rtol=1e-5)

  def test_simple_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 1
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_medium_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 2
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_depthwise_conv2d(self):
    bs = 1
    groups = 32
    rcout = 1
    cin = 1
    helper_test_op([(bs,groups*cin,32,32), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_grouped_conv2d(self):
    bs = 4
    groups = 5
    rcout = 7
    cin = 3
    helper_test_op([(bs,groups*cin,5,5), (groups*rcout,cin,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_fancy_conv2d(self):
    bs = 2
    cin = 3
    cout = 1
    groups = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (groups*cout,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_strided_conv2d_simple(self):
    bs,H,W = 2,3,1
    helper_test_op([(bs,1,5,1), (1,1,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=2).relu(), atol=1e-4)

  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    with self.subTest(stride := 2):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=stride).relu(), atol=1e-4)
    with self.subTest(stride := (2,1)):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=stride).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=(2,1)).relu(), atol=1e-4)

  def test_negative_padding_conv2d(self):
    n,k = 10, 3
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:-1, 1:-1],w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=-1).relu(), atol=1e-4)
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:, 1:],w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=(-1,0,-1,0)).relu(), atol=1e-4)

  def test_simple_padding_conv2d(self):
    p = (1,1,1,1)
    helper_test_op(None,
      lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4, vals=[[[[[2.,3.]]]], [[[[1.]]]]])

  def test_asymmetric_padding_conv2d(self):
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding := p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4)
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4)

  @unittest.skipIf(Device.DEFAULT == "METAL" and CI, "broken in METAL CI")
  def test_padded_conv2d_p21(self):
    bs,cin,H,W,padding = 4, 3, 3, 3, (2,1)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu(), atol=1e-4)

  @unittest.skipIf(Device.DEFAULT == "METAL" and CI, "broken in METAL CI")
  def test_padded_conv2d_p22(self):
    bs,cin,H,W,padding = 4, 3, 3, 3, (2,2)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu(), atol=1e-4)

  def test_padded_conv2d_1x1(self):
    bs,cin,H,W,padding = 4, 3, 1, 1, 2
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu(), atol=1e-4)

  def test_padded_conv2d_bs1(self):
    bs,cin,H,W,padding = 1, 3, 3, 3, 1
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu(), atol=1e-4)

  def test_padding_add(self):
    helper_test_op([(64,64), (60,60)],
      lambda x,w: x+torch.nn.functional.pad(w, (2,2,2,2)),
      lambda x,w: x+w.pad2d((2,2,2,2)))

  def test_dilated_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    for d in [2, (2,1)]:
      with self.subTest(dilation := d):
        helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
          lambda x,w: torch.nn.functional.conv2d(x,w,dilation=dilation).relu(),
          lambda x,w: Tensor.conv2d(x,w,dilation=dilation).relu(), atol=1e-4)

  def test_maxpool2d_simple(self):
    ksz = (2,2)
    helper_test_op([(1,1,2,3)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
      lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_maxpool2d(self):
    for ksz in [(2,2), (3,3), 2, 3, (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_maxpool2d_bigger_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=stride))

  @unittest.skipIf(Device.DEFAULT == "CUDA", "CUDA fails on this")
  def test_maxpool2d_unit_stride(self):
    helper_test_op([(32,2,110,28)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=1),
      lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=1))

  def test_maxpool2d_smaller_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=stride))

  def test_maxpool2d_dilation(self):
    for dilation in [(2, 3), (3, 2), 2, 3]:
      helper_test_op([(32,2,110,28)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), dilation=dilation),
        lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), dilation=dilation))

  def test_avgpool2d(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz), rtol=1e-5)

  def test_global_avgpool2d(self):
    helper_test_op([(32,2,111,28)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(111,28)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(111,28)), rtol=1e-5)

  def test_cat(self):
    for dim in range(-2, 3):
      helper_test_op([(45,65,9), (45,65,9), (45,65,9)], lambda x,y,z: torch.cat((x,y,z), dim), lambda x,y,z: x.cat(y, z, dim=dim))

    with self.assertRaises(AssertionError):
      a = Tensor(3.14)
      a.cat(a)

  def test_multicat(self):
    for dim in range(-1, 2):
      helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: torch.cat((x,y,z), dim), lambda x,y,z: x.cat(y, z, dim=dim))

  def test_stack(self):
    x = Tensor.randn(45, 65, 3)

    for dim in range(-1, 3):
      helper_test_op([(45, 65, 3), (45, 65, 3), (45, 65, 3)], lambda x, y, z: torch.stack((x, y, z), dim=dim), lambda x, y, z: Tensor.stack([x, y, z], dim=dim))

    with self.assertRaises(IndexError):
      Tensor.stack([x], dim=77)

    a = Tensor(3.14)
    np.testing.assert_allclose(Tensor.stack([a, a]).numpy(), Tensor([3.14, 3.14]).numpy())

  def test_repeat(self):
    x = Tensor.randn(4, 6, 3)
    base_repeats = [2, 4, 3]

    for reps in [[], [4], [2, 1], [3, 2, 2]]:
      repeats = base_repeats + reps
      helper_test_op([(4, 6, 3)], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))
      helper_test_op([()], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))

    with self.assertRaises(AssertionError):
      x.repeat((2, 4))

    with self.assertRaises(AssertionError):
      x.repeat((2, 0, 4))

  def test_clip(self):
    helper_test_op([(45,65)], lambda x: x.clip(-2.3, 1.2), lambda x: x.clip(-2.3, 1.2))

  def test_matvecmat(self):
    helper_test_op([(1,128), (128,128), (128,128)], lambda x,y,z: (x@y).relu()@z, atol=1e-4)

  def test_matvec(self):
    helper_test_op([(1,128), (128,128)], lambda x,y: (x@y).relu(), atol=1e-4)

  # this was the failure in llama early realizing freqs_cis
  def test_double_slice(self):
    helper_test_op([(4,4)], lambda x: x[:, 1:2][1:2])
    helper_test_op([(4,4)], lambda x: x[1:3][1:2])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][0:1])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][:, 0:1])

  @unittest.skip("this test is broken #862")
  def test_max_inf(self):
    n = Tensor([1, float("nan")]).max().numpy()
    assert math.isnan(n.item()), f"{n.item()} is not nan"

  def test_inf_where(self):
    x = Tensor.full((3, 3), float("inf"))
    n = (x < 0).where(x, 1).numpy()
    assert np.all(n == 1.)

  def _get_index_randoms(self):
    # indices cannot have gradient
    # TODO currently does not support IndexError for out of bounds idx values
    a = torch.randint(low=-1, high=1, size=(2,1,1,1,1,1), dtype=torch.int64, requires_grad=False)
    b = torch.randint(high=1, size=(1,3,1,1,1,1), dtype=torch.int64, requires_grad=False)
    c = torch.randint(low=-5, high=5, size=(1,1,4,1,1,1), dtype=torch.int64, requires_grad=False)
    d = torch.randint(high=4, size=(2,1,1,5,1,1), dtype=torch.int64, requires_grad=False)
    e = torch.randint(high=1, size=(1,1,1,1,6,1), dtype=torch.int64, requires_grad=False)
    i, j, k, o, p = [Tensor(tor.detach().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False) for tor in [a,b,c,d,e]]
    return a,b,c,d,e,i,j,k,o,p

  def test_slice_fancy_indexing_no_dim_collapse(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    # no dim collapse from int or dim injection from None
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,d,e], lambda x: x[i,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[:,b,c,d,:], lambda x: x[:,j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,...], lambda x: x[i,j,...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,...,e], lambda x: x[i,...,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[...,c,:,e], lambda x: x[...,k,:,p])

  def test_slice_fancy_indexing_dim_collapse_int(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    # dim collapse from int
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,c,d,e], lambda x: x[1,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,3,d,e], lambda x: x[i,j,3,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,2,d,2], lambda x: x[1,j,2,o,2])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,2,2,2,e], lambda x: x[i,2,2,2,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,:,3:11:2,d,0:2], lambda x: x[1,:,3:11:2,o,0:2])

  def test_slice_fancy_indexing_dim_inject_none(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    # dim injection from None
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,c,d,e], lambda x: x[None,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,d,None], lambda x: x[i,j,k,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,None,d,e], lambda x: x[i,j,None,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,c,d,None], lambda x: x[None,j,k,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,:,None,d,e], lambda x: x[i,:,None,o,p])

  def test_slice_fancy_indexing_dim_inject_and_collapse(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    # dim injection and collapse
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,None,d,1], lambda x: x[1,j,None,o,1])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,2,d,None], lambda x: x[None,j,2,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[...,1,d,None], lambda x: x[...,1,o,None])

  def test_slice_fancy_indexing_with_idx(self):
    # indexing using idx with different dim
    helper_test_op([(2,3)], lambda x: x[torch.tensor([[0,0,0],[0,0,0]]), torch.tensor(1)], lambda x: x[Tensor([[0,0,0],[0,0,0]]), Tensor(1)])
    helper_test_op([(2,3)], lambda x: x[torch.tensor([1]), torch.tensor([[0,0,0],[0,0,0]])], lambda x: x[Tensor([1]), Tensor([[0,0,0],[0,0,0]])])

  def test_gather(self):
    # indices cannot have gradient
    # indices cannot be negative (torch gather)
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op([(4,5,6)], lambda x: x.gather(index=b, dim=0), lambda x: x.gather(idx=a, dim=0))
    helper_test_op([(4,5,6)], lambda x: x.gather(index=b, dim=1), lambda x: x.gather(idx=a, dim=1))
    helper_test_op([(4,5,6)], lambda x: x.gather(index=b, dim=2), lambda x: x.gather(idx=a, dim=2))
    helper_test_op([(3,4,5)], lambda x: x.gather(index=b, dim=0), lambda x: x.gather(idx=a, dim=0))
    self.helper_test_exception([(4,5,6)], lambda x: x.gather(index=torch.tensor([1], dtype=torch.int64), dim=0), lambda x: x.gather(idx=Tensor([1], dtype=dtypes.int32), dim=0), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,1,1)], lambda x: x.gather(index=b, dim=0), lambda x: x.gather(idx=a, dim=0), expected=(RuntimeError, AssertionError))

  def test_scaled_product_attention(self):
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64)], lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z), lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z))
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64), (32,8,16,16)], lambda x,y,z,m: torch.nn.functional.scaled_dot_product_attention(x,y,z,attn_mask=m), lambda x,y,z,m: Tensor.scaled_dot_product_attention(x,y,z,attn_mask=m))
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64)], lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z,is_causal=True), lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,is_causal=True))

  def test_binary_crossentropy(self):
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),torch.clip(y,0,1)), lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,torch.clip(y,0,1)), lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,torch.clip(y,0,1)), lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),torch.clip(y,0,1)), lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
