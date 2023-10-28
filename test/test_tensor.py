import numpy as np
import torch
import struct
import unittest, copy
import mmap
from tinygrad.tensor import Tensor, Device
from tinygrad.helpers import dtypes
from extra.gradcheck import numerical_jacobian, jacobian, gradcheck
from extra.utils import temp

x_init = np.random.randn(1,3).astype(np.float32)
U_init = np.random.randn(3,3).astype(np.float32)
V_init = np.random.randn(3,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTinygrad(unittest.TestCase):
  def test_zerodim_initialization(self):
    a = Tensor(55)
    b = Tensor(3.14)

    self.assertEqual(a.shape, ())
    self.assertEqual(b.shape, ())

  def test_plus_equals(self):
    a = Tensor.randn(10,10)
    b = Tensor.randn(10,10)
    c = a + b
    val1 = c.numpy()
    a += b
    val2 = a.numpy()
    np.testing.assert_allclose(val1, val2)

  def test_backward_pass(self):
    def test_tinygrad():
      x = Tensor(x_init, requires_grad=True)
      W = Tensor(W_init, requires_grad=True)
      m = Tensor(m_init)
      out = x.dot(W).relu()
      out = out.log_softmax()
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.numpy(), x.grad.numpy(), W.grad.numpy()

    def test_pytorch():
      x = torch.tensor(x_init, requires_grad=True)
      W = torch.tensor(W_init, requires_grad=True)
      m = torch.tensor(m_init)
      out = x.matmul(W).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.detach().numpy(), x.grad, W.grad

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "this test uses more than 8 bufs which breaks webgpu") #TODO: remove after #1461
  def test_backward_pass_diamond_model(self):
    def test_tinygrad():
      u = Tensor(U_init, requires_grad=True)
      v = Tensor(V_init, requires_grad=True)
      w = Tensor(W_init, requires_grad=True)
      x = u.mul(v).relu()
      y = u.mul(w).relu()
      out = x.add(y).mul(y).relu()
      out = out.log_softmax()
      out = out.sum()
      out.backward()
      return out.numpy(), u.grad.numpy(), v.grad.numpy(), w.grad.numpy()

    def test_pytorch():
      u = torch.tensor(U_init, requires_grad=True)
      v = torch.tensor(V_init, requires_grad=True)
      w = torch.tensor(W_init, requires_grad=True)
      x = u.mul(v).relu()
      y = u.mul(w).relu()
      out = x.add(y).mul(y).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.sum()
      out.backward()
      return out.detach().numpy(), u.grad, v.grad, w.grad

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_nograd(self):
    x = Tensor(x_init, requires_grad=False)
    m = Tensor(m_init, requires_grad=False)
    W = Tensor(W_init, requires_grad=True)
    tmp = x.mul(m)
    mm = tmp.matmul(W)
    out = mm.relu()
    out = out.sum()
    out.backward()
    assert x.grad is None
    assert m.grad is None
    assert tmp.grad is None
    assert mm.grad is not None
    assert W.grad is not None

  def test_dropout(self):
    with Tensor.train():
      n, rate = 1_000_000, 0.1
      w = Tensor.ones(n).dropout(rate)
      non_zeros = np.count_nonzero(w.numpy())
      expected = n * (1 - rate)
      np.testing.assert_allclose(non_zeros, expected, rtol=2e-3)

  def test_jacobian(self):
    W = np.random.RandomState(42069).random((10, 5)).astype(np.float32)
    x = np.random.RandomState(69420).random((1, 10)).astype(np.float32)

    torch_x = torch.tensor(x, requires_grad=True)
    torch_W = torch.tensor(W, requires_grad=True)
    torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)
    PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()

    tiny_x = Tensor(x, requires_grad=True)
    tiny_W = Tensor(W, requires_grad=True)
    tiny_func = lambda x: x.dot(tiny_W).relu().log_softmax()
    J = jacobian(tiny_func, tiny_x)
    NJ = numerical_jacobian(tiny_func, tiny_x)

    np.testing.assert_allclose(PJ, J, atol = 1e-5)
    np.testing.assert_allclose(PJ, NJ, atol = 1e-3)

  def test_gradcheck(self):
    W = np.random.RandomState(1337).random((10, 5)).astype(np.float32)
    x = np.random.RandomState(7331).random((1, 10)).astype(np.float32)

    tiny_x = Tensor(x, requires_grad=True)
    tiny_W = Tensor(W, requires_grad=True)
    tiny_func = lambda x: x.dot(tiny_W).relu().log_softmax()

    self.assertTrue(gradcheck(tiny_func, tiny_x, eps = 1e-3))

    # coarse approx. since a "big" eps and the non-linearities of the model
    self.assertFalse(gradcheck(tiny_func, tiny_x, eps = 1e-5))

  def test_random_fns_are_deterministic_with_seed(self):
    for random_fn in [Tensor.randn, Tensor.normal, Tensor.uniform, Tensor.scaled_uniform, Tensor.glorot_uniform, Tensor.kaiming_normal]:
      with self.subTest(msg=f"Tensor.{random_fn.__name__}"):
        Tensor.manual_seed(1337)
        a = random_fn(10,10).realize()
        Tensor.manual_seed(1337)
        b = random_fn(10,10).realize()
        np.testing.assert_allclose(a.numpy(), b.numpy())

  def test_randn_isnt_inf_on_zero(self):
    # simulate failure case of rand handing a zero to randn
    original_rand, Tensor.rand = Tensor.rand, Tensor.zeros
    try: self.assertNotIn(np.inf, Tensor.randn(16).numpy())
    except: raise
    finally: Tensor.rand = original_rand

  def test_zeros_like_has_same_dtype(self):
    for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
      a = Tensor([1, 2, 3], dtype=datatype)
      b = Tensor.zeros_like(a)
      assert a.dtype == b.dtype, f"a.dtype and b.dtype should be {datatype}"
      assert a.shape == b.shape, f"shape mismatch (Tensor.zeros_like){a.shape} != (torch){b.shape}"

    a = Tensor([1, 2, 3])
    b = Tensor.zeros_like(a, dtype=dtypes.int8)
    assert a.dtype != b.dtype and a.dtype == dtypes.float32 and b.dtype == dtypes.int8, "a.dtype should be float and b.dtype should be char"
    assert a.shape == b.shape, f"shape mismatch (Tensor.zeros_like){a.shape} != (torch){b.shape}"

  def test_ones_like_has_same_dtype_and_shape(self):
    for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
      a = Tensor([1, 2, 3], dtype=datatype)
      b = Tensor.ones_like(a)
      assert a.dtype == b.dtype, f"a.dtype and b.dtype should be {datatype}"
      assert a.shape == b.shape, f"shape mismatch (Tensor.ones_like){a.shape} != (torch){b.shape}"

    a = Tensor([1, 2, 3])
    b = Tensor.ones_like(a, dtype=dtypes.int8)
    assert a.dtype != b.dtype and a.dtype == dtypes.float32 and b.dtype == dtypes.int8, "a.dtype should be float and b.dtype should be char"
    assert a.shape == b.shape, f"shape mismatch (Tensor.ones_like){a.shape} != (torch){b.shape}"

  def test_ndim(self):
    assert Tensor.randn(1).ndim == 1
    assert Tensor.randn(2,2,2).ndim == 3
    assert Tensor.randn(1,1,1,1,1,1).ndim == 6

  def test_argfix(self):
    self.assertEqual(Tensor.zeros().shape, ())
    self.assertEqual(Tensor.ones().shape, ())

    self.assertEqual(Tensor.zeros([]).shape, ())
    self.assertEqual(Tensor.ones([]).shape, ())

    self.assertEqual(Tensor.zeros(tuple()).shape, ())
    self.assertEqual(Tensor.ones(tuple()).shape, ())

    self.assertEqual(Tensor.zeros(1).shape, (1,))
    self.assertEqual(Tensor.ones(1).shape, (1,))

    self.assertEqual(Tensor.zeros(1,10,20).shape, (1,10,20))
    self.assertEqual(Tensor.ones(1,10,20).shape, (1,10,20))

    self.assertEqual(Tensor.zeros([1]).shape, (1,))
    self.assertEqual(Tensor.ones([1]).shape, (1,))

    self.assertEqual(Tensor.zeros([10,20,40]).shape, (10,20,40))
    self.assertEqual(Tensor.ones([10,20,40]).shape, (10,20,40))

  def test_numel(self):
    assert Tensor.randn(10, 10).numel() == 100
    assert Tensor.randn(1,2,5).numel() == 10
    assert Tensor.randn(1,1,1,1,1,1).numel() == 1
    assert Tensor([]).numel() == 0
    # assert Tensor.randn(1,0,2,5) == 0 # TODO: fix empty tensors

  def test_element_size(self):
    for _, dtype in dtypes.fields().items():
      assert dtype.itemsize == Tensor.randn(3, dtype=dtype).element_size(), f"Tensor.element_size() not matching Tensor.dtype.itemsize for {dtype}"

  def test_deepwalk_ctx_check(self):
    layer = Tensor.uniform(1, 1, requires_grad=True)
    x = Tensor.randn(1, 1, 1)
    x.dot(layer).mean().backward()
    x = Tensor.randn(1, 1, 1)
    x.dot(layer).mean().backward()

  def test_zerosized_tensors(self):
    Tensor([]).realize()
    Tensor([]).numpy()

  def test_tensor_ndarray_dtype(self):
    arr = np.array([1]) # where dtype is implicitly int64
    assert Tensor(arr).dtype == dtypes.int64
    assert Tensor(arr, dtype=dtypes.float32).dtype == dtypes.float32 # check if ndarray correctly casts to Tensor dtype
    assert Tensor(arr, dtype=dtypes.float64).dtype == dtypes.float64 # check that it works for something else

  def test_tensor_list_dtype(self):
    arr = [1]
    assert Tensor(arr).dtype == Tensor.default_type
    assert Tensor(arr, dtype=dtypes.float32).dtype == dtypes.float32
    assert Tensor(arr, dtype=dtypes.float64).dtype == dtypes.float64

  def test_tensor_copy(self):
    x = copy.deepcopy(Tensor.ones((3,3,3)))
    np.testing.assert_allclose(x.numpy(), np.ones((3,3,3)))

  def test_copy_from_disk(self):
    t = Tensor.randn(30, device="CPU").to(f"disk:{temp('test_copy_from_disk')}")
    a = t[10:20]
    dev = a.to(Device.DEFAULT)
    np.testing.assert_allclose(a.numpy(), dev.numpy())

  # Regression test for https://github.com/tinygrad/tinygrad/issues/1751
  def test_copy_from_numpy_unaligned(self):
    # 2**15 is the minimum for repro
    arr = np.random.randn(2**15).astype(dtypes.float.np)
    fn = temp('test_copy_from_numpy_unaligned')
    with open(fn, 'wb') as f: f.write(b't' + arr.tobytes())
    with open(fn, "a+b") as f: memview = memoryview(mmap.mmap(f.fileno(), arr.nbytes + 1))
    ua_arr = np.frombuffer(memview[1:], dtype=arr.dtype, count=arr.shape[0])
    np.testing.assert_allclose(arr, ua_arr)
    assert not ua_arr.flags.aligned
    # force device copy - to() is opt'd away - Tensor(dev)/1 is ignored
    np.testing.assert_allclose(ua_arr, (Tensor(ua_arr)/Tensor(1)).numpy())

if __name__ == '__main__':
  unittest.main()
