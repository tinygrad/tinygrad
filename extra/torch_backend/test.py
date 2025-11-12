# simple tests
import unittest
import torch
import numpy as np
from tinygrad.helpers import getenv, GlobalCounters
if getenv("TINY_BACKEND2"):
  import extra.torch_backend.backend2
  device = "cpu"
else:
  import extra.torch_backend.backend
  device = "tiny"

class TestTorchBackend(unittest.TestCase):
  def test_randperm_generator_out(self):
    n = 10
    out = torch.empty(n, dtype=torch.long, device=device)
    res = torch.randperm(n, out=out).cpu().numpy()
    np.testing.assert_equal(set(res), set(range(n)))
    np.testing.assert_equal(out.cpu().numpy(), res)

    res2 = torch.randperm(n).cpu().numpy()
    np.testing.assert_equal(set(res2), set(range(n)))

  def test_numpy_ones(self):
    a = torch.ones(4, device=device)
    np.testing.assert_equal(a.cpu().numpy(), [1,1,1,1])

  def test_numpy_ones(self):
    a = torch.ones(4, dtype=torch.int32, device=device)
    assert a.dtype == torch.int32
    np.testing.assert_equal(a.cpu().numpy(), [1,1,1,1])

  def test_plus(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    c = a+b
    np.testing.assert_equal(c.cpu().numpy(), [2,2,2,2])

  def test_expand(self):
    a = torch.Tensor([1,2,3,4]).to(device)
    out = a.reshape(4,1).expand(4,4)
    np.testing.assert_equal(out.cpu().numpy(), [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])

  def test_batchnorm_unsqueeze(self):
    bn = torch.nn.BatchNorm2d(4).to(device)
    x = torch.randn(8, 4, 3, 3, device=device)
    out = bn(x)
    self.assertEqual(out.shape, x.shape)

  def test_reshape(self):
    a = torch.Tensor([[1,2],[3,4]]).to(device)
    np.testing.assert_equal(a.reshape(4).cpu().numpy(), [1,2,3,4])
    np.testing.assert_equal(a.reshape(2,1,2).cpu().numpy(), [[[1,2]],[[3,4]]])
    np.testing.assert_equal(a.unsqueeze(1).cpu().numpy(), [[[1,2]],[[3,4]]])
    np.testing.assert_equal(a.unsqueeze(1).unsqueeze(1).cpu().numpy(), [[[[1,2]]],[[[3,4]]]])
    np.testing.assert_equal(a.unsqueeze(1).unsqueeze(1).squeeze().cpu().numpy(), [[1,2],[3,4]])

  def test_permute(self):
    a = torch.Tensor([[1,2],[3,4]]).to(device)
    print(a.stride())
    null = a.permute(0,1)
    perm = a.permute(1,0)
    back = perm.permute(1,0)
    np.testing.assert_equal(a.cpu().numpy(), [[1,2],[3,4]])
    np.testing.assert_equal(null.cpu().numpy(), [[1,2],[3,4]])
    np.testing.assert_equal(perm.cpu().numpy(), [[1,3],[2,4]])
    np.testing.assert_equal(back.cpu().numpy(), [[1,2],[3,4]])

  def test_shrink(self):
    a = torch.Tensor([1,2,3,4]).to(device)
    np.testing.assert_equal(a[:3].cpu().numpy(), [1,2,3])
    np.testing.assert_equal(a[1:].cpu().numpy(), [2,3,4])

  def test_as_strided(self):
    a = torch.arange(70, device=device).reshape(1,1,10,7)
    a = a.as_strided((1,1,10,5), (0,0,7,1), storage_offset=0)
    a = a.as_strided((1,1,5,5), (50,50,7,1), storage_offset=21)
    np.testing.assert_equal(a.cpu().numpy().sum(-1), [[[115,150,185,220,255]]])

  def test_plus_inplace(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    a += b
    a += b
    np.testing.assert_equal(a.cpu().numpy(), [3,3,3,3])

  def test_exp2(self):
    a = torch.ones(4, device=device)
    b = a.exp2()
    np.testing.assert_equal(b.cpu().numpy(), [2,2,2,2])

  def test_amax(self):
    x = torch.tensor([[[ 1.5,  2.3,  3.1,  4.7],
                       [ 5.2,  6.8,  7.4,  12.9],
                       [ 9.0, 12.3, 11.6, 10.1]],
                      [[13.2, 16.9, 15.5, 14.1],
                       [17.1, 24.9, 19.8, 20.2],
                       [21.0, 22.3, 23.6, 18.4]]], device=device)

    y1 = torch.amax(x)
    expected = np.array([24.9], dtype=np.float32)
    np.testing.assert_equal(y1.cpu().numpy(), expected)

    y2 = torch.amax(x, dim=(1,2))
    expected = np.array([12.9, 24.9], dtype=np.float32)
    np.testing.assert_equal(y2.cpu().numpy(), expected)

    y3 = torch.amax(x, dim=2)
    expected = np.array([[4.7, 12.9, 12.3], [16.9, 24.9, 23.6]], dtype=np.float32)
    np.testing.assert_equal(y3.cpu().numpy(), expected)


  def test_amin(self):
    x = torch.tensor([[[ 1.5,  2.3,  3.1,  4.7],
                       [ 5.2,  6.8,  7.4,  12.9],
                       [ 9.0, 12.3, 11.6, 10.1]],
                      [[13.2, 16.9, 15.5, 14.1],
                       [17.1, 24.9, 19.8, 20.2],
                       [21.0, 22.3, 23.6, 18.4]]], device=device)

    y1 = torch.amin(x)
    expected = np.array([1.5], dtype=np.float32)
    np.testing.assert_equal(y1.cpu().numpy(), expected)

    y2 = torch.amin(x, dim=(1,2))
    expected = np.array([1.5, 13.2], dtype=np.float32)
    np.testing.assert_equal(y2.cpu().numpy(), expected)

    y3 = torch.amin(x, dim=2)
    expected = np.array([[1.5, 5.2, 9.0], [13.2, 17.1, 18.4]], dtype=np.float32)
    np.testing.assert_equal(y3.cpu().numpy(), expected)

  def test_isfinite(self):
    a = torch.ones(4, device=device)
    np.testing.assert_equal(torch.isfinite(a).cpu().numpy(), [True, True, True, True])

  def test_eq(self):
    a = torch.ones(4, device=device)
    b = torch.ones(4, device=device)
    c = a == b
    print(c.cpu())

  def test_maxpool2d_backward(self):
    x = torch.arange(3*3, dtype=torch.float32, device=device).reshape(1, 1, 3, 3).requires_grad_(True)
    torch.nn.functional.max_pool2d(x, kernel_size=2, stride=1).sum().backward()
    np.testing.assert_equal(x.grad.squeeze().cpu().numpy(), [[0, 0, 0], [0, 1, 1], [0, 1, 1]])

  def test_matmul_backward(self):
    x = torch.randn(3, 4, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(4, 5, device=device, dtype=torch.float32, requires_grad=True)
    z = (x @ y).sum()
    z.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.shape == x.shape
    assert y.grad.shape == y.shape

  def test_matmul_broadcast_backward(self):
    x = torch.randn(2, 3, 4, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(4, 5, device=device, dtype=torch.float32, requires_grad=True)
    z = (x @ y).sum()
    z.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.shape == x.shape
    assert y.grad.shape == y.shape

  def test_copy_cast(self):
    x = torch.zeros(4, device=device, dtype=torch.int64)
    y = torch.ones(4, device=device, dtype=torch.float32).to(dtype=torch.int64)
    res1 = x ^ y # an operation that only works on int types
    print(res1.cpu())
    y = y.cpu().float().to(device=device, dtype=torch.int64)
    res2 = x ^ y
    print(res2.cpu())

  def test_topk(self):
    # test topk return_types
    a = torch.tensor([1, 3, 2, 4], device=device)
    out = torch.topk(a, k=2)
    np.testing.assert_equal(out.values.cpu().numpy(), [4, 3])
    np.testing.assert_equal(out.indices.cpu().numpy(), [3, 1])

  def test_masked_select(self):
    a = torch.tensor([4, 3, 2, 1], device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    out = torch.masked_select(a, mask)
    np.testing.assert_equal(out.cpu().numpy(), [4, 2])
    mask = torch.tensor(True, device=device)
    out = torch.masked_select(a, mask)
    np.testing.assert_equal(out.cpu().numpy(), [4, 3, 2, 1])

  def test_isin_tensor_tensor_out(self):
    a = torch.tensor([1, 2, 3], device=device)
    b = torch.tensor([2, 4], device=device)
    expected_base = torch.tensor([False, True, False], device=device)
    for assume_unique in [False, True]:
      for invert, expected in [(False, expected_base), (True, ~expected_base)]:
        out = torch.empty_like(a, dtype=torch.bool)
        res = torch.ops.aten.isin.Tensor_Tensor_out(a, b, invert=invert, assume_unique=assume_unique, out=out)
        np.testing.assert_equal(out.cpu().numpy(), expected.cpu().numpy())

  def test_uniform(self):
    for torch_dtype in [torch.float32, torch.float16]:
      a = torch.rand(10, 10, device=device, dtype=torch_dtype)
      self.assertEqual(a.dtype, torch_dtype)

  def test_normal(self):
    for torch_dtype in [torch.float32, torch.float16]:
      a = torch.randn(10, 10, device=device, dtype=torch_dtype)
      self.assertEqual(a.dtype, torch_dtype)

  def test_equal(self):
    tensor_a = torch.tensor([[1, 2], [3, 4]], device=device)
    tensor_b = torch.tensor([[1, 2], [3, 4]], device=device)
    tensor_c = torch.tensor([[1, 2], [1, 2]], device=device)
    assert torch.equal(tensor_a, tensor_b)
    assert not torch.equal(tensor_a, tensor_c)

  def test_linalg_eigh(self):
    a = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32, device=device)
    w, v = torch.linalg.eigh(a)
    np.testing.assert_equal(w.cpu().numpy(), [-1, 3])
    recon = (v @ torch.diag(w) @ v.T).cpu().numpy()
    np.testing.assert_allclose(recon, a.cpu().numpy(), atol=1e-6)

  def test_linalg_det(self):
    a = torch.diag(torch.tensor([1,2,3,4,5], dtype = torch.float32, device=device))
    b = torch.linalg.det(a)
    np.testing.assert_equal(b.cpu().numpy(), 120.0)

  def test_linalg_eigh(self):
    a = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32, device=device)
    w, v = torch.linalg.eigh(a)
    np.testing.assert_allclose(w.cpu().numpy(), [-1, 3], rtol=1e-5)
    recon = (v @ torch.diag(w) @ v.T).cpu().numpy()
    np.testing.assert_allclose(recon, a.cpu().numpy(), rtol=1e-5)

  def test_diag_vector_to_matrix(self):
    vec = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float32, device=device)
    mat = torch.diag(vec)
    expected = np.diag([1., 2., 3., 4., 5.])
    np.testing.assert_allclose(mat.cpu().numpy(), expected, rtol=1e-5)
    assert mat.shape == (5, 5)

  def test_diagonal_matrix_to_vector(self):
    mat = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=torch.float32, device=device)
    vec = torch.linalg.diagonal(mat)
    expected = np.array([1., 5., 9.])
    np.testing.assert_allclose(vec.cpu().numpy(), expected, rtol=1e-5)
    assert vec.shape == (3,)
    
  def test_permute(self):
    a = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    b = a.permute(2, 0, 1)  # (2,3,4) -> (4,2,3)
    assert b.shape == (4, 2, 3)
    np.testing.assert_equal(b.cpu().numpy(), a.cpu().numpy().transpose(2, 0, 1))

  def test_linalg_cross(self):
    a = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device)
    b = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)
    cross = torch.linalg.cross(a, b)
    np.testing.assert_equal(cross.cpu().numpy(), np.array([[0, -1, 0], [1, 0, 0]], dtype=np.float32))

  def test_scalar_assign(self):
    a = torch.tensor([1, 2, 3], device=device)
    a[1] = 4
    np.testing.assert_equal(a.cpu().numpy(), [1, 4, 3])

  @unittest.skip("meh")
  def test_str(self):
    a = torch.ones(4, device=device)
    print(str(a))

  def test_floor_div(self):
    a = torch.tensor([10., 7., 5.], device=device)
    b = torch.tensor([3., 2., 2.], device=device)
    result = a // b
    np.testing.assert_equal(result.cpu().numpy(), [3., 3., 2.])

  @unittest.skip("can't run")
  def test_mnist_index(self):
    GlobalCounters.reset()
    from tinygrad.nn.datasets import mnist
    X_train, Y_train, _, _ = mnist()
    X_train = torch.tensor(X_train.float().numpy(), device=device)
    Y_train = torch.tensor(Y_train.cast('int64').numpy(), device=device)
    samples = torch.randint(0, X_train.shape[0], (32,))
    X,Y = X_train[samples], Y_train[samples]
    X.cpu(), Y.cpu()
    self.assertLessEqual(GlobalCounters.global_ops, 10_000_000)

  def _test_diagonal(self, *shape):
    a = torch.randn(*shape, dtype=torch.float32, device=device)
    ref = np.diagonal(a.cpu().numpy(), axis1=-2, axis2=-1)
    diag = torch.linalg.diagonal(a)
    np.testing.assert_equal(diag.cpu().numpy(), ref)
    np.testing.assert_equal(diag[-1].cpu().numpy(), ref[-1])

  def test_diagonal_cube(self): self._test_diagonal(3, 3, 3)
  def test_diagonal_rectangular(self): self._test_diagonal(4, 5, 6)
  def test_diagonal_4d(self): self._test_diagonal(2, 3, 4, 5)


  def test_slice_inplace_zero(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:]
    b.zero_()
    expected = np.array([[1., 1., 1.],
                         [1., 0., 0.],
                         [1., 0., 0.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_slice_inplace_fill(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:]
    b.fill_(5.0)
    expected = np.array([[1., 1., 1.],
                         [1., 5., 5.],
                         [1., 5., 5.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_slice_inplace_mul(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:]
    b *= 2
    expected = np.array([[1., 1., 1.],
                         [1., 2., 2.],
                         [1., 2., 2.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_permute_slice_zero(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:].permute(1, 0)
    b.zero_()
    expected = np.array([[1., 1., 1.],
                         [1., 0., 0.],
                         [1., 0., 0.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_permute_slice_mul(self):
    a = torch.ones((3, 3), device=device)
    b = a[1:, 1:].permute(1, 0)
    b *= 2
    expected = np.array([[1., 1., 1.],
                         [1., 2., 2.],
                         [1., 2., 2.]])
    np.testing.assert_equal(a.cpu().numpy(), expected)

  def test_simple_slice_setitem(self):
    a = torch.tensor([10, 20, 30], device=device)
    a[1] = 99
    np.testing.assert_equal(a.cpu().numpy(), [10, 99, 30])

  def test_2d_slice_setitem(self):
    a = torch.zeros((3, 3), device=device)
    a[1, 2] = 99
    self.assertEqual(a[1, 2].item(), 99)
    self.assertEqual(a.sum().item(), 99)

  def test_view_copy(self):
    a = torch.tensor([10, 20, 30], device=device)
    view = a[1]
    view.copy_(torch.tensor(88, device=device))
    np.testing.assert_equal(a.cpu().numpy(), [10, 88, 30])

  def test_diag_2d_input(self):
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
    d = torch.diag(a)
    np.testing.assert_equal(d.cpu().numpy(), [1, 5, 9])

  def test_diag_1d_input(self):
    a = torch.tensor([1, 2, 3], device=device)
    d = torch.diag(a)
    expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    np.testing.assert_equal(d.cpu().numpy(), expected)

  def test_permute_view_tracking(self):
    a = torch.ones((2, 3, 4), device=device)
    b = a.permute(2, 0, 1)
    self.assertEqual(b.shape, (4, 2, 3))

  def test_detach_view_creation(self):
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = a.detach()
    np.testing.assert_equal(b.cpu().numpy(), [1.0, 2.0, 3.0])

  def test_view_zero_inplace(self):
    a = torch.ones((4, 4), device=device)
    view = a[1:3, 1:3]
    view.zero_()
    self.assertEqual(view.sum().item(), 0)

  def test_view_fill_inplace(self):
    a = torch.zeros((4, 4), device=device)
    view = a[1:3, 1:3]
    view.fill_(5)
    self.assertEqual(view.sum().item(), 20)

  def test_permute_contiguous(self):
    a = torch.tensor([[1, 2], [3, 4]], device=device)
    b = a.permute(1, 0)
    c = b.contiguous()
    expected = [[1, 3], [2, 4]]
    np.testing.assert_equal(c.cpu().numpy(), expected)

  def test_diag_2d_extract_diagonal(self):
    a = torch.tensor([[1, 2], [3, 4]], device=device)
    result = torch.diag(a)
    np.testing.assert_equal(result.cpu().numpy(), [1, 4])


  def test_slice_inplace_multiply_offset_preservation(self):
    a = torch.tensor([1, 2, 3], device=device)
    a[1:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [1, 4, 6])

  def test_slice_inplace_mul_pattern(self):
    a = torch.tensor([1, 2, 3, 4], device=device)
    a[:2] *= 3
    a[2:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [3, 6, 6, 8])

  def test_chained_slice_column(self):
    a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    torch_res = a[:, 1:2][:, 0:1].cpu().numpy()
    cpu_res = torch.arange(16, dtype=torch.float32).reshape(4, 4)[:, 1:2][:, 0:1].numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_slice_with_step(self):
    a = torch.arange(20, dtype=torch.float32, device=device)
    torch_res = a[::2][1:4].cpu().numpy()
    cpu_res = torch.arange(20, dtype=torch.float32)[::2][1:4].numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_dot_vector_matrix(self):
    a = torch.arange(65, dtype=torch.float32, device=device)
    b = torch.arange(65*45, dtype=torch.float32, device=device).reshape(65, 45)
    torch_res = a.matmul(b).reshape(-1).cpu().numpy()
    cpu_res = torch.arange(65, dtype=torch.float32).matmul(torch.arange(65*45, dtype=torch.float32).reshape(65, 45)).numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_alias_passthrough(self):
    a = torch.randn(3, 3, device=device)
    alias_view = torch.ops.aten.alias(a)
    alias_view += 1
    np.testing.assert_equal(a.cpu().numpy(), alias_view.cpu().numpy())

  def test_split_simple_vector(self):
    a = torch.arange(10, dtype=torch.float32, device=device)
    torch_chunks = a.split([1,4,5])
    cpu_chunks = torch.arange(10, dtype=torch.float32).split([1,4,5])
    for tc, cc in zip(torch_chunks, cpu_chunks):
      np.testing.assert_equal(tc.cpu().numpy(), cc.cpu().numpy())

  def test_split_matches_torch(self):
    a = torch.arange(10, dtype=torch.float32, device=device)
    torch_chunks = a.split([1,4,5])
    tiny_chunks = [chunk.cpu().numpy() for chunk in torch_chunks]
    cpu_chunks = [torch.arange(10, dtype=torch.float32).split([1,4,5])[i].numpy() for i in range(3)]
    for tr, cr in zip(tiny_chunks, cpu_chunks): np.testing.assert_equal(tr, cr)

  def test_sum_matches_torch(self):
    a = torch.arange(6, dtype=torch.float32, device=device).reshape(2,3)
    torch_res = a.sum().cpu().numpy()
    cpu_res = torch.arange(6, dtype=torch.float32).reshape(2,3).sum().numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_view_matches_torch(self):
    a = torch.arange(6, dtype=torch.float32, device=device)
    torch_res = a.view(2, 3).cpu().numpy()
    cpu_res = torch.arange(6, dtype=torch.float32).view(2, 3).numpy()
    np.testing.assert_equal(torch_res, cpu_res)

  def test_realize_with_views_offset_preservation(self):
    a = torch.tensor([10, 20, 30, 40], device=device)
    b = a[2:]  # view starting at offset 2
    b *= 5  # triggers realize_with_views
    np.testing.assert_equal(a.cpu().numpy(), [10, 20, 150, 200])
    np.testing.assert_equal(b.cpu().numpy(), [150, 200])

  def test_view_zero_with_indices(self):
    a = torch.tensor([1, 2, 3, 4], device=device)
    a[1:3].zero_()
    np.testing.assert_equal(a.cpu().numpy(), [1, 0, 0, 4])

  def test_view_fill_with_indices(self):
    a = torch.tensor([1, 2, 3, 4], device=device)
    a[::2].fill_(9)
    np.testing.assert_equal(a.cpu().numpy(), [9, 2, 9, 4])

  def test_nested_slice_inplace_ops(self):
    a = torch.tensor([1, 2, 3, 4, 5, 6], device=device)
    a[:3] += 10
    a[3:] *= 2
    np.testing.assert_equal(a.cpu().numpy(), [11, 12, 13, 8, 10, 12])

  def test_diag_1d_still_works(self):
    a = torch.tensor([1, 2, 3], device=device)
    result = torch.diag(a)
    expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    np.testing.assert_equal(result.cpu().numpy(), expected)

  def test_diag_backward(self):
    a = torch.randn(5, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    b.sum().backward()
    assert a.grad is not None

  def test_diagonal_backward(self):
    a = torch.randn(5, 5, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diagonal(a)
    b.sum().backward()
    assert a.grad is not None

  def test_expand_backward(self):
    a = torch.randn(4, 3, 1, 6, dtype=torch.float32, device=device, requires_grad=True)
    b = a.expand(4, 3, 2, 6)
    b.sum().backward()
    assert a.grad is not None

  def test_einsum_backward(self):
    a = torch.randn(10, 10, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.einsum('ij->ji', a)
    b.sum().backward()
    assert a.grad is not None

  def test_diag_backward_gradient_values(self):
    # Test diag from 1D vector -> 2D matrix
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    loss = b.sum()
    loss.backward()
    
    # Gradient should be 1.0 for each element (sum of diagonal matrix puts 1 on each diagonal element)
    expected_grad = torch.ones(3, dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_diagonal_backward_gradient_values(self):
    # Test diagonal from 2D matrix -> 1D vector
    a = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diagonal(a)  # Should be [1.0, 5.0, 9.0]
    loss = b.sum()
    loss.backward()
    
    # Gradient should be 1.0 only on diagonal elements, 0 elsewhere
    expected_grad = torch.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_expand_backward_gradient_values(self):
    # Test expand with dimension size 1 -> N
    a = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = a.expand(3, 4)  # Expand from (3,1) to (3,4)
    loss = b.sum()
    loss.backward()
    
    # Gradient should sum across the expanded dimension
    # Each row is repeated 4 times, so gradient should be 4.0 for each element
    expected_grad = torch.tensor([[4.0], [4.0], [4.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_expand_backward_with_leading_dims(self):
    # Test expand that adds new leading dimensions
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device, requires_grad=True)  # (1, 2)
    b = a.expand(3, 1, 2)  # Add leading dimension
    loss = b.sum()
    loss.backward()
    
    # Gradient should sum across the new leading dimension
    # Each element is repeated 3 times in the new dimension
    expected_grad = torch.tensor([[3.0, 3.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_diag_2d_to_1d_backward(self):
    # Test diagonal extraction from 2D matrix (this goes through diag operation with ndim==2)
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device, requires_grad=True)
    # When calling .diag() on 2D, it extracts diagonal
    b = torch.diag(a)  # Should be [1.0, 4.0]
    loss = b.sum()
    loss.backward()
    
    # Gradient should be 1.0 on diagonal, 0.0 elsewhere
    expected_grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_expand_complex_backward(self):
    # Test expand with both size-1 expansion and leading dimension addition
    a = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device, requires_grad=True)  # (1, 1, 2)
    b = a.expand(2, 3, 2)  # (2, 3, 2) - adds first dim and expands second dim
    loss = b.sum()
    loss.backward()
    
    # Total gradient should be 2*3 = 6 for each element
    expected_grad = torch.tensor([[[6.0, 6.0]]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_diag_backward_with_scaling(self):
    # Test diag backward with non-uniform gradients
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    # Create non-uniform gradient by multiplying
    loss = (b * torch.tensor([[2.0, 0.0, 0.0],
                               [0.0, 3.0, 0.0],
                               [0.0, 0.0, 4.0]], device=device)).sum()
    loss.backward()
    
    # Gradient should be [2.0, 3.0, 4.0] (diagonal of the multiplier)
    expected_grad = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_repeat_basic(self):
    # Test basic repeat operation
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = a.repeat(2, 1)
    expected = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_repeat_multidim(self):
    # Test repeat with multiple dimensions
    a = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    b = a.repeat(2, 3)
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3).repeat(2, 3)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_repeat_backward(self):
    # Test repeat with gradients
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = a.repeat(3, 2)
    loss = b.sum()
    loss.backward()
    # Each element is repeated 3*2 = 6 times
    expected_grad = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_cumsum_1d(self):
    # Test 1D cumsum
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device=device)
    b = torch.cumsum(a, dim=0)
    expected = torch.tensor([1, 3, 6, 10], dtype=torch.float32)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_cumsum_2d(self):
    # Test 2D cumsum along different dimensions
    a = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    b = torch.cumsum(a, dim=0)
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).cumsum(dim=0)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())
    
    c = torch.cumsum(a, dim=1)
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).cumsum(dim=1)
    np.testing.assert_equal(c.cpu().numpy(), expected.numpy())

  def test_cumsum_large(self):
    a = torch.arange(513, dtype=torch.float32, device=device)
    b = torch.cumsum(a, dim=0)
    expected = torch.arange(513, dtype=torch.float32).cumsum(dim=0)
    np.testing.assert_allclose(b.cpu().numpy(), expected.numpy(), rtol=1e-5)

  def test_cumsum_backward(self):
    # Test cumsum with gradients
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.cumsum(a, dim=0)
    loss = b.sum()
    loss.backward()
    # Gradient propagates: last element affects all cumsum outputs
    expected_grad = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_constant_pad_nd_1d(self):
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = torch.nn.functional.pad(a, (1, 2), mode='constant', value=0)
    expected = torch.tensor([0, 1, 2, 3, 0, 0], dtype=torch.float32)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_constant_pad_nd_2d(self):
    a = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    b = torch.nn.functional.pad(a, (1, 1, 1, 1), mode='constant', value=0)
    expected = torch.nn.functional.pad(torch.arange(6, dtype=torch.float32).reshape(2, 3), (1, 1, 1, 1), mode='constant', value=0)
    np.testing.assert_equal(b.cpu().numpy(), expected.numpy())

  def test_constant_pad_nd_2d_backward(self):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.nn.functional.pad(a, (1, 1, 1, 1), mode='constant', value=0)
    loss = b.sum()
    loss.backward()
    # Gradient only flows to original elements, not padding
    expected_grad = torch.ones((2, 2), dtype=torch.float32)
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected_grad.numpy(), rtol=1e-5)

  def test_negative_strides_cumsum_backward(self):
    # Test that cumsum backward doesn't produce negative strides
    a = torch.randn(5, device=device, requires_grad=True)
    b = torch.cumsum(a, dim=0)
    b.sum().backward()
    # Should be able to call .cpu() on gradient without stride issues
    grad = a.grad.cpu().numpy()
    self.assertEqual(len(grad), 5)

  def test_cumsum_fix_gradient_values(self):
    # Test cumsum gradient computation matches expected values
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.cumsum(a, dim=0)
    loss = b.sum()
    loss.backward()
    expected = np.array([4.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(a.grad.cpu().numpy(), expected, rtol=1e-5)

  def test_diag_operations_comprehensive(self):
    # Test diag 1D to 2D
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True)
    b = torch.diag(a)
    expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    np.testing.assert_equal(b.detach().cpu().numpy(), expected)
    
    # Test diag 2D to 1D
    c = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
    d = torch.diag(c)
    np.testing.assert_equal(d.cpu().numpy(), [1, 5, 9])
    
    # Test diagonal operation
    e = torch.randn(5, 5, dtype=torch.float32, device=device, requires_grad=True)
    f = torch.diagonal(e)
    self.assertEqual(f.shape, (5,))

if __name__ == "__main__":
  unittest.main()
