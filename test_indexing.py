import pytest

from tinygrad import Tensor, UOp


def test_vmap():
  def f(x): return x.sum(axis=0)*2

  x = Tensor.ones(3, 10, 2)
  a = UOp.range(3, -1)
  out = f(x[a]).reshape(1, 2).expand(a, 2).contiguous()
  out.realize()
  assert (out==20).all().item()

def test_flat_indexing():
  n, m, p = 20, 10, 4
  x = Tensor([list(range(m))]*n, dtype="float64")
  i = [2*k + m*k for k in range(p)]  # row-major idexing of (k, 2*k)
  assert x.flatten()[i].tolist() == pytest.approx([2.0 * k for k in range(p)])

def test_vmap_after_indexing():
  def fn(x:Tensor)->Tensor: return x[[0,2,4]]
  x = Tensor([[0,1,2,3,4,5]]*3)
  r = UOp.range(3, -1)
  out = fn(x[r]).reshape(1, 3).expand(r, 3).contiguous()
  out.realize()
  assert out.flatten().tolist()==pytest.approx([0,2,4,0,2,4,0,2,4])

def test_indexing_after_vmap():
  def fn(x: Tensor) -> Tensor:
    return Tensor.arange(6) * x

  x = Tensor.ones(3,6)
  r = UOp.range(3, -1)
  i,j = [0,1,2], [0,2,4]
  o = fn(x[r]).reshape(1, 6).expand(r, 6)[i,j]
  assert o.tolist() == pytest.approx([0.0, 2.0, 4.0])


def test_flat_indexing_after_vmap():
  def fn(x: Tensor) -> Tensor:
    return Tensor.arange(6) * x

  x = Tensor.ones(3, 6)
  r = UOp.range(3, -1)
  o = fn(x[r]).reshape(1, 6).expand(r, 6).flatten()
  i = [0, 7, 14]
  assert o[i].tolist() == pytest.approx([0.0, 2.0, 4.0])

# def test_slicing_after_vmap():
#   def fn(x: Tensor) -> Tensor:
#     return Tensor.arange(6) * x
#   x = Tensor.ones(3,6)
#   r = UOp.range(3, -1)
#   i = list(range(p))
#   o = fn(x[r]).reshape(1, 6).expand(r, 6)[i].contiguous()
#   assert o.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
