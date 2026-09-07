import numpy as np
from tinygrad import Tensor,dtypes


def test_float_add_nonuniform_odd_shape():
  a=np.arange(407,dtype=np.float32).reshape(37,11)/13-9
  b=np.cos(np.arange(407,dtype=np.float32)).reshape(37,11)
  np.testing.assert_allclose((Tensor(a)+Tensor(b)).numpy(),a+b,rtol=1e-6,atol=1e-6)


def test_integer_full_width_multiply_and_bitwise_source_negation():
  a=np.array([0x10001,0xffff0001,0xffffffff,0x80008000,123456789],np.uint32)
  b=np.array([0x20003,0x0002ffff,0xffffffff,0x8000ffff,987654321],np.uint32)
  np.testing.assert_array_equal((Tensor(a)*Tensor(b)).numpy(),a*b)
  np.testing.assert_array_equal((~Tensor(a)&Tensor(b)).numpy(),~a&b)


def test_signed_byte_conversion():
  a=np.array([-128,-17,-1,0,1,17,127],np.int8)
  np.testing.assert_array_equal(Tensor(a).cast(dtypes.int32).numpy(),a.astype(np.int32))

def test_float_to_half_conversion_uses_nearest_even():
  values=np.array([1.0006,-1.0006,1.0015,-1.0015,2051,65520],np.float32)
  with np.errstate(over='ignore'):
    expected=values.astype(np.float16).view(np.uint16)
  actual=Tensor(values).cast(dtypes.half).numpy().view(np.uint16)
  np.testing.assert_array_equal(actual,expected)

def test_trigonometric_reduction_preserves_large_arguments():
  values=np.array([0,.5,-.5,10,1e4,1e6,-1e6],np.float32)
  np.testing.assert_allclose(Tensor(values).sin().numpy(),np.sin(values),atol=3e-3,rtol=3e-3)
  np.testing.assert_allclose(Tensor(values).cos().numpy(),np.cos(values),atol=3e-3,rtol=3e-3)


def test_half_gemm_nonuniform():
  a=(np.arange(9*17).reshape(9,17)%11-5).astype(np.float16)/8
  b=(np.arange(17*7).reshape(17,7)%13-6).astype(np.float16)/8
  np.testing.assert_allclose((Tensor(a)@Tensor(b)).numpy(),a@b,rtol=1e-3,atol=2e-3)


def test_reductions_and_divergent_selection():
  a=(np.arange(513,dtype=np.float32)%17)-8
  out=(Tensor(a)>0).where(Tensor(a)*3,Tensor(a)-2)
  np.testing.assert_array_equal(out.numpy(),np.where(a>0,a*3,a-2))
  np.testing.assert_equal(out.sum().numpy(),np.where(a>0,a*3,a-2).sum())


def test_float_gemv_nonuniform():
  a=(np.arange(64,dtype=np.float32)%11-5).reshape(1,64)/8
  b=(np.arange(4096,dtype=np.float32).reshape(64,64)%13-6)/8
  np.testing.assert_allclose((Tensor(a)@Tensor(b)).numpy(),a@b,rtol=1e-5,atol=1e-5)
