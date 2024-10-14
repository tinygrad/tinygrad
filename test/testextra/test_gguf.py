import ctypes
import numpy as np
import unittest
from extra.gguf import GGUFConverters
from tinygrad.tensor import Tensor
import ggml

params = ggml.ggml_init_params(mem_size=0, mem_buffer=None)
ctx = ggml.ggml_init(params)

np.random.seed(1337)
block_count = 4

class TestGGUF(unittest.TestCase):
  def test_dequantization_q4_0(self): self._test_dequantization(ggml.GGML_TYPE_Q4_0)
  def test_dequantization_q8_0(self): self._test_dequantization(ggml.GGML_TYPE_Q8_0)
  def test_dequantization_q6_k(self): self._test_dequantization(ggml.GGML_TYPE_Q6_K)
  def _test_dequantization(self, ttype: int):
    type_traits = ggml.ggml_internal_get_type_traits(ttype)
    n_el, n_bytes = block_count * type_traits.blck_size, block_count * type_traits.type_size

    data_in = (np.random.random((n_el,)).astype(np.float32) * 100).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_q_data, c_dq_data = (ctypes.c_char * n_bytes)(0), (ctypes.c_float * n_el)(0)
    type_traits.from_float(data_in, c_q_data, n_el)
    type_traits.to_float(c_q_data, c_dq_data, n_el)

    q_tensor = Tensor(np.frombuffer(c_q_data, dtype=np.uint8, count=n_bytes))
    dq_tensor = GGUFConverters.converter_map[ttype](q_tensor, n_el).reshape(n_el)

    np.testing.assert_equal(dq_tensor.numpy(), np.frombuffer(c_dq_data, dtype=np.float32))

if __name__ == '__main__':
  unittest.main()
