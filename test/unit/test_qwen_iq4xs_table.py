# qwen4exp._dequant_rows_half: the V3.1 token_embd (IQ4_XS) dequantized to half on the host: bit-equal to the gguf loader's dequantization (DEV=CPU)
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.qwen4exp import _dequant_rows_half

class TestIQ4XSTable(unittest.TestCase):
  def test_chunks_equal_the_loader(self):
    rows, dim = 37, 512  # a chunk of 16 rows twice and a remainder of 5
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, size=(rows, dim//256, 136), dtype=np.uint8)
    raw[..., :2] = rng.uniform(-1, 1, size=(rows, dim//256, 1)).astype(np.float16).view(np.uint8)  # d finite
    flat = Tensor(raw.reshape(-1))
    out = bytearray(rows*dim*2)
    _dequant_rows_half(flat, rows, dim, 23, memoryview(out), chunk=16)
    ref = ggml_data_to_tensor(flat, rows*dim, 23).cast(dtypes.float16).numpy().reshape(-1)
    np.testing.assert_array_equal(np.frombuffer(bytes(out), dtype=np.uint16), ref.view(np.uint16))

if __name__ == "__main__": unittest.main()
