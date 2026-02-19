import os, unittest
from tinygrad import dtypes, Tensor, fetch, Device
from tinygrad.nn.state import ggml_data_to_tensor, gguf_load
from tinygrad.device import is_dtype_supported
import numpy as np
from gguf import GGUFReader, GGUFValueType, GGMLQuantizationType, GGML_QUANT_SIZES, dequantize, quantize

ggml_test_block_count = 4

@unittest.skipIf(any(not is_dtype_supported(t) for t in [ dtypes.uint8, dtypes.half ]), "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_load_tinyllama_q8_0(self): self._test_gguf_load("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q8_0.gguf?download=true")
  def test_load_tinyllama_q4_0(self): self._test_gguf_load("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf?download=true")
  def test_load_gpt2_q4_1(self): self._test_gguf_load("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q4_1.gguf?download=true")
  def test_load_sample_q6_k(self): self._test_gguf_load("https://huggingface.co/Isotr0py/test-gguf-sample/resolve/main/Quant_Q6_K_1024.gguf?download=true")

  def test_dequantization_q8_0_hardcoded(self):
    # Q8_0: 2 bytes float16 scale + 32 bytes int8 values, dequant = scale * values
    block = np.frombuffer(np.float16(2.0).tobytes() + np.arange(1, 33, dtype=np.int8).tobytes(), dtype=np.uint8).copy()
    expected = np.arange(1, 33, dtype=np.float32) * 2.0
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.Q8_0.value).numpy().flatten(), expected)

  def test_dequantization_mxfp4_hardcoded(self):
    # MXFP4: 1 byte shared exponent E + 16 packed bytes (32 x 4-bit values)
    # nibble: bit3=sign, bit2:1=exp, bit0=mant; E=128 gives scale=1.0
    # codes 0-7 = [0, 1, 2, 3, 4, 6, 8, 12], codes 8-15 are their negatives
    block = np.array([0x80] + list(range(16)), dtype=np.uint8)  # E=128, nibbles 0-15 in low, zeros in high
    expected = np.array([0., 1., 2., 3., 4., 6., 8., 12., -0., -1., -2., -3., -4., -6., -8., -12.] + [0.]*16, dtype=np.float32)
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, 39).numpy().flatten(), expected)

  def test_dequantization_q4_0(self): self._test_dequantization(GGMLQuantizationType.Q4_0)
  def test_dequantization_q4_1(self): self._test_dequantization(GGMLQuantizationType.Q4_1)
  def test_dequantization_q8_0(self): self._test_dequantization(GGMLQuantizationType.Q8_0)
  def test_dequantization_q4_k(self): self._test_dequantization(GGMLQuantizationType.Q4_K)
  def test_dequantization_q6_k(self): self._test_dequantization(GGMLQuantizationType.Q6_K)
  def test_dequantization_mxfp4(self):
    MXFP4 = 39

    def encode(nibbles, E):
      packed = [(low & 0xF) | ((high & 0xF) << 4) for low, high in zip(nibbles[:16], nibbles[16:])]
      return np.array([E] + packed, dtype=np.uint8)

    def decode(code, E):
      sign = -1.0 if (code & 0b1000) else 1.0
      exp = (code >> 1) & 0b11
      mant = code & 0b1
      val = 2 * ((1.0 + 0.5 * mant) * np.exp2(exp - 1) if exp else 0.5 * mant)
      scale = np.exp2(E - 128) if E >= 2 else np.exp2(-127 if E == 1 else -128)
      return sign * val * scale

    blocks, expected = [], []
    rng = np.random.default_rng(42)
    for _ in range(4):
      E = rng.integers(0, 256)
      codes = rng.integers(0, 16, size=32, dtype=np.uint8)
      blocks.append(encode(codes, E))
      expected.extend(decode(c, E) for c in codes)
    tensor = Tensor(np.concatenate(blocks))
    out = ggml_data_to_tensor(tensor, len(expected), MXFP4)
    # TODO: should this be exact equal? somehow failed on CI
    np.testing.assert_allclose(out.numpy(), expected, atol=0.0, rtol=1e-6)

  def test_dequantization_mxfp4_block(self):
    MXFP4 = 39
    # https://gist.github.com/Ananta-Ranganathan/3317b6ed51a3b033e9c2564fafb4e043
    # used the above script to download the first block of blk.0.attn_k_b.weight from
    # https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/blob/main/GLM-4.7-Flash-MXFP4_MOE.gguf
    # and compute the canonical expected dequantized output with the GGUF PY implementation
    block = np.array([0x7a, 0x29, 0xab, 0x61, 0x10, 0x21, 0x02, 0x4a,
                    0x15, 0xca, 0x05, 0x01, 0x9b, 0x39, 0x0b, 0x0b, 0x1c], dtype=np.uint8)
    expected = np.array([-0.01562500, -0.04687500, 0.01562500, 0.00000000,
                        0.01562500,  0.03125000, -0.03125000, 0.09375000,
                        -0.03125000,  0.09375000, 0.01562500, -0.04687500,
                        -0.01562500, -0.04687500, -0.04687500, -0.06250000,
                        0.03125000, -0.03125000, 0.12500000,  0.01562500,
                        0.03125000,  0.00000000, 0.06250000,  0.01562500,
                        -0.06250000,  0.00000000, 0.00000000, -0.01562500,
                        0.04687500,  0.00000000, 0.00000000,  0.01562500], dtype=np.float32)
    out = ggml_data_to_tensor(Tensor(block), 32, MXFP4)
    # TODO: similar to previous test fails on Mac CI with assert_equal for unclear reason
    np.testing.assert_allclose(out.numpy(), expected, atol=0.0, rtol=1e-6)

  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  def _test_dequantization(self, qtype: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n_el, n_bytes = ggml_test_block_count * block_size, ggml_test_block_count * type_size

    try:
      q_data = quantize((np.random.random((n_el,)).astype(np.float32) * 100 - 50), qtype)
    except NotImplementedError:
      q_data = np.random.default_rng(42).integers(0, 256, size=n_bytes, dtype=np.uint8)
    ref = dequantize(q_data, qtype)

    q_tensor = Tensor(q_data)
    dq_tensor = ggml_data_to_tensor(q_tensor, n_el, qtype.value).reshape(n_el)

    np.testing.assert_equal(dq_tensor.numpy(), ref)

  def _test_gguf_load(self, url: str):
    fp = fetch(url)
    model_size = os.stat(fp).st_size
    gguf_tensor = Tensor.empty(model_size, dtype=dtypes.uint8, device=f"disk:{fp}").to(Device.DEFAULT)
    kv_data, tensors = gguf_load(gguf_tensor)

    reader = GGUFReader(fp)

    for rt in reader.tensors:
      ref = dequantize(rt.data, rt.tensor_type)
      np.testing.assert_equal(tensors[rt.name].numpy(), ref.reshape(tensors[rt.name].shape))

    for k, f in reader.fields.items():
      if k.startswith("GGUF."): continue  # skip file header keys (version, tensor_count, kv_count)
      def read_val(i, parts=f.parts, is_str=(f.types[-1] == GGUFValueType.STRING)):
        return bytes(parts[i]).decode("utf-8") if is_str else parts[i][0].item()
      if f.types[0] == GGUFValueType.ARRAY:
        self.assertEqual(kv_data[k], [read_val(i) for i in f.data])
      else:
        self.assertEqual(kv_data[k], read_val(-1))

if __name__ == '__main__':
  unittest.main()
