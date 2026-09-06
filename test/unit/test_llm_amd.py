import unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, function
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, q8_quantize, flash_attention
from tinygrad.llm.gguf import ggml_data_to_tensor

class TestQ8Quantize(unittest.TestCase):
  def test_word_quant_weights_use_typed_buffer_view(self):
    for ggml_type, type_size in ((13, 176), (23, 136)):
      with self.subTest(ggml_type=ggml_type):
        raw = Tensor(np.zeros(type_size + 4, dtype=np.uint8), device="CPU").contiguous().realize()[4:]
        decoded = ggml_data_to_tensor(raw, 256, ggml_type).reshape(1, 256)
        linear = Linear(256, 1, bias=False)
        linear.set_quantized(decoded)
        self.assertEqual(linear.ggml_type, ggml_type)
        self.assertEqual(linear.weight.dtype, dtypes.uint32)
        self.assertEqual(linear.weight.nbytes(), type_size)
        self.assertEqual(linear.weight.uop.buf_uop.buffer.offset, 4)

  def test_values_and_scales(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    x = np.linspace(-3.1, 2.7, 64, dtype=np.float32).reshape(2, 32)
    quant, scale, gsum = q8_quantize(Tensor(x), 2, 32)
    scale_np = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True) / 127, 1e-8)
    expected = np.clip(np.rint(x / scale_np), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(2, 32).numpy(), expected)
    np.testing.assert_allclose(scale.numpy(), scale_np, rtol=1e-6)
    # xsum holds the two per-16 sums per 32-wide group
    np.testing.assert_array_equal(gsum.numpy().reshape(2, 2), expected.reshape(2, 2, 16).sum(-1).astype(np.float32))

  def test_quantize_rounding_ties(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    values = np.array([-127,127]+[i+0.5 for i in range(-15,15)],dtype=np.float32)
    quant,_,_ = q8_quantize(Tensor(values),1,32)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(32).numpy(),np.rint(values).astype(np.int8))

  def test_q6_linear_compiles_in_function(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, 210, dtype=np.uint8)
    packed[-2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 256, 14).reshape(1, 256)
    linear = Linear(256, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    self.assertTrue(np.isfinite(run(Tensor.randn(1, 256)).realize().item()))
    # the Q6 weight is repacked: 210-byte blocks padded to 212 (one block = 53 words)
    self.assertEqual(linear.weight.uop.buf_uop.buffer.nbytes, 53*4)
    self.assertEqual(linear.weight.dtype, dtypes.uint32)

  def test_q4_k_linear(self): self._test_quant_linear(12, 144)
  def test_iq4_linear(self): self._test_quant_linear(23, 136)
  def test_q5_linear(self): self._test_quant_linear(13, 176)

  def test_quant_linear_partial_output_tile(self):
    # Cover a sub-tile output, a trailing tile, and IQ4's larger-output tile selection.
    for typ, size, outputs, tokens in ((12, 144, 16, 16), (12, 144, 48, 32), (13, 176, 48, 16), (23, 136, 4112, 32)):
      with self.subTest(ggml_type=typ, out_features=outputs):
        self._test_quant_linear(typ, size, in_features=256, out_features=outputs, token_counts=(tokens,))

  def test_quant_linear_preserves_rope_permutation(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    for typ, size in ((12, 144), (13, 176), (14, 210), (23, 136)):
      with self.subTest(ggml_type=typ):
        packed = rng.integers(0, 256, (16, size), dtype=np.uint8)
        packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ != 14: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
        decoded = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256).half()
        original = decoded.numpy()
        x = rng.normal(size=(3, 256)).astype(np.float16)
        for prefix in (None, 0, 4):
          with self.subTest(prefix=prefix):
            w = decoded.reshape(2, 8, 256)
            if prefix is None:
              weight = w.rearrange("n (h two) d -> n (two h) d", two=2)
            else:
              weight = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1)
            start = prefix or 0
            rows = np.arange(16).reshape(2, 8)
            order = np.concatenate((rows[:, :start], rows[:, start:].reshape(2, -1, 2).transpose(0, 2, 1).reshape(2, -1)), axis=1)
            linear = Linear(256, 16, bias=False)
            linear.weight = weight.reshape(16, 256)
            np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ original[order.flatten()].astype(np.float32).T,
                                       rtol=3e-3, atol=2e-2)
            self.assertIsNone(linear.ggml_type)

  def test_dense_gemv_bias(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    w, bias = rng.normal(size=(32, 128)).astype(np.float16), rng.normal(size=32).astype(np.float16)
    linear = Linear(128, 32)
    linear.weight, linear.bias = Tensor(w), Tensor(bias)
    for tokens in (1, 3):
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, 128)).astype(np.float16)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ w.astype(np.float32).T + bias, rtol=2e-3, atol=2e-3)

  def _test_quant_linear(self, ggml_type, block_bytes, in_features=2048, out_features=64, token_counts=(1, 3, 32, 64, 128)):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (out_features*in_features//256, block_bytes), dtype=np.uint8)
    packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    if ggml_type in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    linear.weight = decoded
    for tokens in token_counts:
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, in_features)).astype(np.float32 if tokens == 3 else np.float16)
        reference_x = x.astype(np.float32)
        if tokens < 16:
          grouped = reference_x.reshape(tokens, -1, 32)
          scale = np.maximum(np.abs(grouped).max(-1, keepdims=True) / 127, 1e-8)
          reference_x = (np.clip(np.rint(grouped/scale), -127, 127)*scale).reshape(tokens, in_features)
        reference_w = weight if tokens < 16 else weight.astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), reference_x @ reference_w.T, rtol=3e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, ggml_type)

  def test_q6_linear_multiple_tokens(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, blocks = 2048, 16*2048//256
    packed = rng.integers(0, 256, blocks*210, dtype=np.uint8)
    for i in range(blocks): packed[i*210+208:i*210+210] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16*in_features, 14).reshape(16, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 14)

    # symbolic token counts take the padded kernel path and give the same results
    generic = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(generic, {"weight":decoded}, verbose=False, realize=False)
    sym = Tensor(np.concatenate([x, np.zeros((1, in_features), np.float32)])).contiguous()[:UOp.variable("tokens", 1, 4).bind(3)]
    np.testing.assert_allclose(generic(sym)[:3].numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertTrue(generic.use_custom_quant)
    self.assertEqual(generic.ggml_type, 14)

  def test_attention_uses_physical_cache_length(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.half).contiguous()
    assigned = Tensor(cache.uop.after(cache[:, :, :, 0:1, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, 1).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

  def test_flash_attention_decode_gqa_output_layout(self): self._test_flash_decode(4, 1, 128, 256, 3)
  def test_flash_attention_decode_large_gqa_group(self): self._test_flash_decode(8, 1, 256, 256, 73)

  def _test_flash_decode(self, heads, kv_heads, dim, n, valid):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = rng.normal(size=(1, heads, 1, dim)).astype(np.float16)
    cache = rng.normal(size=(2, 1, kv_heads, n, dim)).astype(np.float16)
    k, v = (np.repeat(c[0, :, :valid].astype(np.float32), heads//kv_heads, axis=0) for c in cache)
    scores = q[0].astype(np.float32) @ k.transpose(0, 2, 1) / np.sqrt(dim)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    np.testing.assert_allclose(flash_attention(Tensor(q), Tensor(cache), valid).numpy(), expected[None], rtol=2e-3, atol=2e-3)

  def test_prefill_attention_nonfinite_cache_tail(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = Tensor.zeros(1, 2, 32, 128, dtype=dtypes.half)
    values = rng.normal(size=(33, 128)).astype(np.float16)
    expected = np.stack([values[:i+2].astype(np.float32).mean(0) for i in range(32)])[None, None].repeat(2, axis=1)
    for tail in (np.nan, np.inf, -np.inf):
      with self.subTest(tail=tail):
        cache = np.full((2, 1, 1, 64, 128), tail, dtype=np.float16)
        cache[0, :, :, :33] = 0
        cache[1, :, :, :33] = values
        valid = UOp.variable("valid_end", 32, 64).bind(33)
        cache_tensor = Tensor(cache).realize()
        assigned = Tensor(cache_tensor.uop.after(Tensor(valid).uop))
        out = flash_attention(q, assigned, valid)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_flash_attention_decode_beyond_256_chunks(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    n = 257 * 64
    q = Tensor.zeros(1, 1, 1, 32, dtype=dtypes.half).realize()
    k = Tensor.zeros(1, 1, n, 32, dtype=dtypes.half)
    v = Tensor.zeros(1, 1, n-64, 32, dtype=dtypes.half).cat(Tensor.ones(1, 1, 64, 32, dtype=dtypes.half), dim=2)
    cache = Tensor.stack(k, v).contiguous().realize()
    for valid, expected in ((1, 0), (n, 1/257)):
      with self.subTest(valid=valid):
        valid_kv_len = UOp.variable("valid_kv_len", 1, n).bind(valid)
        assigned = Tensor(cache.uop.after(Tensor(valid_kv_len).uop))
        np.testing.assert_allclose(flash_attention(q, assigned, valid_kv_len).numpy(), expected, rtol=2e-3, atol=2e-4)

  def test_flash_attention_decode_long_context_random(self):
    self._test_flash_decode(8, 2, 128, 257*64, 257*64-13)  # past 256 chunks, with a ragged tail

  def test_flash_attention_decode_chunk_round_accumulator_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    valid_kv_len, max_kv_len = 6749, 6784  # three chunk rounds, with a ragged tail
    q = Tensor.zeros(1, 8, 1, 32, dtype=dtypes.half).realize()
    cache = Tensor.stack(Tensor.zeros(1, 1, max_kv_len, 32, dtype=dtypes.half),
                         Tensor.full((1, 1, max_kv_len, 32), 5500, dtype=dtypes.half)).contiguous().realize()
    np.testing.assert_allclose(flash_attention(q, cache, valid_kv_len).numpy(), 5500, rtol=2e-3, atol=2e-3)

  def test_prefill_attention_unaligned_start(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    start_pos = 1718
    q = Tensor.zeros(1, 8, 32, 128)
    old_kv = rng.normal(size=(2, 1, 1, start_pos, 128)).astype(np.float32)
    new_kv = rng.normal(size=(2, 1, 1, 32, 128)).astype(np.float32)
    cache = Tensor.zeros(2, 1, 1, 2048, 128, dtype=dtypes.half).contiguous()
    Tensor.realize(cache[:, :, :, :start_pos].assign(Tensor(old_kv).cast(dtypes.half)))
    sp = UOp.variable("start_pos", 0, 2047).bind(start_pos)
    assigned = Tensor(cache.uop.after(cache[:, :, :, sp:sp+32, :].uop.store(Tensor(new_kv).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, sp+32).realize()
    values = np.concatenate([old_kv[1, 0, 0], new_kv[1, 0, 0]]).astype(np.float16).astype(np.float32)
    expected = np.stack([values[:start_pos+i+1].mean(0) for i in range(32)])[None, None].repeat(8, axis=1)
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

if __name__ == "__main__": unittest.main()
