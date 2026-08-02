import unittest
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.gguf import _GGML_QUANT, ggml_data_to_tensor
from tinygrad.llm.kernels import amd as llm_amd
from tinygrad.llm.model import Embedding, Linear

def q8_activation(x:np.ndarray) -> np.ndarray:
  grouped = x.reshape(*x.shape[:-1], -1, 32)
  scale = np.maximum(np.max(np.abs(grouped), axis=-1, keepdims=True) / 127, 1e-8)
  return (np.clip(np.rint(grouped / scale), -127, 127) * scale).reshape(x.shape)


def random_packed(rng:np.random.Generator, ggml_type:int, elements:int) -> np.ndarray:
  block_size, type_size = _GGML_QUANT[ggml_type]
  blocks = rng.integers(0, 256, size=(elements // block_size, type_size), dtype=np.uint8)
  scales = rng.uniform(0.001, 0.02, size=len(blocks)).astype(np.float16).view(np.uint8).reshape(-1, 2)
  blocks[:, :2] = scales
  if ggml_type in (12, 13): blocks[:, 2:4] = scales
  if ggml_type == 14: blocks[:, -2:] = scales
  return blocks.flatten()


@unittest.skipUnless(Device.DEFAULT == "AMD", "requires DEV=AMD")
class TestLLMQuantAMD(unittest.TestCase):
  @staticmethod
  def assert_q8_equal(result:tuple[Tensor, Tensor, Tensor], expected:np.ndarray):
    grouped = expected.reshape(expected.shape[0], -1, 32)
    scale = np.maximum(np.max(np.abs(grouped), axis=-1) / 127, 1e-8)
    quant = np.clip(np.rint(grouped / scale[..., None]), -127, 127).astype(np.int8)
    np.testing.assert_equal(result[0].numpy().view(np.int8).reshape(grouped.shape), quant)
    np.testing.assert_allclose(result[1].numpy(), scale, rtol=1e-6, atol=1e-8)
    np.testing.assert_equal(result[2].numpy(), quant.astype(np.int32).sum(-1))

  def test_gated_delta_prefill_matches_sequential_reference(self):
    rng = np.random.default_rng(36)
    batch, heads, tokens, dim = 1, 2, 5, 128
    q, k, v = [rng.standard_normal((batch, heads, tokens, dim), dtype=np.float32) for _ in range(3)]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True) / np.float32(np.sqrt(dim))
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    beta, alpha = rng.random((batch, heads, tokens), dtype=np.float32), rng.uniform(0.9, 1, (batch, heads, tokens)).astype(np.float32)
    state = rng.standard_normal((batch, heads, dim, dim), dtype=np.float32).astype(np.float16)
    expected_core, expected_state = np.empty_like(q), state.astype(np.float32)
    for token in range(tokens):
      state_k = np.einsum("bhij,bhj->bhi", expected_state, k[:, :, token])
      state_q = np.einsum("bhij,bhj->bhi", expected_state, q[:, :, token])
      delta = (v[:, :, token] - state_k * alpha[:, :, token, None]) * beta[:, :, token, None]
      expected_core[:, :, token] = state_q * alpha[:, :, token, None] + delta * np.sum(k[:, :, token] * q[:, :, token], axis=-1)[..., None]
      expected_state = expected_state * alpha[:, :, token, None, None] + delta[..., None] * k[:, :, token, None, :]
    core, next_state = llm_amd.gated_delta_prefill(
      *(Tensor(x, device="AMD") for x in (q, k, v, beta, alpha)), Tensor(state, device="AMD"))
    np.testing.assert_allclose(core.numpy(), expected_core, rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(next_state.numpy(), expected_state.astype(np.float16), rtol=2e-4, atol=1e-3)

  def test_q8_quantize_matches_reference(self):
    rng, tokens, in_features = np.random.default_rng(35), 3, 256
    x = rng.standard_normal((tokens, in_features), dtype=np.float32)
    grouped = x.reshape(tokens, -1, 32)
    expected_scale = np.maximum(np.max(np.abs(grouped), axis=-1) / 127, 1e-8)
    expected_quant = np.clip(np.rint(grouped / expected_scale[..., None]), -127, 127).astype(np.int8)
    quant, scale, group_sum = llm_amd.q8_quantize_sum(Tensor(x, device="AMD"), tokens, in_features)
    np.testing.assert_equal(quant.numpy().view(np.int8).reshape(grouped.shape), expected_quant)
    np.testing.assert_allclose(scale.numpy(), expected_scale, rtol=1e-7, atol=0)
    np.testing.assert_equal(group_sum.numpy(), expected_quant.astype(np.int32).sum(-1))

  def test_q4_embedding_matches_reference(self):
    rng, vocab_size, embed_size = np.random.default_rng(34), 16, 256
    raw = random_packed(rng, 12, vocab_size * embed_size)
    expected = ggml_data_to_tensor(Tensor(raw), vocab_size * embed_size, 12).reshape(vocab_size, embed_size).half()
    storage = Tensor(np.concatenate((np.zeros(68, dtype=np.uint8), raw)), dtype=dtypes.uint8, device="AMD").realize()
    embedding = Embedding(vocab_size, embed_size)
    embedding.set_quantized(storage[68:], 12)
    idx = np.array([[7, 1, 15], [0, 4, 7]], dtype=np.int32)
    np.testing.assert_equal(embedding(Tensor(idx, device="AMD")).numpy(), expected.numpy()[idx])

  def test_iq4_lut_is_ready_for_jit_capture(self):
    rng, in_features, out_features = np.random.default_rng(33), 256, 16
    raw = random_packed(rng, 23, out_features * in_features)
    weight = ggml_data_to_tensor(Tensor(raw), out_features * in_features, 23).numpy().reshape(out_features, in_features)
    llm_amd.iq4_half_lut.cache_clear()
    layer = Linear(in_features, out_features, bias=False)
    layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="AMD").realize(), 23)
    @TinyJit
    def run(x:Tensor): return layer(x).realize()
    x = rng.standard_normal((16, in_features), dtype=np.float32)
    expected = x.astype(np.float16).astype(np.float32) @ weight.astype(np.float16).astype(np.float32).T
    for _ in range(2): np.testing.assert_allclose(run(Tensor(x, device="AMD")).numpy(), expected, rtol=1e-5, atol=2e-3)

  def test_packed_linear_offset_matches_reference(self):
    rng = np.random.default_rng(32)
    for ggml_type,in_features in ((8, 256), (12, 256), (13, 256), (14, 256), (23, 256)):
      for tokens in ((1, 16, 32, 64, 128) if ggml_type == 23 else (1, 16, 128) if ggml_type in (12, 13) else
                     (1, 16) if ggml_type == 14 else (1,)):
        raw, out_features = random_packed(rng, ggml_type, 64 * in_features), 64
        weight = ggml_data_to_tensor(Tensor(raw), out_features * in_features, ggml_type).numpy().reshape(out_features, in_features)
        storage = Tensor(np.concatenate((np.zeros(68, dtype=np.uint8), raw)), dtype=dtypes.uint8, device="AMD").realize()
        layer = Linear(in_features, out_features, bias=False)
        layer.set_quantized(storage[68:], ggml_type)
        x = rng.standard_normal((tokens, in_features), dtype=np.float32)
        expected = x.astype(np.float16).astype(np.float32) @ weight.astype(np.float16).astype(np.float32).T \
          if ggml_type in (12, 13, 23) and tokens > 1 else q8_activation(x) @ weight.T
        np.testing.assert_allclose(layer(Tensor(x, device="AMD")).numpy(), expected, rtol=1e-5, atol=2e-3)


