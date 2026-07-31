import functools, sys, unittest
import numpy as np

from tinygrad import Device, Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.llm.gguf import _GGML_QUANT, ggml_data_to_tensor
from tinygrad.llm.kernels.cpu import (attention_decode, attention_prefill, causal_conv_silu, expert_pair, expert_silu,
                              expert_weighted_sum, f16_linear,
                              f16_matvec, gated_delta, gated_delta_prefill, gated_delta_q8, gdn_qkv, iq3_repack, moe_ffn, q6_argmax, q8_batched_pair,
                              q8_gdn_norm_projections, q8_gdn_projections, q8_linear_pair, q8_repack, q8_silu_linear,
                              recurrent_decode_bucket, rmsnorm, rmsnorm_f16_linear, shared_gate,
                              silu, silu_mul, uop_attention_prefill, uop_f16_matvec, uop_linear, uop_moe_ffn, uop_q8_linear_pair,
                              uop_q8_prequant_linear, uop_expert_silu_weighted, weighted_sum)
from tinygrad.llm.kernels.cpu import _dot_bytes_ptr, _dot_nibbles_ptr
from tinygrad.llm.model import biased_sigmoid_topk, pairwise_topk, ExpertWeights, FFNBlock, Linear, Transformer, TransformerConfig
from tinygrad.uop.ops import KernelInfo


def q8_activation(x:np.ndarray) -> np.ndarray:
  grouped = x.reshape(*x.shape[:-1], -1, 32)
  scale = np.maximum(np.max(np.abs(grouped), axis=-1, keepdims=True) / 127, 1e-8)
  return (np.clip(np.rint(grouped / scale), -127, 127) * scale).reshape(x.shape)

def q8k_activation(x:np.ndarray) -> np.ndarray:
  grouped = x.reshape(*x.shape[:-1], -1, 256)
  signed_max = np.take_along_axis(grouped, np.argmax(np.abs(grouped), axis=-1, keepdims=True), axis=-1)
  scale = -signed_max / 127
  inverse = np.divide(1, scale, out=np.zeros_like(scale), where=scale != 0)
  quantized = np.sign(grouped * inverse) * np.floor(np.abs(grouped * inverse) + 0.5)
  return (np.minimum(quantized, 127) * scale).reshape(x.shape)


def random_packed(rng:np.random.Generator, ggml_type:int, elements:int) -> np.ndarray:
  block_size, type_size = _GGML_QUANT[ggml_type]
  blocks = rng.integers(0, 256, size=(elements // block_size, type_size), dtype=np.uint8)
  scales = rng.uniform(0.001, 0.02, size=len(blocks)).astype(np.float16).view(np.uint8).reshape(-1, 2)
  blocks[:, :2] = scales
  if ggml_type == 14: blocks[:, -2:] = scales
  return blocks.flatten()


@unittest.skipUnless(Device.DEFAULT == "AMD", "requires DEV=AMD")
class TestLLMQuantAMD(unittest.TestCase):
  def test_packed_linear_offset_matches_reference(self):
    rng = np.random.default_rng(32)
    for ggml_type,in_features in ((8, 256), (14, 256)):
      raw, out_features = random_packed(rng, ggml_type, 64 * in_features), 64
      weight = ggml_data_to_tensor(Tensor(raw), out_features * in_features, ggml_type).numpy().reshape(out_features, in_features)
      storage = Tensor(np.concatenate((np.zeros(68, dtype=np.uint8), raw)), dtype=dtypes.uint8, device="AMD").realize()
      layer = Linear(in_features, out_features, bias=False)
      layer.set_quantized(storage[68:], ggml_type)
      x = rng.standard_normal((1, in_features), dtype=np.float32)
      np.testing.assert_allclose(layer(Tensor(x, device="AMD")).numpy(), q8_activation(x) @ weight.T, rtol=1e-5, atol=5e-4)

  def test_iq3_expert_prefill_and_decode_match_reference(self):
    rng = np.random.default_rng(31)
    num_experts, in_features, out_features = 2, 256, 16
    raw = random_packed(rng, 21, num_experts * out_features * in_features)
    weight = ggml_data_to_tensor(Tensor(raw, device="CPU"), num_experts * out_features * in_features,
                                 21).numpy().reshape(num_experts, out_features, in_features)
    experts = ExpertWeights(num_experts, in_features, out_features)
    experts.set_quantized(Tensor.empty(num_experts, out_features, in_features),
                          Tensor(raw, dtype=dtypes.uint8, device="AMD").realize(), 21)
    x = rng.standard_normal((2, 1, in_features), dtype=np.float32)
    for sel in (np.array([[1, 0], [0, 1]], dtype=np.int32), np.array([1, 0], dtype=np.int32)):
      activation = x if sel.ndim == 2 else x[:1]
      expected = np.stack([q8_activation(activation).reshape(-1, in_features)[route // 2] @ weight[expert].T
                           for route,expert in enumerate(sel.reshape(-1))]).reshape(*sel.shape, out_features)
      got = experts(Tensor(sel, device="AMD"), Tensor(activation, device="AMD")).numpy()
      np.testing.assert_allclose(got, expected, rtol=1e-5, atol=5e-4)


@unittest.skipUnless(sys.platform.startswith("linux") and Device.DEFAULT == "CPU", "requires DEV=CPU on Linux")
class TestLLMQuantCPU(unittest.TestCase):
  def test_grouped_byte_dot_uop(self):
    rng = np.random.default_rng(20)
    a = rng.integers(-128, 128, 32, dtype=np.int8)
    b = rng.integers(-127, 128, 32, dtype=np.int8)
    out = Tensor.empty(8, dtype=dtypes.int32, device="CPU")
    ta, tb = Tensor(a, device="CPU").realize(), Tensor(b, device="CPU").realize()
    def dot_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
      parts = _dot_bytes_ptr(a[0], b[0])
      return UOp.group(*(out[i].store(parts.index(i)) for i in range(8))).sink(arg=KernelInfo("grouped_byte_dot"))
    got = Tensor.custom_kernel(out, ta, tb, fxn=dot_kernel)[0].numpy()
    expected = (a.astype(np.int32) * b.astype(np.int32)).reshape(8, 4).sum(axis=1)
    np.testing.assert_equal(got, expected)

  def test_scaled_grouped_byte_dot_uop(self):
    rng = np.random.default_rng(22)
    a = rng.integers(-128, 128, 32, dtype=np.int8)
    b = rng.integers(-127, 128, 32, dtype=np.int8)
    out = Tensor.empty(8, dtype=dtypes.int32, device="CPU")
    ta, tb = Tensor(a, device="CPU").realize(), Tensor(b, device="CPU").realize()
    def dot_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
      parts = _dot_bytes_ptr(a[0], b[0]) * 7
      return UOp.group(*(out[i].store(parts.index(i)) for i in range(8))).sink(arg=KernelInfo("scaled_grouped_byte_dot"))
    got = Tensor.custom_kernel(out, ta, tb, fxn=dot_kernel)[0].numpy()
    expected = (a.astype(np.int32) * b.astype(np.int32)).reshape(8, 4).sum(axis=1) * 7
    np.testing.assert_equal(got, expected)

  def test_unpack_lut_dot_uop(self):
    rng = np.random.default_rng(21)
    packed, x = rng.integers(0, 256, 16, dtype=np.uint8), rng.integers(-127, 128, 32, dtype=np.int8)
    values = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
    out = Tensor.empty(8, dtype=dtypes.int32, device="CPU")
    tp, tx = Tensor(packed, device="CPU").realize(), Tensor(x, device="CPU").realize()
    def dot_kernel(out:UOp, packed:UOp, x:UOp) -> UOp:
      parts = _dot_nibbles_ptr(packed[0], x[0], values)
      return UOp.group(*(out[i].store(parts.index(i)) for i in range(8))).sink(arg=KernelInfo("unpack_lut_dot"))
    got = Tensor.custom_kernel(out, tp, tx, fxn=dot_kernel)[0].numpy()
    decoded = np.array(values, dtype=np.int8)[np.concatenate((packed & 15, packed >> 4))]
    expected = (decoded.astype(np.int32) * x.astype(np.int32)).reshape(8, 4).sum(axis=1)
    np.testing.assert_equal(got, expected)

  def test_generate_accepts_different_recurrent_prefill_shapes(self):
    class TinyRecurrentTransformer(Transformer):
      def __init__(self):
        self.max_context, self.has_recurrent_block = 32, True
        self.token_embd = nn.Embedding(4, 1)
        self.blk, self._cached_tokens = [], []
        self._state_checkpoints, self._state_checkpoint_pos = [], 0
        self._save_state_jit = self._restore_state_jit = None
        self._warming_up = False
        self.prefill_jit = TinyJit(self.forward)
        self.flash_prefill_jit = TinyJit(functools.partial(self.forward, use_flash=True))
        self.sample_prefill_jit = TinyJit(functools.partial(self.forward, sample=True))
        self.recurrent_prefill_jits = {}
        self.rollout_jits, self.sample_rollout_jits = {}, {}

      def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
                  kv_len:int|UOp|None=None, valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
        return tokens[:, -1:] + 1

    model = TinyRecurrentTransformer()
    for _ in range(3): self.assertEqual(next(model.generate([1] * 8, chunk_size=8)), 2)
    self.assertEqual(next(model.generate([1] * 3, chunk_size=8)), 2)

  def test_attention_decode_matches_causal_gqa_reference(self):
    rng = np.random.default_rng(0)
    batch, heads, kv_heads, cache_len, head_dim = 1, 16, 2, 4113, 32
    q = rng.standard_normal((batch, heads, 1, head_dim), dtype=np.float32)
    cache = rng.standard_normal((2, batch, kv_heads, cache_len, head_dim), dtype=np.float32).astype(np.float16)
    for pos in (0, 11, 4096):
      with self.subTest(pos=pos):
        start_pos = UOp.variable("start_pos", 0, cache_len-1).bind(pos)
        got = attention_decode(Tensor(q, device="CPU").contiguous(),
                                    Tensor(cache, device="CPU").contiguous(), start_pos).numpy()
        expected = np.empty_like(got)
        for head in range(heads):
          kv_head = head // (heads // kv_heads)
          keys = cache[0, 0, kv_head, :pos+1].astype(np.float32)
          values = cache[1, 0, kv_head, :pos+1].astype(np.float32)
          scores = q[0, head, 0] @ keys.T / np.sqrt(head_dim)
          probs = np.exp(scores - scores.max())
          expected[0, head, 0] = probs / probs.sum() @ values
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

  def test_attention_prefill_matches_causal_gqa_reference(self):
    rng = np.random.default_rng(18)
    batch, heads, tokens, kv_heads, cache_len, head_dim, pos = 1, 4, 5, 2, 19, 256, 3
    q = rng.standard_normal((batch, heads, tokens, head_dim), dtype=np.float32)
    cache = rng.standard_normal((2, batch, kv_heads, cache_len, head_dim), dtype=np.float32).astype(np.float16)
    tq, tcache = Tensor(q, device="CPU").contiguous(), Tensor(cache, device="CPU").contiguous()
    start_pos = UOp.variable("start_pos", 0, cache_len-1).bind(pos)
    got = attention_prefill(tq, tcache, start_pos).numpy()
    expected = np.empty_like(got)
    for head in range(heads):
      kv_head = head // (heads // kv_heads)
      for token in range(tokens):
        keys = cache[0, 0, kv_head, :pos+token+1].astype(np.float32)
        values = cache[1, 0, kv_head, :pos+token+1].astype(np.float32)
        scores = q[0, head, token] @ keys.T / np.sqrt(head_dim)
        probs = np.exp(scores - scores.max())
        expected[0, head, token] = probs / probs.sum() @ values
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(uop_attention_prefill(tq, tcache, start_pos).numpy(), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(uop_attention_prefill(tq[:, :, :4], tcache, start_pos).numpy(), expected[:, :, :4],
                               rtol=2e-5, atol=2e-5)

  def test_recurrent_decode_bucket(self):
    self.assertEqual(recurrent_decode_bucket(1024, 262144, "CPU"), 8192)
    self.assertEqual(recurrent_decode_bucket(8192, 262144, "CPU"), 8192)
    self.assertEqual(recurrent_decode_bucket(90000, 262144, "CPU"), 8192)
    self.assertEqual(recurrent_decode_bucket(90000, 262144, "AMD"), 262144)

  def test_gated_delta_matches_reference(self):
    rng = np.random.default_rng(4)
    batch, heads, dim = 2, 3, 16
    q, k, v = (rng.standard_normal((batch, heads, dim), dtype=np.float32) for _ in range(3))
    beta, alpha = rng.random((batch, heads), dtype=np.float32), rng.random((batch, heads), dtype=np.float32)
    state = rng.standard_normal((batch, heads, dim, dim), dtype=np.float32)
    for state_dtype in (np.float32, np.float16):
      with self.subTest(state_dtype=state_dtype):
        typed_state = state.astype(state_dtype)
        args = (*(Tensor(x, device="CPU") for x in (q, k, v, beta, alpha)), Tensor(typed_state, device="CPU"))
        core, next_state = gated_delta(*args)
        typed_state_k, typed_state_q = (np.einsum("bhij,bhj->bhi", typed_state.astype(np.float32), x) for x in (k, q))
        typed_delta = (v - typed_state_k * alpha[..., None]) * beta[..., None]
        typed_core = typed_state_q * alpha[..., None] + typed_delta * np.sum(k * q, axis=-1, keepdims=True)
        typed_next = typed_state.astype(np.float32) * alpha[..., None, None] + typed_delta[..., None] * k[..., None, :]
        np.testing.assert_allclose(core.numpy(), typed_core, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(next_state.numpy(), typed_next.astype(state_dtype), rtol=2e-5, atol=2e-5)
        norm = nn.RMSNorm(dim, eps=1e-6)
        norm.weight = Tensor(rng.standard_normal(dim, dtype=np.float32), device="CPU").half().realize()
        normalized, _ = gated_delta(*args, norm_weight=norm.weight, norm_eps=norm.eps)
        np.testing.assert_allclose(normalized.numpy(), rmsnorm(norm, core).numpy(), rtol=2e-5, atol=2e-5)
        inplace_state = Tensor(typed_state, device="CPU").realize()
        inplace_core, inplace_next = gated_delta(*(Tensor(x, device="CPU") for x in (q, k, v, beta, alpha)),
                                                       inplace_state, inplace=True)
        Tensor.realize(inplace_core, inplace_next)
        np.testing.assert_allclose(inplace_core.numpy(), typed_core, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(inplace_next.numpy(), typed_next.astype(state_dtype), rtol=2e-5, atol=2e-5)
        np.testing.assert_equal(inplace_state.numpy(), inplace_next.numpy())

  def test_gated_delta_prefill_matches_sequential_reference(self):
    rng = np.random.default_rng(17)
    batch, heads, tokens, dim = 1, 2, 5, 128
    q, k, v = [rng.standard_normal((batch, heads, tokens, dim), dtype=np.float32) for _ in range(3)]
    beta = rng.random((batch, heads, tokens), dtype=np.float32)
    alpha = rng.random((batch, heads, tokens), dtype=np.float32)
    state = rng.standard_normal((batch, heads, dim, dim), dtype=np.float32).astype(np.float16)
    weight = rng.standard_normal(dim, dtype=np.float32).astype(np.float16)
    eps = 1e-6
    expected_core = np.empty_like(q)
    expected_state = state.astype(np.float32)
    for token in range(tokens):
      for b in range(batch):
        for h in range(heads):
          qq, kk, vv = q[b, h, token], k[b, h, token], v[b, h, token]
          aa, bb, current = alpha[b, h, token], beta[b, h, token], expected_state[b, h]
          delta = (vv - (current @ kk) * aa) * bb
          core = (current @ qq) * aa + delta * (kk @ qq)
          expected_state[b, h] = current * aa + delta[:, None] * kk[None, :]
          expected_core[b, h, token] = core / np.sqrt(np.mean(core * core) + eps) * weight
    got_core, got_state = gated_delta_prefill(
      *[Tensor(z, device="CPU") for z in (q, k, v, beta, alpha)], Tensor(state, device="CPU"), Tensor(weight, device="CPU"), eps)
    np.testing.assert_allclose(got_core.numpy(), expected_core, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got_state.numpy(), expected_state.astype(np.float16), rtol=1e-3, atol=1e-3)

  def test_gated_delta_q8_feeds_projection(self):
    rng = np.random.default_rng(18)
    batch, heads, dim = 1, 2, 32
    q, k, v = [Tensor(rng.standard_normal((batch, heads, dim), dtype=np.float32), device="CPU").realize() for _ in range(3)]
    beta, alpha = [Tensor(rng.random((batch, heads), dtype=np.float32), device="CPU").realize() for _ in range(2)]
    state = Tensor(rng.standard_normal((batch, heads, dim, dim), dtype=np.float32), device="CPU").half().realize()
    gate = Tensor(rng.standard_normal((batch, heads, dim), dtype=np.float32), device="CPU").half().realize()
    norm_weight = Tensor(rng.standard_normal(dim, dtype=np.float32), device="CPU").half().realize()
    layer = Linear(heads * dim, 17, bias=False)
    layer.set_quantized(Tensor(random_packed(rng, 8, layer.out_features * layer.in_features),
                               dtype=dtypes.uint8, device="CPU").realize(), 8)

    core, expected_state = gated_delta(q, k, v, beta, alpha, state, norm_weight=norm_weight, norm_eps=1e-6)
    expected = q8_silu_linear(layer, gate.reshape(batch, 1, -1), core.reshape(batch, 1, -1))
    xq, xd, got_state = gated_delta_q8(q, k, v, beta, alpha, state, gate, norm_weight, 1e-6)
    got = uop_q8_prequant_linear(layer, xq, xd).reshape(batch, 1, -1)
    np.testing.assert_allclose(got_state.numpy(), expected_state.numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)

  def test_gdn_qkv_matches_normal_api(self):
    rng = np.random.default_rng(19)
    batch, tokens, k_heads, v_heads, dim = 2, 5, 2, 4, 16
    conv = Tensor(rng.standard_normal((batch, tokens, (2*k_heads+v_heads)*dim), dtype=np.float32), device="CPU")
    q, k, v = conv.split([k_heads*dim, k_heads*dim, v_heads*dim], dim=-1)
    q = (q.reshape(batch, tokens, k_heads, dim) *
         (q.reshape(batch, tokens, k_heads, dim).square().sum(-1, keepdim=True) + 1e-6).rsqrt()).repeat(1, 1, v_heads//k_heads, 1)
    k = (k.reshape(batch, tokens, k_heads, dim) *
         (k.reshape(batch, tokens, k_heads, dim).square().sum(-1, keepdim=True) + 1e-6).rsqrt()).repeat(1, 1, v_heads//k_heads, 1)
    expected = (q.transpose(1, 2) * dim**-0.5, k.transpose(1, 2), v.reshape(batch, tokens, v_heads, dim).transpose(1, 2))
    for got, ref in zip(gdn_qkv(conv, k_heads, v_heads, dim), expected):
      np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=2e-5, atol=2e-5)

  def test_decode_rmsnorm_matches_reference(self):
    rng = np.random.default_rng(5)
    for rows in (1, 32, 128):
      for dtype in (dtypes.float16, dtypes.float32):
        for weight_dtype in (dtypes.float16, dtypes.float32):
          with self.subTest(rows=rows, dtype=dtype, weight_dtype=weight_dtype):
            norm = nn.RMSNorm(64, eps=1e-6)
            norm.weight = Tensor(rng.standard_normal(64, dtype=np.float32), device="CPU").cast(weight_dtype).realize()
            x = Tensor(rng.standard_normal((rows, 64), dtype=np.float32), device="CPU").cast(dtype).realize()
            tol = 5e-4 if dtype == dtypes.float16 else 2e-6
            np.testing.assert_allclose(rmsnorm(norm, x).numpy(), norm(x).numpy(), rtol=tol, atol=tol)

  def test_causal_conv_silu_matches_reference(self):
    rng = np.random.default_rng(15)
    batch, tokens, channels, kernel = 2, 7, 64, 4
    for dtype in (dtypes.float16, dtypes.float32):
      for weight_dtype in (dtypes.float16, dtypes.float32):
        if dtype == weight_dtype == dtypes.float16: continue
        with self.subTest(dtype=dtype, weight_dtype=weight_dtype):
          state = Tensor(rng.standard_normal((batch, kernel - 1, channels), dtype=np.float32), device="CPU").realize()
          x = Tensor(rng.standard_normal((batch, tokens, channels), dtype=np.float32), device="CPU").cast(dtype).realize()
          weight = Tensor(rng.standard_normal((channels, kernel), dtype=np.float32),
                          device="CPU").cast(weight_dtype).realize()
          window = state.cat(x, dim=1)
          expected = functools.reduce(lambda a,b: a+b, (window[:, i:i+tokens] * weight[:, i] for i in range(kernel))).silu()
          np.testing.assert_allclose(causal_conv_silu(state, x, weight).numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)
          np.testing.assert_allclose(causal_conv_silu(state, x, weight.T.contiguous()).numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)

    # This is the vectorized path used by Qwen's prefill.
    state = Tensor(rng.standard_normal((1, kernel - 1, channels), dtype=np.float32), device="CPU").realize()
    x = Tensor(rng.standard_normal((1, tokens, channels), dtype=np.float32), device="CPU").half().realize()
    weight = Tensor(rng.standard_normal((channels, kernel), dtype=np.float32), device="CPU").half().realize()
    expected = causal_conv_silu(state, x, weight).numpy()
    np.testing.assert_allclose(causal_conv_silu(state, x, weight.T.contiguous()).numpy(), expected, rtol=2e-6, atol=2e-6)

  def test_shared_gate_matches_reference(self):
    rng = np.random.default_rng(6)
    for dtype in (dtypes.float16, dtypes.float32):
      with self.subTest(dtype=dtype):
        x = Tensor(rng.standard_normal((3, 64), dtype=np.float32), device="CPU").cast(dtype).realize()
        weight = Tensor(rng.standard_normal(64, dtype=np.float32), device="CPU").half().realize()
        expected = (x * weight).sum(axis=-1, keepdim=True).sigmoid()
        np.testing.assert_allclose(shared_gate(x, weight).numpy(), expected.numpy(), rtol=2e-5, atol=2e-5)

  def test_silu_mul_matches_reference(self):
    rng = np.random.default_rng(10)
    for dtype in (dtypes.float16, dtypes.float32):
      with self.subTest(dtype=dtype):
        gate = Tensor(rng.standard_normal((2, 64), dtype=np.float32), device="CPU").cast(dtype).realize()
        up = Tensor(rng.standard_normal((2, 64), dtype=np.float32), device="CPU").cast(dtype).realize()
        np.testing.assert_equal(silu(gate).numpy(), gate.silu().numpy())
        np.testing.assert_allclose(silu_mul(gate, up).numpy(), (gate.silu() * up).numpy(), rtol=2e-5, atol=2e-5)
    gate = Tensor(rng.standard_normal((2, 64), dtype=np.float32), device="CPU").half().realize()
    up = Tensor(rng.standard_normal((2, 64), dtype=np.float32), device="CPU").realize()
    np.testing.assert_equal(silu_mul(gate, up).numpy(), (gate.silu() * up).numpy())
    gate = Tensor(rng.standard_normal(4096, dtype=np.float32), device="CPU").half().realize()
    up = Tensor(rng.standard_normal(4096, dtype=np.float32), device="CPU").realize()
    np.testing.assert_allclose(silu_mul(gate, up).numpy(), (gate.silu() * up).numpy(), rtol=2e-6, atol=2e-6)

  def test_biased_topk_matches_reference(self):
    rng = np.random.default_rng(7)
    logits = Tensor(rng.standard_normal((1, 2, 256), dtype=np.float32), device="CPU").half().realize()
    bias = Tensor(rng.standard_normal(256, dtype=np.float32), device="CPU").half().realize()
    probs = logits.sigmoid()
    _, expected_sel = pairwise_topk(probs + bias, 8)
    expected = probs.gather(-1, expected_sel)
    expected = expected / expected.sum(axis=-1, keepdim=True)
    got, got_sel = biased_sigmoid_topk(logits, bias, 8, normalize=True)
    np.testing.assert_equal(got_sel.numpy(), expected_sel.numpy().reshape(2, 8))
    np.testing.assert_allclose(got.numpy(), expected.numpy().reshape(2, 8), rtol=5e-4, atol=5e-4)

  def test_packed_linear_matches_q8_activation_reference(self):
    rng = np.random.default_rng(1)
    for ggml_type, in_features in ((8, 64), (14, 256)):
      for tokens in (1, 3, 8):
        with self.subTest(ggml_type=ggml_type, tokens=tokens):
          out_features = 7
          raw = random_packed(rng, ggml_type, out_features * in_features)
          layer = Linear(in_features, out_features, bias=False)
          layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), ggml_type)
          x = rng.standard_normal((tokens, in_features), dtype=np.float32)
          got = layer(Tensor(x, device="CPU")).numpy()
          weight = ggml_data_to_tensor(Tensor(raw), out_features * in_features, ggml_type).numpy().reshape(out_features, in_features)
          np.testing.assert_allclose(got, q8_activation(x) @ weight.T, rtol=1e-5, atol=5e-4)

  def test_q8_linear_pair_matches_separate(self):
    rng = np.random.default_rng(11)
    in_features = 64
    layers = []
    for out_features in (7, 11):
      raw = random_packed(rng, 8, out_features * in_features)
      layer = Linear(in_features, out_features, bias=False)
      layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), 8)
      layers.append(layer)
    for dtype in (dtypes.float16, dtypes.float32):
      with self.subTest(dtype=dtype):
        x = Tensor(rng.standard_normal((1, in_features), dtype=np.float32), device="CPU").cast(dtype).realize()
        got = q8_linear_pair(*layers, x)
        for paired,layer in zip(got, layers): np.testing.assert_allclose(paired.numpy(), layer(x).numpy(), rtol=2e-5, atol=2e-5)
        for paired,layer in zip(q8_linear_pair(*layers, x.reshape(1, 1, in_features)), layers):
          self.assertEqual(paired.shape, (1, 1, layer.out_features))
          np.testing.assert_allclose(paired.numpy(), layer(x).numpy().reshape(1, 1, -1), rtol=2e-5, atol=2e-5)
        for layer in layers: layer.cpu_repacked = q8_repack(layer.weight, layer.out_features, layer.in_features).realize()
        for repacked,original in zip(q8_linear_pair(*layers, x), got): np.testing.assert_equal(repacked.numpy(), original.numpy())
        original_weights = [layer.weight for layer in layers]
        for layer in layers: layer.weight = Tensor.zeros_like(layer.weight).realize()
        for repacked,original in zip(uop_q8_linear_pair(*layers, x), got):
          np.testing.assert_allclose(repacked.numpy(), original.numpy(), rtol=2e-6, atol=1e-5)
        for layer,weight in zip(layers, original_weights): layer.weight = weight
        for layer in layers: layer.cpu_repacked = None

  def test_large_q8_uop_linear_repacked_matches_raw(self):
    rng = np.random.default_rng(20)
    in_features, out_features = 1024, 7
    raw = random_packed(rng, 8, out_features * in_features)
    layer = Linear(in_features, out_features, bias=False)
    layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), 8)
    x = Tensor(rng.standard_normal((1, in_features), dtype=np.float32), device="CPU").realize()
    expected = uop_linear(layer, x).numpy()
    layer.cpu_repacked = q8_repack(layer.weight, out_features, in_features).realize()
    np.testing.assert_allclose(uop_linear(layer, x).numpy(), expected, rtol=2e-6, atol=1e-5)

  def test_q8_batched_pair_matches_separate(self):
    rng = np.random.default_rng(15)
    in_features = 64
    layers = []
    for out_features in (7, 11):
      raw = random_packed(rng, 8, out_features * in_features)
      layer = Linear(in_features, out_features, bias=False)
      layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), 8)
      layers.append(layer)
    for dtype in (dtypes.float16, dtypes.float32):
      with self.subTest(dtype=dtype):
        x = Tensor(rng.standard_normal((2, 3, in_features), dtype=np.float32), device="CPU").cast(dtype).realize()
        got = q8_batched_pair(*layers, x)
        for paired,layer in zip(got, layers): np.testing.assert_allclose(paired.numpy(), layer(x).numpy(), rtol=2e-5, atol=2e-5)
        for layer in layers: layer.cpu_repacked = q8_repack(layer.weight, layer.out_features, layer.in_features).realize()
        for repacked,original in zip(q8_batched_pair(*layers, x), got): np.testing.assert_equal(repacked.numpy(), original.numpy())
        for layer in layers: layer.cpu_repacked = None

  def test_q8_silu_linear_matches_separate(self):
    rng = np.random.default_rng(16)
    in_features, out_features = 64, 11
    layer = Linear(in_features, out_features, bias=False)
    layer.set_quantized(Tensor(random_packed(rng, 8, out_features * in_features), dtype=dtypes.uint8, device="CPU").realize(), 8)
    gate = Tensor(rng.standard_normal((2, 4, in_features), dtype=np.float32), device="CPU").half().realize()
    up = Tensor(rng.standard_normal(gate.shape, dtype=np.float32), device="CPU").realize()
    fused = q8_silu_linear(layer, gate, up)
    separate = layer(silu_mul(gate, up).half())
    linear = layer(gate[:, 0])
    single = q8_silu_linear(layer, gate[:1, :1], up[:1, :1])
    np.testing.assert_equal(fused.numpy(), separate.numpy())
    layer.cpu_repacked = q8_repack(layer.weight, layer.out_features, layer.in_features).realize()
    np.testing.assert_equal(q8_silu_linear(layer, gate, up).numpy(), fused.numpy())
    np.testing.assert_equal(q8_silu_linear(layer, gate[:1, :1], up[:1, :1]).numpy(), single.numpy())
    np.testing.assert_equal(layer(gate[:, 0]).numpy(), linear.numpy())

  def test_q8_gdn_projections_match_separate(self):
    rng = np.random.default_rng(10)
    in_features = 256
    layers = []
    for out_features in (1, 1):
      raw = random_packed(rng, 8, out_features * in_features)
      layer = Linear(in_features, out_features, bias=False)
      layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), 8)
      layers.append(layer)
    x = Tensor(rng.standard_normal((1, in_features), dtype=np.float32), device="CPU").half().realize()
    weight = Tensor(rng.standard_normal((16, in_features), dtype=np.float32), device="CPU").half().realize()
    got = q8_gdn_projections(*layers, weight, x)
    expected = (*q8_linear_pair(*layers, x), x @ weight.T)
    for fused,separate in zip(got, expected): np.testing.assert_allclose(fused.numpy(), separate.numpy(), rtol=1e-3, atol=1e-3)
    for layer in layers: layer.cpu_repacked = q8_repack(layer.weight, layer.out_features, layer.in_features).realize()
    for repacked,original in zip(q8_gdn_projections(*layers, weight, x), got):
      np.testing.assert_equal(repacked.numpy(), original.numpy())
    norm = nn.RMSNorm(in_features, eps=1e-6)
    norm.weight = Tensor(rng.standard_normal(in_features, dtype=np.float32), device="CPU").half().realize()
    raw_x = Tensor(rng.standard_normal((1, in_features), dtype=np.float32), device="CPU").realize()
    expected = q8_gdn_projections(*layers, weight, rmsnorm(norm, raw_x).half())
    for fused,separate in zip(q8_gdn_norm_projections(*layers, weight, raw_x, norm), expected):
      np.testing.assert_equal(fused.numpy(), separate.numpy())

  def test_f16_linear_matches_standard(self):
    rng = np.random.default_rng(13)
    layer = Linear(256, 37, bias=False)
    layer.weight = Tensor(rng.standard_normal((37, 256), dtype=np.float32), device="CPU").half().realize()
    for dtype in (dtypes.float16, dtypes.float32):
      for tokens in (1, 3):
        with self.subTest(dtype=dtype, tokens=tokens):
          x = Tensor(rng.standard_normal((tokens, 256), dtype=np.float32), device="CPU").cast(dtype).realize()
          np.testing.assert_allclose(f16_linear(layer, x).numpy(), layer(x).numpy(), rtol=1e-5, atol=2e-5)
          np.testing.assert_allclose(f16_matvec(x, layer.weight).numpy(), layer(x).numpy(), rtol=1e-5, atol=2e-5)
          np.testing.assert_allclose(uop_f16_matvec(x, layer.weight).numpy(), layer(x).numpy(), rtol=1e-5, atol=2e-5)
    norm = nn.RMSNorm(256, eps=1e-6)
    norm.weight = Tensor(rng.standard_normal(256, dtype=np.float32), device="CPU").half().realize()
    x = Tensor(rng.standard_normal((1, 256), dtype=np.float32), device="CPU").realize()
    normalized, out = rmsnorm_f16_linear(norm, layer, x)
    expected = rmsnorm(norm, x)
    np.testing.assert_allclose(normalized.numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(out.numpy(), f16_linear(layer, expected).numpy(), rtol=1e-5, atol=2e-5)

  def test_q6_argmax_matches_materialized_logits(self):
    rng = np.random.default_rng(8)
    in_features, out_features = 256, 37
    raw = random_packed(rng, 14, out_features * in_features)
    weight = ggml_data_to_tensor(Tensor(raw), out_features * in_features, 14).numpy().reshape(out_features, in_features)
    layer = Linear(in_features, out_features, bias=False)
    layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), 14)
    for _ in range(3):
      x = Tensor(rng.standard_normal((1, in_features), dtype=np.float32), device="CPU").realize()
      expected = int(np.argmax(weight @ q8k_activation(x.numpy()).reshape(-1)))
      self.assertEqual(q6_argmax(layer, x).item(), expected)

  def test_packed_experts_match_reference(self):
    rng = np.random.default_rng(2)
    num_experts, in_features, out_features = 2, 256, 5
    sel, x = np.array([1, 0, 1, 0], dtype=np.int32), rng.standard_normal((4, in_features), dtype=np.float32)
    for ggml_type in (14, 21, 23):
      with self.subTest(ggml_type=ggml_type):
        raw = random_packed(rng, ggml_type, num_experts * out_features * in_features)
        weight = ggml_data_to_tensor(Tensor(raw), num_experts * out_features * in_features,
                                     ggml_type).numpy().reshape(num_experts, out_features, in_features)
        experts = ExpertWeights(num_experts, in_features, out_features)
        experts.set_quantized(Tensor(weight), Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), ggml_type)
        got = experts(Tensor(sel, device="CPU"), Tensor(x, device="CPU")).numpy()
        activation = q8k_activation(x) if ggml_type in (21, 23) else q8_activation(x)
        expected = np.stack([activation[i] @ weight[expert].T for i,expert in enumerate(sel)])
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=5e-4)
        if ggml_type == 21:
          experts.cpu_repacked = iq3_repack(experts.weight, num_experts * out_features, in_features).realize()
          np.testing.assert_allclose(experts(Tensor(sel, device="CPU"), Tensor(x, device="CPU")).numpy(),
                                     expected, rtol=1e-5, atol=5e-4)
        direct_sel = np.array([1, 0], dtype=np.int32)
        direct = experts(Tensor(direct_sel, device="CPU"), Tensor(x[:1], device="CPU")).numpy()
        direct_activation = q8k_activation(x[:1]) if ggml_type in (21, 23) else q8_activation(x[:1])
        direct_expected = np.stack([direct_activation[0] @ weight[expert].T for expert in direct_sel])
        np.testing.assert_allclose(direct, direct_expected, rtol=1e-5, atol=5e-4)
  def test_weighted_expert_sum_matches_reference(self):
    rng = np.random.default_rng(16)
    x = rng.standard_normal((2, 4, 257), dtype=np.float32)
    probs = rng.random((2, 4), dtype=np.float32)
    got = weighted_sum(Tensor(x, device="CPU"), Tensor(probs, device="CPU")).numpy()
    np.testing.assert_allclose(got, (x * probs[..., None]).sum(axis=1), rtol=1e-6, atol=1e-6)

  def test_quantized_expert_weighted_sum_matches_separate(self):
    rng = np.random.default_rng(29)
    num_experts, inputs, routes_per_input, in_features, out_features = 16, 4, 8, 256, 64
    sel = Tensor(rng.integers(0, num_experts, (1, inputs, routes_per_input), dtype=np.int32), device="CPU").realize()
    x = Tensor(rng.standard_normal((*sel.shape, in_features), dtype=np.float32), device="CPU").realize()
    probs = Tensor(rng.random(sel.shape, dtype=np.float32), device="CPU").realize()
    for ggml_type in (14, 23):
      with self.subTest(ggml_type=ggml_type):
        layer = ExpertWeights(num_experts, in_features, out_features)
        layer.set_quantized(Tensor.empty(num_experts, out_features, in_features),
                            Tensor(random_packed(rng, ggml_type, num_experts * in_features * out_features),
                                   dtype=dtypes.uint8, device="CPU").realize(), ggml_type)
        np.testing.assert_allclose(expert_weighted_sum(layer, sel, x, probs).numpy(), weighted_sum(layer(sel, x), probs).numpy(),
                                   rtol=5e-6, atol=1e-5)

  def test_fused_expert_silu_matches_separate(self):
    rng = np.random.default_rng(12)
    num_experts, in_features, out_features = 3, 256, 7
    sel = Tensor(np.array([2, 0], dtype=np.int32), device="CPU")
    x = Tensor(rng.standard_normal((1, in_features), dtype=np.float32), device="CPU").realize()
    for ggml_type in (14, 21, 23):
      with self.subTest(ggml_type=ggml_type):
        experts = []
        for _ in range(2):
          raw = random_packed(rng, ggml_type, num_experts * out_features * in_features)
          weight = ggml_data_to_tensor(Tensor(raw), num_experts * out_features * in_features,
                                       ggml_type).reshape(num_experts, out_features, in_features)
          expert = ExpertWeights(num_experts, in_features, out_features)
          expert.set_quantized(weight, Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), ggml_type)
          experts.append(expert)
        gate, up = expert_pair(*experts, sel, x)
        np.testing.assert_allclose(expert_silu(*experts, sel, x).numpy(), silu_mul(gate, up).numpy(), rtol=1e-4, atol=1e-4)
        if ggml_type == 21:
          direct_expected = expert_silu(*experts, sel, x).numpy()
          batch_sel = Tensor(np.array([[2, 2], [1, 2]], dtype=np.int32), device="CPU")
          batch_x = Tensor(rng.standard_normal((2, in_features), dtype=np.float32), device="CPU").realize()
          expected = expert_silu(*experts, batch_sel, batch_x).numpy()
          batch_gate, batch_up = expert_pair(*experts, batch_sel, batch_x)
          np.testing.assert_allclose(expected, silu_mul(batch_gate, batch_up).numpy(), rtol=1e-4, atol=1e-4)
          for expert in experts:
            expert.cpu_repacked = iq3_repack(expert.weight, expert.num_experts * expert.out_features, expert.in_features).realize()
          np.testing.assert_allclose(expert_silu(*experts, sel, x).numpy(), direct_expected, rtol=1e-5, atol=1e-4)
          np.testing.assert_allclose(expert_silu(*experts, batch_sel, batch_x).numpy(), expected, rtol=1e-5, atol=1e-4)

  def test_expert_silu_weighted_reuses_routes(self):
    rng = np.random.default_rng(34)
    num_experts, in_features, hidden, out_features = 3, 256, 256, 64
    layers = []
    for ggml_type,layer_in,layer_out in ((21, in_features, hidden), (21, in_features, hidden), (23, hidden, out_features)):
      layer = ExpertWeights(num_experts, layer_in, layer_out)
      raw = random_packed(rng, ggml_type, num_experts * layer_in * layer_out)
      weight = ggml_data_to_tensor(Tensor(raw), num_experts * layer_in * layer_out, ggml_type).reshape(
        num_experts, layer_out, layer_in)
      layer.set_quantized(weight, Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), ggml_type)
      if ggml_type == 21: layer.cpu_repacked = iq3_repack(layer.weight, num_experts * layer_out, layer_in).realize()
      layers.append(layer)
    sel = Tensor(np.array([[2, 0], [1, 2]], dtype=np.int32), device="CPU")
    x = Tensor(rng.standard_normal((2, in_features), dtype=np.float32), device="CPU").realize()
    probs = Tensor(rng.random(sel.shape, dtype=np.float32), device="CPU").realize()
    expected = expert_weighted_sum(layers[2], sel, expert_silu(layers[0], layers[1], sel, x), probs).numpy()
    got = uop_expert_silu_weighted(layers[0], layers[1], layers[2], sel, x, probs).numpy()
    np.testing.assert_allclose(got, expected, rtol=5e-6, atol=1e-5)

  def test_fused_moe_matches_separate_quantized_layers(self):
    rng = np.random.default_rng(15)
    dim = hidden = 256
    config = TransformerConfig(1, dim, hidden, 1, 1, 1e-6, 32, dim, 1e6, dim, dim,
                               num_experts=3, num_experts_per_tok=2, shared_expert_dim=hidden)
    block = FFNBlock(config)
    routed_weights = {}
    for name,expert,ggml_type in (("gate", block.ffn_gate_exps, 21), ("up", block.ffn_up_exps, 21),
                                  ("down", block.ffn_down_exps, 23)):
      elements = expert.num_experts * expert.in_features * expert.out_features
      raw = random_packed(rng, ggml_type, elements)
      weight = ggml_data_to_tensor(Tensor(raw), elements, ggml_type).reshape(
        expert.num_experts, expert.out_features, expert.in_features)
      routed_weights[name] = weight.numpy().astype(np.float32)
      expert.set_quantized(weight, Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), ggml_type)
      if ggml_type == 21: expert.cpu_repacked = iq3_repack(expert.weight, expert.num_experts * expert.out_features, expert.in_features).realize()
    for layer in (block.ffn_gate_shexp, block.ffn_up_shexp, block.ffn_down_shexp):
      elements = layer.in_features * layer.out_features
      raw = random_packed(rng, 8, elements)
      layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), 8)
      layer.cpu_repacked = q8_repack(layer.weight, layer.out_features, layer.in_features).realize()
    block.ffn_gate_inp_shexp["weight"] = Tensor(rng.standard_normal(dim, dtype=np.float32), device="CPU").half().realize()

    x = Tensor(rng.standard_normal((1, 2, dim), dtype=np.float32), device="CPU").realize()
    probs = Tensor(np.array([[[0.7, 0.3], [0.4, 0.6]]], dtype=np.float32), device="CPU").realize()
    sel = Tensor(np.array([[[2, 0], [1, 2]]], dtype=np.int32), device="CPU").realize()
    selected = sel.numpy().reshape(-1)
    quantized_x = q8k_activation(x.numpy()).reshape(2, dim)
    gate = np.stack([routed_weights["gate"][expert] @ quantized_x[route // 2] for route,expert in enumerate(selected)])
    up = np.stack([routed_weights["up"][expert] @ quantized_x[route // 2] for route,expert in enumerate(selected)])
    routed_hidden = silu_mul(Tensor(gate, device="CPU"), Tensor(up, device="CPU")).numpy()
    quantized_hidden = q8k_activation(routed_hidden)
    routed = np.stack([routed_weights["down"][expert] @ quantized_hidden[route] for route,expert in enumerate(selected)])
    routed = Tensor((routed.reshape(2, 2, dim) * probs.numpy().reshape(2, 2, 1)).sum(1).reshape(1, 2, dim), device="CPU")
    shared_gate_out, shared_up = block.ffn_gate_shexp(x), block.ffn_up_shexp(x)
    shared = block.ffn_down_shexp(silu_mul(shared_gate_out, shared_up))
    expected = routed + shared * shared_gate(x, block.ffn_gate_inp_shexp["weight"])
    original = moe_ffn(block, x, probs, sel).numpy()
    np.testing.assert_allclose(original, expected.numpy(), rtol=1e-4, atol=2e-2)
    for expert in (block.ffn_gate_exps, block.ffn_up_exps):
      expert.cpu_repacked = iq3_repack(expert.weight, expert.num_experts * expert.out_features, expert.in_features).realize()
    np.testing.assert_equal(moe_ffn(block, x, probs, sel).numpy(), original)
    np.testing.assert_allclose(uop_moe_ffn(block, x[:, :1], probs[:, :1], sel[:, :1]).numpy(), original[:, :1],
                               rtol=1e-4, atol=2e-2)

  def test_fused_moe_q6_down_matches_separate(self):
    rng = np.random.default_rng(16)
    dim = hidden = 256
    config = TransformerConfig(1, dim, hidden, 1, 1, 1e-6, 32, dim, 1e6, dim, dim,
                               num_experts=3, num_experts_per_tok=2, shared_expert_dim=hidden)
    block = FFNBlock(config)
    for expert,ggml_type in ((block.ffn_gate_exps, 21), (block.ffn_up_exps, 21), (block.ffn_down_exps, 14)):
      elements = expert.num_experts * expert.in_features * expert.out_features
      raw = random_packed(rng, ggml_type, elements)
      weight = ggml_data_to_tensor(Tensor(raw), elements, ggml_type).reshape(
        expert.num_experts, expert.out_features, expert.in_features)
      expert.set_quantized(weight, Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), ggml_type)
    for expert in (block.ffn_gate_exps, block.ffn_up_exps):
      expert.cpu_repacked = iq3_repack(expert.weight, expert.num_experts * expert.out_features, expert.in_features).realize()
    for layer in (block.ffn_gate_shexp, block.ffn_up_shexp, block.ffn_down_shexp):
      elements = layer.in_features * layer.out_features
      raw = random_packed(rng, 8, elements)
      layer.set_quantized(Tensor(raw, dtype=dtypes.uint8, device="CPU").realize(), 8)
    block.ffn_gate_inp_shexp["weight"] = Tensor(rng.standard_normal(dim, dtype=np.float32), device="CPU").half().realize()

    x = Tensor(rng.standard_normal((1, 1, dim), dtype=np.float32), device="CPU").realize()
    probs = Tensor(np.array([[[0.7, 0.3]]], dtype=np.float32), device="CPU").realize()
    sel = Tensor(np.array([[[2, 0]]], dtype=np.int32), device="CPU").realize()
    hidden = expert_silu(block.ffn_gate_exps, block.ffn_up_exps, sel, x.unsqueeze(2))
    routed = weighted_sum(block.ffn_down_exps(sel, hidden), probs)
    gate, up = block.ffn_gate_shexp(x), block.ffn_up_shexp(x)
    shared = block.ffn_down_shexp(silu_mul(gate, up)) * shared_gate(x, block.ffn_gate_inp_shexp["weight"])
    np.testing.assert_allclose(moe_ffn(block, x, probs, sel).numpy(), (routed + shared).numpy(), rtol=3e-4, atol=1e-4)

if __name__ == "__main__":
  unittest.main()
