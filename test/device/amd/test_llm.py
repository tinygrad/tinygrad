import functools, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, function, Device, TinyJit
from tinygrad.helpers import DEV
from tinygrad.llm.kernels.amd import (Linear, ExpertWeights, amd_custom_kernels_supported, flash_attention, gated_delta_prefill, reduce_scatter,
                                      _is_rdna4, _is_cdna, topk_indices, partial_sum, allgather_last, indexer_scores, mla_topk_attention, kda_decode,
                                      _mailbox_shape)
from tinygrad.llm.model import MLATransformerBlock, TransformerConfig, IndexerConfig, Transformer, SSMConfig, AttentionResidual
from tinygrad.llm.kernels.amd_gguf import (q8_quantize, expert_linear, expert_tiles, moe_route, q8_0_linear, head_linear, QUANT_SIZES,
  HALFWORD_QUANTS, SMALL_BLOCK_QUANTS, Q4_K, IQ3_S, IQ3_XXS)
from tinygrad.llm.gguf import ggml_data_to_tensor
from test.runtime.test_llm_quantized import QuantLinearMixin

class TestQ8Quantize(QuantLinearMixin, unittest.TestCase):
  def test_wmma_architecture_selection(self):
    if not amd_custom_kernels_supported(device:=Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    self.assertEqual(_is_rdna4(device), Device[device].target[0] == 12)

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

  def test_quantize_nonunit_rounding_ties(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    values = np.zeros((4, 32), dtype=np.float16)
    values[:, 0] = [1.8837890625, 2.525390625, 2.23828125, 1.736328125]
    values[:, 1], values[:, 2] = values[:, 0]/2, -values[:, 0]/2
    values[3, 3:5] = [-0.3076171875, -0.6904296875]
    expected_scale = np.abs(values.astype(np.float32)).max(-1, keepdims=True)/127
    quant, scale, _ = q8_quantize(Tensor(values), 4, 32)
    np.testing.assert_array_equal(scale.numpy(), expected_scale)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(4, 32).numpy(), np.rint(values/expected_scale).astype(np.int8))

  def test_q6_linear_compiles_in_function(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, 210, dtype=np.uint8)
    packed[-2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (2, 0))).contiguous().realize()[2:]
    decoded = ggml_data_to_tensor(raw, 256, 14).reshape(1, 256)
    linear = Linear(256, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    self.assertTrue(np.isfinite(run(Tensor.randn(1, 256)).realize().item()))
    self.assertEqual(linear.weight.nbytes(), 210)
    self.assertEqual(linear.weight.dtype, dtypes.uint16)

  def test_q4_k_linear(self): self._test_quant_linear(12, 144)
  def test_iq4_linear(self): self._test_quant_linear(23, 136)
  def test_q5_linear(self): self._test_quant_linear(13, 176)
  def test_head_linear(self):  # each head's rows on its own weights (an mla's absorbed v_b): as x @ w.T of the decoded weights in half
    rng = np.random.default_rng(3)
    for ggml_type, H, out_features, in_features in ((8, 4, 512, 128), (6, 4, 512, 128), (Q4_K, 3, 128, 512)):  # (Q8_0, Q5_0: k_b, Q4_K: v_b)
      with self.subTest(ggml_type=ggml_type):
        block = {8: (34, 32), 6: (22, 32)}.get(ggml_type, (QUANT_SIZES[ggml_type], 256))  # (bytes, weights)
        packed = rng.integers(0, 256, (H*out_features*in_features // block[1], block[0]), dtype=np.uint8)
        packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if ggml_type == Q4_K: packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
        w = ggml_data_to_tensor(raw, H*out_features*in_features, ggml_type).reshape(H, out_features, in_features)
        x = rng.standard_normal((1, H, 128, in_features)).astype(np.float16)
        want = x.astype(np.float32) @ w.numpy().astype(np.float16).astype(np.float32).transpose(0, 2, 1)
        # (a symbolic count of rows: those there are)
        q8 = 2e-2 * np.abs(want).max()
        for T, n, atol in ((128, 128, 2e-2), (UOp.variable("rows", 1, 128).bind(77), 77, 2e-2), (1, 1, q8), (2, 2, q8)):
          got = head_linear(Tensor(x)[:, :, :T], w)
          assert got is not None
          np.testing.assert_allclose(got.pad_to((1, H, 128, out_features)).numpy()[:, :, :n], want[:, :, :n], rtol=3e-3, atol=atol)

  def test_q5_0_linear(self): self._test_quant_linear(6, 176, in_features=288, out_features=16, token_counts=(1, 3, 16))
  def test_q8_0_linear_small_blocks(self): self._test_quant_linear(8, 272, in_features=288, out_features=16, token_counts=(1, 16))
  def test_q6_linear_odd_blocks(self): self._test_quant_linear(14, 210, in_features=768, out_features=3, token_counts=(1, 3, 16))
  def test_q2_k_linear(self): self._test_quant_linear(10, 84, in_features=768, token_counts=(1, 3, 16))
  def test_q3_k_linear(self): self._test_quant_linear(11, 110, in_features=768, token_counts=(1, 3, 16))
  def test_iq2_xs_linear(self): self._test_quant_linear(17, 74, in_features=768, token_counts=(1, 3, 16))
  def test_iq3_xxs_linear(self): self._test_quant_linear(18, 98, in_features=768, token_counts=(1, 3, 16))
  def test_iq4_nl_linear(self): self._test_quant_linear(20, 144, in_features=768, token_counts=(1, 3, 16))
  def test_iq3_s_linear(self): self._test_quant_linear(21, 110, in_features=768, token_counts=(1, 3, 16))
  def test_iq2_s_linear(self): self._test_quant_linear(22, 82, in_features=768, token_counts=(1, 3, 16))

  def test_wmma_linear_symbolic(self):
    with patch("tinygrad.llm.kernels.amd_gguf.q8_quantize", side_effect=AssertionError("expected WMMA")):
      for typ in QUANT_SIZES:
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=48, token_counts=(3, 17), bias=True, symbolic=True)

  def test_mixed_quant_linear_wmma_tiles(self):
    with patch("tinygrad.llm.kernels.amd_gguf.q8_quantize", side_effect=AssertionError("expected WMMA")):
      for typ in HALFWORD_QUANTS:
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=768, out_features=64, token_counts=(16, 32, 64), bias=True)

  def test_wmma_dequantization_basis(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    from gguf import dequantize, GGMLQuantizationType
    rng = np.random.default_rng(1907)
    scales = np.array([0, 2**-24, -2**-24, 2**-14, -0.00035, .001, .0037, -.125, 1, -4, 8], np.float16)
    identity = Tensor(np.eye(768, dtype=np.float16)).realize()
    for typ, size in QUANT_SIZES.items():
      with self.subTest(ggml_type=typ):
        packed = rng.integers(0, 256, (48*3, size), dtype=np.uint8)
        blocks = packed.reshape(-1, SMALL_BLOCK_QUANTS[typ]*2) if typ in SMALL_BLOCK_QUANTS else packed
        offset = size-2 if typ in (11, 14) else 80 if typ == 10 else 0
        blocks[:, offset:offset+2] = np.resize(scales, len(blocks)).view(np.uint8).reshape(-1, 2)
        if typ in (10, 12, 13):
          offset = 82 if typ == 10 else 2
          blocks[:, offset:offset+2] = np.resize(scales[::-1], len(blocks)).view(np.uint8).reshape(-1, 2)
        expected = dequantize(packed.flatten(), GGMLQuantizationType(typ)).reshape(48, 768).astype(np.float16).astype(np.float32).T
        # Odd super-block counts and halfword-only alignment exercise both row and block offsets.
        pad = 2 if typ in HALFWORD_QUANTS else 4
        raw = Tensor(np.pad(packed.flatten(), (pad, 0))).realize()[pad:]
        linear = Linear(768, 48, bias=False)
        linear.weight = ggml_data_to_tensor(raw, 48*768, typ).reshape(48, 768)
        actual = linear(identity).numpy()
        self.assertEqual(linear.ggml_type, typ)
        # Bare RDNA3 WMMA has a small FP32 accumulation bias even for identity products.
        np.testing.assert_array_equal(actual.astype(np.float16), expected.astype(np.float16))
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2**-36)

  def test_quant_linear_unaligned_storage_fallback(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    from gguf import dequantize, GGMLQuantizationType
    rng = np.random.default_rng(73)
    x = rng.normal(size=(16, 256)).astype(np.float32)
    for typ, offset in ((12, 1), (12, 2), (12, 3), (21, 1), (21, 3)):
      with self.subTest(ggml_type=typ, offset=offset):
        packed = rng.integers(0, 256, (16, QUANT_SIZES[typ]), dtype=np.uint8)
        packed[:, :2] = np.array([.001], np.float16).view(np.uint8)
        if typ == 12: packed[:, 2:4] = np.array([.0002], np.float16).view(np.uint8)
        weight = dequantize(packed.flatten(), GGMLQuantizationType(typ)).reshape(16, 256)
        raw = Tensor(np.pad(packed.flatten(), (offset, 0))).realize()[offset:]
        linear = Linear(256, 16, bias=False)
        linear.weight = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), x @ weight.T, rtol=2e-5, atol=1e-5)
        self.assertIsNone(linear.ggml_type)

  def test_mixed_quant_linear_split_k(self):
    for typ in (10, 11, 17, 18, 20, 21, 22):
      with self.subTest(ggml_type=typ):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=1280, out_features=3, token_counts=(1,))

  def test_quant_linear_decode_row_groups(self):
    # One, two, and four waves per workgroup; ragged split-K and multiple tokens must stay independent.
    for typ in (12, 13, 14, 23):
      for outputs in (3, 6, 12):
        with self.subTest(ggml_type=typ, out_features=outputs):
          self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=1280, out_features=outputs, token_counts=(1, 3))

  def test_quant_linear_bias(self):
    for typ in (12, 21, 23):
      with self.subTest(ggml_type=typ):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=16, token_counts=(1, 3, 16), bias=True)

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

  def test_quant_linear_rejects_unaligned_rows_and_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for width in (128, 256):
      with self.subTest(width=width):
        packed = np.zeros((2*width//256, 136), dtype=np.uint8)
        packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        packed[:, 8:] = np.arange(128, dtype=np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).realize()[4:]
        weight = ggml_data_to_tensor(raw, 2*width, 23).reshape(2, width)
        if width == 256: weight = weight.int().float()
        expected = weight.numpy().sum(-1)[None]
        linear = Linear(width, 2, bias=False)
        linear.weight = weight
        np.testing.assert_allclose(linear(Tensor.ones(1, width)).numpy(), expected, rtol=1e-3, atol=1e-3)
        self.assertIsNone(linear.ggml_type)

  def test_dense_gemv_preserves_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1)
    linear.weight = Tensor.full((1, 128), 0.75).contiguous().realize().int().float()
    linear.bias = Tensor.full((1,), 0.75).contiguous().realize().int().float()
    np.testing.assert_array_equal(linear(Tensor.ones(1, 128)).numpy(), 0)

  def test_dense_gemv_float32_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1, bias=False)
    linear.weight = Tensor.full((1, 128), 1/128, dtype=dtypes.float32).realize()
    np.testing.assert_array_equal(linear(Tensor.full((1, 128), 65536, dtype=dtypes.float32)).numpy(), 65536)

  def test_gated_delta_state_and_precision(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for case in ("view", "reset", "half"):
      with self.subTest(case=case):
        q = Tensor.full((1, 1, 1, 32), 256 if case == "half" else 1, dtype=dtypes.half if case == "half" else dtypes.float32)
        state = Tensor.full((1, 1, 32, 4), int(case == "reset"), dtype=dtypes.float32).contiguous().realize().transpose(-1, -2)
        if case != "view": state = state.contiguous().realize()
        start = Tensor(UOp.variable("start_pos", 0, 10).bind(0)) if case == "reset" else None
        beta = Tensor.full((1, 1, 1), 1/2097152 if case == "half" else 1, dtype=dtypes.float32)
        if case != "reset":
          message = "recurrent state must be contiguous" if case == "view" else "recurrent Q/K must be float32"
          with self.assertRaisesRegex(AssertionError, message):
            gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
          continue
        out = gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
        np.testing.assert_array_equal(out.numpy(), 32)
        np.testing.assert_array_equal(state.numpy(), 1)

  def _test_moe_decode(self, typ:int):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(5)
    E, T, K, IN, OUT, size = 4, 1, 2, 256, 16, QUANT_SIZES[typ]
    blocks = rng.integers(0, 256, (E*OUT*IN//256, size), dtype=np.uint8)
    blocks[:, :2] = np.array([0.001], np.float16).view(np.uint8)
    blocks[:, 2:4] = np.array([0.0001], np.float16).view(np.uint8)
    decoded = ggml_data_to_tensor(Tensor(blocks.flatten()).contiguous().realize(), E*OUT*IN, typ).reshape(E, OUT, IN)
    w = decoded.numpy()
    act = rng.standard_normal((1, T, K, IN)).astype(np.float32)
    sel = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])[None].astype(np.int32)
    ew = ExpertWeights(E, IN, OUT)
    ew.weight = decoded.half()
    ew.set_quantized()
    self.assertEqual(ew.ggml_type, typ)
    got = ew(Tensor(sel), Tensor(act)).numpy()[0]
    for t in range(T):
      for j in range(K):
        ref = w[sel[0, t, j]] @ act[0, t, j]
        np.testing.assert_allclose(got[t, j], ref, rtol=2e-2, atol=2e-2*np.abs(ref).max())

  def test_moe_decode(self): self._test_moe_decode(Q4_K)
  def test_moe_decode_iq3_s(self): self._test_moe_decode(IQ3_S)

  def test_gated_delta_random_scan(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32) for _ in range(2))
    v = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    beta = rng.uniform(0, 1, (1, 2, 4)).astype(np.float32)
    beta[:, :, -1] = 0  # padded step must leave the recurrent state unchanged
    initial = rng.normal(0, 0.1, (1, 2, 8, 32)).astype(np.float32)
    for per_channel in (False, True):
      alpha = rng.uniform(0.8, 1, (1, 2, 4, 32) if per_channel else (1, 2, 4)).astype(np.float32)
      alpha[:, :, -1] = 1
      for start in (0, 3):
        with self.subTest(per_channel=per_channel, start=start):
          expected_state = np.zeros_like(initial) if start == 0 else initial.copy()
          outputs = []
          for t in range(4):
            decay = alpha[:, :, t, None, :] if per_channel else alpha[:, :, t, None, None]
            expected_state *= decay
            delta = (v[:, :, t] - (expected_state*k[:, :, t, None]).sum(-1)) * beta[:, :, t, None]
            expected_state += delta[..., None]*k[:, :, t, None]
            outputs.append((expected_state*q[:, :, t, None]).sum(-1))
          state = Tensor(initial).contiguous().realize()
          out = gated_delta_prefill(*map(Tensor, (q, k, v, beta, alpha)), state, Tensor(UOp.variable("start", 0, 3).bind(start)))
          np.testing.assert_allclose(out.numpy(), np.stack(outputs, axis=2), rtol=2e-4, atol=2e-6)
          np.testing.assert_allclose(state.numpy(), expected_state, rtol=2e-4, atol=2e-6)

  def test_gated_delta_kept_states(self): # a short scan keeps the states after its first tokens (a speculative verify's rollback)
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(1)
    q, k = (rng.normal(0, 0.1, (1, 2, 3, 32)).astype(np.float32) for _ in range(2))
    v, beta, alpha = rng.normal(size=(1, 2, 3, 8)).astype(np.float32), rng.uniform(0, 1, (1, 2, 3)).astype(np.float32), \
      rng.uniform(0.8, 1, (1, 2, 3, 32)).astype(np.float32)
    initial = rng.normal(0, 0.1, (1, 2, 8, 32)).astype(np.float32)
    expected, states = initial.copy(), []
    for t in range(3):
      expected *= alpha[:, :, t, None, :]
      expected += ((v[:, :, t] - (expected*k[:, :, t, None]).sum(-1)) * beta[:, :, t, None])[..., None]*k[:, :, t, None]
      states.append(expected.copy())
    state, kept = Tensor(initial).contiguous().realize(), Tensor.empty(2, 1, 2, 8, 32).realize()
    gated_delta_prefill(*map(Tensor, (q, k, v, beta, alpha)), state, Tensor(UOp.variable("start", 0, 3).bind(1)), kept).realize()
    np.testing.assert_allclose(kept.numpy(), np.stack(states[:2]), rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(state.numpy(), states[2], rtol=2e-4, atol=2e-6)

  def test_q8_0_linear(self): # a float weight quantized to q8_0 blocks, its gemv
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(3)
    w, x = rng.normal(size=(96, 256)).astype(np.float32), rng.normal(size=(2, 256)).astype(np.float32)
    out = q8_0_linear(Tensor(w), Tensor.empty(1).device)(Tensor(x)).numpy()
    np.testing.assert_allclose(out, x @ w.T, rtol=0.02, atol=0.02 * np.abs(x @ w.T).max())

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

  def test_attention_fallback_shapes(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for tokens, capacity, dim in ((1, 65, 64), (32, 64, 32), (32, 64, 384), (32, 64, 512)):
      with self.subTest(tokens=tokens, capacity=capacity, dim=dim):
        valid = 33
        cache = np.full((2, 1, 1, capacity, dim), np.nan, dtype=np.float16)
        cache[0, :, :, :valid] = 0
        cache[1, :, :, :valid] = np.arange(valid)[:, None]
        q = Tensor.zeros(1, 2, tokens, dim, dtype=dtypes.half)
        expected = np.broadcast_to(np.arange(valid-tokens, valid)[None, None, :, None]/2, q.shape)
        np.testing.assert_allclose(flash_attention(q, Tensor(cache), valid).numpy(), expected, rtol=1e-3, atol=1e-3)

  def test_attention_uses_physical_cache_length(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.half).contiguous()
    assigned = Tensor(cache.uop.after(cache[:, :, :, 0:1, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, 1).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

  def test_flash_attention_decode_symbolic_gqa(self):
    with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected custom decode")):
      self._test_flash_decode(8, 2, 256, 128, 37, symbolic=True)

  def test_flash_attention_decode_gqa_tail(self): self._test_flash_decode(3, 1, 192, 64, 37)

  def test_flash_attention_decode_gqa_output_layout(self): self._test_flash_decode(4, 1, 128, 256, 3)
  def test_flash_attention_decode_large_gqa_group(self): self._test_flash_decode(8, 1, 256, 256, 73)

  def _test_flash_decode(self, heads, kv_heads, dim, n, valid, symbolic=False):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = rng.normal(size=(1, heads, 1, dim)).astype(np.float16)
    cache = rng.normal(size=(2, 1, kv_heads, n, dim)).astype(np.float16)
    k, v = (np.repeat(c[0, :, :valid].astype(np.float32), heads//kv_heads, axis=0) for c in cache)
    scores = q[0].astype(np.float32) @ k.transpose(0, 2, 1) / np.sqrt(dim)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    cache_tensor = Tensor(cache)
    if symbolic:
      start_pos = UOp.variable("start_pos", 0, n-1).bind(valid-1)
      valid = start_pos + 1
      cache_tensor = cache_tensor.realize()
    np.testing.assert_allclose(flash_attention(Tensor(q), cache_tensor, valid).numpy(), expected[None], rtol=2e-3, atol=2e-3)

  def test_attention_symbolic_length_jit(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    cache = np.zeros((2, 1, 1, 128, 128), dtype=np.float16)
    cache[1] = np.arange(128, dtype=np.float16)[:, None]
    kv = Tensor(cache).realize()
    for tokens in (1, 32):
      q = Tensor(np.zeros((1, 2, tokens, 128), dtype=np.float16)).realize()
      run = TinyJit(lambda q, kv, end: flash_attention(q, kv, end).realize())
      for valid in (tokens, 33, 97, 128, 64, tokens):
        with self.subTest(tokens=tokens, valid=valid):
          end = UOp.variable("valid_length", tokens, 128).bind(valid)
          expected = np.broadcast_to(np.arange(valid-tokens, valid)[None, None, :, None]/2, q.shape)
          np.testing.assert_allclose(run(q, kv, end).numpy(), expected, rtol=1e-3, atol=1e-3)

  def test_prefill_attention_random(self):
    if not amd_custom_kernels_supported(d:=Tensor.empty(1).device) or _is_cdna(d): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(42)
    cases = ((2, 4, 2, 32, 64, 45, 64, False), (1, 3, 1, 64, 128, 83, 128, False),
             (1, 2, 1, 96, 256, 149, 192, False), (1, 4, 2, 17, 128, 128, 128, True), (1, 2, 1, 3, 64, 33, 64, True),
             (2, 3, 1, 64, 96, 99, 128, False), (1, 2, 1, 32, 320, 33, 64, False))
    for batch, heads, kv_heads, tokens, dim, valid, capacity, symbolic in cases:
      with self.subTest(batch=batch, tokens=tokens, dim=dim, symbolic=symbolic):
        q = rng.normal(size=(batch, heads, tokens, dim)).astype(np.float16)
        cache = rng.normal(size=(2, batch, kv_heads, capacity, dim)).astype(np.float16)
        cache[:, :, :, valid:] = np.nan
        k, v = (np.repeat(c[:, :, :valid].astype(np.float32), heads//kv_heads, axis=1) for c in cache)
        scores = q.astype(np.float32) @ k.swapaxes(-1, -2) / np.sqrt(dim)
        scores = np.where(np.arange(valid)[None, :] <= np.arange(valid-tokens, valid)[:, None], scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        expected = (probs / probs.sum(-1, keepdims=True)) @ v
        query, kv = Tensor(q), Tensor(cache).realize()
        end = UOp.variable("valid", 1, capacity).bind(valid)
        if symbolic:
          count = UOp.variable("query_tokens", 1, 32).bind(tokens)
          query = Tensor(np.pad(q, ((0, 0), (0, 0), (0, 32-tokens), (0, 0)))).contiguous()[:, :, :count]
        @function(allow_implicit=True)
        def run(q:Tensor, kv:Tensor, end:UOp): return flash_attention(q, kv, end)
        with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected WMMA attention")):
          actual = run(query, kv, end)[:, :, :tokens].numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-3)

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
    if not amd_custom_kernels_supported(d:=Tensor.empty(1).device): self.skipTest("RDNA3/4 or CDNA3/4 required")
    if _is_cdna(d) and DEV.interface.startswith("MOCK"): self.skipTest("too slow on the CDNA emulator")
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

class TestExpertLinear(unittest.TestCase):
  # a pool of slots, each an expert's gate and up rows then its down rows (lat -> 2*units -> lat)
  LAT, UNITS, NSLOTS = 3584, 512, 20
  ROW_UP, ROW_DOWN = LAT // 256 * QUANT_SIZES[IQ3_XXS], UNITS // 256 * QUANT_SIZES[IQ3_XXS]
  PART = 2 * UNITS * ROW_UP + LAT * ROW_DOWN

  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    self.rng = np.random.default_rng(0)
    self.raw = self.rng.integers(0, 256, self.NSLOTS * self.PART, dtype=np.uint8)
    self.raw.reshape(-1, QUANT_SIZES[IQ3_XXS])[:, :2] = np.array([0.01], np.float16).view(np.uint8)  # sane block scales
    self.pool = Tensor(self.raw).realize()
    self.sel = [5, 0, 19, 7, 3, 11, 2, 18, 9, 1, 14, 6, 17, 4, 12, 8]
    self.slots = Tensor(np.array(self.sel, np.int32)).realize()

  def weight(self, slot:int, off:int, rows:int, cols:int) -> np.ndarray:
    blob = Tensor(self.raw[slot * self.PART + off:slot * self.PART + off + rows * cols // 256 * QUANT_SIZES[IQ3_XXS]])
    return ggml_data_to_tensor(blob, rows * cols, IQ3_XXS).reshape(rows, cols).numpy()

  def test_gate_up_shared_input(self):
    x = self.rng.standard_normal((1, self.LAT)).astype(np.float32)
    out = expert_linear(self.pool, self.slots, Tensor(x), 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART).numpy()
    for k, s in enumerate(self.sel):
      ref = self.weight(s, 0, 2 * self.UNITS, self.LAT) @ x[0]
      np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_down_per_expert_input(self):
    h, off = self.rng.standard_normal((len(self.sel), self.UNITS)).astype(np.float32), 2 * self.UNITS * self.ROW_UP
    out = expert_linear(self.pool, self.slots, Tensor(h), self.LAT, self.UNITS, IQ3_XXS, self.PART, off).numpy()
    for k, s in enumerate(self.sel):
      ref = self.weight(s, off, self.LAT, self.UNITS) @ h[k]
      np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_pool_sharded(self): # device 1's pool is device 0's a slot on: each device its own gate/up rows on x, then its down rows on its own h
    devs = tuple(f"{Device.DEFAULT.split(':')[0]}:{i}" for i in range(2))
    pool = Tensor(np.concatenate([self.raw, np.roll(self.raw, -self.PART)])).shard(devs, 0)
    x, slots = Tensor(self.rng.standard_normal((1, self.LAT)).astype(np.float32)), self.slots.shard(devs)
    h, off = self.rng.standard_normal((2, len(self.sel), self.UNITS)).astype(np.float32), 2 * self.UNITS * self.ROW_UP
    gu = expert_linear(pool, slots, x.shard(devs), 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART).numpy()
    down = expert_linear(pool, slots, Tensor(h).shard(devs, 0), self.LAT, self.UNITS, IQ3_XXS, self.PART, off).numpy()
    for g in range(2):
      for k, s in enumerate(self.sel):
        for out, ref in ((gu[g, k], self.weight((s + g) % self.NSLOTS, 0, 2 * self.UNITS, self.LAT) @ x.numpy()[0]),
                         (down[g, k], self.weight((s + g) % self.NSLOTS, off, self.LAT, self.UNITS) @ h[g, k])):
          np.testing.assert_allclose(out, ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_act_in_the_q8(self): # the down gemv on act(gate, up) of the gate and up gemv's partial sums: as on the act computed apart
    x, off = Tensor(self.rng.standard_normal((1, self.LAT)).astype(np.float32)), 2 * self.UNITS * self.ROW_UP
    def act(g, u): return g.sigmoid() * u
    gu = expert_linear(self.pool, self.slots, x, 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, split=True)
    out = expert_linear(self.pool, self.slots, gu, self.LAT, self.UNITS, IQ3_XXS, self.PART, off, act=act).numpy()
    summed = gu.sum(-1)
    ref = expert_linear(self.pool, self.slots, act(summed[:, :self.UNITS], summed[:, self.UNITS:]), self.LAT, self.UNITS, IQ3_XXS, self.PART, off)
    np.testing.assert_allclose(out, ref.numpy(), rtol=1e-3, atol=1e-3 * np.abs(ref.numpy()).max())

  def test_weighted_sum(self): # a token's rows (j*tokens + t) summed weighted in the down gemv: as summed apart
    h, off = Tensor(self.rng.standard_normal((len(self.sel), self.UNITS)).astype(np.float32)), 2 * self.UNITS * self.ROW_UP
    w = self.rng.standard_normal(len(self.sel)).astype(np.float32)
    out = expert_linear(self.pool, self.slots, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off, w=Tensor(w), rows=2).numpy()
    ref = expert_linear(self.pool, self.slots, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off).numpy()
    ref = (ref * w[:, None]).reshape(-1, 2, self.LAT).sum(0)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3 * np.abs(ref).max())

  def test_a_slots_rows(self): # per tokens a slot (their weights loaded once): as with each slot repeated
    x, off, per = Tensor(self.rng.standard_normal((2, self.LAT)).astype(np.float32)), 2 * self.UNITS * self.ROW_UP, 2
    rep = Tensor(np.repeat(np.array(self.sel, np.int32), per))
    out = expert_linear(self.pool, self.slots, x, 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, per=per).numpy()
    np.testing.assert_allclose(out, expert_linear(self.pool, rep, x, 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART).numpy(), rtol=1e-5, atol=1e-5)
    h = Tensor(self.rng.standard_normal((len(self.sel) * per, self.UNITS)).astype(np.float32))
    w = Tensor(self.rng.standard_normal(len(self.sel) * per).astype(np.float32))
    out = expert_linear(self.pool, self.slots, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off, w=w, rows=per, per=per).numpy()
    ref = expert_linear(self.pool, rep, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off, w=w, rows=per).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())

  def test_expert_tiles(self): # tiles of rows an expert's (16 or 64, f16 WMMA), x's rows from a table (-1: x[0]) or its own: as the weights decoded
    if not _is_rdna4(self.pool.device): self.skipTest("RDNA4 required")
    for tile in (16, 64):
      with self.subTest(tile=tile):
        tiles, off, n = Tensor(np.array(self.sel[:2] + [-1], np.int32)), 2 * self.UNITS * self.ROW_UP, 3 * tile
        x = self.rng.standard_normal((5, self.LAT)).astype(np.float32)
        xr = self.rng.integers(-1, 5, n).astype(np.int32)
        out = expert_tiles(self.pool, tiles, Tensor(x), 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, xrow=Tensor(xr), tile=tile).float().numpy()
        h = self.rng.standard_normal((n, self.UNITS)).astype(np.float32)
        down = expert_tiles(self.pool, tiles, Tensor(h), self.LAT, self.UNITS, IQ3_XXS, self.PART, off, tile=tile).float().numpy()
        def act(g, u): return g * (u / 25).tanh()
        gu = expert_tiles(self.pool, tiles, Tensor(x), 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, xrow=Tensor(xr), act=act,
                          tile=tile).float().numpy()
        live = np.where(np.arange(n) % 32 < 12, 0, -1).astype(np.int32)  # (rows 16..31 of every 32 padding: their 16 left out, 0)
        kept = np.repeat(live[::16] >= 0, 16)
        for got, want in ((expert_tiles(self.pool, tiles, Tensor(x), 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, xrow=Tensor(xr), tile=tile,
                                        live=Tensor(live)), out),
                          (expert_tiles(self.pool, tiles, Tensor(h), self.LAT, self.UNITS, IQ3_XXS, self.PART, off, tile=tile,
                                        live=Tensor(live)), down)):
          got = got.float().numpy()
          np.testing.assert_array_equal(got[kept], want[kept])
          np.testing.assert_array_equal(got[~kept], 0)
        for t, s in enumerate(self.sel[:2]):
          rows = slice(tile * t, tile * t + tile)
          ref = x[np.maximum(xr[rows], 0)].astype(np.float16).astype(np.float32) @ self.weight(s, 0, 2 * self.UNITS, self.LAT).T
          np.testing.assert_allclose(out[rows], ref, rtol=2e-2, atol=2e-3 * np.abs(ref).max())
          np.testing.assert_allclose(gu[rows], ref[:, :self.UNITS] * np.tanh(ref[:, self.UNITS:] / 25), rtol=2e-2, atol=2e-3 * np.abs(ref).max())
          ref = h[rows].astype(np.float16).astype(np.float32) @ self.weight(s, off, self.LAT, self.UNITS).T
          np.testing.assert_allclose(down[rows], ref, rtol=2e-2, atol=2e-3 * np.abs(ref).max())

  def test_live_rows(self): # the rows of weight 0 skipped: a slot's gate and up rows where all its tokens' are (0 out), the down rows' sums
    x, off, per = Tensor(self.rng.standard_normal((2, self.LAT)).astype(np.float32)), 2 * self.UNITS * self.ROW_UP, 2
    w = self.rng.standard_normal(len(self.sel) * per).astype(np.float32)
    w[[0, 1, 6, 7, 9, 30, 31]] = 0  # (slots 0, 3 and 15 dead, a token of slot 4)
    gu = expert_linear(self.pool, self.slots, x, 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, per=per, live=Tensor(w)).numpy()
    ref = expert_linear(self.pool, self.slots, x, 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, per=per).numpy()
    dead = (w.reshape(-1, per) == 0).all(1).repeat(per)
    np.testing.assert_array_equal(gu[dead], 0)
    np.testing.assert_allclose(gu[~dead], ref[~dead], rtol=1e-5, atol=1e-5)
    h = Tensor(self.rng.standard_normal((len(self.sel) * per, self.UNITS)).astype(np.float32))
    out = expert_linear(self.pool, self.slots, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off, w=Tensor(w), rows=per, per=per).numpy()
    ref = expert_linear(self.pool, self.slots, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off, per=per).numpy()
    ref = (ref * w[:, None]).reshape(-1, per, self.LAT).sum(0)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3 * np.abs(ref).max())

  def test_live_token_entries(self): # per=1: a zero-weight entry must skip its NaN weights even if the neighboring token's entry is live
    off = 2 * self.UNITS * self.ROW_UP
    weights = self.rng.uniform(0.1, 1, len(self.sel)).astype(np.float32)
    weights[::3] = 0
    raw = self.raw.copy().reshape(self.NSLOTS, -1)
    for slot in np.array(self.sel)[weights == 0]:
      raw[slot].reshape(-1, QUANT_SIZES[IQ3_XXS])[:, :2] = np.array([np.nan], np.float16).view(np.uint8)
    pool = Tensor(raw.flatten()).realize()
    for tokens, entries in ((1, 16), (2, 16), (4, 16), (2, 6)):  # six entries: 48 groups, a final half-wave
      w, slots = Tensor(weights[:entries]), self.slots[:entries]
      x = Tensor(self.rng.standard_normal((tokens, self.LAT)).astype(np.float32))
      gu = expert_linear(pool, slots, x, 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART, live=w).numpy()
      ref = expert_linear(self.pool, slots, x, 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART).numpy()
      np.testing.assert_array_equal(gu[weights[:entries] == 0], 0)
      np.testing.assert_allclose(gu[weights[:entries] != 0], ref[weights[:entries] != 0], rtol=1e-5, atol=1e-5)
      h = Tensor(self.rng.standard_normal((entries, self.UNITS)).astype(np.float32))
      out = expert_linear(pool, slots, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off, w=w, rows=tokens).numpy()
      ref = expert_linear(self.pool, slots, h, self.LAT, self.UNITS, IQ3_XXS, self.PART, off).numpy()
      ref = (ref * weights[:entries, None]).reshape(-1, tokens, self.LAT).sum(0)
      np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3 * np.abs(ref).max())

  def test_pool_past_2_31_words(self): # a slot past 2**31 words of the pool (64 bit offsets), the pool's zeros up to its end
    nslots = (1 << 32) // self.PART + 2
    pool = Tensor.zeros(nslots * self.PART // 2, dtype=dtypes.uint16).contiguous().realize()
    self.assertEqual(pool[-4:].tolist(), [0] * 4)
    last = nslots - 1
    pool[last * self.PART // 2:(last + 1) * self.PART // 2].assign(Tensor(self.raw[:self.PART].view(np.uint16))).realize()
    x = self.rng.standard_normal((1, self.LAT)).astype(np.float32)
    out = expert_linear(pool, Tensor([last], dtype=dtypes.int32), Tensor(x), 2 * self.UNITS, self.LAT, IQ3_XXS, self.PART).numpy()
    ref = self.weight(0, 0, 2 * self.UNITS, self.LAT) @ x[0]
    np.testing.assert_allclose(out[0], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

class TestMoeRoute(unittest.TestCase):
  def test_invalid_route(self):
    T, E, K = 5, 256, 8
    logits = np.zeros((T, E), np.float32)
    logits[1, 119], logits[2], logits[3] = np.nan, np.nan, -np.inf
    weights, pos, entries, _, counts = (t.numpy() for t in moe_route(Tensor(logits), Tensor.zeros(E), K, 16))
    self.assertEqual(int(counts.sum()), 2*K)
    np.testing.assert_equal(pos.reshape(T, K)[1:4], -1)
    np.testing.assert_equal(weights[1:4], 0)
    np.testing.assert_equal(np.sort(entries[entries >= 0]), np.r_[0:K, 4*K:5*K])
    np.testing.assert_allclose(weights[[0, 4]].sum(-1), 1)

  def test_route(self): # the host's route(): top k of sigmoid + bias (ties: the lower expert), weights over the k's sum; rows sorted into tiles
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    T, E, K, tile = 300, 896, 16, 64
    rng = np.random.default_rng(0)
    logits, bias = rng.standard_normal((T, E)).astype(np.float32), (rng.standard_normal(E) * 0.1).astype(np.float32)
    logits[:, 100:140] = logits[:, :40]  # (ties)
    bias[100:140] = bias[:40]
    w, pos, entries, tiles, counts_gpu = (t.numpy() for t in moe_route(Tensor(logits), Tensor(bias), K, tile))
    sig = np.float32(1) / (np.float32(1) + np.exp(-logits))
    sel = np.lexsort((np.broadcast_to(np.arange(E), (T, E)), -(sig + bias)), axis=-1)[:, :K]
    top = np.take_along_axis(sig, sel, 1)
    total = np.zeros(T, np.float32)
    for j in range(K): total += top[:, j]
    np.testing.assert_allclose(w, top / total[:, None], rtol=1e-6)
    i = np.arange(T * K)
    np.testing.assert_equal(entries[pos], i)
    np.testing.assert_equal(tiles[pos // tile], sel.reshape(-1))
    counts = np.bincount(sel.reshape(-1), minlength=E)
    np.testing.assert_equal(counts_gpu, counts)
    np.testing.assert_equal(np.sort(tiles[tiles >= 0]), np.repeat(np.arange(E), (counts + tile - 1) // tile))
    self.assertEqual((entries >= 0).sum(), T * K)
    live = 211  # (the tokens from live on not routed: no rows, weight 0)
    Tensor.realize(*(outs:=moe_route(Tensor(logits), Tensor(bias), K, tile, UOp.variable("live", 1, T).bind(live))))
    w2, pos2, entries2, _, counts2 = (t.numpy() for t in outs)
    np.testing.assert_equal(pos2.reshape(T, K)[live:], -1)
    np.testing.assert_equal(w2[live:], 0)
    np.testing.assert_equal(counts2, np.bincount(sel[:live].reshape(-1), minlength=E))
    np.testing.assert_equal(np.sort(entries2[entries2 >= 0]), np.arange(live * K))

class TestTopk(unittest.TestCase):
  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 or CDNA3/4 required")

  def test_topk_indices(self): # each row's k largest of its first S scores (ties: the earlier, all of them while S <= k), S symbolic
    rng = np.random.default_rng(0)
    for R, N in ((1, 4096), (3, 5000)):
      s = rng.standard_normal((R, N)).astype(np.float32)
      s[0, :3000:7], s[-1, 100:200] = 0.5, -np.inf
      v, x = UOp.variable("S", 1, N), Tensor(s).pad((None, (0, -N % 1024))).contiguous().realize()
      for S in (1, 100, 2048, 2049, 3001, N):
        got = topk_indices(x, 2048, v.bind(S)).numpy()
        for r in range(R):
          self.assertEqual(set(got[r, :min(2048, S)].tolist()), set(np.lexsort((np.arange(S), -s[r, :S]))[:2048].tolist()), f"{R=} {S=} {r=}")

  def test_topk_long(self): # 262144 keys: a row of ties, one of -inf, one of +inf at the chunks' edges
    R, N = 4, 262144
    s = np.random.default_rng(41).standard_normal((R, N)).astype(np.float32)
    s[0, :4000:3], s[1, :], s[2, 100:200] = 0.5, 0.25, -np.inf
    s[3, [65535, 65536, 131071, 131072, N - 1]] = np.inf
    v, x = UOp.variable("S", 1, N), Tensor(s).realize()
    for S in (1, 2049, N - 3, N):
      got = topk_indices(x, 2048, v.bind(S)).numpy()
      for r in range(R): self.assertEqual(set(got[r, :min(2048, S)].tolist()), set(np.lexsort((np.arange(S), -s[r, :S]))[:2048].tolist()))

  def test_indexer_scores(self): # a row's (token t at start + t) score of each key: the heads' sum of w relu(q . key), -inf past the token
    rng = np.random.default_rng(0)
    for T, H, start in ((1, 32, 2999), (2, 32, 1000), (5, 4, 40)):
      q, w = rng.standard_normal((1, T, H, 128)).astype(np.float16), rng.standard_normal((1, T, H)).astype(np.float32)
      keys = rng.standard_normal((1, 4096, 128)).astype(np.float16)
      v = UOp.variable("start_pos", 0, 4095)
      got = indexer_scores(Tensor(q), Tensor(keys).realize(), Tensor(w), v.bind(start), v.bind(start) + T).numpy()[:, :start + T]
      want = (np.maximum(np.einsum("thd,sd->ths", q[0].astype(np.float32), keys[0, :start + T].astype(np.float32)), 0) * w[0][..., None]).sum(1)
      want[np.arange(start + T) > start + np.arange(T)[:, None]] = -np.inf
      np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3, err_msg=f"{T=} {H=}")

  def test_mla_gathered_keys(self): # attention over the keys of the indexer's positions: as over the cache masked to them
    cfg = TransformerConfig(num_blocks=1, dim=64, hidden_dim=64, n_heads=4, n_kv_heads=1, norm_eps=1e-5, vocab_size=16, head_dim=80, rope_theta=1e4,
                            rope_dim=16, v_head_dim=16, max_context=64, q_lora_rank=32, kv_lora_rank=64, mla_nope=True,
                            indexer=IndexerConfig(4, 16, 8))
    rng, blk = np.random.default_rng(0), MLATransformerBlock(cfg)
    for v in nn.state.get_parameters(blk): v.replace(Tensor(rng.standard_normal(v.shape).astype(np.float32) * 0.3).cast(v.dtype))
    for T, pos in ((1, 3), (2, 7), (1, 40), (2, 50), (5, 2), (16, 40)):  # (a chunk's rows too)
      x, p = Tensor(rng.standard_normal((1, T, 64)).astype(np.float32)), UOp.variable("start_pos", 0, 63).bind(pos)
      blk._init_state(x)
      cache = rng.standard_normal(blk.cache_k.shape).astype(np.float16)
      cache[:, :, pos + T:] = np.nan  # (past the keys there are: never read)
      blk.cache_k.assign(Tensor(cache)).realize()
      blk.indexer.cache_kg.assign(Tensor(rng.standard_normal(blk.indexer.cache_kg.shape).astype(np.float16))).realize()
      idx = blk._indexer_mask(x, (p,), None).numpy()
      idx[..., min(8, pos + T):] = 63  # (the slots past the keys there are: unset, here the last position's)
      sel, idx = idx[0, :, :min(8, pos + T)], Tensor(idx)
      mask = np.full((1, 1, T, pos + T), -np.inf, np.float32)
      for t in range(T): mask[0, 0, t, sel[t][sel[t] <= pos + t]] = 0
      np.testing.assert_allclose(blk._attention(x, (p,), idx).numpy(), blk._attention(x, (pos,), Tensor(mask)).numpy(), rtol=1e-4, atol=1e-4)

  def test_mla_every_position(self): # the top-k attention kernel without positions: causal over every key there is
    rng, H, D, V, N = np.random.default_rng(1), 4, 80, 64, 64
    cache = rng.standard_normal((1, 1, N, D)).astype(np.float16)
    for T, start in ((1, 0), (2, 30), (5, 40)):
      q = rng.standard_normal((1, H, T, D)).astype(np.float32)
      v = UOp.variable("start_pos", 0, N - 8).bind(start)
      got = mla_topk_attention(Tensor(q), Tensor(cache).realize(), None, v, V, 0.1).numpy()
      k = cache[0, 0, :start + T].astype(np.float32)
      s = q[0] @ k.T * 0.1 + np.where(np.arange(start + T) > start + np.arange(T)[:, None], -np.inf, 0)
      p = np.exp(s - s.max(-1, keepdims=True))
      np.testing.assert_allclose(got[0], (p / p.sum(-1, keepdims=True)) @ k[:, :V], rtol=1e-4, atol=1e-4)

  def test_mla_wmma(self): # a chunk's rows on the tensor cores (RDNA4): as the attention over their positions' keys (queries, probabilities: half)
    if not _is_rdna4(Tensor.empty(1).device): self.skipTest("RDNA4 required")
    rng, H, D, V, N, K, T = np.random.default_rng(2), 24, 576, 512, 256, 64, 32
    cache = (rng.standard_normal((1, 1, N, D)) * 0.5).astype(np.float16)
    for start, positions in ((0, True), (100, True), (100, False), (200, True)):
      q = (rng.standard_normal((1, H, T, D)) * 0.5).astype(np.float32)
      idx = np.stack([np.concatenate([p:=np.sort(rng.choice(min(start + t + 1, N), min(K, start + t + 1), replace=False)), np.full(K - len(p), -1)])
                      for t in range(T)]).astype(np.int32)
      v = UOp.variable("start_pos", 0, N - T).bind(start)
      got = mla_topk_attention(Tensor(q), Tensor(cache).realize(), Tensor(idx[None]) if positions else None, v, V, 0.05).numpy()[0]
      k = cache[0, 0].astype(np.float32)
      for t in range(T):
        keys = idx[t][(idx[t] >= 0) & (idx[t] <= start + t)] if positions else np.arange(start + t + 1)
        s = q[0, :, t] @ k[keys].T * 0.05
        p = np.exp(s - s.max(-1, keepdims=True))
        np.testing.assert_allclose(got[:, t], (p / p.sum(-1, keepdims=True)) @ k[keys, :V], rtol=2e-3, atol=2e-3, err_msg=f"{start=} {t=}")

  def test_mla_every_key(self): # no indexer: the indexed kernel over every position, as the dense attention masked causal
    cfg = TransformerConfig(num_blocks=1, dim=64, hidden_dim=64, n_heads=4, n_kv_heads=1, norm_eps=1e-5, vocab_size=16, head_dim=80, rope_theta=1e4,
                            rope_dim=16, v_head_dim=16, max_context=64, q_lora_rank=32, kv_lora_rank=64, mla_nope=True)
    rng, blk = np.random.default_rng(0), MLATransformerBlock(cfg)
    for v in nn.state.get_parameters(blk): v.replace(Tensor(rng.standard_normal(v.shape).astype(np.float32) * 0.3).cast(v.dtype))
    for T, pos in ((1, 3), (2, 7), (2, 50), (16, 40)):
      x, p = Tensor(rng.standard_normal((1, T, 64)).astype(np.float32)), UOp.variable("start_pos", 0, 63).bind(pos)
      blk._init_state(x)
      blk.cache_k.assign(Tensor(rng.standard_normal(blk.cache_k.shape).astype(np.float16))).realize()
      mask = np.where(np.arange(pos + T) <= pos + np.arange(T)[:, None], 0, -np.inf).astype(np.float32).reshape(1, 1, T, pos + T)
      np.testing.assert_allclose(blk._attention(x, (p,)).numpy(), blk._attention(x, (pos,), Tensor(mask)).numpy(), rtol=1e-4, atol=1e-4)

  def test_kda_decode(self): # a kda block's tokens in kda_decode (heads of 32) as the generic ops on the CPU: outputs, states, rollbacks
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 or CDNA required")
    for batch, groups in ((1, 2), (2, 2), (1, 1)):
      with self.subTest(batch=batch, groups=groups):
        ssm = SSMConfig(conv_kernel=4, state_size=32, group_count=groups, time_step_rank=2, inner_size=64, kda=True, gate_lower_bound=-5.0,
                        full_rank=(True, False))
        cfg = TransformerConfig(num_blocks=1, dim=64, hidden_dim=64, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=16, head_dim=32,
                                rope_theta=1e4, rope_dim=32, v_head_dim=32, max_context=16, ssm_layers=(True,), ssm=ssm)
        rng, models = np.random.default_rng(0), [Transformer(cfg) for _ in range(2)]
        for ps in zip(*(nn.state.get_parameters(m.blk[0]) for m in models)):
          arr = rng.standard_normal(ps[0].shape).astype(np.float32) * 0.3
          for p, dev in zip(ps, (Device.DEFAULT, "CPU")): p.replace(Tensor(arr, device=dev).cast(p.dtype))
        xs = rng.standard_normal((5, batch, 8, 64)).astype(np.float32)
        def steps(m:Transformer, dev:str) -> list[np.ndarray]:  # (a token from 0, verify sizes 2/3/4 rolled back to slots 0/1/2, a chunk's rows)
          blk, out = m.blk[0], []
          blk.keep = 3
          for x, T, pos in zip(xs, (1, 2, 3, 4, 8), (0, 1, 2, 4, 7)):
            x = Tensor(x[:, :T], device=dev)
            blk._init_state(x)
            out.append(blk._attention(x, (pos,)).numpy())
            if T in (2, 3, 4):
              m.rollback(T-1)
              out.extend(s.numpy().copy() for s in blk._states())
          return out + [s.numpy() for s in blk._states()]
        for got, want in zip(*(steps(m, dev) for m, dev in zip(models, (Device.DEFAULT, "CPU")))):
          np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

  def test_kda_decode_q8(self): # kda_decode's q8 of its outputs as q8_quantize's of them
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 or CDNA required")
    for D in (32, 64, 128):
      with self.subTest(dim=D):
        rng, H, R, T = np.random.default_rng(2), 2, 3, 4
        def t(*shape): return Tensor(rng.standard_normal(shape).astype(np.float32)).realize()
        states = [t(1, H, D, D), *(t(1, R, H*D) for _ in range(3))]
        pos = UOp.variable("start_pos", 0, 15).bind(4)
        inputs = [*(t(1, T, H*D) for _ in range(4)), t(1, T, H), t(1, T, H*D)]
        weights, dt, a, nw = [t(H*D, R+1) for _ in range(3)], t(H*D), t(H, 1), t(D)
        zq:tuple[Tensor, ...] = ()
        for token in range(T):
          zq, states = kda_decode(*inputs, weights, dt, a, nw, pos+token, -5.0, 1e-5, states, token=token, z=zq)
        Tensor.realize(*zq)
        for got, want in zip(zq[1:], q8_quantize(zq[0], T, H*D)): np.testing.assert_array_equal(got.numpy(), want.numpy())

class TestResidualMix(unittest.TestCase):
  def test_mix_and_norm(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 or CDNA required")
    rng = np.random.default_rng(3)
    for B, T, streams, dim in ((1, 1, 1, 256), (2, 2, 3, 512), (1, 2, 4, 7168), (1, 3, 14, 256), (1, 4, 15, 256)):
      with self.subTest(shape=(B, T, streams, dim)):
        xs, wm, wn = (Tensor(rng.standard_normal(shape).astype(np.float32)).realize() for shape in ((B, T, streams, dim), (dim,), (dim,)))
        scores = (xs.rmsnorm(eps=1e-5) * wm).sum(-1, keepdim=True)
        expected = ((scores.softmax(2) * xs).sum(2)).rmsnorm(eps=1e-5) * wn
        cfg = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=dim, n_heads=1, n_kv_heads=1, norm_eps=1e-5, vocab_size=16, head_dim=dim,
                                rope_theta=1e4, rope_dim=dim, v_head_dim=dim)
        mix, norm = AttentionResidual(cfg), nn.RMSNorm(dim, 1e-5)
        mix.weight, norm.weight = wm, wn
        np.testing.assert_allclose(mix.mix(xs, norm).numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)

  def test_one_pass(self): # a token's streams' sums of squares and dots with the mix weight, then their mix and its sum of squares
    rng = np.random.default_rng(4)
    for streams, dim, count, rows in ((9, 7168, 9, None), (9, 7168, 6, (2, 3)), (14, 256, 1, (0, 5)), (3, 512, 2, None)):
      with self.subTest(shape=(streams, dim, count, rows)):
        xs, wm, wn = (rng.standard_normal(shape).astype(np.float32) for shape in ((1, 5, streams, dim), (dim,), (dim,)))
        x = xs[0].astype(np.float64)[slice(None) if rows is None else slice(rows[0], rows[0] + rows[1])]
        score = (x @ wm) / np.sqrt((x * x).sum(-1) / dim + 1e-5)
        e = np.exp(np.where(np.arange(streams) < count, score, -np.inf) - score[:, :count].max(-1, keepdims=True))
        h = (e[..., None] * x).sum(1) / e.sum(-1, keepdims=True)
        expected = h / np.sqrt((h * h).sum(-1, keepdims=True) / dim + 1e-6) * wn
        cfg = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=dim, n_heads=1, n_kv_heads=1, norm_eps=1e-5, vocab_size=16, head_dim=dim,
                                rope_theta=1e4, rope_dim=dim, v_head_dim=dim)
        mix, norm = AttentionResidual(cfg), nn.RMSNorm(dim, 1e-6)
        mix.weight, norm.weight = Tensor(wm), Tensor(wn)
        y = mix.mix(Tensor(xs).realize(), norm, UOp.variable("count", 1, streams).bind(count), rows)
        np.testing.assert_allclose(y[0].numpy(), expected, rtol=1e-4, atol=1e-4)


class TestPartialSum(unittest.TestCase):
  def test_chained_in_a_jit(self): # every gpu's sums in gpu order, bitwise, over replays: one shot and two shot
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    if not amd_custom_kernels_supported(devs) or not _is_rdna4(devs): self.skipTest("4 RDNA4 gpus required")
    for n in (2048, 14336):
      p = np.random.default_rng(n).standard_normal((4, n)).astype(np.float32)
      parts, ref = Tensor(p).shard(devs, 0).realize(), np.zeros(n, np.float32)
      @TinyJit
      def chain(y:Tensor, parts=parts) -> Tensor:
        for _ in range(16): y = partial_sum(parts + y * 0.25)
        return y.realize()
      y = Tensor.zeros(n).contiguous().to(devs).realize()
      for _ in range(4): y = chain(y)
      for _ in range(4 * 16): ref = (((q:=p + ref * np.float32(0.25))[0] + q[1]) + q[2]) + q[3]
      for i in range(4): np.testing.assert_array_equal(Tensor(y.uop.mselect(i)).numpy().view(np.uint32), ref.view(np.uint32))

  def test_reduce_scatter(self): # the partials' sum split on its rows over the gpus (the ring's order, in half), in a jit over replays
    for ndev in (2, 3, 4):
      devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(ndev))
      p = np.random.default_rng(ndev).standard_normal((ndev, 16 * ndev, 64)).astype(np.float32)
      rs = TinyJit(lambda x: reduce_scatter(x * 2).realize())
      for _ in range(3): y = rs(Tensor(p).shard(devs, 0).realize())
      self.assertEqual((y.shape, y.uop.axis), ((16 * ndev, 64), 0))
      q = (p * 2).astype(np.float16 if amd_custom_kernels_supported(devs) else np.float32).reshape(ndev, ndev, 16, 64)
      ring = [functools.reduce(np.add, [q[(t + k) % ndev, t] for k in (*range(1, ndev), 0)]) for t in range(ndev)]
      np.testing.assert_array_equal(y.numpy(), np.concatenate(ring).astype(np.float32))

class TestAllgather(unittest.TestCase):
  def test_allgather_last(self): # a last axis shard on every gpu whole (4 RDNA4 gpus: their mailboxes, else a copy in half), over replays
    for ndev in (2, 4):
      devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(ndev))
      for shape in ((1, 3584), (2, 7168), (3, 72), (3, 3584)):
        x = Tensor(p:=np.random.default_rng(ndev).standard_normal(shape).astype(np.float32)).shard(devs, 1).realize()
        gather = TinyJit(lambda x: allgather_last(x * 2).contiguous().realize())
        for _ in range(3): y = gather(x)
        self.assertIsNone(y.uop.axis)
        # (the mailboxes' bitwise, a copy's in half)
        want = p * 2 if _mailbox_shape(x, int(np.prod(x.uop.max_shard_shape))) else (p * 2).astype(np.float16).astype(np.float32)
        for i in range(ndev): np.testing.assert_array_equal(Tensor(y.uop.mselect(i)).numpy(), want)
      self.assertIs(allgather_last(z:=Tensor.ones(4, 8).shard(devs, 0)), z)  # (not sharded on its last axis: as it is)

if __name__ == "__main__": unittest.main()
