import gc, unittest, weakref
import numpy as np
from tinygrad import Tensor, UOp, dtypes, function, Device
from tinygrad.helpers import Context
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, QUANT_SIZES, HALFWORD_QUANTS, iq4_half_lut, _iq_grid
from tinygrad.llm.gguf import ggml_data_to_tensor

class QuantLinearMixin:
  def _test_quant_linear(self, ggml_type, block_bytes, in_features=2048, out_features=64, token_counts=(1, 3, 32, 64, 128),
                         bias=False, custom=True, symbolic=False):
    if custom and not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (out_features*in_features//256, block_bytes), dtype=np.uint8)
    if ggml_type in (11, 14): packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    elif ggml_type == 10:
      packed[:, 80:82] = np.array([0.001], dtype=np.float16).view(np.uint8)
      packed[:, 82:84] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    elif ggml_type == 20: packed.reshape(-1, 18)[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    else: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    if ggml_type in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    linear.weight = decoded
    bias_value = rng.normal(size=out_features).astype(np.float32) if bias else 0
    if bias: linear.bias = Tensor(bias_value)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    for tokens in token_counts:
      # TODO: z3 cannot model the integer ORs in custom IQ3_S/IQ2_S lookup indices.
      # Compile locally so the CHECK_OOB override also applies to compilation.
      with self.subTest(tokens=tokens), Context(**({"CHECK_OOB": 0, "PARALLEL": 0} if custom and ggml_type in (21, 22) else {})):
        x = rng.normal(size=(tokens, in_features)).astype(np.float32 if tokens == 3 else np.float16)
        reference_x = x.astype(np.float32)
        wmma = custom and (32 if symbolic else tokens) % 16 == 0 and out_features % 16 == 0
        if wmma: reference_x = x.astype(np.float16).astype(np.float32)
        if custom and not wmma:
          grouped = reference_x.reshape(tokens, -1, 32)
          scale = np.maximum(np.abs(grouped).max(-1, keepdims=True) / 127, 1e-8)
          reference_x = (np.clip(np.rint(grouped/scale), -127, 127)*scale).reshape(tokens, in_features)
        reference_w = weight.astype(np.float16).astype(np.float32) if wmma else weight
        inp = Tensor(x) if not symbolic else Tensor(np.pad(x, ((0, 32-tokens), (0, 0)))).contiguous()[:
          UOp.variable("wmma_tokens", 1, 32).bind(tokens)]
        actual = (run if tokens == 1 or symbolic else linear)(inp)[:tokens].numpy()
        self.assertEqual(linear.ggml_type, ggml_type if custom else None)
        np.testing.assert_allclose(actual, reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
        if not symbolic and tokens == 3 and ggml_type not in (12, 13, 14, 23):
          sym = Tensor(np.pad(x, ((0, 1), (0, 0)))).contiguous()[:UOp.variable("tokens", 1, 4).bind(3)]
          np.testing.assert_allclose(linear(sym)[:3].numpy(), reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, ggml_type if custom else None)

class TestQ8Quantize(QuantLinearMixin, unittest.TestCase):
  def test_quant_tables_not_retained(self):
    # one _iq_grid table and the iq4 lut cover both table creation paths
    for typ in (18, 23):
      table = (iq4_half_lut(Device.DEFAULT) if typ == 23 else _iq_grid(Device.DEFAULT, typ)).realize()
      ref = weakref.ref(table)
      del table
      gc.collect()
      self.assertIsNone(ref())

  def test_quant_weights_share_storage(self):
    for ggml_type, type_size in QUANT_SIZES.items():
      with self.subTest(ggml_type=ggml_type):
        packed = np.arange(type_size + 4, dtype=np.uint8)
        raw = Tensor(packed).realize()[4:]
        if raw.uop.contiguous_view() is None: self.skipTest("requires buffer views")
        decoded = ggml_data_to_tensor(raw, 256, ggml_type).reshape(1, 256)
        linear = Linear(256, 1, bias=False)
        linear.set_quantized(decoded)
        linear.weight.realize()
        self.assertEqual(linear.ggml_type, ggml_type)
        self.assertEqual(linear.weight.dtype, dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32)
        self.assertEqual(linear.weight.nbytes(), type_size)
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), packed[4:])
        raw.assign(raw.full_like(1)).realize()
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), np.ones(type_size, dtype=np.uint8))

  def test_quant_linear_fallback(self):
    if amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("run with DISABLE_AMD_KERNELS=1")
    # per-type dequant math on the generic path is covered by test_gguf, spot check a representative set here
    for typ in (12, 14, 17, 23):
      with self.subTest(ggml_type=typ):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=16, token_counts=(1, 3), bias=True, custom=False)

if __name__ == "__main__": unittest.main()
