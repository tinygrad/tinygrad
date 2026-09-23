import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device, TinyJit, nn, UOp
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES, HALFWORD_QUANTS, amd_custom_kernels_supported, flash_attention
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.parallel import _local_shard, load_sharded, shard_config, sum_shards

class TestTensorParallel(unittest.TestCase):
  devices = ('CPU', 'CPU:1')

  def setUp(self):
    if Device.DEFAULT != 'CPU': self.skipTest('CPU reference tests; use TestTensorParallelAMD for GPU integration')

  def test_segmented_shard(self):
    x = Tensor(np.arange(24*8, dtype=np.float32).reshape(24, 8), device='CPU')
    local = _local_shard(x, self.devices, 0, (8, 8, 8))
    for i, device in enumerate(self.devices):
      actual = Tensor(local.uop.mselect(i)).to(device).numpy()
      expected = np.concatenate([x.numpy()[s+i*4:s+(i+1)*4] for s in (0, 8, 16)])
      np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(sum_shards(local).to('CPU').numpy(),
                                  np.concatenate([x.numpy()[s:s+4]+x.numpy()[s+4:s+8] for s in (0, 8, 16)]))

  def test_uneven_heads(self):
    with self.assertRaisesRegex(AssertionError, 'uneven'):
      shard_config(self.config(), ('CPU', 'CPU:1', 'CPU:2'))

  @staticmethod
  def config(hybrid=False, repeats=2):
    return TransformerConfig(num_blocks=2, dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, norm_eps=1e-6, vocab_size=64,
      head_dim=32, rope_theta=10000, rope_dim=16, v_head_dim=32, max_context=64, qk_norm=32,
      attn_output_gate=hybrid, ssm_layers=(True, False) if hybrid else (),
      ssm=SSMConfig(conv_kernel=4, state_size=32, group_count=2, time_step_rank=2*repeats, inner_size=64*repeats) if hybrid else None)

  def test_transformer(self): self._test_model(False)
  def test_hybrid(self): self._test_model(True)
  def test_hybrid_three_groups(self): self._test_model(True, repeats=3)
  def test_nondefault_devices(self):
    self.devices = ('CPU:1', 'CPU:2')
    self._test_model(False)

  def _test_model(self, hybrid, repeats=2):
    config = self.config(hybrid, repeats)
    reference, parallel = Transformer(config), Transformer(config, self.devices)
    rng = np.random.default_rng(42)
    state = {k:Tensor(rng.normal(0, .05, v.shape).astype(np.float32), device='CPU') for k,v in nn.state.get_state_dict(reference).items()}
    for k in state:
      if 'norm' in k: state[k] = state[k] + 1
      if 'ssm_a' in k: state[k] = -state[k].abs()
    nn.state.load_state_dict(reference, state, verbose=False, realize=False)
    with patch('tinygrad.llm.parallel.amd_custom_kernels_supported', return_value=True):
      load_sharded(parallel, state.copy(), config, self.devices)

    def hidden(model, tokens, start):
      x = model.token_embd(tokens.to(model.token_embd.weight.device)).float()
      if model.devices: x = x.to(model.devices)
      for block in model.blk: x = block(x, start)
      return x.to(Device.DEFAULT).realize()
    ref_run, tp_run = TinyJit(lambda t, p: hidden(reference, t, p)), TinyJit(lambda t, p: hidden(parallel, t, p))
    # Reset, chunked prefill, decode, and reset after divergent history exercise both caches and recurrent state.
    for pos, tokens in [(0, [1, 2, 3]), (3, [4, 5]), (5, [6]), (6, [7]), (7, [8]), (8, [9]), (0, [10])]:
      t = Tensor([tokens], device=Device.DEFAULT)
      p = UOp.variable('start', 0, 63).bind(pos)
      actual = tp_run(t, p) if len(tokens) == 1 else hidden(parallel, t, p)
      expected = ref_run(t, p) if len(tokens) == 1 else hidden(reference, t, p)
      np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=3e-4, atol=3e-4)
    # Also test the vocabulary-sharded output projection and gathering before sampling.
    for model in (reference, parallel): model._cached_tokens = []
    self.assertEqual(list(zip(range(4), reference.generate([1, 2, 3]))), list(zip(range(4), parallel.generate([1, 2, 3]))))

def quantized_linear(typ, axis, devices):
  rng = np.random.default_rng(42)
  raw = rng.integers(0, 256, (64*4, QUANT_SIZES[typ]), dtype=np.uint8)
  blocks = raw.reshape(-1, 18) if typ == 20 else raw
  offset = QUANT_SIZES[typ]-2 if typ in (11, 14) else 80 if typ == 10 else 0
  blocks[:, offset:offset+2] = np.array([.001], np.float16).view(np.uint8)
  if typ in (10, 12, 13): blocks[:, 82 if typ == 10 else 2:84 if typ == 10 else 4] = np.array([.0002], np.float16).view(np.uint8)
  pad = 2 if typ in HALFWORD_QUANTS else 4
  storage = Tensor(np.pad(raw.flatten(), (pad, 0)), device='CPU').realize()[pad:]
  weight = ggml_data_to_tensor(storage, 64*1024, typ).reshape(64, 1024)
  local = Linear(1024//2 if axis == 1 else 1024, 64//2 if axis == 0 else 64, bias=False)
  name = 'ffn_down' if axis == 1 else 'ffn_gate'
  with patch('tinygrad.llm.parallel.amd_custom_kernels_supported', return_value=True):
    load_sharded({name:local}, {name+'.weight':weight}, TestTensorParallel.config(), devices)
  return local, weight

class TestPackedSharding(unittest.TestCase):
  def test_quant_formats(self):
    for typ in QUANT_SIZES:
      for axis in (0, 1):
        with self.subTest(typ=typ, axis=axis):
          local, weight = quantized_linear(typ, axis, ('CPU', 'CPU:1'))
          self.assertEqual(local.ggml_type, typ)
          self.assertEqual(local.weight.nbytes(), 64*1024//2//256 * QUANT_SIZES[typ])
          expected = weight.half().numpy()
          for i in range(2):
            raw = Tensor(local.weight.uop.mselect(i)).to('CPU').bitcast('uint8')
            actual = ggml_data_to_tensor(raw, 64*1024//2, typ).reshape(local.out_features, local.in_features).half().numpy()
            np.testing.assert_array_equal(actual, np.split(expected, 2, axis=axis)[i])

class TestTensorParallelAMD(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not amd_custom_kernels_supported(Device.DEFAULT): raise unittest.SkipTest('RDNA3/4 required, run in one process with two visible GPUs')
    if Device[Device.DEFAULT].count() < 2: raise unittest.SkipTest('two visible GPUs required')
    cls.devices = (Device.DEFAULT, f'{Device.DEFAULT}:1')
    for device in cls.devices: Device[device].synchronize()

  @classmethod
  def tearDownClass(cls):
    for device in cls.devices: Device[device].synchronize()

  def test_dense_and_hybrid(self):
    test = TestTensorParallel()
    test.devices = self.devices
    for hybrid in (False, True): test._test_model(hybrid)
    test._test_model(True, repeats=3)

  def test_flash_attention_262k(self):
    # Match Qwen's rank-local attention shape, and test ragged lengths all the way to the end of the physical cache.
    n = 262144
    cache = Tensor.zeros(2, 1, 2, n, 256, dtype='float16', device=self.devices).realize()
    values = (np.arange(n, dtype=np.float32) / n).astype(np.float16)
    cache[1].assign(Tensor(values, device=self.devices).reshape(1, 1, n, 1).expand(1, 2, n, 256)).realize()
    for tokens, length in ((1, 1), (1, 65), (1, 6749), (1, n-1), (1, n), (32, n-1)):
      q = Tensor.zeros(1, 12, tokens, 256, device=self.devices).realize()
      end = UOp.variable('kv_end', 1, n).bind(length)
      out = flash_attention(q, cache, end).realize()
      expected = np.array([values[:i+1].astype(np.float32).mean() for i in range(length-tokens, length)]).reshape(1, 1, tokens, 1)
      for i in range(2):
        np.testing.assert_allclose(Tensor(out.uop.mselect(i)).to(self.devices[0]).numpy(),
                                   np.broadcast_to(expected, out.shape), rtol=2e-3, atol=2e-4)

  def test_quantized_matmuls(self):
    for typ in QUANT_SIZES:
      for axis in (0, 1):
        for tokens in (1, 32):
          with self.subTest(typ=typ, axis=axis, tokens=tokens):
            local, weight = quantized_linear(typ, axis, self.devices)
            x = Tensor(np.random.default_rng(1).normal(size=(tokens, 1024)).astype(np.float32), device='CPU')
            inp = _local_shard(x, self.devices, 1) if axis == 1 else x.to(self.devices)
            out = local(inp)
            actual = sum_shards(out) if axis == 1 else Tensor(out.uop.unshard(1))
            if tokens == 32: reference_x, reference_w = x.half().numpy().astype(np.float32), weight.half().numpy().astype(np.float32)
            else:
              grouped = x.numpy().reshape(tokens, -1, 32)
              scale = np.maximum(np.abs(grouped).max(-1, keepdims=True) / 127, 1e-8)
              reference_x = (np.clip(np.rint(grouped/scale), -127, 127)*scale).reshape(tokens, 1024)
              reference_w = weight.numpy()
            np.testing.assert_allclose(actual.to(self.devices[0]).numpy(), reference_x @ reference_w.T, rtol=3e-3, atol=2e-2)

if __name__ == '__main__': unittest.main()
