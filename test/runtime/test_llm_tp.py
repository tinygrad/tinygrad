import itertools, pathlib, tempfile, unittest
import numpy as np
from gguf import GGUFWriter
from tinygrad import Device, Tensor, nn
from tinygrad.helpers import DEV
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, shard_gguf
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES
from test.helpers import not_support_multi_device

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestTensorParallel(unittest.TestCase):
  def test_packed_layout(self):
    devices = (Device.DEFAULT, f"{Device.DEFAULT}:1")
    kv = {'general.architecture':'qwen35', 'qwen35.ssm.group_count':2, 'qwen35.ssm.time_step_rank':4, 'qwen35.ssm.state_size':1,
          'qwen35.ssm.inner_size':4, 'qwen35.attention.head_count':4, 'qwen35.attention.head_count_kv':2, 'qwen35.feed_forward_length':8,
          'tokenizer.ggml.tokens':['a', 'b']}
    for typ, name in itertools.product((12, 14, 23), ('ffn_gate', 'ffn_down', 'attn_qkv', 'ssm_out', 'ssm_norm')):
      with self.subTest(typ=typ, name=name):
        raw = np.random.default_rng(0).integers(0, 0x3c, (8, 4, QUANT_SIZES[typ]), dtype=np.uint8)
        key = f'blk.0.{name}.weight'
        w = shard_gguf({key: (Tensor(raw.flatten(), device='CPU'), (8, 1024), typ)}, kv, devices)[key]
        # Q|K|V0|V1: split each group, including the corresponding input columns of ssm_out.
        expected = [raw]*2 if name == 'ssm_norm' else np.split(raw, 2, axis=1) if name == 'ffn_down' else \
                   [raw[:, 0::2], raw[:, 1::2]] if name == 'ssm_out' else [raw[0::2], raw[1::2]] if name == 'attn_qkv' else np.split(raw, 2)
        self.assertEqual(w.shape, (expected[0].shape[0], expected[0].shape[1]*256))
        storage = next(u for u in w.uop.toposort() if u.op is Ops.BUFFER)
        np.testing.assert_array_equal(Tensor(storage.unshard(0)).numpy(), np.concatenate([p.flatten() for p in expected]))
        layer = Linear(*w.shape[::-1], bias=False)
        layer.set_quantized(w.half())
        self.assertEqual(layer.ggml_type, None if storage.contiguous_view_offset() is None else typ)

  @unittest.skipIf(DEV.interface.startswith("MOCK"), "too heavy for mock GPUs")
  def test_model(self):
    rng = np.random.default_rng(42)
    config = TransformerConfig(num_blocks=2, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=64, head_dim=64,
      v_head_dim=64, rope_theta=10000, rope_dim=16, max_context=64, qk_norm=64, attn_output_gate=True, ssm=SSMConfig(4, 32, 2, 4, 128),
      ssm_layers=(True, False))
    with tempfile.TemporaryDirectory() as folder:
      writer = GGUFWriter(path:=pathlib.Path(folder)/'model.gguf', 'qwen35')
      for key,value in {'context_length':64, 'embedding_length':256, 'feed_forward_length':512, 'block_count':2, 'full_attention_interval':2,
                        'ssm.conv_kernel':4, 'ssm.state_size':32, 'ssm.group_count':2, 'ssm.time_step_rank':4, 'ssm.inner_size':128,
                        'attention.head_count':4, 'attention.head_count_kv':2, 'attention.key_length':64, 'rope.dimension_count':16}.items():
        writer.add_uint32('qwen35.'+key, value)
      writer.add_float32('qwen35.rope.freq_base', 10000)
      writer.add_float32('qwen35.attention.layer_norm_rms_epsilon', 1e-5)
      writer.add_array('tokenizer.ggml.tokens', [str(i) for i in range(64)])
      for name,weight in nn.state.get_state_dict(Transformer(config)).items():
        value = -rng.random(weight.shape) if name.endswith('ssm_a') else rng.normal(1, .1, weight.shape) if 'norm' in name else \
                rng.normal(0, .5 if 'token_embd' in name else .05, weight.shape)
        writer.add_tensor(name.replace('ffn_norm', 'post_attention_norm'), value.astype(np.float32))
      writer.write_header_to_file()
      writer.write_kv_data_to_file()
      writer.write_tensors_to_file()
      writer.close()
      single, parallel = Transformer.from_gguf(path, 64)[0], Transformer.from_gguf(path, 64, shard=2)[0]
      self.assertEqual(parallel.blk[0].ffn_norm.weight.device, (Device.DEFAULT, f'{Device.DEFAULT}:1'))
      prompt = [int(x) for x in rng.integers(0, 64, 40)]
      self.assertEqual(list(itertools.islice(parallel.generate(list(prompt)), 6)),
                       list(itertools.islice(single.generate(list(prompt)), 6)))

if __name__ == '__main__': unittest.main()
