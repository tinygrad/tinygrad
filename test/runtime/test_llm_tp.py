import itertools, pathlib, tempfile, unittest
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, GGML_QUANT_SIZES
from tinygrad import Device, Tensor, nn
from tinygrad.helpers import DEV
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.gguf import gguf_load, gguf_parse, gguf_shard
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES
from test.helpers import not_support_multi_device

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestGGUFShard(unittest.TestCase):
  Q = GGMLQuantizationType
  # name: (shape, type, split)
  tensors = {'rows': ((8, 1024), Q.Q4_K, (0, (1,))), 'cols': ((8, 1024), Q.Q6_K, (1, (1,))),
             'fused_rows': ((8, 1024), Q.Q4_K, (0, (1, 1, 2))), 'fused_cols': ((8, 1024), Q.Q8_0, (1, (1, 1))),
             'experts_rows': ((4, 8, 512), Q.Q4_K, (1, (1,))), 'experts_cols': ((4, 8, 512), Q.Q8_0, (2, (1,))),
             'vector': ((8,), Q.F32, (0, (1,))), 'copied': ((8, 512), Q.Q4_K, None)}
  devices = (Device.DEFAULT, f"{Device.DEFAULT}:1")

  @classmethod
  def setUpClass(cls):
    cls.folder = tempfile.TemporaryDirectory()
    writer, rng = GGUFWriter(path:=pathlib.Path(cls.folder.name)/'model.gguf', 'test'), np.random.default_rng(0)
    for name, (shape, typ, _) in cls.tensors.items():
      block, size = GGML_QUANT_SIZES[typ]
      # every byte < 0x3c: any fp16 scale read from a block is finite
      raw = rng.integers(0, 0x3c, (*shape[:-1], shape[-1]//block*size), dtype=np.uint8) if block > 1 else rng.normal(size=shape).astype(np.float32)
      writer.add_tensor(name, raw, raw_dtype=typ)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    cls.full = {k: v.numpy() for k,v in gguf_load(path)[1].items()}
    cls.entries, cls.splits = gguf_parse(path)[1], {name: spec for name, (_, _, spec) in cls.tensors.items() if spec is not None}
  @classmethod
  def tearDownClass(cls): cls.folder.cleanup()

  def test_shards(self):
    n, out = len(self.devices), gguf_shard(self.entries, self.devices, self.splits)
    for name, (_, _, spec) in self.tensors.items():
      with self.subTest(name=name):
        # the data of every device, stacked on a new axis
        got, full = Tensor(out[name].unsqueeze(0).uop.unshard(0)).to(Device.DEFAULT).numpy(), self.full[name]
        if spec is None: expected = [full]*n
        else:
          # device d gets its piece of every part
          axis, parts = spec
          pieces = np.split(full, np.cumsum([full.shape[axis]*p//sum(parts) for p in parts])[:-1], axis)
          expected = [np.concatenate([np.split(p, n, axis)[d] for p in pieces], axis) for d in range(n)]
        np.testing.assert_array_equal(got, np.stack(expected))

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestTensorParallel(unittest.TestCase):
  def test_packed_layout(self):
    devices = (Device.DEFAULT, f"{Device.DEFAULT}:1")
    # every weight is (8, 1024): 8 rows of 4 quantization blocks. name -> (axis, relative sizes of the fused parts along axis)
    #   ffn_gate (0, (1,)):       rows split in halves, device 0 gets rows 0-3, device 1 rows 4-7
    #   ffn_down (1, (1,)):       columns split in halves, device 0 gets blocks 0-1, device 1 blocks 2-3
    #   attn_qkv (0, (1,1,2)):    rows are Q|K|V of 2, 2 and 4 rows, every part is halved: device 0 gets rows 0,2,4,5, device 1 rows 1,3,6,7
    #   ssm_out  (1, (1,1)):      columns are the inputs from V0|V1 of 2 blocks each: device 0 gets blocks 0,2, device 1 blocks 1,3
    #   ssm_norm is not in the map: copied to both devices
    shard_map = {'ffn_gate.weight': (0, (1,)), 'ffn_down.weight': (1, (1,)), 'attn_qkv.weight': (0, (1, 1, 2)), 'ssm_out.weight': (1, (1, 1))}
    for typ, name in itertools.product((12, 14, 23), ('ffn_gate', 'ffn_down', 'attn_qkv', 'ssm_out', 'ssm_norm')):
      with self.subTest(typ=typ, name=name):
        raw = np.random.default_rng(0).integers(0, 0x3c, (8, 4, QUANT_SIZES[typ]), dtype=np.uint8)
        key = f'{name}.weight'
        w = gguf_shard({key: (Tensor(raw.flatten(), device='CPU'), (8, 1024), typ)}, devices, shard_map)[key]
        # every part is split on its own: Q|K|V of attn_qkv, the V0|V1 input columns of ssm_out
        expected = [raw]*2 if name == 'ssm_norm' else np.split(raw, 2, axis=1) if name == 'ffn_down' else \
                   [raw[:, 0::2], raw[:, 1::2]] if name == 'ssm_out' else [raw[[0,2,4,5]], raw[[1,3,6,7]]] if name == 'attn_qkv' else np.split(raw, 2)
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
