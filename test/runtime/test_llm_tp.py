import unittest, tempfile, pathlib, itertools
import numpy as np
from tinygrad import Tensor, UOp, nn, Device
from tinygrad.helpers import DEV
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, shard_gguf, allreduce, gather
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES, amd_custom_kernels_supported
from test.helpers import not_support_multi_device

SSM_KV = {'general.architecture':'qwen35', 'qwen35.ssm.group_count':2, 'qwen35.ssm.time_step_rank':4, 'qwen35.ssm.state_size':1,
          'qwen35.ssm.inner_size':4, 'qwen35.attention.head_count':4, 'qwen35.attention.head_count_kv':2, 'qwen35.feed_forward_length':8,
          'tokenizer.ggml.tokens':['a', 'b']}
MOCKGPU = DEV.interface.startswith("MOCK")
def devices() -> tuple[str, ...]: return (Device.DEFAULT, f"{Device.DEFAULT}:1")
def packed(typ:int, rows:int, cols:int) -> np.ndarray:
  # every byte < 0x3c: any fp16 scale read from the block is finite and < 1
  return np.random.default_rng(0).integers(0, 0x3c, (rows, cols//256, QUANT_SIZES[typ]), dtype=np.uint8)

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestShardGGUF(unittest.TestCase):
  def test_layout(self):
    for typ, name in itertools.product((12, 14, 23), ('ffn_gate.weight', 'ffn_down.weight', 'attn_qkv.weight', 'ffn_norm.weight')):
      with self.subTest(typ=typ, name=name):
        raw = packed(typ, 8, 1024)
        w = shard_gguf({f'blk.0.{name}': (Tensor(raw.flatten(), device='CPU'), (8, 1024), typ)}, SSM_KV, devices())[f'blk.0.{name}']
        self.assertEqual(w.shape, (8, 1024) if name == 'ffn_norm.weight' else (8, 512) if name == 'ffn_down.weight' else (4, 1024))
        # attn_qkv is Q|K|V0|V1 with two rows each: every device gets one row of each part
        expected = [raw]*2 if name == 'ffn_norm.weight' else np.split(raw, 2, axis=1) if name == 'ffn_down.weight' else \
                   [raw[0::2], raw[1::2]] if name == 'attn_qkv.weight' else np.split(raw, 2, axis=0)
        storage = next(u for u in w.uop.toposort() if u.op is Ops.BUFFER)
        for rank in range(2): np.testing.assert_array_equal(Tensor(storage.mselect(rank)).numpy(), expected[rank].flatten())
        # the local weight is recognized as its packed format and runs on the same storage (a view, devices without views keep it decoded)
        layer = Linear(*w.shape[::-1], bias=False)
        layer.set_quantized(w.half())
        self.assertEqual(layer.ggml_type, None if storage.contiguous_view_offset() is None else typ)

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestTensorParallel(unittest.TestCase):
  def test_linear(self):
    def per_device(*parts:np.ndarray) -> Tensor:
      # one tensor holding a different slice on every device
      bufs = [Tensor(p.flatten()).to(d).realize().uop for p,d in zip(parts, devices())]
      return Tensor(UOp.from_buffer(UOp.mstack(*bufs).buffer)).reshape(parts[0].shape)
    rng = np.random.default_rng(0)
    w1, w2, w3 = [rng.normal(size=s).astype(np.float32) for s in ((64, 32), (32, 64), (16, 32))]
    up, down, head = Linear(32, 32, bias=False), Linear(32, 32, bias=False), Linear(32, 8, bias=False)
    # column-parallel layers hold a slice of the output rows, the row-parallel layer a slice of the input columns
    up.weight, down.weight, head.weight = per_device(*np.split(w1, 2, 0)), per_device(*np.split(w2, 2, 1)), per_device(*np.split(w3, 2, 0))
    x = rng.normal(size=(1, 8, 32)).astype(np.float32)
    for n, symbolic in ((8, False), (5, True)):
      with self.subTest(n=n, symbolic=symbolic):
        tokens = UOp.variable('tokens', 1, 8).bind(n) if symbolic else n
        out = allreduce(down(up(Tensor(x).to(devices())[:, :tokens]).relu()))
        logits = gather(head(out)[:, -1, :])
        ref = np.maximum(x[:, :n] @ w1.T, 0) @ w2.T
        np.testing.assert_allclose(out.to(devices()[0])[:, :n].numpy(), ref, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(logits.numpy(), ref[:, -1] @ w3.T, atol=1e-4, rtol=1e-4)

  @unittest.skipIf(MOCKGPU, "too heavy for mock GPUs")
  def test_quantized_linear(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    rng = np.random.default_rng(0)
    for typ, name in itertools.product(QUANT_SIZES, ('ffn_gate.weight', 'ffn_down.weight')):
      raw = packed(typ, 64, 1024)
      single = Linear(1024, 64, bias=False)
      # the reference runs the same packed kernel on one device
      single.weight = ggml_data_to_tensor(Tensor(raw.flatten()).realize(), 64*1024, typ).reshape(64, 1024).half()
      w = shard_gguf({name: (Tensor(raw.flatten(), device='CPU'), (64, 1024), typ)}, SSM_KV, devices())[name]
      layer = Linear(*w.shape[::-1], bias=False)
      layer.weight = w.half()
      for tokens in (1, 3, 32):
        with self.subTest(typ=typ, name=name, tokens=tokens):
          x = Tensor(rng.normal(size=(1, tokens, 1024)).astype(np.float32)).realize()
          # row-parallel layers see their slice of the input features on each device
          xs = x.to(devices()) if name == 'ffn_gate.weight' else Tensor(UOp.from_buffer(UOp.mstack(*(x[..., i*512:(i+1)*512].contiguous()
            .flatten().to(d).realize().uop for i,d in enumerate(devices()))).buffer)).reshape(1, tokens, 512)
          if tokens == 3: x, xs = [t.pad_to((1, 32, t.shape[-1]))[:, :UOp.variable('tokens', 1, 32).bind(3)] for t in (x, xs)]
          out = layer(xs)
          out = Tensor(out.uop.unshard(2)) if name == 'ffn_gate.weight' else allreduce(out)
          ref = single(x)[:, :tokens].numpy()
          np.testing.assert_allclose(out.to(devices()[0])[:, :tokens].numpy(), ref, atol=1e-4*np.abs(ref).max(), rtol=1e-3)
          self.assertEqual((layer.ggml_type, single.ggml_type), (typ, typ))

  @unittest.skipIf(MOCKGPU, "too heavy for mock GPUs")
  def test_model(self):
    from gguf import GGUFWriter
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
      self.assertEqual(parallel.blk[0].ffn_norm.weight.device, devices())
      prompt = [int(x) for x in rng.integers(0, 64, 20)]
      # chunked prefill with a symbolic token count, then decode
      self.assertEqual(list(itertools.islice(parallel.generate(list(prompt), chunk_size=8), 6)),
                       list(itertools.islice(single.generate(list(prompt), chunk_size=8), 6)))

if __name__ == '__main__':
  unittest.main()
