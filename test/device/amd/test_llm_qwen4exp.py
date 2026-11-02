import pathlib, re, tempfile, unittest
import numpy as np
from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType as Q
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, HCConfig, IndexerConfig, PLEConfig
from test.helpers import not_support_multi_device

VOCABS, MULS, EOS, PROMPT = (13, 17), (13703, 5749, 811), 63, [11, 15, 3, 63, 9, 8, 63, 12, 20, 33, 7]

def load_tiny_qwen4exp(shard:int=1, max_context:int=32) -> Transformer:
  # GDN then QSA with the PLE, two hyper-connection streams, one KV head
  config = TransformerConfig(num_blocks=2, dim=64, hidden_dim=32, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=64, head_dim=32,
    rope_theta=10000, rope_dim=16, v_head_dim=32, max_context=32, num_experts=4, num_experts_per_tok=2, shared_expert_dim=32,
    shared_expert_gate=True, attn_output_gate=True, ssm=SSMConfig(4, 32, 1, 2, 64, sigmoid_gate=True), ssm_layers=(True, False),
    hc=HCConfig(2, 0, 0.0, low_rank=8), indexer=IndexerConfig(2, 32, 8, 4), ple=PLEConfig(1, 32, MULS, VOCABS, (0, 13), EOS, 4))
  writer = GGUFWriter(path:=pathlib.Path(tempfile.mkdtemp())/"model.gguf", "qwen4exp")
  for k, v in {"block_count": 2, "context_length": 32, "embedding_length": 64, "attention.head_count": 2, "attention.head_count_kv": 1,
               "attention.key_length": 32, "attention.value_length": 32, "rope.dimension_count": 16, "expert_count": 4, "expert_used_count": 2,
               "expert_feed_forward_length": 32, "expert_shared_feed_forward_length": 32, "full_attention_interval": 2, "ssm.conv_kernel": 4,
               "ssm.state_size": 32, "ssm.group_count": 1, "ssm.time_step_rank": 2, "ssm.inner_size": 64, "hyper_connection.count": 2,
               "hyper_connection.low_rank": 8, "attention.indexer.head_count": 2, "attention.indexer.key_length": 32,
               "attention.indexer.top_k": 8, "ple.eos_token_id": EOS, "ple.conv_kernel": 4}.items(): writer.add_uint32(f"qwen4exp.{k}", v)
  writer.add_float32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-5)
  for k, v in {"attention.compress_ratios": (0, 4), "ple.layers": (1,), "ple.layer_multipliers": MULS, "ple.head_vocab_sizes": VOCABS,
               "ple.head_offsets": (0, 13)}.items(): writer.add_key_value(f"qwen4exp.{k}", list(v), GGUFValueType.ARRAY, GGUFValueType.UINT32)
  writer.add_array("tokenizer.ggml.tokens", [str(i) for i in range(64)])
  rng = np.random.default_rng(0)
  for name, weight in nn.state.get_state_dict(Transformer(config)).items():
    value = -rng.random(weight.shape) if name.endswith("ssm_a") else rng.normal(1, .05, weight.shape) if "norm" in name else \
            rng.normal(0, .5 if "token_embd" in name else .1, weight.shape)
    writer.add_tensor(re.sub(r"(hc_attn|hc_ffn|output_hc|\.ple)\.", r"\1_", name), value.astype(np.float16))
  table = rng.integers(0, 0x7a, (sum(VOCABS), 18), dtype=np.uint8)
  table[:, :2] = np.asarray([1e-2], np.float16).view(np.uint8)  # IQ4_NL block scales
  writer.add_tensor("per_layer_token_embd.weight", table, raw_dtype=Q.IQ4_NL)
  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_tensors_to_file()
  writer.close()
  return Transformer.from_gguf(path, max_context=max_context, shard=shard)[0]

class TestQwen4Exp(unittest.TestCase):
  def test_generate(self):
    model = load_tiny_qwen4exp()
    gen = model.generate(list(PROMPT), chunk_size=8)
    out = [next(gen) for _ in range(6)]
    # the host buffer holds every position's n-gram table rows: t0*m0 ^ t1*m1 for the bigram head, ^ t2*m2 for the trigram head,
    # predecessors before the start or an EOS are EOS
    toks, table, rows = PROMPT + out, model.ple_table.numpy().reshape(-1, 18), model.ple_rows.numpy()[0]
    for p in range(len(toks)):
      ctx = [EOS if p < i or EOS in toks[p-i:p] else toks[p-i] for i in range(3)]
      bi = ctx[0]*MULS[0] ^ ctx[1]*MULS[1]
      np.testing.assert_array_equal(rows[p], table[[bi % 13, 13 + (bi ^ ctx[2]*MULS[2]) % 17]])
    # chunked prefill and token by token decode agree
    gen = load_tiny_qwen4exp().generate(list(PROMPT), chunk_size=1)
    self.assertEqual(out, [next(gen) for _ in range(6)])

  def test_ple_rows_views(self):
    model = load_tiny_qwen4exp()
    model.ple_rows = Tensor.empty(1, len(PROMPT), len(VOCABS), 18, dtype="uint8", device="CPU").realize()
    model._write_ple_rows(PROMPT, 0)
    expected = model.ple_rows.numpy()
    # Both the table and destination may be views into larger host buffers (e.g. a Tensor-backed GGUF).
    model.ple_table = model.ple_table.pad((7, 11)).contiguous().realize()[7:-11]
    rows = Tensor.full(model.ple_rows.numel()+18, 255, dtype="uint8", device="CPU").contiguous().realize()
    model.ple_rows = rows[9:-9].reshape(model.ple_rows.shape)
    model._write_ple_rows(PROMPT, 0)
    np.testing.assert_array_equal(model.ple_rows.numpy(), expected)
    np.testing.assert_array_equal(rows.numpy()[:9], 255)
    np.testing.assert_array_equal(rows.numpy()[-9:], 255)

  def test_partial_indexer_pool(self):
    prompt = (PROMPT * 3)[:29]
    out = list(load_tiny_qwen4exp(max_context=31).generate(list(prompt), chunk_size=8))
    self.assertEqual(len(out), 2)
    gen = load_tiny_qwen4exp().generate(list(prompt), chunk_size=8)
    self.assertEqual(out, [next(gen) for _ in out])

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_shard_two(self):
    # the KV head is repeated so each device holds one
    single, parallel = (load_tiny_qwen4exp(shard).generate(list(PROMPT), chunk_size=8) for shard in (1, 2))
    self.assertEqual([next(single) for _ in range(4)], [next(parallel) for _ in range(4)])

if __name__ == "__main__": unittest.main()
