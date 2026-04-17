import json, math, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

os.environ["WQKV"] = "1"

from tinygrad import Tensor, nn
from tinygrad.helpers import getenv as cached_getenv
from tinygrad.nn.state import safe_save

import examples.mlperf.llama as llama_helpers
from examples.mlperf.dataloader import iterate_llama2_70b_lora_dataset
from examples.mlperf.model_eval import eval_llama2_70b_lora
from examples.mlperf.models.flat_llama import FlatTransformer
from extra.models.llama import Transformer


class FakeTokenizer:
  def bos_id(self): return 101
  def eos_id(self): return 102
  def encode(self, text:str): return list(range(1, len(text.split()) + 1))


def split_attention_state(ref:Transformer):
  state = nn.state.get_state_dict(ref)
  split_state = {
    "tok_embeddings.weight": state["tok_embeddings.weight"],
    "norm.weight": state["norm.weight"],
    "output.weight": state["output.weight"],
  }
  for i, layer in enumerate(ref.layers):
    attn = layer.attention
    q_dim = attn.n_heads * attn.head_dim
    kv_dim = attn.n_kv_heads * attn.head_dim
    wq, wk, wv = state[f"layers.{i}.attention.wqkv.weight"].split([q_dim, kv_dim, kv_dim], dim=0)
    split_state[f"layers.{i}.attention.wq.weight"] = wq
    split_state[f"layers.{i}.attention.wk.weight"] = wk
    split_state[f"layers.{i}.attention.wv.weight"] = wv
    split_state[f"layers.{i}.attention.wo.weight"] = state[f"layers.{i}.attention.wo.weight"]
    split_state[f"layers.{i}.feed_forward.w1.weight"] = state[f"layers.{i}.feed_forward.w1.weight"]
    split_state[f"layers.{i}.feed_forward.w2.weight"] = state[f"layers.{i}.feed_forward.w2.weight"]
    split_state[f"layers.{i}.feed_forward.w3.weight"] = state[f"layers.{i}.feed_forward.w3.weight"]
    split_state[f"layers.{i}.attention_norm.weight"] = state[f"layers.{i}.attention_norm.weight"]
    split_state[f"layers.{i}.ffn_norm.weight"] = state[f"layers.{i}.ffn_norm.weight"]
  return split_state


class TestMLPerfLlama2LoRAEval(unittest.TestCase):
  def setUp(self):
    cached_getenv.cache_clear()

  def tearDown(self):
    cached_getenv.cache_clear()

  def test_llama2_70b_lora_encode_masks_prompt_tokens(self):
    tokenizer = FakeTokenizer()
    input_ids, labels = llama_helpers.llama2_70b_lora_encode_sample(tokenizer, "source text", "target summary")
    prompt_len = 1 + len(tokenizer.encode(llama_helpers.llama2_70b_lora_prompt("source text")))
    self.assertEqual(labels[:prompt_len], [-1] * prompt_len)
    self.assertEqual(labels[-1], -1)
    self.assertEqual(input_ids[-1], tokenizer.eos_id())
    self.assertTrue(all(x != -1 for x in labels[prompt_len:-1]))

  def test_iterate_llama2_70b_lora_dataset_packs_token_stream(self):
    with tempfile.TemporaryDirectory(prefix="llama2-lora-dataset-") as tmpdir:
      dataset_path = Path(tmpdir) / "validation.jsonl"
      rows = [
        {"input_ids": [1, 2, 3], "labels": [-1, 2, -1]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, -1]},
      ]
      dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

      batches = list(iterate_llama2_70b_lora_dataset(dataset_path, bs=1, seqlen=4))

    self.assertEqual(len(batches), 1)
    tokens, labels = batches[0]
    self.assertListEqual(tokens.numpy().tolist(), [[1, 2, 3, 4]])
    self.assertListEqual(labels.numpy().tolist(), [[-1, 2, -1, 4]])

  def test_iterate_llama2_70b_lora_dataset_respects_split_semantics_and_sample_caps(self):
    with tempfile.TemporaryDirectory(prefix="llama2-lora-dataset-") as tmpdir:
      train_path = Path(tmpdir) / "train.jsonl"
      train_path.write_text(json.dumps({"input": "source text", "output": "target summary"}) + "\n")
      seqlen = len(llama_helpers.llama2_70b_lora_encode_sample(FakeTokenizer(), "source text", "target summary", mask_prompt_labels=False)[0])

      train_tokens, train_labels = next(iter(iterate_llama2_70b_lora_dataset(train_path, bs=1, seqlen=seqlen, tokenizer=FakeTokenizer(), val=False, samples=1)))
      self.assertNotIn(-1, train_labels.numpy().tolist()[0])
      self.assertEqual(train_tokens.shape[0], 1)

      val_tokens, val_labels = next(iter(iterate_llama2_70b_lora_dataset(train_path, bs=1, seqlen=seqlen, tokenizer=FakeTokenizer(), val=True, samples=1)))
      self.assertIn(-1, val_labels.numpy().tolist()[0])
      self.assertEqual(val_tokens.shape[0], 1)

  def test_eval_llama2_70b_lora_loads_base_and_adapter(self):
    Tensor.manual_seed(42)
    tiny_params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000)
    spec = llama_helpers.llama_benchmark_config("llama2_70b_lora")
    spec = {**spec, "model_params": tiny_params, "real_vocab_size": tiny_params["vocab_size"], "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}

    with tempfile.TemporaryDirectory(prefix="llama2-lora-eval-") as tmpdir:
      tmpdir = Path(tmpdir)
      base_model = Transformer(**tiny_params, disable_kv_cache=True)
      base_path = tmpdir / "base.safetensors"
      safe_save(split_attention_state(base_model), base_path.as_posix())

      adapter_model = FlatTransformer(**tiny_params, max_context=8, lora_rank=4, lora_alpha=8, lora_dropout=0.0)
      adapter_model.wqkv_lora_b.assign(Tensor.ones(*adapter_model.wqkv_lora_b.shape, dtype=adapter_model.wqkv_lora_b.dtype))
      adapter_model.wo_lora_b.assign(Tensor.ones(*adapter_model.wo_lora_b.shape, dtype=adapter_model.wo_lora_b.dtype))
      adapter_path = tmpdir / "adapter.safetensors"
      safe_save(adapter_model.adapter_state_dict(), adapter_path.as_posix())

      dataset_path = tmpdir / "validation.jsonl"
      dataset_path.write_text(json.dumps({"input_ids": [1, 2, 3, 4, 5], "labels": [-1, -1, 3, 4, -1]}) + "\n")

      env = {
        "BS": "1",
        "SEQLEN": "5",
        "DATASET_PATH": dataset_path.as_posix(),
        "MODEL_PATH": base_path.as_posix(),
        "ADAPTER_CKPT": adapter_path.as_posix(),
        "LLAMA_LORA_RANK": "4",
        "LLAMA_LORA_ALPHA": "8",
        "LLAMA_LORA_DROPOUT": "0",
      }
      with patch.dict(os.environ, env, clear=False), patch.object(llama_helpers, "llama_benchmark_config", side_effect=lambda model_name, small=False: spec):
        eval_loss = eval_llama2_70b_lora()

    self.assertTrue(math.isfinite(eval_loss))


if __name__ == "__main__":
  unittest.main()
