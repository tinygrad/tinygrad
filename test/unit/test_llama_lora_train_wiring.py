import os
os.environ["WQKV"] = "1"

import sys, tempfile, types
import unittest
from pathlib import Path
from unittest.mock import patch

from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict, safe_save

if "tqdm" not in sys.modules:
  class _TqdmStub:
    def __call__(self, iterable=None, *args, **kwargs): return iterable if iterable is not None else []
    @staticmethod
    def write(*args, **kwargs): pass
  sys.modules["tqdm"] = types.SimpleNamespace(tqdm=_TqdmStub())

from examples.mlperf.llama import llama_benchmark_config
from examples.mlperf.model_train import _llama_checkpoint_path, _llama_configure_trainable_params, _llama_load_model_checkpoint, _llama_sequences_seen
from examples.mlperf.models.flat_llama import FlatTransformer


class TestLlamaLoRATrainWiring(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(42)
    self.params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)

  def test_adapter_only_trainables(self):
    model = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    trainable_params, trainable_names = _llama_configure_trainable_params(model, adapter_only=True)

    self.assertEqual(len(trainable_params), 4)
    self.assertSetEqual(trainable_names, {"wqkv_lora_a", "wqkv_lora_b", "wo_lora_a", "wo_lora_b"})
    self.assertSetEqual({name for name, tensor in get_state_dict(model).items() if tensor.requires_grad}, trainable_names)

  def test_load_model_checkpoint_with_model_prefix(self):
    base = FlatTransformer(**self.params)
    lora = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    Tensor.manual_seed(123)
    base.wqkv.assign(Tensor.rand(*base.wqkv.shape, dtype=base.wqkv.dtype))

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = Path(tmpdir) / "llama.safe"
      safe_save({f"model.{k}": v for k, v in get_state_dict(base).items()}, ckpt.as_posix())
      _llama_load_model_checkpoint(lora, ckpt.as_posix(), strict=False)

    diff = (lora.wqkv - base.wqkv).abs().max().item()
    self.assertLess(diff, 1e-8)

  def test_resume_sequence_count(self):
    self.assertEqual(_llama_sequences_seen(0, bs=3, grad_acc=4), 0)
    self.assertEqual(_llama_sequences_seen(1, bs=3, grad_acc=4), 3)
    self.assertEqual(_llama_sequences_seen(2, bs=3, grad_acc=4), 6)
    self.assertEqual(_llama_sequences_seen(5, bs=3, grad_acc=4), 42)

  def test_llama3_benchmark_config_uses_llama31_identity(self):
    with patch.dict(os.environ, {"LLAMA3_SIZE": "8B"}, clear=False):
      spec = llama_benchmark_config("llama3", small=False)
    self.assertEqual(spec["submission_benchmark"], "llama3.1_8b")
    self.assertEqual(spec["checkpoint_prefix"], "llama3")
    self.assertEqual(spec["result_prefix"], "llama31")
    self.assertEqual(spec["real_vocab_size"], 32000)
    self.assertEqual(spec["model_params"]["vocab_size"], 32000)

  def test_llama2_benchmark_config_uses_lora_identity(self):
    spec = llama_benchmark_config("llama2_70b_lora", small=False)
    self.assertEqual(spec["submission_benchmark"], "llama2_70b_lora")
    self.assertEqual(spec["checkpoint_prefix"], "llama2_70b_lora")
    self.assertEqual(spec["result_prefix"], "llama2_70b_lora")
    self.assertEqual(spec["real_vocab_size"], 32000)
    self.assertEqual(spec["model_params"]["dim"], 8192)
    self.assertEqual(spec["model_params"]["n_kv_heads"], 8)
    self.assertEqual(spec["model_params"]["n_layers"], 80)
    self.assertEqual(spec["model_params"]["hidden_dim"], 28672)
    self.assertEqual(spec["model_params"]["rope_theta"], 10000)

  def test_checkpoint_path_uses_benchmark_prefix(self):
    self.assertEqual(_llama_checkpoint_path("17", "llama3"), Path("./ckpts/llama3_17.safe"))
    self.assertEqual(_llama_checkpoint_path("17", "llama2_70b_lora", "_state.safe"), Path("./ckpts/llama2_70b_lora_17_state.safe"))


if __name__ == "__main__":
  unittest.main()
