import json, os
os.environ["WQKV"] = "1"

import sys, tempfile, types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv as cached_getenv
from tinygrad.nn.state import get_state_dict, safe_load, safe_save

if "tqdm" not in sys.modules:
  class _TqdmStub:
    def __call__(self, iterable=None, *args, **kwargs): return iterable if iterable is not None else []
    @staticmethod
    def write(*args, **kwargs): pass
  sys.modules["tqdm"] = types.SimpleNamespace(tqdm=_TqdmStub())

from examples.mlperf.llama import llama_benchmark_config
from examples.mlperf.model_train import (
  _llama_checkpoint_path,
  _llama_checkpoint_state,
  _llama_configure_trainable_params,
  _llama_load_model_checkpoint,
  _llama_load_model_state_dict,
  _llama_load_training_checkpoint,
  _llama_sequences_seen,
  _llama_train_microbatches_seen,
  _llama_training_checkpoint_state,
  train_llama2_70b_lora,
)
from examples.mlperf.models.flat_llama import FlatTransformer
from extra.models.llama import Transformer


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


class TestLlamaLoRATrainWiring(unittest.TestCase):
  def setUp(self):
    cached_getenv.cache_clear()
    Tensor.manual_seed(42)
    self.params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)

  def tearDown(self):
    cached_getenv.cache_clear()

  def test_adapter_only_trainables(self):
    model = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    trainable_params, trainable_names = _llama_configure_trainable_params(model, adapter_only=True)

    self.assertEqual(len(trainable_params), 4)
    self.assertSetEqual(trainable_names, {"wqkv_lora_a", "wqkv_lora_b", "wo_lora_a", "wo_lora_b"})
    self.assertSetEqual({name for name, tensor in get_state_dict(model).items() if tensor.requires_grad}, trainable_names)

  def test_adapter_only_trainables_with_int8_base(self):
    model = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0, base_quantize="int8")
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

  def test_adapter_checkpoint_saves_float32_and_loads_target_dtype(self):
    source = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    target = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    for name, tensor in source.adapter_state_dict().items():
      setattr(source, name, tensor.cast(dtypes.bfloat16).realize())
    for name, tensor in target.adapter_state_dict().items():
      setattr(target, name, tensor.cast(dtypes.bfloat16).realize())
    source.shard(("CPU:0", "CPU:1"), mp=True)
    target.shard(("CPU:0", "CPU:1"), mp=True)

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt = Path(tmpdir) / "llama2_70b_lora_1.safe"
      safe_save(_llama_checkpoint_state(source, adapter_only=True), ckpt.as_posix())
      saved_state = safe_load(ckpt)
      self.assertSetEqual({tensor.dtype for tensor in saved_state.values()}, {dtypes.float32})
      _llama_load_model_checkpoint(target, ckpt.as_posix(), strict=False)

    for name, tensor in target.adapter_state_dict().items():
      self.assertEqual(tensor.dtype, dtypes.bfloat16, name)
      self.assertIsInstance(tensor.device, tuple, name)

  def test_adapter_checkpoint_state_fallback_preserves_target_dtype(self):
    source = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    target = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    for name, tensor in source.adapter_state_dict().items():
      setattr(source, name, tensor.cast(dtypes.bfloat16).realize())
    for name, tensor in target.adapter_state_dict().items():
      setattr(target, name, tensor.cast(dtypes.bfloat16).realize())

    state_dict = {f"model.{k}": v for k, v in _llama_checkpoint_state(source, adapter_only=True).items()}
    _llama_load_model_state_dict(target, state_dict, strict=False, verbose=False)

    for name, tensor in target.adapter_state_dict().items():
      self.assertEqual(tensor.dtype, dtypes.bfloat16, name)

  def test_adapter_training_checkpoint_saves_float32_and_loads_target_dtype(self):
    from examples.mlperf.lr_schedulers import CosineAnnealingLRWithWarmup
    from examples.mlperf.optim import GradAccClipAdamW

    source = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    target = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    for model in [source, target]:
      for name, tensor in model.adapter_state_dict().items():
        setattr(model, name, tensor.cast(dtypes.bfloat16).realize())
      model.shard(("CPU:0", "CPU:1"), mp=True)

    source_optim_params, _ = _llama_configure_trainable_params(source, adapter_only=True)
    source_optim = GradAccClipAdamW(source_optim_params, lr=1e-3, grad_acc=1, clip_norm=0.3)
    source_scheduler = CosineAnnealingLRWithWarmup(source_optim, 1e-3, 0.0, 0, 4)
    target_optim_params, _ = _llama_configure_trainable_params(target, adapter_only=True)
    target_optim = GradAccClipAdamW(target_optim_params, lr=1e-3, grad_acc=1, clip_norm=0.3)
    target_scheduler = CosineAnnealingLRWithWarmup(target_optim, 1e-3, 0.0, 0, 4)

    state_dict = _llama_training_checkpoint_state(source, source_optim, source_scheduler, adapter_only=True)
    self.assertSetEqual({tensor.dtype for name, tensor in state_dict.items() if name.startswith("model.")}, {dtypes.float32})
    self.assertFalse(any(k.startswith("optimizer.params.") for k in state_dict))
    _llama_load_training_checkpoint(target, target_optim, target_scheduler, state_dict, adapter_only=True)

    for name, tensor in target.adapter_state_dict().items():
      self.assertEqual(tensor.dtype, dtypes.bfloat16, name)
      self.assertIsInstance(tensor.device, tuple, name)

  def test_stochastic_round_uses_shared_noise_for_replicated_mp_tensors(self):
    from examples.mlperf.optim import stochastic_round_bf16

    Tensor.manual_seed(123)
    x = Tensor.full((32,), 1.001, dtype=dtypes.float32).shard(("CPU:0", "CPU:1"), axis=None)
    rounded = stochastic_round_bf16(x).realize()

    replica0 = Tensor(rounded.uop.mselect(0)).numpy()
    replica1 = Tensor(rounded.uop.mselect(1)).numpy()
    np.testing.assert_array_equal(replica0, replica1)

  def test_resume_sequence_count(self):
    self.assertEqual(_llama_sequences_seen(0, bs=3, grad_acc=4), 0)
    self.assertEqual(_llama_sequences_seen(1, bs=3, grad_acc=4), 3)
    self.assertEqual(_llama_sequences_seen(2, bs=3, grad_acc=4), 6)
    self.assertEqual(_llama_sequences_seen(5, bs=3, grad_acc=4), 42)
    self.assertEqual(_llama_train_microbatches_seen(0, grad_acc=4), 0)
    self.assertEqual(_llama_train_microbatches_seen(1, grad_acc=4), 1)
    self.assertEqual(_llama_train_microbatches_seen(2, grad_acc=4), 2)
    self.assertEqual(_llama_train_microbatches_seen(5, grad_acc=4), 14)

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
    self.assertEqual(_llama_checkpoint_path("17", "llama2_70b_lora", "_state.safe", ckpt_dir="/tmp/llama"),
                     Path("/tmp/llama/llama2_70b_lora_17_state.safe"))

  def test_llama2_70b_lora_training_writes_adapter_only_checkpoints(self):
    tiny_params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000)
    spec = llama_benchmark_config("llama2_70b_lora")
    spec = {**spec, "model_params": tiny_params, "real_vocab_size": tiny_params["vocab_size"], "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}

    with tempfile.TemporaryDirectory(prefix="llama2-lora-train-") as tmpdir:
      tmpdir = Path(tmpdir)
      dataset_dir = tmpdir / "dataset"
      dataset_dir.mkdir()
      (dataset_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8, 9, 10], "labels": [6, 7, 8, 9, 10]},
      ]))
      (dataset_dir / "validation.jsonl").write_text(json.dumps({"input_ids": [1, 2, 3, 4, 5], "labels": [-1, -1, 3, 4, -1]}) + "\n")

      base_model = Transformer(**tiny_params, disable_kv_cache=True)
      base_path = tmpdir / "base.safetensors"
      safe_save(split_attention_state(base_model), base_path.as_posix())

      env = {
        "BS": "1",
        "CKPT": "1",
        "DATASET_PATH": dataset_dir.as_posix(),
        "END_LR": "0.0",
        "EVAL_BS": "1",
        "EVAL_FREQ": "1",
        "EVAL_TARGET": "0.0",
        "GRADIENT_ACC_STEPS": "1",
        "LLAMA_LORA_ALPHA": "8",
        "LLAMA_LORA_DROPOUT": "0",
        "LLAMA_LORA_RANK": "4",
        "LR": "1e-3",
        "MAX_STEPS": "1",
        "MODEL_PATH": base_path.as_posix(),
        "SAVE_CKPT_DIR": (tmpdir / "saved").as_posix(),
        "SEQLEN": "5",
        "WARMUP_STEPS": "0",
      }
      original_config = llama_benchmark_config
      with patch.dict(os.environ, env, clear=False), patch("examples.mlperf.llama.llama_benchmark_config",
           side_effect=lambda model_name, small=False: spec if model_name == "llama2_70b_lora" else original_config(model_name, small=small)):
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
          train_llama2_70b_lora()
        finally:
          os.chdir(cwd)

      model_ckpt = safe_load(tmpdir / "saved" / "llama2_70b_lora_1.safe")
      trainer_ckpt = safe_load(tmpdir / "saved" / "llama2_70b_lora_1_state.safe")
      self.assertSetEqual(set(model_ckpt.keys()), {"wqkv_lora_a", "wqkv_lora_b", "wo_lora_a", "wo_lora_b"})
      self.assertNotIn("model.wqkv", trainer_ckpt)
      self.assertIn("rng.seed", trainer_ckpt)
      self.assertFalse(any(k.startswith("optimizer.params.") for k in trainer_ckpt))
      self.assertTrue(any(k.startswith("optimizer.") for k in trainer_ckpt))
      self.assertTrue(any(k.startswith("rng.device.") for k in trainer_ckpt))
      self.assertTrue(any(k.startswith("scheduler.") for k in trainer_ckpt))
      self.assertSetEqual({k for k in trainer_ckpt if k.startswith("model.")},
                          {"model.wqkv_lora_a", "model.wqkv_lora_b", "model.wo_lora_a", "model.wo_lora_b"})

  def test_llama2_70b_lora_training_writes_adapter_only_checkpoints_with_int8_base(self):
    tiny_params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000)
    spec = llama_benchmark_config("llama2_70b_lora")
    spec = {**spec, "model_params": tiny_params, "real_vocab_size": tiny_params["vocab_size"], "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}

    with tempfile.TemporaryDirectory(prefix="llama2-lora-train-int8-") as tmpdir:
      tmpdir = Path(tmpdir)
      dataset_dir = tmpdir / "dataset"
      dataset_dir.mkdir()
      (dataset_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8, 9, 10], "labels": [6, 7, 8, 9, 10]},
      ]))
      (dataset_dir / "validation.jsonl").write_text(json.dumps({"input_ids": [1, 2, 3, 4, 5], "labels": [-1, -1, 3, 4, -1]}) + "\n")

      base_model = Transformer(**tiny_params, disable_kv_cache=True)
      base_path = tmpdir / "base.safetensors"
      safe_save(split_attention_state(base_model), base_path.as_posix())

      env = {
        "BS": "1",
        "CKPT": "1",
        "DATASET_PATH": dataset_dir.as_posix(),
        "END_LR": "0.0",
        "EVAL_BS": "1",
        "EVAL_FREQ": "1",
        "EVAL_TARGET": "0.0",
        "GRADIENT_ACC_STEPS": "1",
        "LLAMA_BASE_QUANTIZE": "int8",
        "LLAMA_LORA_ALPHA": "8",
        "LLAMA_LORA_DROPOUT": "0",
        "LLAMA_LORA_RANK": "4",
        "LR": "1e-3",
        "MAX_STEPS": "1",
        "MODEL_PATH": base_path.as_posix(),
        "SEQLEN": "5",
        "WARMUP_STEPS": "0",
      }
      original_config = llama_benchmark_config
      with patch.dict(os.environ, env, clear=False), patch("examples.mlperf.llama.llama_benchmark_config",
           side_effect=lambda model_name, small=False: spec if model_name == "llama2_70b_lora" else original_config(model_name, small=small)):
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
          train_llama2_70b_lora()
        finally:
          os.chdir(cwd)

      model_ckpt = safe_load(tmpdir / "ckpts" / "llama2_70b_lora_1.safe")
      trainer_ckpt = safe_load(tmpdir / "ckpts" / "llama2_70b_lora_1_state.safe")
      self.assertSetEqual(set(model_ckpt.keys()), {"wqkv_lora_a", "wqkv_lora_b", "wo_lora_a", "wo_lora_b"})
      self.assertNotIn("model.wqkv", trainer_ckpt)
      self.assertIn("rng.seed", trainer_ckpt)
      self.assertFalse(any(k.startswith("optimizer.params.") for k in trainer_ckpt))

  def test_adapter_only_training_checkpoint_restores_rng_state(self):
    from examples.mlperf.lr_schedulers import CosineAnnealingLRWithWarmup
    from examples.mlperf.optim import GradAccClipAdamW

    model = FlatTransformer(**self.params, lora_rank=8, lora_alpha=16, lora_dropout=0.1)
    optim_params, _ = _llama_configure_trainable_params(model, adapter_only=True)
    optim = GradAccClipAdamW(optim_params, lr=1e-3, grad_acc=1, clip_norm=0.3)
    for p in optim.params:
      p.grad = Tensor.zeros(p.shape, dtype=p.dtype, device=p.device).contiguous()
    scheduler = CosineAnnealingLRWithWarmup(optim, 1e-3, 0.0, 0, 4)

    Tensor.manual_seed(1234)
    state_dict = _llama_training_checkpoint_state(model, optim, scheduler, adapter_only=True)
    expected = Tensor.rand(4).numpy().tolist()
    Tensor.rand(32).realize()

    _llama_load_training_checkpoint(model, optim, scheduler, state_dict, adapter_only=True)
    restored = Tensor.rand(4).numpy().tolist()
    self.assertListEqual(restored, expected)

  def test_llama2_70b_lora_resume_matches_uninterrupted_training(self):
    tiny_params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000)
    spec = llama_benchmark_config("llama2_70b_lora")
    spec = {**spec, "model_params": tiny_params, "real_vocab_size": tiny_params["vocab_size"], "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.1}

    with tempfile.TemporaryDirectory(prefix="llama2-lora-resume-") as tmpdir:
      tmpdir = Path(tmpdir)
      dataset_dir = tmpdir / "dataset"
      dataset_dir.mkdir()
      rows = [
        {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8, 9, 10], "labels": [6, 7, 8, 9, 10]},
        {"input_ids": [11, 12, 13, 14, 15], "labels": [11, 12, 13, 14, 15]},
      ]
      (dataset_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
      (dataset_dir / "validation.jsonl").write_text(json.dumps({"input_ids": [1, 2, 3, 4, 5], "labels": [-1, -1, 3, 4, -1]}) + "\n")

      base_model = Transformer(**tiny_params, disable_kv_cache=True)
      base_path = tmpdir / "base.safetensors"
      safe_save(split_attention_state(base_model), base_path.as_posix())

      base_env = {
        "BS": "1",
        "CKPT": "1",
        "DATASET_PATH": dataset_dir.as_posix(),
        "END_LR": "0.0",
        "EVAL_FREQ": "999",
        "EVAL_TARGET": "-1.0",
        "GRADIENT_ACC_STEPS": "1",
        "LLAMA_LORA_ALPHA": "8",
        "LLAMA_LORA_DROPOUT": "0.1",
        "LLAMA_LORA_RANK": "4",
        "LR": "1e-3",
        "MODEL_PATH": base_path.as_posix(),
        "SEQLEN": "5",
        "WARMUP_STEPS": "0",
      }

      original_config = llama_benchmark_config

      baseline_dir = tmpdir / "baseline"
      baseline_dir.mkdir()
      with (
        patch.dict(os.environ, base_env | {"MAX_STEPS": "2"}, clear=False),
        patch("examples.mlperf.llama.llama_benchmark_config",
              side_effect=lambda model_name, small=False: spec if model_name == "llama2_70b_lora" else original_config(model_name, small=small)),
      ):
        cached_getenv.cache_clear()
        cwd = os.getcwd()
        os.chdir(baseline_dir)
        try:
          train_llama2_70b_lora()
        finally:
          os.chdir(cwd)

      resume_dir = tmpdir / "resume"
      resume_dir.mkdir()
      resume_ckpt_dir = resume_dir / "saved"
      with (
        patch.dict(os.environ, base_env | {"MAX_STEPS": "2", "SAMPLES": "1", "SAVE_CKPT_DIR": resume_ckpt_dir.as_posix()}, clear=False),
        patch("examples.mlperf.llama.llama_benchmark_config",
              side_effect=lambda model_name, small=False: spec if model_name == "llama2_70b_lora" else original_config(model_name, small=small)),
      ):
        cached_getenv.cache_clear()
        cwd = os.getcwd()
        os.chdir(resume_dir)
        try:
          train_llama2_70b_lora()
        finally:
          os.chdir(cwd)

      with (
        patch.dict(os.environ, base_env | {"MAX_STEPS": "2", "RESUME_CKPT": "1", "SAVE_CKPT_DIR": resume_ckpt_dir.as_posix()}, clear=False),
        patch("examples.mlperf.llama.llama_benchmark_config",
              side_effect=lambda model_name, small=False: spec if model_name == "llama2_70b_lora" else original_config(model_name, small=small)),
      ):
        cached_getenv.cache_clear()
        cwd = os.getcwd()
        os.chdir(resume_dir)
        try:
          train_llama2_70b_lora()
        finally:
          os.chdir(cwd)

      baseline_ckpt = safe_load(baseline_dir / "ckpts" / "llama2_70b_lora_2.safe")
      resumed_ckpt = safe_load(resume_ckpt_dir / "llama2_70b_lora_2.safe")
      self.assertSetEqual(set(baseline_ckpt.keys()), set(resumed_ckpt.keys()))
      for key in baseline_ckpt:
        diff = np.abs(baseline_ckpt[key].numpy() - resumed_ckpt[key].numpy()).max()
        self.assertLess(float(diff), 1e-8, key)


if __name__ == "__main__":
  unittest.main()
