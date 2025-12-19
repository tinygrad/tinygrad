import math
from pathlib import Path

import numpy as np
import pytest

from tinygrad import Tensor
from examples.mlperf.model_train import train_flux_text_to_image_tiny
from examples.mlperf.model_eval import eval_flux_text_to_image_tiny


def _write_shard(path:Path, latent:np.ndarray, text:np.ndarray, bucket:np.ndarray|None=None):
  path.parent.mkdir(parents=True, exist_ok=True)
  payload:dict[str, np.ndarray] = {"latent": latent, "text": text}
  if bucket is not None:
    payload["timestep_bucket"] = bucket
  np.savez(path, **payload)


def _prepare_shards(base:Path, rng:np.random.Generator, bucket_count:int):
  train_latent = rng.standard_normal((12, 4, 4, 4), dtype=np.float32)
  train_text = rng.standard_normal((12, 3, 16), dtype=np.float32)
  eval_latent = rng.standard_normal((bucket_count, 4, 4, 4), dtype=np.float32)
  eval_text = rng.standard_normal((bucket_count, 3, 16), dtype=np.float32)
  buckets = np.arange(bucket_count, dtype=np.int64)
  _write_shard(base / "train" / "shard0.npz", train_latent, train_text)
  _write_shard(base / "eval" / "shard0.npz", eval_latent, eval_text, buckets)


def _configure_flux_env(monkeypatch:pytest.MonkeyPatch, base:Path, bucket_count:int, dataset_size:int):
  monkeypatch.setenv("FLUX_TRAIN_DIR", str(base / "train"))
  monkeypatch.setenv("FLUX_EVAL_DIR", str(base / "eval"))
  monkeypatch.setenv("FLUX_DATASET_SIZE", str(dataset_size))
  monkeypatch.setenv("FLUX_EVAL_DATASET_SIZE", str(bucket_count))
  monkeypatch.setenv("FLUX_BUCKET_COUNT", str(bucket_count))
  monkeypatch.setenv("FLUX_STEPS", "1")
  monkeypatch.setenv("FLUX_LOG_INTERVAL", "1")
  monkeypatch.setenv("FLUX_LATENT_CHANNELS", "4")
  monkeypatch.setenv("FLUX_TEXT_DIM", "16")
  monkeypatch.setenv("BS", "4")
  monkeypatch.setenv("EVAL_BS", "4")
  monkeypatch.setenv("SEED", "0")
  monkeypatch.setenv("FLUX_EVAL_NOISE_SEED", "0")


def test_flux_text_to_image_pipeline(tmp_path:Path, monkeypatch:pytest.MonkeyPatch):
  rng = np.random.default_rng(0)
  base_primary = tmp_path / "primary"
  _prepare_shards(base_primary, rng, bucket_count=8)
  _configure_flux_env(monkeypatch, base_primary, bucket_count=8, dataset_size=8)

  Tensor.manual_seed(0)
  loss = train_flux_text_to_image_tiny()
  assert loss is not None and math.isfinite(loss)

  val_loss, bucket_means = eval_flux_text_to_image_tiny()
  assert math.isfinite(val_loss)
  assert len(bucket_means) == 8
  for bucket_val in bucket_means:
    assert math.isfinite(bucket_val)

  base_secondary = tmp_path / "secondary"
  _prepare_shards(base_secondary, np.random.default_rng(1), bucket_count=4)
  _configure_flux_env(monkeypatch, base_secondary, bucket_count=4, dataset_size=4)
  _, bucket_override = eval_flux_text_to_image_tiny()
  assert len(bucket_override) == 4
