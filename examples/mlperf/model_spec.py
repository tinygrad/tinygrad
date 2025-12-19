# load each model here, quick benchmark
import os
from typing import Optional
from tinygrad import Tensor, GlobalCounters
from tinygrad.helpers import getenv
import numpy as np

def test_model(model, *inputs):
  GlobalCounters.reset()
  out = model(*inputs)
  if isinstance(out, Tensor): out = out.numpy()
  # TODO: return event future to still get the time_sum_s without DEBUG=2
  print(f"{GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.time_sum_s*1000:.2f} ms")

def spec_resnet():
  # Resnet50-v1.5
  from extra.models.resnet import ResNet50
  mdl = ResNet50()
  img = Tensor.randn(1, 3, 224, 224)
  test_model(mdl, img)

def spec_retinanet():
  # Retinanet with ResNet backbone
  from extra.models.resnet import ResNet50
  from extra.models.retinanet import RetinaNet
  mdl = RetinaNet(ResNet50(), num_classes=91, num_anchors=9)
  img = Tensor.randn(1, 3, 224, 224)
  test_model(mdl, img)

def spec_unet3d():
  # 3D UNET
  from extra.models.unet3d import UNet3D
  mdl = UNet3D()
  #mdl.load_from_pretrained()
  img = Tensor.randn(1, 1, 128, 128, 128)
  test_model(mdl, img)

def spec_rnnt():
  from extra.models.rnnt import RNNT
  mdl = RNNT()
  #mdl.load_from_pretrained()
  x = Tensor.randn(220, 1, 240)
  y = Tensor.randn(1, 220)
  test_model(mdl, x, y)

def spec_bert():
  from extra.models.bert import BertForQuestionAnswering
  mdl = BertForQuestionAnswering()
  #mdl.load_from_pretrained()
  x = Tensor.randn(1, 384)
  am = Tensor.randn(1, 384)
  tt = Tensor(np.random.randint(0, 2, (1, 384)).astype(np.float32))
  test_model(mdl, x, am, tt)

def spec_mrcnn():
  from extra.models.mask_rcnn import MaskRCNN, ResNet
  mdl = MaskRCNN(ResNet(50, num_classes=None, stride_in_1x1=True))
  #mdl.load_from_pretrained()
  x = Tensor.randn(3, 224, 224)
  test_model(mdl, [x])

class _SampleLimitedLoader:
  def __init__(self, loader, sample_limit:int):
    self.loader = loader
    self.sample_limit = sample_limit

  def __iter__(self):
    emitted = 0
    for latents, text, bucket in self.loader:
      if emitted >= self.sample_limit:
        break
      take = min(latents.shape[0], self.sample_limit - emitted)
      yield latents[:take], text[:take], None if bucket is None else bucket[:take]
      emitted += take
      if emitted >= self.sample_limit:
        break

def _parse_limit(value:str|int|None) -> Optional[int]:
  if value in (None, "", 0, "0"):
    return None
  return int(value)

def _limit_loader(loader, sample_limit:Optional[int]):
  if sample_limit is None:
    return loader
  return _SampleLimitedLoader(loader, sample_limit)

def _env_int(name:str, default:int) -> int:
  return int(os.environ.get(name, default))

def flux_text_to_image_tiny_entry():
  from models.flux_schnell_tiny import FluxSchnellTiny
  from examples.mlperf.dataloader_text_to_image import create_text_to_image_eval_loader, create_text_to_image_train_loader

  train_dir = os.environ.get("FLUX_TRAIN_DIR", os.environ.get("FLUX_DATA_DIR", "data/flux_text_to_image/train"))
  eval_dir = os.environ.get("FLUX_EVAL_DIR", train_dir)
  bucket_count = _env_int("FLUX_BUCKET_COUNT", 8)
  latent_channels = _env_int("FLUX_LATENT_CHANNELS", 4)
  text_dim = _env_int("FLUX_TEXT_DIM", 16)
  train_limit = _parse_limit(os.environ.get("FLUX_DATASET_SIZE", ""))
  eval_limit = _parse_limit(os.environ.get("FLUX_EVAL_DATASET_SIZE", ""))

  def model_ctor():
    return FluxSchnellTiny(latent_channels=latent_channels, text_embedding_dim=text_dim)

  def build_train_loader(*, batch_size:int, data_dir:str|None=None, seed:int|None=None):
    loader = create_text_to_image_train_loader(
      data_dir=data_dir or train_dir,
      batch_size=batch_size,
      seed=seed if seed is not None else int(getenv("SEED", 0)),
    )
    return _limit_loader(loader, train_limit)

  def build_eval_loader(*, batch_size:int, data_dir:str|None=None, seed:int|None=None, shuffle:bool=False):
    loader = create_text_to_image_eval_loader(
      data_dir=data_dir or eval_dir,
      batch_size=batch_size,
      seed=seed if seed is not None else int(getenv("SEED", 0)),
      shuffle=shuffle,
    )
    return _limit_loader(loader, eval_limit)

  def build_loss_fn(mode:str, *, seed:int|None=None, eval_noise_seed:int|None=None):
    if mode not in ("train", "eval"):
      raise ValueError(f"unsupported loss mode '{mode}'")
    if mode == "train":
      time_seed = seed if seed is not None else int(getenv("SEED", 0))
      time_rng = np.random.default_rng(time_seed)
      def train_loss_fn(model, batch):
        latents_np, text_np, bucket_np = batch
        if bucket_np is not None:
          raise ValueError("train batches must not include timestep buckets")
        latents = Tensor(latents_np)
        text = Tensor(text_np)
        times = Tensor(time_rng.random((latents.shape[0], 1), dtype=np.float32))
        noise = Tensor.randn(*latents.shape)
        target = noise - latents
        latent_mix = latents + times.reshape(-1, 1, 1, 1) * target
        pred = model(latent_mix, text, times)
        diff = pred - target
        reduce_axes = tuple(range(1, diff.ndim))
        per_sample = (diff * diff).mean(axis=reduce_axes)
        return per_sample.mean(), None
      return train_loss_fn
    eval_seed = eval_noise_seed if eval_noise_seed is not None else int(getenv("FLUX_EVAL_NOISE_SEED", 1337))
    eval_noise_rng = np.random.default_rng(eval_seed)
    def eval_loss_fn(model, batch):
      latents_np, text_np, bucket_np = batch
      if bucket_np is None:
        raise ValueError("eval batches must include timestep buckets")
      latents = Tensor(latents_np)
      text = Tensor(text_np)
      times = Tensor((bucket_np.astype(np.float32) / float(bucket_count)).reshape(-1, 1))
      noise_vals = eval_noise_rng.standard_normal(latents_np.shape).astype(np.float32)
      noise = Tensor(noise_vals)
      target = noise - latents
      latent_mix = latents + times.reshape(-1, 1, 1, 1) * target
      pred = model(latent_mix, text, times)
      diff = pred - target
      reduce_axes = tuple(range(1, diff.ndim))
      per_sample = (diff * diff).mean(axis=reduce_axes)
      return per_sample.mean(), (per_sample.numpy(), bucket_np)
    return eval_loss_fn

  def metric_fn(payloads):
    if not payloads:
      raise ValueError("no validation payloads provided")
    losses = np.concatenate([loss for loss, _ in payloads], axis=0)
    buckets = np.concatenate([bucket for _, bucket in payloads], axis=0)
    bucket_means = []
    for idx in range(bucket_count):
      mask = buckets == idx
      if not mask.any():
        raise ValueError(f"timestep bucket {idx} missing in evaluation batch")
      bucket_means.append(float(losses[mask].mean()))
    return float(np.mean(bucket_means)), bucket_means

  return {
    "name": "flux_text_to_image_tiny",
    "model_ctor": model_ctor,
    "train_dataloader": build_train_loader,
    "eval_dataloader": build_eval_loader,
    "loss_fn": build_loss_fn,
    "metric_fn": metric_fn,
  }

MODEL_REGISTRY = {
  "flux_text_to_image_tiny": flux_text_to_image_tiny_entry,
}

def get_model_entry(name:str):
  if name not in MODEL_REGISTRY:
    raise KeyError(f"unknown MLPerf model spec '{name}'")
  return MODEL_REGISTRY[name]()

if __name__ == "__main__":
  # inference only for now
  Tensor.training = False

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,mrcnn").split(","):
    nm = f"spec_{m}"
    if nm in globals():
      print(f"testing {m}")
      globals()[nm]()
