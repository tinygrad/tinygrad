import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import numpy as np

ArrayBatch = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]

@dataclass(frozen=True)
class _Shard:
  latent: np.ndarray
  text: np.ndarray
  timestep_bucket: Optional[np.ndarray]

def _load_shard(path:str) -> _Shard:
  with np.load(path, allow_pickle=False) as data:
    latent = np.array(data["latent"], dtype=np.float32, copy=True)
    text = np.array(data["text"], dtype=np.float32, copy=True)
    bucket = None
    if "timestep_bucket" in data:
      bucket = np.array(data["timestep_bucket"], dtype=np.int64, copy=True)
  return _Shard(latent=latent, text=text, timestep_bucket=bucket)

def _list_shards(data_dir:Path, pattern:str="*.npz") -> List[str]:
  shards = sorted(str(p) for p in data_dir.glob(pattern))
  if not shards:
    raise FileNotFoundError(f"no shards matching {pattern} under {data_dir}")
  return shards

class TextToImageShardLoader:
  def __init__(self, data_dir:str, batch_size:int, seed:int=0,
               shuffle:bool=True, include_timestep:bool=False):
    self.paths = _list_shards(Path(data_dir))
    self.batch_size = batch_size
    self.seed = seed
    self.shuffle = shuffle
    self.include_timestep = include_timestep

  def __iter__(self) -> Generator[ArrayBatch, None, None]:
    rng = random.Random(self.seed)
    shard_paths = list(self.paths)
    if self.shuffle:
      rng.shuffle(shard_paths)
    for path in shard_paths:
      shard = _load_shard(path)
      count = shard.latent.shape[0]
      indices = list(range(count))
      if self.shuffle:
        rng.shuffle(indices)
      for start in range(0, count, self.batch_size):
        batch_idx = indices[start:start+self.batch_size]
        if not batch_idx:
          continue
        latents = shard.latent[batch_idx]
        text = shard.text[batch_idx]
        bucket = None
        if self.include_timestep:
          bucket_vals = shard.timestep_bucket
          if bucket_vals is None:
            bucket_vals = np.zeros(count, dtype=np.int64)
          bucket = bucket_vals[batch_idx]
        yield latents, text, bucket

def create_text_to_image_train_loader(data_dir:str, batch_size:int, seed:int=0) -> Iterable[ArrayBatch]:
  return TextToImageShardLoader(data_dir, batch_size, seed=seed, shuffle=True, include_timestep=False)

def create_text_to_image_eval_loader(data_dir:str, batch_size:int, seed:int=0,
                                     shuffle:bool=False) -> Iterable[ArrayBatch]:
  return TextToImageShardLoader(data_dir, batch_size, seed=seed, shuffle=shuffle, include_timestep=True)
