import hashlib
import time
from extra.tinysha.tinysha import tiny_keccak
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


def tree_hash(self: Tensor, cfg: str | tuple[int, int] = "sha3_256"):
  rate = { "sha3_224": 144, "sha3_256": 136, "sha3_384": 104, "sha3_512": 72 }.get(cfg, cfg[0])
  blk_size = rate - 1 # rate - 1 is max per permutation
  b_out = (200-rate)//2

  while self.shape[-1] != b_out:
    pad_size = 0 if self.shape[-1] % blk_size == 0 else blk_size - self.shape[-1] % blk_size
    self = self.pad(tuple(None for _ in range(len(self.shape) - 1)) + ((0, pad_size),))
    self = self.reshape(*self.shape[:-1], -1, blk_size)
    self = tiny_keccak(self, cfg).flatten(-2).realize()

  return self

test_data = Tensor.randint(2**12, low=0, high=255, dtype=dtypes.uint8).realize()

time_start = time.time()
print(tree_hash(test_data).numpy().tobytes().hex())
print(f"tree took {time.time() - time_start}s")

time_start = time.time()
print(tiny_keccak(test_data).numpy().tobytes().hex())
print(f"normal took {time.time() - time_start}s")

time_start = time.time()
print(hashlib.sha3_256(test_data.data()).hexdigest())
print(f"hashlib took {time.time() - time_start}s")
