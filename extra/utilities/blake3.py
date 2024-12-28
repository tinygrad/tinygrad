from math import ceil, log2
from typing import Tuple
from tinygrad import Tensor, TinyJit, Variable, dtypes
from tinygrad.helpers import ceildiv

class BLAKE3:
  """BLAKE3 hashing algorithm. Paper: https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf."""
  IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
  PAD, DEFAULT_LEN, PERMUTATIONS = 66, 65, Tensor([2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8], dtype=dtypes.uint32)

  def compress_blocks(self, states: Tensor, data: Tensor, chain_vals: Tensor) -> Tensor:
    def rotr(x: Tensor, n: int) -> Tensor: return (x << (32 - n)) | (x >> n)
    for i in range(7):
      for j, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
        mx, my = data[j * 2], data[j * 2 + 1]
        for m in (mx, my):
          states[a] = (states[a] + states[b] + m).cast(dtypes.uint32)
          states[d] = rotr(states[d] ^ states[a], 16 if m is mx else 8).cast(dtypes.uint32)
          states[c] = (states[c] + states[d]).cast(dtypes.uint32)
          states[b] = rotr(states[b] ^ states[c], 12 if m is mx else 7).cast(dtypes.uint32)
      if i < 6: data = data[self.PERMUTATIONS]
    return (states[:8] ^ states[8:]).cat(chain_vals[:8] ^ states[8:])

  @TinyJit
  def init_chain_vals(self, data: Tensor, info: Tensor) -> Tuple[Tensor, Tensor]:
    chain_vals = self.IV.reshape(1, 8, 1).expand(16, 8, info.shape[-1]).contiguous()
    counts = Tensor.arange(0, data.shape[-1], dtype=dtypes.uint32).reshape(-1, 1).expand(-1, 16).reshape(-1, 16, 1).permute(1, 2, 0)
    counts = counts.cat(Tensor.zeros(chain_vals.shape[0], 1, chain_vals.shape[-1], dtype=dtypes.uint32), dim=1)
    lengths = (info == self.DEFAULT_LEN).where(64, info)
    flags = Tensor.zeros((16, 1, info.shape[-1]), dtype=dtypes.uint32).contiguous()
    flags[-1, 0] = flags[-1, 0] + 2 # chunk end flag
    flags = (flags + 2 * ((flags != 2) * (info < self.DEFAULT_LEN))) # chunk end flag for partial final chunk
    flags[0] = flags[0] + 1 # chunk start flag
    flags = (flags + (8 * (((info < self.PAD).sum() <= 16) * (info < self.DEFAULT_LEN)))).cast(dtypes.uint32) # root flag
    states = (chain_vals.cat(chain_vals[:, :4], counts, lengths, flags, dim=1) * (info < self.PAD).cast(dtypes.uint32))
    for i in range(16):
      next_state = states[i] if i == 0 else states[i-1, :8].cat(states[i, 8:])
      states[i] = self.compress_blocks(next_state, data[i], chain_vals[i])
    states = states * (info < self.PAD)
    end_block = (states * (info < self.DEFAULT_LEN)).sum(0)
    return (states[-1, :] | end_block)[:8].realize()

  @classmethod # JIT doesn't like making n_tree_steps for the loop a Variable, this is a workaround
  def create_jitted_tree_hash(cls, n_tree_steps: int) -> Tensor:
    def tree_hash(self, chain_vals: Tensor) -> Tensor:
      for _ in range(n_tree_steps):
        stacked = chain_vals.transpose().reshape(-1, 16).transpose().reshape(2, 8, -1)
        stacked_mask = stacked.any(1)
        final_step = chain_vals[0, :3].prod().cast(dtypes.bool).neg()
        pair_mask, remainder_mask = (stacked_mask[0] * stacked_mask[1]), (stacked_mask[0] ^ stacked_mask[1])
        paired, remainder = (stacked * pair_mask).reshape(16, -1), (stacked * remainder_mask).reshape(16, -1)[:8]
        flags = final_step.where(12, Tensor.full((1, paired.shape[-1]), 4, dtype=dtypes.uint32))
        iv = self.IV.reshape(8, 1).expand(8, paired.shape[-1])
        counts = Tensor.zeros((2, paired.shape[-1]), dtype=dtypes.uint32)
        lengths = Tensor.full((1, paired.shape[-1]), 64, dtype=dtypes.uint32)
        states = iv.cat(iv[:4], counts, lengths, flags, dim=0)
        chain_vals = ((self.compress_blocks(states, paired, iv) * pair_mask)[:8] + remainder).realize()
        chain_vals = chain_vals.pad((None, (0, chain_vals.shape[1])))
      return chain_vals.realize()
    return tree_hash

  def tensor_to_blake_input(self, tensor: Tensor, padded_input_size: int) -> Tuple[Tensor, Tensor, Variable]:
    assert padded_input_size % 1024 == 0, "padded_input_size must be divisible by 1024"
    data = tensor.flatten().pad(((0, (padded_input_size // tensor.element_size()) - tensor.shape[0],),), value=0)
    data = data.bitcast(dtypes.uint32).reshape(-1, 16, 16).permute(1, 2, 0).contiguous()
    final_chunk_len = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 1024 or 1024)
    n_end_blocks = ceildiv(final_chunk_len, 64) or 1
    n_chunks = max(1, ceildiv(tensor.nbytes(), 1024))
    info = Tensor.full((16, 1, data.shape[-1]), fill_value=self.DEFAULT_LEN, dtype=dtypes.uint32).contiguous()
    info[n_end_blocks - 1, :, n_chunks - 1] = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 64) or 64
    info[n_end_blocks:, :, n_chunks - 1:] = info[:, :, n_chunks:] = self.PAD
    n_steps = Variable(min_val=0, max_val=log2(padded_input_size), name="n_steps").bind(ceil(log2(max(n_chunks, 1))))
    return data, info, n_steps

  def hash(self, tensor: Tensor, padded_input_size: int = 1024**2 * 512) -> str:
    data, info, n_tree_steps = self.tensor_to_blake_input(tensor, padded_input_size)
    chain_vals = self.init_chain_vals(data, info)
    chain_vals = self.tree_hashes[n_tree_steps.val](self, chain_vals) if n_tree_steps.val > 0 else chain_vals
    return chain_vals[:, 0].flatten().bitcast(dtypes.uint8).data().tobytes().hex()

BLAKE3.tree_hashes = {n_tree_steps: BLAKE3.create_jitted_tree_hash(n_tree_steps) for n_tree_steps in range(1, 54)}

if __name__ == "__main__":
  import time
  import sys

  arg = sys.argv[1]
  arg_bytes = int(arg) * 1024 ** 2
  padded_input_size = ceildiv(arg_bytes, 1024) * 1024
  def benchmark_size(size_bytes):
    print(f"\nBenchmarking {size_bytes / 1024 / 1024 :.1f} MB...")
    data = Tensor.rand(size_bytes // 2, dtype=dtypes.float16)
    size = data.numel() * data.element_size()

    start = time.time()
    BLAKE3().hash(data, padded_input_size=padded_input_size)
    end = time.time()

    elapsed = end - start
    throughput = size / elapsed / 1e6  # MB/s
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} MB/s")

  size_mb = float(sys.argv[1])
  size = int(size_mb * 1024 * 1024)

  for i in range(5):
    benchmark_size(size)