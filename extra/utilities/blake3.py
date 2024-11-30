import math
from typing import Dict, List, Optional, Tuple
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine import jit
from tinygrad.helpers import ceildiv
from tinygrad.tensor import Tensor

class BLAKE3:
  def __init__(self, std_sizes: Optional[List[int]] = None):
    self.IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
    self.std_sizes = std_sizes or [1024 * (1024**2)] #, 1024**3, 2 * (1024**3), 4 * (1024**3)] # size rounding for JIT consistency

  @jit.TinyJit
  def mix(self, states: Tensor, chunks: Tensor) -> Tensor:
    def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
    for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
      mx, my = chunks[i * 2], chunks[i * 2 + 1]
      for m in (mx, my):
        states[a] = states[a] + states[b] + m
        states[d] = rotr(states[d] ^ states[a], 16 if m is mx else 8)
        states[c] = states[c] + states[d]
        states[b] = rotr(states[b] ^ states[c], 12 if m is mx else 7)

  def compress_chunks(self, states: Tensor, chunks: Tensor, chain_vals: Tensor, len_info: Dict) -> Tensor:
    n_end_blocks, n_chunks = len_info["n_end_blocks"], len_info["n_chunks"]
    for i in range(16): # parallel over chunks, sequential over blocks
      compressed = self.compress_blocks(states[i].contiguous(), chunks[i].contiguous(), chain_vals[i])
      if i < chunks.shape[1] - 1: states[i + 1, :8] = compressed[:8] # propagate chain vals
      if i == n_end_blocks - 1: final_chain_val = compressed[:8, n_chunks - 1].reshape(-1, 1) # for partial chunks
    return compressed[:8, :n_chunks] if n_end_blocks == 16 else compressed[:8, :n_chunks - 1].cat(final_chain_val, dim=-1)

  def compress_blocks(self, states: Tensor, chunks: Tensor, chain_vals: Tensor) -> Tensor:
    for _ in range(6):
      self.mix(states, chunks)
      chunks[:] = chunks[[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]]
    self.mix(states, chunks)
    states[:8] = states[:8] ^ states[8:]
    states[8:] = chain_vals[:8] ^ states[8:]
    return states

  def tensor_to_blake_data(self, tensor: Tensor) -> Tuple[Tensor, int, int]:
    data = tensor.flatten().bitcast(dtypes.uint8)
    pad_amt = min(size for size in self.std_sizes if size >= tensor.nbytes()) - tensor.nbytes()
    data = data.pad(((0, pad_amt),), value=0).bitcast(dtypes.uint32).reshape(-1, 16, 16).permute(1, 2, 0)
    print(f"post padding data size MB: {data.numel() * data.element_size() / 1024 / 1024 :.1f}")
    final_chunk_len = 0 if tensor.nbytes() == 0 else (tensor.nbytes() % 1024 or 1024)
    n_end_blocks, end_block_len = ceildiv(final_chunk_len, 64) or 1, 0 if tensor.nbytes() == 0 else tensor.nbytes() % 64 or 64
    n_chunks = max(1, ceildiv(tensor.nbytes(), 1024))
    return data.contiguous(), {"n_end_blocks": n_end_blocks, "end_block_len": end_block_len, "n_chunks": n_chunks}

  def pairwise_concat(self, chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    leftover_chunk = chain_vals[:, -1:] if chain_vals.shape[1] % 2 else None
    chain_vals = chain_vals[:, :-1 if leftover_chunk is not None else None]
    return chain_vals.permute(1, 0).reshape(-1, 16).transpose().contiguous(), leftover_chunk

  def create_state(self, iv: Tensor, count: Optional[int], len_info: Dict, flags: Tensor) -> Tensor:
    n_blocks, _, total_chunks = iv.shape
    n_end_blocks, end_block_len, n_chunks = len_info["n_end_blocks"], len_info["end_block_len"], len_info["n_chunks"]
    if count is not None:
      counts = Tensor.arange(count, count + total_chunks, dtype=dtypes.uint32).reshape(-1, 1).expand(-1, 16).reshape(-1, 16, 1)
      counts = counts.permute(1, 2, 0).cat(Tensor.zeros(n_blocks, 1, total_chunks, dtype=dtypes.uint32), dim=1)
    else:
      counts = Tensor.zeros((n_blocks, 2, total_chunks), dtype=dtypes.uint32)
    lengths = Tensor.full((n_blocks, 1, total_chunks), fill_value=64, dtype=dtypes.uint32).contiguous()
    if end_block_len is not None: lengths[n_end_blocks - 1, :, n_chunks - 1] = end_block_len
    return iv.cat(iv[:, :4], counts, lengths, flags, dim=1)

  def create_flags(self, data: Tensor, len_info: Dict, root: bool, parent: bool, final_step: bool) -> Tensor:
    block_end_idx = len_info["n_end_blocks"] - 1 if len_info["n_end_blocks"] is not None else -1
    chunk_end_idx = len_info["n_chunks"] - 1
    flags = Tensor.zeros((16, 1, data.shape[-1]), dtype=dtypes.uint32).contiguous()
    flags[0] = flags[0] + 1 # chunk start flag
    flags[-1, 0, :chunk_end_idx] = flags[-1, 0, :chunk_end_idx] + 2 # chunk end flag
    flags[block_end_idx:, :, chunk_end_idx] = flags[block_end_idx:, :, chunk_end_idx] + 2 # final chunk end flag for partial chunk
    if parent: flags[:] = 4 # parent flag
    if root: flags[block_end_idx, :, chunk_end_idx] = flags[block_end_idx, :, chunk_end_idx] + 8 # root flag
    if final_step: flags[:] = 12
    return flags

  def compress(self, data, compressor, count, len_info, root, parent = False, final_step = False) -> Tensor:
    iv = self.IV.reshape(1, 8, 1).expand(16, 8, data.shape[-1]).contiguous()
    states = self.create_state(iv, count, len_info, self.create_flags(data, len_info, root, parent, final_step))
    return compressor(states, data, iv, len_info)

  def compress_tree(self, states, data, iv, _): return self.compress_blocks(states[-1].contiguous(), data, iv[0])

  def hash(self, tensor: Tensor) -> str:
    data, len_info = self.tensor_to_blake_data(tensor)
    chain_vals = self.compress(data, self.compress_chunks, 0, len_info, len_info["n_chunks"] == 1)
    n_steps = math.ceil(math.log2(max(chain_vals.shape[-1], 1)))
    len_info["end_block_len"] = None # don't need end block length for tree hashing
    for i in range(n_steps): # tree-hash chain value pairs ~halving them in each step
      chain_vals, leftover_chain_val = self.pairwise_concat(chain_vals)
      pre_pad_size = chain_vals.shape[1]
      pad_size = data.shape[2] - pre_pad_size # padding for JIT consistency
      chain_vals = chain_vals.pad(((0, 0), (0, pad_size)), value=0) if i < n_steps - 1 else chain_vals.expand(-1, pre_pad_size + pad_size)
      chain_vals = self.compress(chain_vals.contiguous(), self.compress_tree, None, len_info, i == n_steps - 1, True, i == n_steps - 1)
      chain_vals = chain_vals[:8, :pre_pad_size]
      if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=1)
    return chain_vals.flatten().bitcast(dtypes.uint8).data().tobytes().hex()

if __name__ == "__main__":
  import time
  import sys

  arg = sys.argv[1]

  if arg == "warmup":
    # warmup the JIT
    print("\nWarming up...")
    def warmup(size):
      print(f"Warming up {size / 1024 / 1024 :.1f} MB...")
      warmup_data = Tensor.rand(size // 2, dtype=dtypes.float16)
      BLAKE3().hash(warmup_data)
    for size in BLAKE3().std_sizes: warmup(size)
  else:
    def benchmark_size(size_bytes):
      print(f"\nBenchmarking {size_bytes / 1024 / 1024 :.1f} MB...")
      data = Tensor.rand(size_bytes // 2, dtype=dtypes.float16)
      size = data.numel() * data.element_size()

      start = time.time()
      BLAKE3().hash(data)
      end = time.time()

      elapsed = end - start
      throughput = size / elapsed / 1e6  # MB/s
      print(f"Time: {elapsed:.2f}s")
      print(f"Throughput: {throughput:.1f} MB/s")

    size_mb = float(sys.argv[1])
    size = int(size_mb * 1024 * 1024)

    benchmark_size(size)
