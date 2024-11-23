import math
from typing import Optional, Tuple, Union
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]

def mix(states: Tensor, chunks: Tensor) -> Tensor:
  def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))
  for i, (a,b,c,d) in enumerate([(0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15), (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)]):
    mx, my = chunks[:, i*2], chunks[:, i*2+1]
    for m in (mx, my):
      states[:, a] = states[:, a] + states[:, b] + m
      states[:, d] = rotr(states[:, d] ^ states[:, a], 16 if m is mx else 8)
      states[:, c] = states[:, c] + states[:, d]
      states[:, b] = rotr(states[:, b] ^ states[:, c], 12 if m is mx else 7)

def compress_chunks(states: Tensor, chunks: Tensor, chain_vals: Tensor, final_chunk_n_blocks: int):
  n_blocks = chunks.shape[1]
  # parallel over chunks, sequential over blocks
  for i in range(n_blocks):
    compressed = compress_blocks(states[:, i].contiguous(), chunks[:, i].contiguous(), chain_vals[:, i].contiguous())
    next_chain_vals = compressed[:, :8]
    # add chain value dependency to the next block
    if i < n_blocks - 1:
      states[:, i + 1, :8] = next_chain_vals
  if final_chunk_n_blocks == 16:
    return next_chain_vals
  else:
    # handle partial final chunk
    cvs_for_full_chunks = next_chain_vals[:-1]
    partial_chunk_end_idx = final_chunk_n_blocks - 1 if chunks.shape[0] == 1 else final_chunk_n_blocks
    cv_for_partial_chunk = states[-1:, partial_chunk_end_idx, :8]
    return cvs_for_full_chunks.cat(cv_for_partial_chunk)

def compress_blocks(states: Tensor, chunks: Tensor, chain_vals: Tensor) -> Tensor:
  for _ in range(6):
    mix(states, chunks)
    chunks.replace(chunks[:, [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]])
  mix(states, chunks)
  states[:, :8] = states[:, :8] ^ states[:, 8:]
  states[:, 8:] = chain_vals[:, :8] ^ states[:, 8:]
  return states

def bytes_to_chunks(text_bytes: bytes) -> Tuple[Tensor, int, int]:
  n_bytes = len(text_bytes)
  n_chunks = max((n_bytes + 1024 - 1) // 1024, 1)
  n_blocks = 1024 // 64
  n_words = 64 // 4 # 32-bit words
  chunks = Tensor.zeros(n_chunks, n_blocks, n_words, dtype=dtypes.uint32).contiguous()
  unpadded_len = 0
  # chunks
  for i in range(0, max(n_bytes, 1), 1024):
    chunk = text_bytes[i: i + 1024]
    unpadded_len = len(chunk)
    chunk = chunk.ljust(1024, b"\0")
    # blocks
    for j in range(16):
      block_start = j * 64
      bw_bytes = chunk[block_start:block_start + 64]
      # words
      block_words = [int.from_bytes(bw_bytes[i: i + 4], "little") for i in range(0, len(bw_bytes), 4)]
      chunks[i // 1024, j] = Tensor(block_words, dtype=dtypes.uint32)
  n_end_blocks = max(1, (unpadded_len // 64) + (1 if unpadded_len % 64 else 0))
  end_block_len = 64 if unpadded_len % 64 == 0 and unpadded_len else unpadded_len % 64
  return chunks, n_end_blocks, end_block_len

def pairwise_concat(chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  if chain_vals.shape[0] % 2 != 0: chain_vals, leftover_chunk = chain_vals[:-1], chain_vals[-1:]
  else: leftover_chunk = None
  return (chain_vals.reshape(-1, 16), leftover_chunk)

def create_state(iv: Tensor, count: Optional[int], end_block_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_chunks, n_blocks = iv.shape[0], iv.shape[1]
  if count is not None:
    counts = Tensor.arange(count, count + n_chunks, dtype=dtypes.uint32)
    counts = counts.reshape(n_chunks, 1).expand(n_chunks, n_blocks).reshape(n_chunks, n_blocks, 1)
    counts = counts.cat(Tensor.zeros(n_chunks, n_blocks, 1, dtype=dtypes.uint32), dim=-1)
  else:
    counts = Tensor.zeros(n_chunks, n_blocks, 2, dtype=dtypes.uint32)
  lengths = Tensor.full((n_chunks, n_blocks, 1), fill_value=64, dtype=dtypes.uint32).contiguous()
  if end_block_len is not None: lengths[-1, n_end_blocks - 1] = end_block_len
  return iv.cat(iv[:, :, :4], counts, lengths, flags, dim=-1)

def create_flags(chunks: Tensor, n_end_blocks: Optional[int], is_root: bool, are_parents: bool = False) -> Tensor:
  n_chunks, n_blocks = chunks.shape[:2]
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags = Tensor.zeros((n_chunks, n_blocks, 1), dtype=dtypes.uint32).contiguous()
  flags[:, 0] = flags[:, 0] + 1 # chunk start flag
  flags[:-1, -1] = flags[:-1, -1] + 2 # chunk end flag
  flags[-1, end_idx:] = flags[-1, end_idx:] + 2 # chunk end flag
  if are_parents: flags[:, :, :] = 4 # parent flag
  if is_root: flags[0, end_idx] = flags[0, end_idx] + 8 # root flag
  return flags

def init_compress(input: Union[str, bytes, Tensor]) -> Tensor:
  input_bytes = input.encode("utf-8") if isinstance(input, str) else input if isinstance(input, bytes) else b""
  chunks, n_end_blocks, end_block_len = bytes_to_chunks(input_bytes)
  n_chunks, n_blocks, _ = chunks.shape
  iv = Tensor(IV, dtype=dtypes.uint32).expand(n_chunks, n_blocks, -1).contiguous()
  flags = create_flags(chunks, n_end_blocks, is_root=n_chunks == 1)
  states = create_state(iv, 0, end_block_len, n_end_blocks, flags)
  return compress_chunks(states, chunks, iv, n_end_blocks)

def blake3(data: Union[str, bytes, Tensor]) -> str:
  """Hash an input string, bytes or Tensor in parallel using the BLAKE3 hashing algorithm."""
  chain_vals = init_compress(data)
  tree_levels = math.ceil(math.log2(max(chain_vals.shape[0], 1)))
  for i in range(tree_levels): # tree-hash chain value pairs ~halving them in each step
    chain_vals, leftover_chain_val = pairwise_concat(chain_vals)
    n_chain_vals, n_blocks = chain_vals.shape[0], chain_vals.shape[1]
    init_chain_vals = Tensor(IV, dtype=dtypes.uint32).expand(n_chain_vals, n_blocks, -1).contiguous()
    flags = create_flags(chain_vals, None, are_parents=True, is_root=i == tree_levels - 1)
    states = create_state(init_chain_vals, None, None, None, flags)[:, -1].contiguous()
    chain_vals = compress_blocks(states, chain_vals, init_chain_vals[:, 0])[:, :8]
    if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=0)
  return chain_vals[0].flatten().numpy().tobytes()[:32].hex()
