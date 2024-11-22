import math
import os
from typing import Optional, Tuple, Union
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

MD_LEN = 32
BLOCK_BYTES = 64
CHUNK_BYTES = 1024
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

def rotr(x: Tensor, n: int) -> Tensor: return ((x << (32 - n)) | (x >> n))

def mix(states: Tensor, a: int, b: int, c: int, d: int, mx: Tensor, my: Tensor) -> Tensor:
  states[:, a] = states[:, a] + (states[:, b] + mx)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 16)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 12)
  states[:, a] = states[:, a] + (states[:, b] + my)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 8)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 7)

def mix_round(states: Tensor, chunks: Tensor) -> Tensor:
  mix(states, 0, 4, 8, 12, chunks[:, 0], chunks[:, 1])
  mix(states, 1, 5, 9, 13, chunks[:, 2], chunks[:, 3])
  mix(states, 2, 6, 10, 14, chunks[:, 4], chunks[:, 5])
  mix(states, 3, 7, 11, 15, chunks[:, 6], chunks[:, 7])
  mix(states, 0, 5, 10, 15, chunks[:, 8], chunks[:, 9])
  mix(states, 1, 6, 11, 12, chunks[:, 10], chunks[:, 11])
  mix(states, 2, 7, 8, 13, chunks[:, 12], chunks[:, 13])
  mix(states, 3, 4, 9, 14, chunks[:, 14], chunks[:, 15])

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
    mix_round(states, chunks)
    chunks.replace(chunks[:, MSG_PERMUTATION])
  mix_round(states, chunks)
  states[:, :8] = states[:, :8] ^ states[:, 8:]
  states[:, 8:] = chain_vals[:, :8] ^ states[:, 8:]
  return states

def bytes_to_chunks(text_bytes: bytes) -> Tuple[Tensor, int, int]:
  n_bytes = len(text_bytes)
  n_chunks = max((n_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES, 1)
  n_blocks = CHUNK_BYTES // BLOCK_BYTES
  n_words = BLOCK_BYTES // 4 # 32-bit words
  chunks = Tensor.zeros(n_chunks, n_blocks, n_words, dtype=dtypes.uint32).contiguous()
  unpadded_len = 0
  # chunks
  for i in range(0, max(len(text_bytes), 1), CHUNK_BYTES):
    chunk = text_bytes[i: i + CHUNK_BYTES]
    unpadded_len = len(chunk)
    chunk = chunk.ljust(CHUNK_BYTES, b"\0")
    # blocks
    for j in range(16):
      block_start = j * BLOCK_BYTES
      bw_bytes = chunk[block_start:block_start + BLOCK_BYTES]
      # words
      block_words = [int.from_bytes(bw_bytes[i: i + 4], "little") for i in range(0, len(bw_bytes), 4)]
      chunks[i // CHUNK_BYTES, j] = Tensor(block_words, dtype=dtypes.uint32)
  n_end_blocks = max(1, (unpadded_len // BLOCK_BYTES) + (1 if unpadded_len % BLOCK_BYTES else 0))
  end_block_len = BLOCK_BYTES if unpadded_len % BLOCK_BYTES == 0 and unpadded_len else unpadded_len % BLOCK_BYTES
  return chunks, n_end_blocks, end_block_len

def pairwise_concat(chain_vals: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  if chain_vals.shape[0] % 2 != 0:
    leftover_chunk = chain_vals[-1:]
    chain_vals = chain_vals[:-1]
  else:
    leftover_chunk = None
  return chain_vals.reshape(-1, 16), leftover_chunk

def create_state(iv: Tensor, count: Optional[int], end_block_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_chunks, n_blocks = iv.shape[0], iv.shape[1]
  if count is not None:
    counts = Tensor.arange(count, count + n_chunks, dtype=dtypes.uint32)
    counts = counts.reshape(n_chunks, 1).expand(n_chunks, n_blocks).reshape(n_chunks, n_blocks, 1)
    counts = counts.cat(Tensor.zeros(n_chunks, n_blocks, 1, dtype=dtypes.uint32), dim=-1)
  else:
    counts = Tensor.zeros(n_chunks, n_blocks, 2, dtype=dtypes.uint32)
  lengths = Tensor.full((n_chunks, n_blocks, 1), fill_value=BLOCK_BYTES, dtype=dtypes.uint32).contiguous()
  if end_block_len is not None:
    lengths[-1, n_end_blocks - 1] = end_block_len
  states = iv.cat(iv[:, :, :4], counts, lengths, flags, dim=-1)
  return states

def create_flags(chunks: Tensor, n_end_blocks: Optional[int], is_root: bool, are_parents: bool = False) -> Tensor:
  n_chunks, n_blocks = chunks.shape[0], chunks.shape[1]
  flags = Tensor.zeros((n_chunks, n_blocks, 1), dtype=dtypes.uint32).contiguous()
  flags[:, 0] = flags[:, 0] + CHUNK_START
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags[:-1, -1] = flags[:-1, -1] + CHUNK_END
  flags[-1, end_idx:] = flags[-1, end_idx:] + CHUNK_END
  if are_parents: flags[:, :, :] = PARENT
  if is_root: flags[0, end_idx] = flags[0, end_idx] + ROOT
  return flags

def init_compress(input: Union[str, bytes], counter = 0, not_root: bool = False) -> Tensor:
  input_bytes = input.encode("utf-8") if isinstance(input, str) else input if isinstance(input, bytes) else b""
  chunks, n_end_blocks, end_block_len = bytes_to_chunks(input_bytes)
  n_chunks, n_blocks, _ = chunks.shape
  initial_chain_vals = Tensor(IV, dtype=dtypes.uint32).expand(n_chunks, n_blocks, -1).contiguous()
  flags = create_flags(chunks, n_end_blocks, is_root=n_chunks == 1 and not not_root)
  states = create_state(initial_chain_vals, counter, end_block_len, n_end_blocks, flags)
  return compress_chunks(states, chunks, initial_chain_vals, n_end_blocks)

def init_compress_file(file: str, bufsize: int) -> Tensor:
  chain_vals = Tensor.zeros(0, 8, dtype=dtypes.uint32)
  file_size = os.path.getsize(file)
  with open(file, "rb") as f:
    while chunk_bytes := f.read(bufsize):
      counter = chain_vals.shape[0]
      # don't set the root flag if the file is larger than one chunk, more data is coming
      not_root = file_size > CHUNK_BYTES
      chain_vals = chain_vals.cat(init_compress(chunk_bytes, counter, not_root=not_root))
  return chain_vals

def blake3(text: Optional[Union[str, bytes]] = None, file: Optional[str] = None, bufsize: int = 8 * 1024 * 1024) -> str:
  """
  Hash an input string, bytes, or file in parallel using the BLAKE3 hashing algorithm.
  When hashing a file, the file is read in chunks of `bufsize` bytes.
  """
  assert text is not None or file is not None, "Either text or a file must be provided"
  assert bufsize % CHUNK_BYTES == 0, f"bufsize must be a multiple of {CHUNK_BYTES}"
  # compress input chunks into an initial set of chain values
  chain_vals = init_compress(text) if isinstance(text, (str, bytes)) else init_compress_file(file, bufsize)
  tree_levels = math.ceil(math.log2(max(chain_vals.shape[0], 1)))
  # tree-hash pairs of chain values, ~halving the number of them in each step until one remains
  for i in range(tree_levels):
    chain_vals, leftover_chain_val = pairwise_concat(chain_vals)
    n_chain_vals, n_blocks = chain_vals.shape[0], chain_vals.shape[1]
    init_chain_vals = Tensor(IV, dtype=dtypes.uint32).expand(n_chain_vals, n_blocks, -1).contiguous()
    flags = create_flags(chain_vals, None, are_parents=True, is_root=i == tree_levels - 1)
    states = create_state(init_chain_vals, None, None, None, flags)[:, -1].contiguous()
    chain_vals = compress_blocks(states, chain_vals, init_chain_vals[:, 0])[:, :8]
    if leftover_chain_val is not None: chain_vals = chain_vals.cat(leftover_chain_val, dim=0)
  return chain_vals[0].flatten().numpy().tobytes()[:MD_LEN].hex()
