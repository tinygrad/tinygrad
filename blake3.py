import math
from typing import Optional, Tuple
import unittest
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

MD_LEN = 32
BLOCK_BYTES = 64
CHUNK_BYTES = 1024
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

def rotr(x: Tensor, n: int): return (x << (32 - n)) | (x >> n)

def mix(states: Tensor, a: int, b: int, c: int, d: int, mx: Tensor, my: Tensor) -> Tensor:
  states[:, a] = states[:, a] + (states[:, b] + mx)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 16)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 12)
  states[:, a] = states[:, a] + (states[:, b] + my)
  states[:, d] = rotr(states[:, d] ^ states[:, a], 8)
  states[:, c] = states[:, c] + states[:, d]
  states[:, b] = rotr(states[:, b] ^ states[:, c], 7)

def round(states: Tensor, chunks: Tensor) -> Tensor:
  mix(states, 0, 4, 8, 12, chunks[:, 0], chunks[:, 1])
  mix(states, 1, 5, 9, 13, chunks[:, 2], chunks[:, 3])
  mix(states, 2, 6, 10, 14, chunks[:, 4], chunks[:, 5])
  mix(states, 3, 7, 11, 15, chunks[:, 6], chunks[:, 7])
  mix(states, 0, 5, 10, 15, chunks[:, 8], chunks[:, 9])
  mix(states, 1, 6, 11, 12, chunks[:, 10], chunks[:, 11])
  mix(states, 2, 7, 8, 13, chunks[:, 12], chunks[:, 13])
  mix(states, 3, 4, 9, 14, chunks[:, 14], chunks[:, 15])

def compress_chunks(states: Tensor, chunks: Tensor, chain_vals: Tensor, final_chunk_n_blocks: int):
  for i in range(states.shape[1]): # parallel over chunks, sequential within blocks
    compressed = compress_blocks(states[:, i].contiguous(), chunks[:, i].contiguous(), chain_vals[:, i].contiguous())
    next_chain_vals = compressed[:, :8]
    if i < states.shape[1] - 1:
      chain_vals[:, i + 1] = next_chain_vals
      states[:, i + 1, :8] = next_chain_vals
  if final_chunk_n_blocks == 16: return next_chain_vals
  else: return chain_vals[:-1, -1].cat(chain_vals[-1, final_chunk_n_blocks], dim=0)

def compress_blocks(states: Tensor, chunks: Tensor, chain_vals: Tensor) -> Tensor:
  for i in range(6):
    round(states, chunks)
    chunks.replace(chunks[:, MSG_PERMUTATION])
  round(states, chunks)
  states[:, :8] = states[:, :8] ^ states[:, 8:]
  states[:, 8:] = chain_vals[:, :8] ^ states[:, 8:]
  return states

def bytes_to_chunks(text_bytes: bytes) -> Tuple[Tensor, int, int]:
  n_bytes = len(text_bytes)
  n_chunks = max((n_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES, 1)
  chunks = Tensor.zeros(n_chunks, 16, 16, dtype=dtypes.uint32).contiguous()
  unpadded_len = 0
  # chunks
  for i in range(0, max(len(text_bytes), 1), CHUNK_BYTES):
    chunk = text_bytes[i:i + CHUNK_BYTES]
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

def pairwise_concat(chain_vals: Tensor) -> Tensor:
  if chain_vals.shape[0] % 2 != 0:
    chain_vals = chain_vals.cat(Tensor.zeros(1, 8, dtype=dtypes.uint32), dim=0)
  return chain_vals.reshape(math.ceil(chain_vals.shape[0] / 2), 16)

def create_state(chain_vals: Tensor, iv: Tensor, counter: Optional[int], last_len: Optional[int], n_end_blocks: Optional[int], flags: Tensor) -> Tensor:
  n_chunks, n_blocks, _ = chain_vals.shape
  if counter is not None:
    counts = Tensor.arange(counter, counter + n_chunks, dtype=dtypes.uint32).reshape(n_chunks, 1).expand(n_chunks, n_blocks).reshape(n_chunks, n_blocks, 1)
    counts = Tensor.zeros(n_chunks, n_blocks, 1, dtype=dtypes.uint32).cat(counts, dim=-1)
  else:
    counts = Tensor.zeros(n_chunks, n_blocks, 2, dtype=dtypes.uint32)
  lengths = Tensor.full((n_chunks, n_blocks, 1), fill_value=BLOCK_BYTES, dtype=dtypes.uint32).contiguous()
  if last_len is not None: lengths[:, n_end_blocks - 1] = last_len
  states = chain_vals.cat(iv[:, :, :4], counts, lengths, flags, dim=-1)
  return states

def create_flags(n_chunks: int, n_blocks: int, n_end_blocks: Optional[int], parents: bool, root: bool) -> Tensor:
  flags = Tensor.zeros((n_chunks, n_blocks, 1), dtype=dtypes.uint32).contiguous()
  flags[:, 0] = flags[:, 0] + CHUNK_START
  end_idx = n_end_blocks - 1 if n_end_blocks is not None else -1
  flags[:, end_idx] = flags[:, end_idx] + CHUNK_END
  if parents: flags[:, :, :] = PARENT
  if root: flags[:, end_idx] = flags[:, end_idx] + ROOT
  return flags

def tiny_blake3(text: str) -> str:
  text_bytes = text.encode("utf-8") if text else b""
  chunks, n_end_blocks, end_block_len = bytes_to_chunks(text_bytes)
  # intitial compression
  iv = IV.expand(chunks.shape[0], chunks.shape[1], -1).contiguous()
  flags = create_flags(chunks.shape[0], chunks.shape[1], n_end_blocks, False, chunks.shape[0] == 1)
  states = create_state(iv, iv, 0, end_block_len, n_end_blocks, flags)
  print(f"states before compress_chunks()\n{states.numpy()}")
  chain_vals = compress_chunks(states, chunks, iv, n_end_blocks)
  tree_levels = math.ceil(math.log2(max(chunks.shape[0], 1)))
  # tree hash
  for i in range(tree_levels):
    chunks = pairwise_concat(chain_vals)
    iv = IV.expand(chunks.shape[0], chunks.shape[1], -1).contiguous()
    flags = create_flags(chunks.shape[0], chunks.shape[1], n_end_blocks, True, i == tree_levels - 1)
    states = create_state(iv, iv, None, end_block_len, n_end_blocks, flags)[:, 0]
    print(f"state before compress()\n{states.numpy()}")
    print(f"blocks before compress()\n{chunks.numpy()}")
    chain_vals = compress_blocks(states, chunks, iv[:, 0])
    print(f"state after compress()\n{chain_vals.numpy()}")
  hash = chain_vals[0].flatten().numpy().tobytes()[:MD_LEN].hex()
  return hash

class TestBLAKE3(unittest.TestCase):
  def test_empty(self):
    actual = tiny_blake3("")
    exp = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    self.assertEqual(actual, exp)

  def test_single_char(self):
    actual = tiny_blake3("a")
    exp = "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
    self.assertEqual(actual, exp)

  def test_short(self):
    actual = tiny_blake3("abcdefg")
    exp = "e2d18d70db12705e1845faf500de1198a5ba1483729d97936f1d2b760968312e"
    self.assertEqual(actual, exp)

  def test_block(self):
    text = "abcd" * (64 // 4)
    actual = tiny_blake3(text)
    exp = '0ef2431cde7c3268b417ea0e8c692dafa8211df7d59f09fdb23df4d73a3bd43d'
    self.assertEqual(actual, exp)

  def test_block_plus_one(self):
    text = "a" * (BLOCK_BYTES + 1)
    actual = tiny_blake3(text)
    exp = 'f345679d9055e53939e92c04ff4f6c9d824b849810d4b598f54baa23336cde99'
    self.assertEqual(actual, exp)

  def test_multiple_blocks(self):
    text = ("a" * BLOCK_BYTES) + ("b" * BLOCK_BYTES)
    actual = tiny_blake3(text)
    exp = 'f27ee0ad41ba8d44a592347ad98c260260d36a59aae97b8e8abc51a3f087bff7'
    self.assertEqual(actual, exp)
    text = ("a" * BLOCK_BYTES) + ("b" * BLOCK_BYTES) + ("c" * BLOCK_BYTES) + ("d" * BLOCK_BYTES)
    actual = tiny_blake3(text)
    exp = 'a9089941f4dc9da1f32e5b037cfe53b2b07feb7ab2ef562444af540333a9e605'
    self.assertEqual(actual, exp)

  def test_full_chunk(self):
    text = "abcd" * (CHUNK_BYTES // 4)
    self.assertEqual(len(text), CHUNK_BYTES)
    actual = tiny_blake3(text)
    exp = '1913be5b8328ba32db70a3f5435e3103b5816b06491a36b948f1f7d29191b177'
    self.assertEqual(actual, exp)

  def test_two_chunks(self):
    text = ("abcd" * (CHUNK_BYTES // 4)) * 2
    actual = tiny_blake3(text)
    exp = '916e101f75a1f6e8d5485d1d983d804051f6b92fff635c2b2f5282864f51853b'
    self.assertEqual(actual, exp)

  def test_odd_chunks(self):
    text = ("abcd" * (CHUNK_BYTES // 4)) * 9
    actual = tiny_blake3(text)
    exp = '25e5cebf882b2b65eb56e881d2ba69fd92dc6e58c1a8d94bdc3d04229e83b553'
    self.assertEqual(actual, exp)

  def test_large(self):
    pass

if __name__ == "__main__":
  # unittest.main()
  specific_test = unittest.TestLoader().loadTestsFromName('test_two_chunks', TestBLAKE3)
  unittest.TextTestRunner().run(specific_test)
