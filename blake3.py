from typing import Dict, List
import unittest

import numpy as np

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

# BLAKE3 algorithm
# - call hasher.update() with bytes
# - check against the current chunk state
# - want = number of bytes to complete a full chunk
# - take = min(want, len(input_bytes))
# - if chunk_state.num_bytes === 1024 call compress()
# - call add_chunk_chaining_value()
# - create the state matrix 4x4 32 bit words
# [ chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3] ]
# [ chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7] ]
# [       IV[0],             IV[1],             IV[2],          IV[3]         ]
# [     counter[0],       counter[1],     block_len,            flags         ]
# - first 8 words are from the 8 word chaining value
# - if this is the first chunk then the chaining value is taken from IV
# - pass this through 7 round keyed permutation steps to mix up the state matrix
# - return the state matrix
# - update the chaining value
# - purpose of the chaining value is to make each chunk dependent on prior chunks
# - changing one chunk propogates changes through the entire hash (so-called "avalanche" effect)
# - if a subtree can be formed, create a parent chaining value

# params
MD_LEN = 32
HEIGHT, WIDTH = 4, 4
KEY_LEN = 32
BLOCK_LEN = 64
CHUNK_LEN = 1024
# flags
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
KEYED_HASH = 1 << 4
# input chaining values
IV = Tensor([
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
], dtype=dtypes.uint32)
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]


def mask32(x: int) -> int:
  return x & 0xFFFFFFFF


def add32(x: int, y: int) -> int:
  return mask32(x + y)


def rightrotate32(x: int, n: int) -> int:
  return mask32(x << (32 - n)) | (x >> n)


def mix(states: Tensor, a: int, b: int, c: int, d: int, mx: int, my: int) -> Tensor:
  states[:, a] = add32(states[:, a], add32(states[:, b], mx))
  states[:, d] = rightrotate32(states[:, d] ^ states[:, a], 16)
  states[:, c] = add32(states[:, c], states[:, d])
  states[:, b] = rightrotate32(states[:, b] ^ states[:, c], 12)
  states[:, a] = add32(states[:, a], add32(states[:, b], my))
  states[:, d] = rightrotate32(states[:, d] ^ states[:, a], 8)
  states[:, c] = add32(states[:, c], states[:, d])
  states[:, b] = rightrotate32(states[:, b] ^ states[:, c], 7)


def round(states: Tensor, chunks: Tensor) -> Tensor:
  # Mix the columns.
  mix(states, 0, 4, 8, 12, chunks[:, 0], chunks[:, 1])
  mix(states, 1, 5, 9, 13, chunks[:, 2], chunks[:, 3])
  mix(states, 2, 6, 10, 14, chunks[:, 4], chunks[:, 5])
  mix(states, 3, 7, 11, 15, chunks[:, 6], chunks[:, 7])
  # Mix the diagonals.
  mix(states, 0, 5, 10, 15, chunks[:, 8], chunks[:, 9])
  mix(states, 1, 6, 11, 12, chunks[:, 10], chunks[:, 11])
  mix(states, 2, 7, 8, 13, chunks[:, 12], chunks[:, 13])
  mix(states, 3, 4, 9, 14, chunks[:, 14], chunks[:, 15])


def permute(chunks: Tensor) -> Tensor:
  original = chunks.clone()
  for i in range(16):
    chunks[:, i] = original[:, MSG_PERMUTATION[i]]
  return chunks


def create_state(chain: Tensor, iv: Tensor, counts: Tensor, block_len: Tensor, flags: Tensor):
  states = chain.cat(iv, counts, block_len, flags, dim=-1)
  return states


def round_permute(states: Tensor, chunks: Tensor):
  round(states, chunks)  # round 1
  permute(chunks)
  round(states, chunks)  # round 2
  permute(chunks)
  round(states, chunks)  # round 3
  permute(chunks)
  round(states, chunks)  # round 4
  permute(chunks)
  round(states, chunks)  # round 5
  permute(chunks)
  round(states, chunks)  # round 6
  permute(chunks)
  round(states, chunks)  # round 7


def compress(
    chunks: Tensor,
    chain_vals: Tensor,
    iv: Tensor,
    counts: Tensor,
    block_len: Tensor,
    flags: Tensor,
) -> Tensor:
  states = create_state(chain_vals, iv, counts, block_len, flags)
  round_permute(states, chunks)
  for i in range(8):
    states[:, i] = states[:, i] ^ states[:, i + 8]
    states[:, i + 8] = states[:, i + 8] ^ chain_vals[:, i]
  return states


def chunk_bytes(text_bytes: bytes) -> Tensor:
  n_bytes = len(text_bytes)
  n_chunks = max((n_bytes + CHUNK_LEN - 1) // CHUNK_LEN, 1)
  chunks = Tensor.zeros(n_chunks, CHUNK_LEN, dtype=dtypes.uint8).contiguous()
  for i in range(0, max(len(text_bytes), 1), CHUNK_LEN):
    chunk = text_bytes[i:i + CHUNK_LEN].ljust(CHUNK_LEN, b"\0")
    chunks[i // CHUNK_LEN] = Tensor(chunk, dtype=dtypes.uint8)
  return chunks


def pair_chaining_values(chain_vals: Tensor) -> Tensor:
  """
  Pairs the chaining values and concatenates them.
  """
  n_chunks = chain_vals.shape[0]
  assert chain_vals.shape == (n_chunks, 8)
  return chain_vals.reshape(-1 // 2, 2, 4).flatten(1)


def tinygrad_blake3(text: str) -> str:
  """
  Uses a recursive divide and conquer approach to compress the input text in parallel.
  """
  text_bytes = text.encode("utf-8") if text else b""
  chunks = chunk_bytes(text_bytes)
  last_len = len(text_bytes) % CHUNK_LEN
  n_chunks = chunks.shape[0]
  iv = IV.expand(n_chunks, -1)
  chain = iv
  cnt = 0
  counts = Tensor.arange(cnt, n_chunks, 1, dtype=dtypes.uint32).reshape(-1, 1)
  counts = counts.cat(Tensor.zeros_like(counts, dtype=dtypes.uint32), dim=-1)
  lens = Tensor([[BLOCK_LEN] * (n_chunks - 1) +
                [last_len]], dtype=dtypes.uint32)
  flags = Tensor.zeros(n_chunks, 1, dtype=dtypes.uint32).contiguous()
  flags[0] = flags[0] + CHUNK_START
  flags[-1] = flags[-1] + CHUNK_END
  compressed = compress(chunks, chain, iv[:, :4], counts, lens, flags)
  cnt += n_chunks
  chain = compressed[:, :8]
  flags[0] = flags[0] + ROOT
  # extra permutes to return to the original chunk state
  permute(chunks)
  permute(chunks)
  compressed = compress(chunks, iv, iv[:, :4], counts, lens, flags)
  return compressed[0].flatten().numpy().tobytes()[:MD_LEN].hex()


class TestBLAKE3(unittest.TestCase):
  def test_text_to_chunks(self):
    """Test converting input text to chunks"""
    text = b""
    expected = Tensor([b"\0" * CHUNK_LEN], dtype=dtypes.uint8)
    actual = chunk_bytes(text)
    np.testing.assert_equal(actual.numpy(), expected.numpy())
    text = b"a" * (CHUNK_LEN - 1)
    expected = Tensor([b"a" * (CHUNK_LEN - 1) + b"\0"], dtype=dtypes.uint8)
    actual = chunk_bytes(text)
    np.testing.assert_equal(actual.numpy(), expected.numpy())
    text = b"a" * CHUNK_LEN + b"b" * CHUNK_LEN
    expected = Tensor([b"a" * CHUNK_LEN, b"b" * CHUNK_LEN], dtype=dtypes.uint8)
    actual = chunk_bytes(text)
    np.testing.assert_equal(actual.numpy(), expected.numpy())

  def _setup_test_inputs(self, text: str) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    text_bytes = text.encode("utf-8") if text else b""
    final_len = len(text_bytes) % CHUNK_LEN
    chunks = chunk_bytes(text_bytes)
    n_chunks = chunks.shape[0]
    chaining_values = IV.expand(1, -1)
    iv = chaining_values[:, :4]
    counts = Tensor.arange(0, n_chunks, 1, dtype=dtypes.uint32).reshape(-1, 1)
    counts = counts.cat(counts >> 32, dim=-1)
    flags = Tensor.zeros(n_chunks, 1, dtype=dtypes.uint32).contiguous()
    flags[0, 0] = flags[0, 0] + CHUNK_START
    flags[-1, 0] = flags[-1, 0] + CHUNK_END
    block_len = Tensor([final_len], dtype=dtypes.uint32).expand(n_chunks, -1)
    return chunks, n_chunks, chaining_values, iv, counts, block_len, flags

  def test_create_state(self):
    _, _, chaining_values, iv, counts, block_len, flags = self._setup_test_inputs(
        "a")
    states = create_state(chaining_values, iv, counts, block_len, flags)
    expected = Tensor([[1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924,
                      528734635, 1541459225, 1779033703, 3144134277, 1013904242, 2773480762, 0, 0, 1, 3]], dtype=dtypes.uint32)
    np.testing.assert_equal(states.numpy(), expected.numpy())

  def test_round(self):
    """Test the round function with known input and output values"""
    state = Tensor([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225, 1779033703, 3144134277, 1013904242, 2773480762, 0, 0, 1, 3],  # fourth row
                   dtype=dtypes.uint32).unsqueeze(0)
    chunks = Tensor([97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    dtype=dtypes.uint32).unsqueeze(0)
    expected = Tensor([1197540149, 369699509, 4114839138, 3226644656, 2715381399, 1053419351, 925057643, 3011426483, 2698030395, 591305675, 1733393876, 3237318155, 3541682352, 3711187737, 2111353108, 4049535030],
                      dtype=dtypes.uint32)
    round(state, chunks)
    np.testing.assert_equal(state[0].numpy(), expected.numpy())

  def test_permute(self):
    """Test the permute function with known input and output values"""
    chunks = Tensor([1684234849, 6776421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    dtype=dtypes.uint32).unsqueeze(0)
    expected = Tensor([0, 0, 0, 0, 0, 1684234849, 0, 0, 6776421, 0, 0, 0, 0, 0, 0, 0],
                      dtype=dtypes.uint32)
    permute(chunks)
    np.testing.assert_equal(chunks[0].numpy(), expected.numpy())

  def test_round_permute(self):
    """Test the round_permute function with known input and output values"""
    state = Tensor([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225, 1779033703, 3144134277, 1013904242, 2773480762, 0, 0, 7, 3],
                   dtype=dtypes.uint32).unsqueeze(0)
    chunks = Tensor([1684234849, 6776421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    dtype=dtypes.uint32).unsqueeze(0)
    round_permute(state, chunks)
    expected_state = Tensor([1505703587, 3043897677, 297679660, 1360396464, 4153763876, 3213303786, 2247926175, 698554836, 2797293838, 1713178191, 2926895767, 2158408814, 615604690, 1510005954, 4016387937, 3875951467],
                            dtype=dtypes.uint32)
    np.testing.assert_equal(state[0].numpy(), expected_state.numpy())

  def test_empty(self):
    actual = tinygrad_blake3("")
    expected = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    self.assertEqual(actual, expected)

  def test_single_char(self):
    actual = tinygrad_blake3("a")
    expected = "17762fddd969a453925d65717ac3eea21320b66b54342fde15128d6caf21215f"
    self.assertEqual(actual, expected)

  def test_single_block(self):
    text = "abcd" * (CHUNK_LEN // 4)
    self.assertEqual(len(text), CHUNK_LEN)
    actual = tinygrad_blake3(text)
    expected = '1913be5b8328ba32db70a3f5435e3103b5816b06491a36b948f1f7d29191b177'
    self.assertEqual(actual, expected)


if __name__ == "__main__":
  unittest.main()
  hash = tinygrad_blake3("a")
  print(hash)
