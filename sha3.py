import hashlib
from typing import List, Tuple, Union

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
import unittest

height, width = 5, 5
n_cells = height * width
rate_bytes = 136
rate_uints = rate_bytes // 8
num_rounds = 24
md_len = 32
rho_offsets = [
    [0, 1, 62, 28, 27],
    [36, 44, 6, 55, 20],
    [3, 10, 43, 25, 39],
    [41, 45, 15, 21, 8],
    [18, 2, 61, 56, 14],
]
pi_offsets = [
    [0, 10, 20, 5, 15],
    [16, 1, 11, 21, 6],
    [7, 17, 2, 12, 22],
    [23, 8, 18, 3, 13],
    [14, 24, 9, 19, 4],
]
iota_round_constants = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]


def message_to_bytearray(message: str) -> Tuple[bytearray, int]:
  encoded = message.encode("utf-8")
  ljust = ((len(encoded) + rate_bytes - 1) // rate_bytes) * rate_bytes
  return (bytearray(encoded.ljust(ljust, b'\0')), len(encoded))


# constants for converting bytes to uint64
POWERS = Tensor([256 ** i for i in range(8)], dtype=dtypes.uint64)
PADDING_UINT = Tensor.zeros((1, 1, n_cells - rate_uints), dtype=dtypes.uint64)


def messages_to_blocks(messages: List[str]) -> Tuple[Tensor, int]:
  """
  Convert a list of equal length messages to a tensor of width x height blocks compatible with the state size.
  returns: ([num_blocks, batch_size, height, width], num_bytes)
  """
  batch_size = len(messages)
  byte_arrays = []
  for msg in messages:
    arr, msg_num_bytes = message_to_bytearray(msg)
    byte_arrays.append(arr)
  msg_bytes = Tensor(byte_arrays, dtype=dtypes.uint8)
  num_blocks = len(byte_arrays[0]) // rate_bytes
  byte_blocks = msg_bytes.reshape(batch_size, num_blocks, rate_uints, 8)
  # convert each block of 8 bytes to a uint64
  uint64_blocks = byte_blocks @ POWERS
  # pad to fit the full state size
  state_padding = PADDING_UINT.expand(batch_size, num_blocks, -1)
  padded_blocks = Tensor.cat(uint64_blocks, state_padding, dim=-1)
  padded_blocks = padded_blocks.permute(1, 0, 2)
  padded_blocks = padded_blocks.reshape(num_blocks, batch_size, height, width)
  return padded_blocks, msg_num_bytes


def rotl64(n: Union[int, Tensor], shifts: int):
  lshift = (n << shifts) & ((1 << 64) - 1)
  rshift = n >> (64 - shifts)
  return lshift | rshift


def theta(states: Tensor) -> Tensor:
  C = states[:, 0] ^ states[:, 1] ^ states[:, 2] ^ states[:, 3] ^ states[:, 4]
  left = C.roll(shifts=-4, dims=1)
  right = C.roll(shifts=-1, dims=1)
  rotated = rotl64(right, 1)
  D = left ^ rotated
  return states ^ D.unsqueeze(1)


def rho_pi(states: Tensor) -> Tensor:
  batch_size = states.shape[0]
  rho_pi_state = Tensor.zeros(
      (batch_size, n_cells), dtype=dtypes.uint64).contiguous()
  for i in range(n_cells):
    x, y = divmod(i, width)
    rho_offset = rho_offsets[x][y]
    rotated = rotl64(states[:, x, y], rho_offset)
    pi_offset = pi_offsets[x][y]
    rho_pi_state[:, pi_offset] = rotated
  return rho_pi_state.view(batch_size, height, width)


def chi(states: Tensor) -> Tensor:
  shift1 = states.roll(shifts=-1, dims=2)
  shift2 = states.roll(shifts=-2, dims=2)
  return states ^ ((shift1 ^ Tensor.full_like(shift1, -1)) & shift2)


def iota(states: Tensor, round_idx: int) -> Tensor:
  states[:, 0, 0] ^= iota_round_constants[round_idx]
  return states


def keccak_round(states: Tensor, round_idx: int) -> Tensor:
  """
  One round of batched Keccak-f
  """
  theta_state = theta(states)
  rho_pi_state = rho_pi(theta_state)
  chi_state = chi(rho_pi_state)
  iota_state = iota(chi_state, round_idx)
  return iota_state


def keccak_f(states: Tensor) -> Tensor:
  """
  Batched Keccack-f function
  states: [batch_size, height, width]
  """
  new_states = states
  for round_idx in range(num_rounds):
    new_states = keccak_round(new_states, round_idx)
  return new_states


def pad(states: Tensor, pad_idx_bytes: int):
  """
  Add padding bytes as defined by the SHA3 standard
  """
  # adjust padding offsets to pick out the correct byte inside the uint64
  padpoint_uint = (pad_idx_bytes % rate_bytes) // 8
  padpoint_shift = ((pad_idx_bytes % rate_bytes) % 8) * 8
  x, y = divmod(padpoint_uint, width)
  states[:, x, y] = states[:, x, y] ^ (0x06 << padpoint_shift)
  x, y = divmod((rate_bytes // 8) - 1, width)
  states[:, x, y] = states[:, x, y] ^ (0x80 << 56)
  return states


def digest(state: Tensor) -> str:
  return state.flatten().numpy().tobytes()[:md_len].hex()


def tinygrad_sha3_256(messages: List[str]) -> List[str]:
  """
  Hash a batch of equal length messages using SHA3-256.
  """
  batch_size = len(messages)
  states = Tensor.zeros((batch_size, height, width), dtype=dtypes.uint64).contiguous()
  batched_blocks, msg_bytes = messages_to_blocks(messages)
  n_blocks = batched_blocks.shape[0]
  # absorb phase
  for block_idx in range(0, n_blocks):
    blocks = batched_blocks[block_idx]
    states = states ^ blocks
    if block_idx < n_blocks - 1 or (block_idx == n_blocks - 1 and msg_bytes % rate_bytes == 0):
      states = keccak_f(states)
  # squeeze phase
  states = pad(states, min(msg_bytes % rate_bytes, rate_bytes - 1))
  states = keccak_f(states)
  return [digest(state) for state in states]


def hashlib_sha3_256(messages: List[str]) -> List[str]:
  results = []
  for message in messages:
    hash_obj = hashlib.sha3_256()
    message_bytes = message.encode("utf-8")
    hash_obj.update(message_bytes)
    hash_hex = hash_obj.hexdigest()
    results.append(hash_hex)
  return results


class TestSHA3(unittest.TestCase):
  def test_message_to_bytearray(self):
    """Test converting a message to a bytearray"""
    msg = "a"
    arr, msg_len = message_to_bytearray(msg)
    self.assertEqual(msg_len, 1)
    self.assertEqual(len(arr), rate_bytes)
    self.assertEqual(arr[1:], b'\0' * (rate_bytes - 1))
    msg = "a" * (rate_bytes + 1)
    arr, msg_len = message_to_bytearray(msg)
    self.assertEqual(msg_len, rate_bytes + 1)
    self.assertEqual(arr[rate_bytes + 1:], b'\0' * (rate_bytes - 1))
    self.assertEqual(len(arr), rate_bytes * 2)
    msg = "界"
    arr, msg_len = message_to_bytearray(msg)
    self.assertEqual(msg_len, 3)
    self.assertEqual(len(arr), rate_bytes)
    self.assertEqual(arr[3:], b'\0' * (rate_bytes - 3))

  def test_rate_bytes_length(self):
    """Test string exactly rate_bytes length"""
    text = ["a" * rate_bytes]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_empty_string(self):
    """Test SHA3-256 hash of empty string"""
    text = [""]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_short_string(self):
    """Test SHA3-256 hash of short string"""
    text = ["abc"]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_longer_than_rate_bytes(self):
    """Test string longer than rate_bytes"""
    text = ["a" * (rate_bytes + 1)]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_multiple_rate_bytes(self):
    """Test string multiple of rate_bytes"""
    text = ["c" * (rate_bytes * 2)]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_multiple_rate_bytes_2(self):
    """Test string multiple of rate_bytes different characters"""
    text = [("c" * rate_bytes) + ("a" * rate_bytes)]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_special_chars(self):
    """Test string with special characters"""
    text = ["!@#$%^&*()"]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_unicode(self):
    """Test string with unicode characters"""
    text = ["界"]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_long_mixed_string(self):
    """Test long string"""
    text = ["Hello123!@#$" * 100 + "世界" * 50 + "αβγδ" * 25 + "🌟🌍" * 10]
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_batch(self):
    """Test batching"""
    texts = ['hello' * 100, 'world' * 100, 'tests' * 100, 'reall' * 100, 'cools' * 100] * 100
    tinygrad_hashes = tinygrad_sha3_256(texts)
    hashlib_hashes = hashlib_sha3_256(texts)
    self.assertEqual(tinygrad_hashes, hashlib_hashes)


if __name__ == "__main__":
  unittest.main()
