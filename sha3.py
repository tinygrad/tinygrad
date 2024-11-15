import hashlib
from typing import Union
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
import unittest

height, width = 5, 5
rate_bytes = 136
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


def absorb_block(state: Tensor, block: bytearray) -> Tensor:
  for idx in range(0, rate_bytes, 8):
    block_val = block[idx: idx + 8]
    uint64 = int.from_bytes(block_val, "little")
    state[idx // 8] = state[idx // 8] ^ uint64
  return state.view(height, width)


def rotl64(n: Union[int, Tensor], shifts: int):
  lshift = (n << shifts) & ((1 << 64) - 1)
  rshift = n >> (64 - shifts)
  return lshift | rshift


def theta(state: Tensor) -> Tensor:
  C = state[0] ^ state[1] ^ state[2] ^ state[3] ^ state[4]
  left = C.roll(shifts=-4, dims=0)
  right = C.roll(shifts=-1, dims=0)
  rotated = rotl64(right, 1)
  D = left ^ rotated
  return state ^ D


def rho_pi(state: Tensor) -> Tensor:
  """
  Combined rho and pi steps
  """
  rho_pi_state = Tensor.zeros(
      height * width, dtype=dtypes.uint64).contiguous()
  for i in range(height * width):
    x, y = divmod(i, width)
    rotated = rotl64(state[x][y], rho_offsets[x][y])
    pos = pi_offsets[x][y]
    rho_pi_state[pos] = rotated
  return rho_pi_state.view(height, width)


def chi(state: Tensor) -> Tensor:
  shift1 = state.roll(shifts=-1, dims=1)
  shift2 = state.roll(shifts=-2, dims=1)
  # ~shift1
  return state ^ ((shift1 ^ Tensor.full_like(shift1, -1)) & shift2)


def iota(state: Tensor, round_idx: int) -> Tensor:
  state[0][0] ^= iota_round_constants[round_idx]
  return state


def keccak_round(state: Tensor, round_idx: int) -> Tensor:
  """
  One round of Keccak-f
  """
  state = theta(state)
  rho_pi_state = rho_pi(state)
  chi_state = chi(rho_pi_state)
  return iota(chi_state, round_idx)


def keccak_f(state: Tensor) -> Tensor:
  """
  Keccack-f function
  """
  for round_idx in range(24):
    state = keccak_round(state, round_idx)
  return state


def pad(state: Tensor, pad_idx_bytes: int):
  # adjust padding offsets to get the correct byte inside the uint64
  padpoint_uint = (pad_idx_bytes % rate_bytes) // 8
  padpoint_shift = ((pad_idx_bytes % rate_bytes) % 8) * 8
  state[padpoint_uint] = state[padpoint_uint] ^ (0x06 << padpoint_shift)
  state[(rate_bytes // 8) - 1] = state[(rate_bytes // 8) - 1] ^ (0x80 << 56)
  return state


def visualize_state(state: Tensor):
  """
  Print the state for debugging
  """
  for x in range(height):
    for y in range(width):
      print(f"{state.view(height, width)[x][y].numpy():016x}", end=" ")
    print()


def digest(state: Tensor) -> str:
  return state.flatten().numpy().tobytes()[:md_len].hex()


def tinygrad_sha3_256(message: str) -> str:
  msg_bytes = bytearray(message.encode("utf-8"))
  state = Tensor.zeros(height * width, dtype=dtypes.uint64).contiguous()

  # absorb
  for block_idx in range(0, len(msg_bytes), rate_bytes):
    block = msg_bytes[block_idx: block_idx + rate_bytes]
    state = absorb_block(state.view(height * width), block)
    bytes_absorbed = min(block_idx + 1 * rate_bytes, len(msg_bytes))
    if bytes_absorbed % rate_bytes == 0:
      state = keccak_f(state)

  # squeeze
  state = state.view(height * width)
  state = pad(state, min(len(msg_bytes) % rate_bytes, rate_bytes - 1))
  state = keccak_f(state.view(height, width))

  # digest
  return digest(state)


def hashlib_sha3_256(message: str) -> str:
  """
  Compare against hashlib
  """
  hash_obj = hashlib.sha3_256()
  message_bytes = message.encode("utf-8")
  hash_obj.update(message_bytes)
  hash_hex = hash_obj.hexdigest()
  return hash_hex


class TestSHA3(unittest.TestCase):
  def test_rate_bytes_length(self):
    """Test string exactly rate_bytes length"""
    text = "a" * rate_bytes
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_empty_string(self):
    """Test SHA3-256 hash of empty string"""
    self.assertEqual(tinygrad_sha3_256(""), hashlib_sha3_256(""))

  def test_short_string(self):
    """Test SHA3-256 hash of short string"""
    self.assertEqual(tinygrad_sha3_256("abc"), hashlib_sha3_256("abc"))

  def test_longer_than_rate_bytes(self):
    """Test string longer than rate_bytes"""
    text = "a" * (rate_bytes + 1)
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_multiple_rate_bytes(self):
    """Test string multiple of rate_bytes"""
    text = "c" * (rate_bytes * 2)
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_multiple_rate_bytes_2(self):
    """Test string multiple of rate_bytes different characters"""
    text = ("c" * rate_bytes) + ("a" * rate_bytes)
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))

  def test_special_chars(self):
    """Test string with special characters"""
    self.assertEqual(
        tinygrad_sha3_256("!@#$%^&*()"), hashlib_sha3_256("!@#$%^&*()")
    )

  def test_unicode(self):
    """Test string with unicode characters"""
    self.assertEqual(
        tinygrad_sha3_256("Hello 世界"), hashlib_sha3_256("Hello 世界")
    )

  def test_long_mixed_string(self):
    """Test long string"""
    text = "Hello123!@#$" * 100 + "世界" * 50 + "αβγδ" * 25 + "🌟🌍" * 10
    self.assertEqual(tinygrad_sha3_256(text), hashlib_sha3_256(text))


if __name__ == "__main__":
  unittest.main()
