from tinygrad import Variable
from tinygrad.helpers import fetch, getenv
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

# def _esch256_chunk(states: list[Tensor], chunk: Tensor) -> list[Tensor]:
#   chunk = chunk.bitcast(dtypes.uint32).contiguous()
#
#   tx = ell(chunk[:, 0] ^ chunk[:, 2])
#   ty = ell(chunk[:, 1] ^ chunk[:, 3])
#
#   for i in range(0, 4, 2):
#     states[i] = states[i] ^ chunk[:, i] ^ ty
#     states[i + 1] = states[i + 1] ^ chunk[:, i + 1] ^ tx
#
#   for i in range(4, 6, 2):
#     states[i] = states[i] ^ ty
#     states[i + 1] = states[i + 1] ^ tx
#
#   return states
# def _esch256_last_chunk(states: list[Tensor], chunk: Tensor) -> list[Tensor]:
#   if chunk.shape[-1] < 16:
#     chunk = chunk.pad((None, (0, 1)), value=0x80).pad((None, (0, 16 - (chunk.shape[-1] + 1))))
#     states[5] = states[5] ^ (1 << 24)
#   else:
#     states[5] = states[5] ^ (2 << 24)
#   return _esch256_chunk(states, chunk)
# def esch256(msg: Tensor) -> Tensor:
#   states = list(Tensor.zeros((msg.shape[0], 12), dtype=dtypes.uint32, device=msg.device).split(1, dim=1))
#
#   msg_chunks = msg.shape[1] // 16
#   for i in range(msg_chunks - 1):
#     states = _esch256_chunk(states, msg[:, i*16:(i+1)*16])
#     states = sparkle(states, 6, 7)
#
#   states = _esch256_last_chunk(states, msg[:, (msg_chunks-1)*16:])
#   states = sparkle(states, 6, 11)
#
#   out = Tensor.cat(*states, dim=1)[:, :4]
#   states = sparkle(states, 6, 7)
#   out = out.cat(Tensor.cat(*states, dim=1)[:, :4], dim=1)
#
#   return out
#
# def rotr(x: Tensor | int, k: int) -> Tensor | int:
#   if k == 0: return x
#   return (x >> k) | (x << (32 - k))
# def ell(x: Tensor) -> Tensor:
#   return rotr(x ^ (x << 16), 16)
#
# def _alzette_round(x: Tensor, y: Tensor, s: int, t: int, c: int) -> tuple[Tensor, Tensor]:
#   x = x + rotr(y, s)
#   y = y ^ rotr(x, t)
#   x = x ^ c
#   return x.contiguous(), y.contiguous()
# def alzette(x: Tensor, y: Tensor, c: int) -> tuple[Tensor, Tensor]:
#   x, y = _alzette_round(x, y, 31, 24, c)
#   x, y = _alzette_round(x, y, 17, 17, c)
#   x, y = _alzette_round(x, y, 0, 31, c)
#   x, y = _alzette_round(x, y, 24, 16, c)
#   return x, y
#
# def diffusion(states: list[Tensor], nb: int) -> list[Tensor]:
#   tx = x0 = states[0]
#   ty = y0 = states[1]
#   for i in range(2, nb, 2):
#     tx = tx ^ states[i]
#     ty = ty ^ states[i + 1]
#   tx = ell(tx)
#   ty = ell(ty)
#
#   for i in range(2, nb, 2):
#     states[i - 2] = states[i + nb] ^ states[i] ^ ty
#     states[i + nb] = states[i]
#     states[i - 1] = states[i + nb + 1] ^ states[i + 1] ^ tx
#     states[i + nb + 1] = states[i + 1]
#   states[nb - 2] = states[nb] ^ x0 ^ ty
#   states[nb] = x0
#   states[nb - 1] = states[nb + 1] ^ y0 ^ tx
#   states[nb + 1] = y0
#
#   return states
#
# SPARKLE_CONSTANTS = [
#   0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738,
#   0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D
# ]
# def sparkle(states: list[Tensor], nb: int, ns: int) -> list[Tensor]:
#   for i in range(ns):
#     states[1] = states[1] ^ SPARKLE_CONSTANTS[i % len(SPARKLE_CONSTANTS)]
#     states[3] = states[3] ^ i
#
#     for j in range(0, nb * 2, 2):
#       states[j], states[j + 1] = alzette(states[j], states[j + 1], SPARKLE_CONSTANTS[j >> 1])
#
#     states = diffusion(states, nb)
#   return states
#
# def test_kat(kat, i):
#   kat = kat.split("\n")
#   msg = kat[1].split(" = ")[1]
#   md = kat[2].split(" = ")[1].lower()
#   msg_hex = Tensor([bytes.fromhex(msg)], dtype=dtypes.uint8)
#   mdc = esch256(msg_hex).data().tobytes().hex()
#   assert md == mdc, f"failed on KAT {i}, {md} != {mdc}"
#
# if __name__ == "__main__":
#   import zipfile
#   from urllib.request import Request
#   kats_zip = fetch("https://csrc.nist.gov/CSRC/media/Projects/lightweight-cryptography/documents/finalist-round/updated-submissions/sparkle.zip")
#   kats = zipfile.ZipFile(kats_zip).open("sparkle/Implementations/crypto_hash/esch256v2/LWC_HASH_KAT_256.txt").read().decode()
#   kats = kats.split("\n\n")
#
#   if kat_idx := getenv("KAT", 0):
#     test_kat(kats[kat_idx], kat_idx)
#   else:
#     for i, kat in enumerate(kats):
#       if not kat: continue
#       test_kat(kat, i)

def rotl64(x: Tensor | int, k: int) -> Tensor | int:
  if k == 0: return x
  return (x << k) | (x >> (64 - k))

RNDC = [
  0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
  0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
  0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
  0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
  0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
  0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
  0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
ROTC = [
  1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
  27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44,
]
PERM = [
  10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
  15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1,
]
def _keccak_round(state: list[Tensor], rndc: int) -> list[Tensor]:
  # theta
  c = [state[i + 0] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20] for i in range(5)]
  c = [s.contiguous() for s in c]

  for i in range(5):
    t = c[(i - 1) % 5] ^ rotl64(c[(i + 1) % 5], 1)
    for j in range(0, 25, 5):
      state[i + j] = state[i + j] ^ t

  # rho & pi
  t = state[1]
  for i in range(24):
    t2 = state[j := PERM[i]]
    state[j] = rotl64(t, ROTC[i])
    t = t2

  # chi
  for j in range(0, 25, 5):
    b = [state[j + i] for i in range(5)]
    for i in range(5):
      state[j + i] = state[j + i] ^ ((~b[(i + 1) % 5]) & b[(i + 2) % 5])

  # iota
  state[0] = state[0] ^ rndc

  state = [s.contiguous() for s in state]

  return state
def _keccakf1600(state: list[Tensor]) -> list[Tensor]:
  for round in range(24):
    state = _keccak_round(state, RNDC[round])
  return state
def kaccak(rate: int, capacity: int, msg: Tensor, delim: int, output_len: int) -> Tensor:
  assert rate + capacity == 1600 and rate % 8 == 0, "rate + capacity must be 1600 and rate must be a multiple of 8"
  brate, wrate = rate // 8, rate // 64

  state = list(Tensor.zeros((msg.shape[0], 25), dtype=dtypes.uint64, device=msg.device).split(1, dim=1))

  # pad up to multiple of brate
  msg = msg.pad((None, (0, 1)), value=delim)
  msg = msg.pad((None, (0, brate - (msg.shape[1] % brate) - 1)), value=0)
  msg = msg.pad((None, (0, 1)), value=0x80)
  chunks = msg.bitcast(dtypes.uint64).split(wrate, dim=1)

  for chunk in chunks:
    if chunk.shape[1] == 0: break
    for j in range(chunk.shape[1]):
      state[j] = state[j] ^ chunk[:, j]
    state = _keccakf1600(state)

  obsize = min(brate, output_len)
  out = Tensor.cat(*state, dim=1).bitcast(dtypes.uint8)[:, :obsize]
  output_len -= obsize
  while output_len > 0:
    state = _keccakf1600(state)
    obsize = min(brate, output_len)
    out = out.cat(Tensor.cat(*state, dim=1).bitcast(dtypes.uint8)[:, :obsize], dim=1)
    output_len -= obsize
  return out

def shake128(msg: Tensor, output_len: int = 16) -> Tensor:
  return kaccak(1344, 256, msg, 0x1f, output_len)

def string_to_uint8_tensor(s: str) -> Tensor:
  b = s.encode()
  return Tensor([b], dtype=dtypes.uint8)

if __name__ == "__main__":
  s = string_to_uint8_tensor("")
  print(s.data().tobytes().hex())
  a = shake128(s)
  print(a.data().tobytes().hex())
