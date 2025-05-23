import math
from tinygrad import Variable
from tinygrad.helpers import fetch, getenv
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

def rotl64(x: Tensor | int, k: int) -> Tensor | int:
  if k == 0: return x
  return (x << k) | (x >> (64 - k))

RNDC = Tensor([
  0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
  0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
  0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
  0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
  0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
  0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
  0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
], dtype=dtypes.uint64)
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
def kaccak(capacity: int, msg: Tensor, delim: int, output_len: int) -> Tensor:
  rate = 1600 - capacity
  brate, wrate = rate // 8, rate // 64

  state = list(Tensor.zeros((msg.shape[0], 25), dtype=dtypes.uint64, device=msg.device).contiguous().split(1, dim=1))

  # pad up to multiple of brate
  msg = msg.pad((None, (0, 1)), value=delim)
  if msg.shape[1] % 168 != 0: msg = msg.pad((None, (0, brate - (msg.shape[1] % brate))), value=0)
  chunks = msg.contiguous().bitcast(dtypes.uint64).split(wrate, dim=1)

  for i,chunk in enumerate(chunks):
    for j in range(chunk.shape[1]):
      state[j] = state[j] ^ chunk[:, j]
    if i == len(chunks) - 1:
      state[j] = state[j] ^ 0x8000000000000000
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
  return kaccak(256, msg, 0x1f, output_len)

def string_to_uint8_tensor(s: str) -> Tensor:
  b = s.encode()
  return Tensor([b], dtype=dtypes.uint8)

def test_kat(kat, i):
  print(f"running KAT {i}")
  kat = kat.split("\r\n")
  mlen = int(kat[0].split(" = ")[1])
  msg = kat[1].split(" = ")[1]
  md = kat[2].split(" = ")[1].lower()
  msgt = Tensor([bytes.fromhex(msg)], dtype=dtypes.uint8)[:, :mlen]
  mdc = shake128(msgt).data().tobytes().hex()
  assert md == mdc, f"failed on KAT {i}, {md} != {mdc}"

if __name__ == "__main__":
  import zipfile
  kats_zip = fetch("https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/sha3/shakebytetestvectors.zip")
  kats = zipfile.ZipFile(kats_zip).open("SHAKE128ShortMsg.rsp").read().decode()
  kats = kats.split("\r\n\r\n")[2:-1]

  if kat_idx := getenv("KAT", 0):
    test_kat(kats[kat_idx], kat_idx)
  else:
    for i, kat in enumerate(kats[getenv("KAT_START", 0):getenv("KAT_END", len(kats))]):
      if not kat: continue
      test_kat(kat, i)
