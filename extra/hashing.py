from tinygrad import Variable
from tinygrad.helpers import fetch, getenv
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit

def _esch256_chunk(states: list[Tensor], chunk: Tensor) -> list[Tensor]:
  chunk = chunk.bitcast(dtypes.uint32).contiguous()

  tx = ell(chunk[:, 0] ^ chunk[:, 2])
  ty = ell(chunk[:, 1] ^ chunk[:, 3])

  for i in range(0, 4, 2):
    states[i] = states[i] ^ chunk[:, i] ^ ty
    states[i + 1] = states[i + 1] ^ chunk[:, i + 1] ^ tx

  for i in range(4, 6, 2):
    states[i] = states[i] ^ ty
    states[i + 1] = states[i + 1] ^ tx

  return states
def _esch256_last_chunk(states: list[Tensor], chunk: Tensor) -> list[Tensor]:
  if chunk.shape[-1] < 16:
    chunk = chunk.pad((None, (0, 1)), value=0x80).pad((None, (0, 16 - (chunk.shape[-1] + 1))))
    states[5] = states[5] ^ (1 << 24)
  else:
    states[5] = states[5] ^ (2 << 24)
  return _esch256_chunk(states, chunk)
def esch256(msg: Tensor) -> Tensor:
  states = list(Tensor.zeros((msg.shape[0], 12), dtype=dtypes.uint32, device=msg.device).split(1, dim=1))

  msg_chunks = msg.shape[1] // 16
  for i in range(msg_chunks - 1):
    states = _esch256_chunk(states, msg[:, i*16:(i+1)*16])
    states = sparkle(states, 6, 7)

  states = _esch256_last_chunk(states, msg[:, (msg_chunks-1)*16:])
  states = sparkle(states, 6, 11)

  out = Tensor.cat(*states, dim=1)[:, :4]
  states = sparkle(states, 6, 7)
  out = out.cat(Tensor.cat(*states, dim=1)[:, :4], dim=1)

  return out

def rotr(x: Tensor | int, k: int) -> Tensor | int:
  if k == 0: return x
  return (x >> k) | (x << (32 - k))
def ell(x: Tensor) -> Tensor:
  return rotr(x ^ (x << 16), 16)

def _alzette_round(x: Tensor, y: Tensor, s: int, t: int, c: int) -> tuple[Tensor, Tensor]:
  x = x + rotr(y, s)
  y = y ^ rotr(x, t)
  x = x ^ c
  return x.contiguous(), y.contiguous()
def alzette(x: Tensor, y: Tensor, c: int) -> tuple[Tensor, Tensor]:
  x, y = _alzette_round(x, y, 31, 24, c)
  x, y = _alzette_round(x, y, 17, 17, c)
  x, y = _alzette_round(x, y, 0, 31, c)
  x, y = _alzette_round(x, y, 24, 16, c)
  return x, y

def diffusion(states: list[Tensor], nb: int) -> list[Tensor]:
  tx = x0 = states[0]
  ty = y0 = states[1]
  for i in range(2, nb, 2):
    tx = tx ^ states[i]
    ty = ty ^ states[i + 1]
  tx = ell(tx)
  ty = ell(ty)

  for i in range(2, nb, 2):
    states[i - 2] = states[i + nb] ^ states[i] ^ ty
    states[i + nb] = states[i]
    states[i - 1] = states[i + nb + 1] ^ states[i + 1] ^ tx
    states[i + nb + 1] = states[i + 1]
  states[nb - 2] = states[nb] ^ x0 ^ ty
  states[nb] = x0
  states[nb - 1] = states[nb + 1] ^ y0 ^ tx
  states[nb + 1] = y0

  return states

SPARKLE_CONSTANTS = [
  0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738,
  0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D
]
def sparkle(states: list[Tensor], nb: int, ns: int) -> list[Tensor]:
  for i in range(ns):
    states[1] = states[1] ^ SPARKLE_CONSTANTS[i % len(SPARKLE_CONSTANTS)]
    states[3] = states[3] ^ i

    for j in range(0, nb * 2, 2):
      states[j], states[j + 1] = alzette(states[j], states[j + 1], SPARKLE_CONSTANTS[j >> 1])

    states = diffusion(states, nb)
  return states

def test_kat(kat, i):
  kat = kat.split("\n")
  msg = kat[1].split(" = ")[1]
  md = kat[2].split(" = ")[1].lower()
  msg_hex = Tensor([bytes.fromhex(msg)], dtype=dtypes.uint8)
  mdc = esch256(msg_hex).data().tobytes().hex()
  assert md == mdc, f"failed on KAT {i}, {md} != {mdc}"

if __name__ == "__main__":
  import zipfile
  from urllib.request import Request
  kats_zip = fetch(Request("https://csrc.nist.gov/CSRC/media/Projects/lightweight-cryptography/documents/finalist-round/updated-submissions/sparkle.zip"))
  kats = zipfile.ZipFile(kats_zip).open("sparkle/Implementations/crypto_hash/esch256v2/LWC_HASH_KAT_256.txt").read().decode()
  kats = kats.split("\n\n")

  if kat_idx := getenv("KAT", 0):
    test_kat(kats[kat_idx], kat_idx)
  else:
    for i, kat in enumerate(kats):
      if not kat: continue
      test_kat(kat, i)
