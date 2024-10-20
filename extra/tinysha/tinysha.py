from typing import Tuple, Union
import unittest
import random
import hashlib

from math import prod
from tinygrad.dtype import dtypes
from tinygrad import Tensor
from tinygrad.helpers import tqdm

def tiny_keccak(self: Tensor, cfg: Union[str, Tuple[int, int]] = "sha3_256"):
  tkwargs = dict(device=self.device, dtype=dtypes.uint64)
  rot_offsets = [44, 43, 21, 14, 28, 20, 3, 45, 61, 1, 6, 25, 8, 18, 27, 36, 10, 15, 56, 62, 55, 39, 41, 2]
  rot_offsets_vecs = Tensor([[0, 1]] + [ [1<<v, 1<<(64-v)] for v in rot_offsets ], **tkwargs).transpose()
  reorder_matrix = Tensor([0, 6, 12, 18, 24, 3, 9, 10, 16, 22, 1, 7, 13, 19, 20, 4, 5, 11, 17, 23, 2, 8, 14, 15, 21], **tkwargs).one_hot(25)
  round_const_masks = Tensor([ 1, 0x8082, 0x800000000000808a, 0x8000000080008000, 0x808b, 0x80000001, 0x8000000080008081, 0x8000000000008009, 0x8a,
  0x88, 0x80008009, 0x8000000a, 0x8000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
  0x800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x80000001, 0x8000000080008008 ], **tkwargs).unsqueeze(1).pad((None, (0, 24)))

  rate, dsbyte = { "sha3_224": (144, 0x06), "sha3_256": (136, 0x06), "sha3_384": (104, 0x06), "sha3_512": (72, 0x06) }.get(cfg, cfg)
  data, lower_shape, nb_out = self.bitcast(dtypes.uint8).reshape(prod(self.shape[:-1]), -1), self.shape[:-1], (200-rate)//2
  data_pad = rate - (data.shape[-1] % rate)
  # pad batches then pad blocks
  data = data.pad((None, (0, data_pad))).reshape(data.shape[0], -1, rate).pad((None, None, (0, 200 - rate))).flatten(1)

  # create pad mask
  lbe = data.shape[1] - 200 + rate - data_pad
  if data_pad == 1: p = [(lbe, 0), (1, dsbyte^0x80), (data.shape[-1] - lbe - 1, 0)]
  else: p = [(lbe, 0), (1, dsbyte), (data.shape[-1] + rate - 202 - lbe, 0), (1, 0x80), (200 - rate, 0)]
  pad_mask = Tensor.cat(*(Tensor(v, dtype=dtypes.uint8, device=data.device).expand(l) for l, v in p))

  data = (data ^ pad_mask).reshape(data.shape[0], -1, 200).bitcast(dtypes.uint64)

  state = Tensor.zeros((data.shape[0], 25), **tkwargs)
  for k in range(data.shape[-2]):
    state = state.xor(data[:,k].reshape(-1,25))
    for i in range(24): # f1600
      p = state.reshape((-1, 5, 5)).transpose(2, 1)
      t1 = p[:,:,0].xor(p[:,:,1]).xor(p[:,:,2]).xor(p[:,:,3]).xor(p[:,:,4]).roll(-1, 1) # xor reduce
      state = state.xor(t1.roll(2, 1).xor((t1 << 1) ^ (t1 >> 63)).unsqueeze(2).expand((-1,-1,5)).transpose(2, 1).flatten(1))
      state = reorder_matrix.where(state.unsqueeze(-1).expand((-1, -1, 25)).transpose(2, 1), Tensor.zeros(*state.shape, 25, **tkwargs)).sum(2)
      state = state.mul(rot_offsets_vecs[0]).xor(state.div(rot_offsets_vecs[1], upcast=False)).reshape((-1,5,5))
      state = state.xor(state.roll(shifts=-1, dims=2).xor(-1).bitwise_and(state.roll(shifts=-2, dims=2))).flatten(1) ^ round_const_masks[i]
  return state.bitcast(dtypes.uint8)[:,:nb_out].reshape(*lower_shape, nb_out)

class TestKeccak(unittest.TestCase):
  def setUp(self) -> None: random.seed(1337)

  def test_shape_keeping(self):
    s = (1, 2, 3, 4)
    for i in range(len(s)):
      si = s[i:]
      out_shape = tiny_keccak(Tensor.randint(*si, high=255, dtype=dtypes.uint8)).shape
      self.assertTupleEqual(si[:-1], out_shape[:-1])

  def test_sha3_224(self): self._test_preset("sha3_224", [143, 144])
  def test_sha3_256(self): self._test_preset("sha3_256", [135, 136])
  def test_sha3_384(self): self._test_preset("sha3_384", [103, 104])
  def test_sha3_512(self): self._test_preset("sha3_512", [71, 72])
  def _test_preset(self, name: str, special_sizes: list[int]):
    hasher: type[hashlib._Hash] = getattr(hashlib, name)

    for n in tqdm(special_sizes + [1, 128]):
      a, b = random.randbytes(n), random.randbytes(n)

      ha_ref, hb_ref = hasher(a).digest(), hasher(b).digest()
      tres = tiny_keccak(Tensor.stack(*(Tensor(d) for d in (a, b))), name)
      ha, hb = tres[0].numpy().tobytes(), tres[1].numpy().tobytes()

      self.assertEqual(ha_ref, ha)
      self.assertEqual(ha_ref, tiny_keccak(Tensor(a), name).numpy().tobytes())
      self.assertEqual(hb_ref, hb)

if __name__ == "__main__":
  unittest.main()