import unittest
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.qwen4exp import ExpertWeights, expert_slicing, expert_slice_table, _expert_first, _slice_table

@unittest.skipUnless(Device.DEFAULT == "AMD" and getenv("TEST_GPUS", 0) >= 6, "requires six AMD GPUs")
class TestExpertSlices(unittest.TestCase):
  def test_down_projection_partition(self):
    devs = tuple(Device.canonicalize(f"AMD:{i}") for i in range(6))
    if any(Device[d].target[0] not in (11, 12) for d in devs): self.skipTest("requires RDNA3/4")
    rng = np.random.default_rng(18)
    E, D, H, T, k = 12, 256, 640, 5, 3
    quant = rng.integers(-8, 9, size=(E, D, H), dtype=np.int8)
    scales = np.full((E, D, H//32, 1), 1/128, dtype=np.float16)
    packed = np.concatenate((scales.view(np.uint8), quant.reshape(E, D, H//32, 32).view(np.uint8)), axis=-1)
    selections = np.array([rng.choice(E, size=k, replace=False) for _ in range(T)], dtype=np.int32)
    for size in (32, 64, 128, 320, 640):
      with self.subTest(slice_size=size):
        slices = H//size
        parts = expert_slicing(E, slices, len(devs))
        assert parts is not None
        tables = [expert_slice_table(p, E, slices) for p in parts]
        width = len(tables[0][0])
        raw = np.stack([np.concatenate([packed[u//slices, :, (u%slices)*size//32:(u%slices+1)*size//32].ravel()
                                       for u in part]) for part in parts])
        layer = ExpertWeights(E*slices, size, D)
        layer.ggml_type, layer.slices, layer.experts, layer.local_experts = 8, slices, E, len(parts[0])
        layer.weight = Tensor(raw).shard(devs, axis=0).flatten().bitcast(dtypes.uint16).contiguous().realize()
        layer.first, layer.slice_table = _expert_first((0,)*len(devs), devs), _slice_table(E, slices, devs)
        sel, _ = layer.select_slices(Tensor(selections[None]).to(devs), Tensor.ones(1, T, k).to(devs))
        sel = sel.contiguous().realize()
        @TinyJit
        def run(x): return layer(sel, x).sum(-2).realize()
        for _ in range(3):
          # Exactly representable Q8 activations and power-of-two weight scales give an exact float32 reference.
          full = rng.integers(-15, 16, size=(T, k, H)).astype(np.float32)
          full[..., ::32] = 127
          xs = np.zeros((len(devs), 1, T, k*width, size), dtype=np.float32)
          expected = np.zeros((len(devs), T, D), dtype=np.float32)
          for rank, (part, table) in enumerate(zip(parts, tables)):
            for t in range(T):
              for j, expert in enumerate(selections[t]):
                for slot, local in enumerate(table[expert]):
                  if local < 0: continue
                  s = part[local] % slices
                  activation = full[t, j, s*size:(s+1)*size]
                  xs[rank, 0, t, j*width+slot] = activation
                  expected[rank, t] += quant[expert, :, s*size:(s+1)*size].astype(np.float32) @ activation / 128
          result = run(Tensor(xs).shard(devs, axis=0).contiguous().realize())
          for rank, buf in enumerate(result.uop.base.buffer.bufs):
            np.testing.assert_array_equal(np.frombuffer(buf.as_memoryview(), dtype=np.float32).reshape(T, D), expected[rank])
          reference = sum(np.einsum("tdh,th->td", quant[selections[:, j]].astype(np.float32), full[:, j])/128 for j in range(k))
          np.testing.assert_array_equal(expected.sum(0), reference)

if __name__ == "__main__": unittest.main()
