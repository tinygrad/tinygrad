import os, unittest

os.environ["DEV"] = "NULL:HIP:gfx950"
os.environ["WQKV"] = "1"
os.environ["AMAX_UOP"] = "1"
os.environ["FP8"] = "1"
os.environ["ASM_GEMM"] = "1"
os.environ["HK_FLASH_ATTENTION"] = "0"
os.environ["DEFAULT_FLOAT"] = "bfloat16"
os.environ["OPTIM_DTYPE"] = "bfloat16"
os.environ["ALL2ALL"] = "1"
os.environ["USE_ATOMICS"] = "1"
os.environ["ALLREDUCE_CAST"] = "1"
os.environ["NULL_ALLOW_COPYOUT"] = "1"
os.environ["ALLOW_DEVICE_USAGE"] = "1"

from tinygrad import Tensor, dtypes, TinyJit
import tinygrad.device as device_mod
from tinygrad.device import Device
from tinygrad.helpers import Context
from examples.mlperf.models.flat_llama import FP8_MAX, matmul, apply_grad

device_mod.ALLOW_DEVICE_USAGE = True


class TinyFFN:
  def __init__(self, dim=4096, hidden_dim=14336, vocab=256):
    self.w1 = Tensor.empty(hidden_dim, dim, dtype=dtypes.bfloat16)
    self.w2 = Tensor.empty(dim, hidden_dim, dtype=dtypes.bfloat16)
    self.w3 = Tensor.empty(hidden_dim, dim, dtype=dtypes.bfloat16)
    self.out = Tensor.empty(vocab, dim, dtype=dtypes.bfloat16)

    def _amax(): return Tensor.full((), FP8_MAX).contiguous().requires_grad_(False)
    self._fp8_amax = {name: [_amax()] for name in ("x1", "w1", "x2", "w2", "x3", "w3")}

  def shard(self, devices:tuple[str, ...]):
    for t in (self.w1, self.w2, self.w3, self.out): t.shard_(devices, axis=None).realize()
    for name in self._fp8_amax:
      self._fp8_amax[name][0] = self._fp8_amax[name][0].to(devices).contiguous().requires_grad_(False)

  def __call__(self, x:Tensor) -> Tensor:
    a = self._fp8_amax
    x_w1, *ret = matmul(x, self.w1, amax_x=a["x1"][0], amax_w=a["w1"][0])
    a["x1"][0].assign(ret[0]); a["w1"][0].assign(ret[1])
    x_w3, *ret = matmul(x.contiguous_backward(), self.w3, amax_x=a["x3"][0], amax_w=a["w3"][0])
    a["x3"][0].assign(ret[0]); a["w3"][0].assign(ret[1])
    h, *ret = matmul(x_w1.silu() * x_w3, self.w2, amax_x=a["x2"][0], amax_w=a["w2"][0])
    a["x2"][0].assign(ret[0]); a["w2"][0].assign(ret[1])
    return matmul(h.contiguous().contiguous_backward(), self.out, fp8=False)[0]


class TestFlatLlamaHKAmax(unittest.TestCase):
  def test_hk_amax_feedforward_backward_dp_scheduler_regression(self):
    def run_minibatch(hk_amax:int):
      devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
      model = TinyFFN()
      model.shard(devices)

      grad_targets = [model.w1, model.w3]
      grads = [Tensor.zeros(p.shape, dtype=p.dtype, device=p.device).contiguous() for p in grad_targets]
      fp8_amax = [t for ts in model._fp8_amax.values() for t in ts]
      x = Tensor.randn(2, 256, 4096, dtype=dtypes.bfloat16)
      y = Tensor.randint(2, 256, low=0, high=256, dtype=dtypes.int)

      with Context(HK_AMAX=hk_amax):
        @TinyJit
        def minibatch(x:Tensor, y:Tensor):
          x = x.to(None).shard(devices, axis=0)
          y = y.to(None).shard(devices, axis=0)
          loss = model(x).sparse_categorical_crossentropy(y)
          for g, new_g in zip(grads, loss.gradient(*grad_targets)):
            apply_grad(g, new_g.uop)
          return loss.flatten().float().to("CPU").realize(*grads, *fp8_amax)

        minibatch(x, y)

    run_minibatch(0)
    run_minibatch(1)


if __name__ == "__main__":
  unittest.main()
