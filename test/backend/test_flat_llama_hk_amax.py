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
from tinygrad.nn.state import get_parameters
import examples.mlperf.models.flat_llama as flat_llama_mod

device_mod.ALLOW_DEVICE_USAGE = True


class TestFlatLlamaHKAmax(unittest.TestCase):
  def test_hk_amax_feedforward_backward_dp_scheduler_regression(self):
    def run_minibatch(hk_amax:int):
      devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
      params = dict(dim=4096, hidden_dim=14336, n_heads=32, n_kv_heads=8, n_layers=1, norm_eps=1e-5, vocab_size=128256, rope_theta=500000,
                    max_context=256)
      model = flat_llama_mod.FlatTransformer(**params)
      for v in get_parameters(model): v.assign(Tensor.empty(v.shape, dtype=v.dtype))
      model.shard(devices)

      grad_targets = [model.w1[0], model.w3[0]]
      grads = [Tensor.zeros(p.shape, dtype=p.dtype, device=p.device).contiguous() for p in grad_targets]
      fp8_amax = [t for ts in model._fp8_amax.values() for t in ts]
      tokens = Tensor.randint(2, 257, low=0, high=model.vocab_size, dtype=dtypes.int)

      with Context(HK_AMAX=hk_amax):
        @TinyJit
        def minibatch(tokens:Tensor):
          tokens = tokens.to(None).shard(devices, axis=0)
          logits = model(tokens[:, :-1])
          loss = logits.sparse_categorical_crossentropy(tokens[:, 1:])
          for g, new_g in zip(grads, loss.gradient(*grad_targets)):
            flat_llama_mod.apply_grad(g, new_g.uop)
          return loss.flatten().float().to("CPU").realize(*grads, *fp8_amax)

        minibatch(tokens)

    run_minibatch(0)
    run_minibatch(1)


if __name__ == "__main__":
  unittest.main()
