import unittest
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.kimi_k3 import _shard_kimi_k3
from test.unit.test_llm_k3 import small_k3_config
from tinygrad.llm.model import Transformer

class TestKimiK3TP8(unittest.TestCase):
  @staticmethod
  def _model():
    model = Transformer(small_k3_config())
    for name,value in nn.state.get_state_dict(model).items():
      fill = 127 if name.endswith("weight_scale") else 0
      dtype = value.dtype if value.dtype is dtypes.uint8 else dtypes.bfloat16
      value.replace(Tensor.full(value.shape, fill, dtype=dtype, device="NULL"))
    _shard_kimi_k3(model, tuple(f"NULL:{i}" for i in range(8)))
    return model

  def test_prefill_decode_and_jit_replay(self):
    devices = tuple(f"NULL:{i}" for i in range(8))
    model = self._model()
    temperature = Tensor([0.0], device=devices)
    self.assertEqual(model(Tensor([[1, 2]], dtype=dtypes.int32, device=devices), 0, temperature).realize().shape, (1, 1))
    model(Tensor([[1, 2]], dtype=dtypes.int32, device=devices), 0, temperature).realize()
    self.assertEqual(model(Tensor([[3]], dtype=dtypes.int32, device=devices), 2, temperature).realize().shape, (1, 1))
    model(Tensor([[4]], dtype=dtypes.int32, device=devices), 3, temperature).realize()
    self.assertEqual(model.blk[0].recurrent_state.uop.axis, 1)
    self.assertEqual(model.blk[1].cache_k.dtype, dtypes.bfloat16)

if __name__ == "__main__": unittest.main()
