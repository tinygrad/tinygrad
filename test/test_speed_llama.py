# NOTE: this only tests the speed of the LLaMA codegen, it doesn't actually run the net
import unittest, time
from examples.llama import Transformer, args_7B
from test.test_net_speed import start_profile, stop_profile
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.lazy import Device
from tinygrad.state import get_state_dict

class TestLLaMASpeed(unittest.TestCase):
  def test_llama_compile(self):
    # TODO: with default device
    old_default = Device.DEFAULT
    Device.DEFAULT = "FAKE"
    print(Device.DEFAULT)
    print("testing llama python run time")
    model = Transformer(**args_7B)
    print("built model")
    # assign fake tensors to the values
    for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype))
    print("assigned empty tensors")

    model(Tensor([[1]]), 0).realize()
    print("did warmup")

    pr = None if getenv("NOPROFILE") else start_profile()
    tms = [time.perf_counter()]
    for i in range(5):
      model(Tensor([[2]]), i).realize()
      tms.append(time.perf_counter())
    if pr: stop_profile(pr, sort='time')
    print("runtime in ms:", ', '.join("%.2f"%((tms[i+1]-tms[i])*1000) for i in range(len(tms)-1)))

    Device.DEFAULT = old_default

if __name__ == '__main__':
  unittest.main()
