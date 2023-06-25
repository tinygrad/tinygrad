# NOTE: this only tests the speed of the LLaMA codegen, it doesn't actually run the net
import unittest, time
from examples.llama import Transformer, args_7B
from test.test_net_speed import start_profile, stop_profile
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.lazy import Device
from tinygrad.state import get_state_dict
from tinygrad.ops import Compiled

class TestLLaMASpeed(unittest.TestCase):
  @unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "only test for compiled backends")
  def test_llama_compile(self):
    # TODO: with default device
    old_default = Device.DEFAULT
    Device.DEFAULT = "FAKE"

    # use the codegen from the real device
    Device['fake'].codegen = Device[old_default].codegen
    print("using", Device['fake'].codegen)

    print("testing llama python run time")
    model = Transformer(**args_7B)
    print("built model")
    # assign fake tensors to the values
    for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype))
    print("assigned empty tensors, doing warmup")

    def run_llama(st, empty_method_cache=True):
      #print(f"clearing {len(Device['fake'].method_cache)} from method cache")
      if empty_method_cache: Device['fake'].method_cache.clear()
      tms = [time.perf_counter()]
      for i in range(5):
        model(Tensor([[2]]), i).realize()
        tms.append(time.perf_counter())
      print(f"{st:15s} runtime in ms:", ', '.join("%.2f"%((tms[i+1]-tms[i])*1000) for i in range(len(tms)-1)))

    run_llama("codegen")
    run_llama("methodcache", False)

    pr = start_profile()
    run_llama("profile")
    stop_profile(pr, sort='time', frac=0.1)

    # reset device
    Device.DEFAULT = old_default

if __name__ == '__main__':
  unittest.main()
