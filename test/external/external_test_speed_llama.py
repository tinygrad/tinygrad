# NOTE: this only tests the speed of the LLaMA codegen, it doesn't actually run the net
import unittest, time
import numpy as np
from examples.llama import Transformer, MODEL_PARAMS
from test.test_net_speed import start_profile, stop_profile
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.nn.state import get_state_dict
from tinygrad.ops import Compiled
from tinygrad.helpers import dtypes, prod
from tinygrad.runtime.lib import RawBuffer

class FakeProgram:
  def __init__(self, name:str, prg:str): pass
  def __call__(self, *bufs, global_size, local_size, wait=False): pass

class RawFakeBuffer(RawBuffer):
  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs): return cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
  def toCPU(self): return np.empty(self.size, dtype=self.dtype.np)

class TestLLaMASpeed(unittest.TestCase):
  @unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "only test for compiled backends")
  def test_llama_compile(self):
    backup_program = Device[Device.DEFAULT].runtime
    backup_buffer = Device[Device.DEFAULT].buffer
    Device[Device.DEFAULT].runtime = FakeProgram
    Device[Device.DEFAULT].buffer = RawFakeBuffer

    print("testing llama python run time")
    model = Transformer(**MODEL_PARAMS["1"]["7B"]["args"])
    print("built model")
    # assign fake tensors to the values
    for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype))
    print("assigned empty tensors, doing warmup")

    def run_llama(st, empty_method_cache=True):
      if empty_method_cache: Device[Device.DEFAULT].method_cache.clear()
      tms = [time.perf_counter()]
      for i in range(10):
        model(Tensor([[2]]), i).realize()
        tms.append(time.perf_counter())
      timings = [(tms[i+1]-tms[i])*1000 for i in range(len(tms)-1)]
      print(f"{st:15s} mean runtime: {sum(timings)/len(timings):7.2f}ms, runs: ", ", ".join(f'{x:7.2f}' for x in timings))

    run_llama("codegen")
    run_llama("methodcache", False)

    pr = start_profile()
    run_llama("profile")
    stop_profile(pr, sort='time', frac=0.1)

    Device[Device.DEFAULT].runtime = backup_program
    Device[Device.DEFAULT].buffer = backup_buffer

if __name__ == '__main__':
  unittest.main()
