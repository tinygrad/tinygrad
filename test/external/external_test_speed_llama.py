# NOTE: this only tests the speed of the LLaMA codegen, it doesn't actually run the net
import unittest, time
from examples.llama import Transformer, MODEL_PARAMS
from tinygrad.tensor import Tensor
from tinygrad import Device
from tinygrad.nn.state import get_state_dict
from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import Profiling

class FakeProgram:
  def __init__(self, name:str, prg:bytes): pass
  def __call__(self, *bufs, global_size, local_size, wait=False): pass

class FakeAllocator(Allocator):
  def _alloc(self, sz): return None
  def copyin(self, dest, src:memoryview): pass

class TestLLaMASpeed(unittest.TestCase):
  @unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "only test for compiled backends")
  def test_llama_compile(self):
    backup_program = Device[Device.DEFAULT].runtime
    backup_allocator = Device[Device.DEFAULT].allocator
    Device[Device.DEFAULT].runtime = FakeProgram
    Device[Device.DEFAULT].allocator = FakeAllocator()

    print("testing llama python run time")
    model = Transformer(**MODEL_PARAMS["1"]["7B"]["args"])
    print("built model")
    # assign fake tensors to the values
    for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype))
    print("assigned empty tensors, doing warmup")

    def run_llama(st, empty_method_cache=True):
      if empty_method_cache: Device[Device.DEFAULT].get_runner.cache_clear()
      tms = [time.perf_counter()]
      for i in range(10):
        model(Tensor([[1,2,3,4]]), i).realize()
        tms.append(time.perf_counter())
      timings = [(tms[i+1]-tms[i])*1000 for i in range(len(tms)-1)]
      print(f"{st:15s} mean runtime: {sum(timings)/len(timings):7.2f}ms, runs: ", ", ".join(f'{x:7.2f}' for x in timings))

    run_llama("codegen")
    run_llama("methodcache", False)

    with Profiling(sort='time', frac=0.1):
      run_llama("profile")

    Device[Device.DEFAULT].runtime = backup_program
    Device[Device.DEFAULT].allocator = backup_allocator

if __name__ == '__main__':
  unittest.main()
