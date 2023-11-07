#!/usr/bin/env python
import unittest, gc
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.helpers import GlobalCounters
from tinygrad.runtime.lib import RawBuffer, LRUAllocator
from tinygrad.helpers import dtypes, prod
from tinygrad.ops import Device
from test.helpers import derandomize_model

from examples.llama import Transformer

ALLOCATED_DEV_BUFS = 0
class FakeDeviceBuffer:
  def __init__(self, sz, dt, device):
    self.id = 1
    self.size = sz
    self.dtype = dt
    self.device = device

    global ALLOCATED_DEV_BUFS
    ALLOCATED_DEV_BUFS += 1
class FakeAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs): return FakeDeviceBuffer(size, dtype, device)
  def _do_free(self, buf):
    buf.id -= 1
    assert buf.id == 0, f"Free should be called once, but {buf.id}"
  def __del__(self): # Fake allocator should clear all buffers after each test.
    for v in self.cached_buffers.values():
      for buf, _ in v: self._free_buffer(buf)

FAKE_GLOBAL_ALLOCATOR = None
class FakeBuffer(RawBuffer):
  def __init__(self, size, dtype, device='0'):
    global FAKE_GLOBAL_ALLOCATOR
    super().__init__(size, dtype, allocator=FAKE_GLOBAL_ALLOCATOR, **{'device': device})
    assert self._buf.size == size and self._buf.dtype == dtype and self._buf.device == device, "This allocator requires 100% match of dtype and size."
  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs): return cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
  def toCPU(self): return np.empty(self.size, dtype=self.dtype.np)
class FakeProgram:
  def __init__(self, name:str, prg:str): pass
  def __call__(self, *bufs, global_size, local_size, wait=False): pass

def helper_test_correctness(gen, train):
  from tinygrad.runtime.ops_gpu import CL, CLAllocator
  old_alloc = CL.cl_allocator
  CL.cl_allocator = CLAllocator(0)
  no_alloc_result = train(*gen()).numpy()
  Device[Device.DEFAULT].synchronize()
  CL.cl_allocator = CLAllocator(512<<30) # Test cache correctness, so cache as much as possible, 512gb
  for _ in range(4):
    GlobalCounters.reset()
    np.testing.assert_allclose(train(*gen()).numpy(), no_alloc_result, rtol=1e-3, atol=1e-5)
    Device[Device.DEFAULT].synchronize()
  assert len(CL.cl_allocator.cached_buffers) != 0, "Cache must be used"
  CL.cl_allocator = old_alloc

def __helper_test_alloc_count(gen, train):
  was_alloc = ALLOCATED_DEV_BUFS
  for _ in range(2):
    train(*gen())
  return ALLOCATED_DEV_BUFS - was_alloc

def helper_test_alloc_count(mm, gen, train):
  global FAKE_GLOBAL_ALLOCATOR
  backup_program = Device[Device.DEFAULT].runtime
  backup_buffer = Device[Device.DEFAULT].buffer
  Device[Device.DEFAULT].runtime = FakeProgram
  Device[Device.DEFAULT].buffer = FakeBuffer
  Device[Device.DEFAULT].method_cache.clear()
  FAKE_GLOBAL_ALLOCATOR = FakeAllocator(16<<30)
  new_allocs = __helper_test_alloc_count(gen, train)
  Device[Device.DEFAULT].method_cache.clear()
  FAKE_GLOBAL_ALLOCATOR = FakeAllocator(0)
  old_allocs = __helper_test_alloc_count(gen, train)
  print(f"{mm}: llama: old allocs count {old_allocs}, new allocs count {new_allocs}")
  assert new_allocs < old_allocs, f"Hmm, doesn't cache work any more?"
  Device[Device.DEFAULT].runtime = backup_program
  Device[Device.DEFAULT].buffer = backup_buffer
  FAKE_GLOBAL_ALLOCATOR = None

def check_gc():
  if Device.DEFAULT == "GPU":
    gc.collect() # Need to collect Tensors.
    from extra.introspection import print_objects
    assert print_objects() == 0

class TestAllocators(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
  def test_lru_allocator_tiny_llama(self):
    old_type = Tensor.default_type
    Tensor.default_type = dtypes.float16

    args_tiny = {"dim": 1024, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    def __test():
      model = Transformer(**args_tiny)
      derandomize_model(model)
      def test(t): return model(t, 0).realize()
      helper_test_correctness(lambda: (Tensor([[1,]]),), test)
    __test()
    Tensor.default_type = old_type
    check_gc()

  @unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
  def test_lru_allocator_tiny_llama_alloc_counts(self):
    args_tiny = {"dim": 1024, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    def test_alloc_count(t):
      model = Transformer(**args_tiny)
      for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype))
      return model(t, 0).realize()
    helper_test_alloc_count("llama", lambda: (Tensor([[2,]]),), test_alloc_count)
    check_gc()

  @unittest.skip("huge for CI")
  def test_stable_diffusion(self):
    from examples.stable_diffusion import UNetModel
    model = UNetModel()
    derandomize_model(model)
    def test(t, t2): return model(t, 801, t2).realize()
    helper_test_correctness(lambda: (Tensor.randn(1, 4, 16, 16),Tensor.randn(1, 77, 768)), test)

if __name__ == "__main__":
  unittest.main()
