#!/usr/bin/env python
import unittest
from tinygrad.runtime.lib import RawBuffer, LRUAllocator
from tinygrad.helpers import dtypes
from tinygrad.ops import ASTRunner
from tinygrad.jit import CacheCollector
from weakref import ref

class FakeDeviceBuffer:
  def __init__(self, sz, dt, device):
    self.size = sz
    self.dtype = dt
    self.device = device

class FakeAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs): return FakeDeviceBuffer(size, dtype, device)

FAKE_GLOBAL_ALLOCATOR = None
class FakeBuffer(RawBuffer):
  def __init__(self, size, dtype, device='0'):
    global FAKE_GLOBAL_ALLOCATOR
    super().__init__(size, dtype, allocator=FAKE_GLOBAL_ALLOCATOR, **{'device': device})
    assert self._buf.size == size and self._buf.dtype == dtype and self._buf.device == device, "This allocator requires 100% match of dtype and size."

def alloc(allocator, size, dtype, **kwargs):
  global FAKE_GLOBAL_ALLOCATOR
  FAKE_GLOBAL_ALLOCATOR = allocator
  buf = FakeBuffer(size, dtype, **kwargs)
  assert buf.dtype == dtype and buf.size == size
  FAKE_GLOBAL_ALLOCATOR = None
  return buf

def anybuf(size, dtype):
  return FakeBuffer(size, dtype)

def add_to_cache(bufs):
  CacheCollector.add(ASTRunner("", None), bufs, None)
  return bufs[0]

def add_to_cache_refed(bufs):
  CacheCollector.add(None, bufs, None)
  return bufs[0], [ref(buf) for buf in bufs]

def get_bufs_count(cache):
  ss = set()
  for (_,bufs,_) in cache:
    for buf in bufs: ss.add(buf)
  return len(ss)

class TestCacheCollector(unittest.TestCase):
  def test_cache_collector_optimization(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256 << 30)
    inps = [FakeBuffer(64, dtypes.float32) for _ in range(2)]
    CacheCollector.start()
    out = add_to_cache([FakeBuffer(32, dtypes.float32), inps[0]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out, inps[1]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    assert cache[-1][1][0] == out, "Output does not match."
    assert get_bufs_count(cache) == 5, "Should have 5 buffers in total"
    # This is not worth added complexity on real models
    # assert cache[-1][1][0] == cache[0][1][0], "Should reuse final output buffer as output in 1st kernel"
    FAKE_GLOBAL_ALLOCATOR = None

  def test_cache_collector_cycle_avoidance(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256 << 30)
    inps = [FakeBuffer(64, dtypes.float32) for _ in range(2)]
    CacheCollector.start()
    # Output buffer here cannot be shared with final output buffer, since we could get a cycle the next step as inps[1] has the same shape and dtype.
    out = add_to_cache([FakeBuffer(64, dtypes.float32), inps[0]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out, inps[1]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    out = add_to_cache([FakeBuffer(64, dtypes.float32), out])
    out = add_to_cache([FakeBuffer(64, dtypes.float32), out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    assert cache[-1][1][0] == out, "Output does not match."
    assert cache[-1][1][0] != cache[0][1][0] and cache[0][1][0] == cache[3][1][0], "Output buffers from 1st and 4th kernel could not be the same as the 5th."
    assert get_bufs_count(cache) == 6, "Should have 6 buffers in total"
    FAKE_GLOBAL_ALLOCATOR = None

  def test_cache_collector_all_alive(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256 << 30)
    inps = [FakeBuffer(64, dtypes.float32) for _ in range(2)]
    outs = [FakeBuffer(128, dtypes.float32) for _ in range(4)]
    CacheCollector.start()
    out = add_to_cache([outs[0], inps[0]])
    out = add_to_cache([outs[1], out, inps[1]])
    out = add_to_cache([outs[2], out])
    out = add_to_cache([outs[3], out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    assert cache[0][1][0] == outs[0], "Output0 should be on its place."
    assert cache[1][1][0] == outs[1], "Output1 should be on its place."
    assert cache[2][1][0] == outs[2], "Output2 should be on its place."
    assert cache[3][1][0] == outs[3], "Output3 should be on its place."
    assert cache[-1][1][0] == out, "Output does not match."
    assert get_bufs_count(cache) == len(outs) + len(inps), "Nothing to optimize, since buffers are alive and might be used as outputs"
    FAKE_GLOBAL_ALLOCATOR = None

  def test_cache_collector_middle_input(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256 << 30)
    inps = [FakeBuffer(64, dtypes.float32) for _ in range(2)]
    outs = [FakeBuffer(32, dtypes.float32) for _ in range(1)]
    CacheCollector.start()
    out = add_to_cache([FakeBuffer(32, dtypes.float32), inps[0]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out, inps[1]])
    out,refs2 = add_to_cache_refed([outs[0], out, FakeBuffer(32, dtypes.float32)])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    assert cache[2][1][2] == refs2[2](), "Input should be captured."
    assert cache[0][1][0] != cache[2][1][2], "None of outputs buffer should reuse new_input."
    assert cache[1][1][0] != cache[2][1][2], "None of outputs buffer should reuse new_input."
    assert cache[3][1][0] != cache[2][1][2], "None of outputs buffer should reuse new_input."
    assert cache[4][1][0] != cache[2][1][2], "None of outputs buffer should reuse new_input."
    assert cache[-1][1][0] == out, "Output does not match."
    assert get_bufs_count(cache) == 7
    FAKE_GLOBAL_ALLOCATOR = None

  def test_cache_collector_multidev(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256 << 30)
    inps = [FakeBuffer(64, dtypes.float32, '1') for _ in range(2)]
    CacheCollector.start()
    out = add_to_cache([FakeBuffer(32, dtypes.float32, '1'), inps[0]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32, '1'), out, inps[1]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32, '1'), out])
    out = add_to_cache([FakeBuffer(32, dtypes.float32, '2'), out])
    out = add_to_cache([FakeBuffer(32, dtypes.float32, '2'), out])
    out = add_to_cache([FakeBuffer(32, dtypes.float32, '2'), out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    for i in range(3):
      assert cache[i][1][0]._device == '1', f"Device does not match {i}, has {cache[i][1][0]._device}."
    for i in range(3, 6):
      assert cache[i][1][0]._device == '2', f"Device does not match {i}, has {cache[i][1][0]._device}."
    assert get_bufs_count(cache) == 7
    FAKE_GLOBAL_ALLOCATOR = None

  def test_cache_collector_anybufs_inputs(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256 << 30)
    inps = [FakeBuffer(64, dtypes.float32, '1') for _ in range(2)]
    CacheCollector.start()
    out = add_to_cache([FakeBuffer(32, dtypes.float32), inps[0]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out, inps[1]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), 32, None])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out, 58, None])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    assert get_bufs_count(cache) == 8
    FAKE_GLOBAL_ALLOCATOR = None

  def test_cache_collector_optimize_when_not_cached_anymore(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256)
    inps = [FakeBuffer(64, dtypes.float32) for _ in range(2)]
    CacheCollector.start()
    # Output buffer here cannot be shared with final output buffer, since we could get a cycle the next step as inps[1] has the same shape and dtype.
    out = add_to_cache([FakeBuffer(240, dtypes.float32), inps[0]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out, inps[1]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    out = add_to_cache([FakeBuffer(240, dtypes.float32), out])
    out = add_to_cache([FakeBuffer(64, dtypes.float32), out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    assert cache[-1][1][0] == out, "Output does not match."
    assert get_bufs_count(cache) == 6, "Should have 6 buffers in total"
    assert cache[0][1][0] == cache[3][1][0], "Output buffers from 1st and 4th should be the same"
    FAKE_GLOBAL_ALLOCATOR = None

  def test_cache_collector_mark_output_buffer(self):
    global FAKE_GLOBAL_ALLOCATOR
    FAKE_GLOBAL_ALLOCATOR = FakeAllocator(256)
    output_buffer = FakeBuffer(240, dtypes.float32)
    inps = [FakeBuffer(64, dtypes.float32) for _ in range(2)]
    CacheCollector.start()
    # Output buffer here cannot be shared with final output buffer, since we could get a cycle the next step as inps[1] has the same shape and dtype.
    out = add_to_cache([FakeBuffer(240, dtypes.float32), inps[0]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out, inps[1]])
    out = add_to_cache([FakeBuffer(32, dtypes.float32), out])
    CacheCollector._mark_output_buffer(output_buffer)
    out = add_to_cache([output_buffer, out])
    out = add_to_cache([FakeBuffer(64, dtypes.float32), out])
    cache = CacheCollector.finish()
    assert cache[0][1][1] == inps[0], "Input should be on its place."
    assert cache[1][1][2] == inps[1], "Input should be on its place."
    assert cache[-1][1][0] == out, "Output does not match."
    assert cache[0][1][0] != cache[3][1][0], "Cannot reuse 4th output buffer, it's an output buffer which might ovewrite itself"
    assert get_bufs_count(cache) == 6, "Should have 6 buffers in total"
    FAKE_GLOBAL_ALLOCATOR = None

if __name__ == "__main__":
  unittest.main()
