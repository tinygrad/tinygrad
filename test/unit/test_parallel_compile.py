"""Tests for parallel kernel compilation."""
import unittest
from tinygrad.engine.parallel_compile import parallel_compile

class ClangCompiler:
  """Simple test compiler that just returns source as bytes."""
  def __init__(self, cachekey=None):
    self.cachekey = cachekey
  def compile(self, src: str) -> bytes:
    return src.encode()

class TestParallelCompile(unittest.TestCase):
  def test_parallel_compile_empty(self):
    """Test empty task list."""
    result = parallel_compile([])
    self.assertEqual(result, {})

  def test_parallel_compile_single(self):
    """Test single task (should use sequential path)."""
    tasks = [(b"key1", "source1", ClangCompiler, None)]
    result = parallel_compile(tasks, num_workers=4)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[b"key1"], b"source1")

  def test_parallel_compile_multiple(self):
    """Test multiple tasks with parallel compilation."""
    tasks = [
      (b"key1", "source1", ClangCompiler, None),
      (b"key2", "source2", ClangCompiler, None),
      (b"key3", "source3", ClangCompiler, None),
      (b"key4", "source4", ClangCompiler, None),
    ]
    result = parallel_compile(tasks, num_workers=2)
    self.assertEqual(len(result), 4)
    for i in range(1, 5):
      key = f"key{i}".encode()
      expected = f"source{i}".encode()
      self.assertEqual(result[key], expected)

  def test_parallel_compile_with_cachekey(self):
    """Test compilation with cachekey parameter."""
    tasks = [(b"key1", "source1", ClangCompiler, "my_cachekey")]
    result = parallel_compile(tasks, num_workers=1)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[b"key1"], b"source1")

  def test_sequential_fallback_for_small_batch(self):
    """Test that small batches use sequential compilation."""
    tasks = [
      (b"key1", "source1", ClangCompiler, None),
      (b"key2", "source2", ClangCompiler, None),
    ]
    # With only 2 tasks and 4 workers, should use sequential path
    result = parallel_compile(tasks, num_workers=4)
    self.assertEqual(len(result), 2)

if __name__ == "__main__":
  unittest.main()
