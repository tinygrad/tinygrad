import os
import unittest

import tinygrad.distributed as dist
from tinygrad.tensor import Tensor


class DistributedRuntimeTest(unittest.TestCase):
  def setUp(self) -> None:
    self._saved_env = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE")}
    for key in self._saved_env:
      os.environ.pop(key, None)
    dist.destroy_process_group()

  def tearDown(self) -> None:
    dist.destroy_process_group()
    for key, value in self._saved_env.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value

  def test_init_process_group_reads_environment(self) -> None:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="dummy")
    self.assertTrue(dist.is_initialized())
    self.assertEqual(dist.get_rank(), 0)
    self.assertEqual(dist.get_world_size(), 1)

  def test_init_process_group_cannot_reinitialize(self) -> None:
    dist.init_process_group(backend="dummy", rank=0, world_size=1)
    with self.assertRaises(RuntimeError):
      dist.init_process_group(backend="dummy", rank=0, world_size=1)

  def test_dummy_backend_collective_operations_identity(self) -> None:
    dist.init_process_group(backend="dummy", rank=0, world_size=1)
    tensor = Tensor([1, 2, 3], device="CPU").contiguous()
    tensor.realize()
    self.assertEqual(dist.all_reduce(tensor).tolist(), [1, 2, 3])
    self.assertEqual(dist.all_gather(tensor).tolist(), [1, 2, 3])
    self.assertEqual(dist.broadcast(tensor, src_rank=0).tolist(), [1, 2, 3])
    dist.send(tensor, dst_rank=0)
    received = dist.recv(tensor, src_rank=0)
    self.assertEqual(received.tolist(), [1, 2, 3])


if __name__ == "__main__":
  unittest.main()
