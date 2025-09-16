import os
import unittest

import tinygrad.distributed as dist
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.nn.optim import SGD
from tinygrad.helpers import Context, DISABLE_COMPILER_CACHE


@unittest.skipIf(Device.DEFAULT != "CPU", "dummy backend tests require CPU default device")
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
    tensor = Tensor([1, 2, 3])
    tensor.realize()
    self.assertEqual(dist.all_reduce(tensor).tolist(), [1, 2, 3])
    self.assertEqual(dist.all_gather(tensor).tolist(), [1, 2, 3])
    self.assertEqual(dist.broadcast(tensor, src_rank=0).tolist(), [1, 2, 3])
    dist.send(tensor, dst_rank=0)
    received = dist.recv(tensor, src_rank=0)
    self.assertEqual(received.tolist(), [1, 2, 3])


@unittest.skipIf(Device.DEFAULT != "CPU", "zero optimizer test requires CPU default device")
class ZeroOptimizerTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls._old_training = Tensor.training
    Tensor.training = True
    import tinygrad.helpers as helpers
    cls._helpers = helpers
    cls._old_cachedb = helpers.CACHEDB
    cls._old_db_connection = helpers._db_connection
    helpers.CACHEDB = os.path.abspath(".tinygrad_cache.db")
    helpers._db_connection = None

  @classmethod
  def tearDownClass(cls) -> None:
    Tensor.training = cls._old_training
    cls._helpers.CACHEDB = cls._old_cachedb
    cls._helpers._db_connection = cls._old_db_connection

  def setUp(self) -> None:
    dist.destroy_process_group()
    dist.init_process_group(backend="dummy")

  def tearDown(self) -> None:
    dist.destroy_process_group()

  def test_zero_optimizer_step_updates_parameter(self) -> None:
    param = Tensor([0.0], requires_grad=True)
    optimizer = SGD([param], lr=0.1, fused=False)
    zero = dist.ZeroOptimizer(optimizer)
    param.grad = Tensor([1.0])
    with Context(DISABLE_COMPILER_CACHE=1):
      zero.step()
    self.assertAlmostEqual(param.item(), -0.1, places=6)
    zero.zero_grad()
    self.assertIsNone(param.grad)


if __name__ == "__main__":
  unittest.main()
