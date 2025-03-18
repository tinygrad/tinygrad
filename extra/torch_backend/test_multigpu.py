import unittest, os
from tinygrad.helpers import getenv
import torch
import tinygrad.frontend.torch
torch.set_default_device("tiny")
import numpy as np

def _distributed_test_runner(rank, test_class, test_name, world_size, backend, master_addr, master_port, extra_args, extra_kwargs):
  os.environ["MASTER_ADDR"] = master_addr
  os.environ["MASTER_PORT"] = master_port
  torch.multiprocessing.current_process().name = "MainProcess" # TODO:
  torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
  try:
    instance = test_class()
    if hasattr(instance, 'setUp'): instance.setUp()
    test_func = getattr(instance, test_name)
    test_func(*extra_args, rank=rank, world_size=world_size, _distributed=True, **extra_kwargs)
    if hasattr(instance, 'tearDown'): instance.tearDown()
  finally:
    torch.distributed.destroy_process_group()

def distributed_test(world_size, backend="cpu:gloo,tiny:tiny", master_addr="localhost", master_port="29500"):
  def decorator(func):
    def wrapper(*args, **kwargs):
      if kwargs.pop("_distributed", False): return func(*args, **kwargs)
      torch.multiprocessing.spawn(
        _distributed_test_runner,
        args=(args[0].__class__, func.__name__, world_size, backend, master_addr, master_port, args[1:], kwargs),
        nprocs=world_size,
        join=True
      )
    return wrapper
  return decorator

GPUS = getenv("GPUS", 1)

@unittest.skipIf(GPUS<=1, "only single GPU")
class TestTorchBackendMultiGPU(unittest.TestCase):
  def test_transfer(self):
    a = torch.Tensor([[1,2],[3,4]]).to("tiny:0")
    b = torch.Tensor([[3,2],[1,0]]).to("tiny:1")
    self.assertNotEqual(a.device, b.device)
    np.testing.assert_array_equal(a.cpu(), a.to("tiny:1").cpu())
    np.testing.assert_array_equal(b.cpu(), b.to("tiny:1").cpu())

  def test_basic_ops(self):
    a = torch.Tensor([[1,2],[3,4]]).to("tiny:0")
    b = torch.Tensor([[3,2],[1,0]]).to("tiny:1")
    c1 = a + b.to("tiny:0")
    c2 = b + a.to("tiny:1")
    np.testing.assert_array_equal(c1.cpu(), torch.full((2,2),4).cpu())
    np.testing.assert_array_equal(c1.cpu(), c2.cpu())

  @distributed_test(world_size=GPUS)
  def test_broadcast(self, rank, world_size):
    device = torch.device("tiny", rank)
    a = torch.arange(4, device=device) + rank*4
    b = torch.arange(4, device=device) + (world_size-1)*4
    torch.distributed.broadcast(a, src=world_size-1)
    np.testing.assert_array_equal(a.cpu(), b.cpu())

  @distributed_test(world_size=GPUS)
  def test_allgather(self, rank, world_size):
    device = torch.device("tiny", rank)
    tensors = [torch.zeros((2,2), device=device) for _ in range(world_size)]
    torch.distributed.all_gather(tensors, torch.ones((2,2), device=device)+rank)
    for i,t in enumerate(tensors):
      np.testing.assert_array_equal(t.cpu(), i+1)

  @distributed_test(world_size=GPUS)
  def test_allreduce(self, rank, world_size):
    device = torch.device("tiny", rank)
    b = torch.arange(12, device=device).reshape((2,2,3))
    a = (rank+1) * b
    torch.distributed.all_reduce(a, torch.distributed.ReduceOp.SUM)
    np.testing.assert_array_equal(a.cpu(), b.cpu()*(world_size*(world_size+1))//2)

if __name__ == "__main__":
  unittest.main()

