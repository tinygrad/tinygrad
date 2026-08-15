import unittest
import functools
from tinygrad import Tensor, Variable, UOp, function
from tinygrad.uop.ops import KernelInfo
from tinygrad.schedule import schedule_cache

def custom_set0_kernel(A:UOp, _B:UOp=None, num:int=0) -> UOp:
  return A[0].set(num).sink(arg=KernelInfo(f"custom_set0_{num}"))

def custom_set0_backward(grad_a:UOp, _) -> tuple[None, UOp]:
  x = Tensor.invalids(*grad_a.shape, dtype=grad_output.dtype, device=grad_output.device)
  x = Tensor.custom_kernel(x, fxn=functools.partial(custom_set0_kernel, num=0))[0]
  return None, (x * Tensor(grad_a, device=grad_output.device)).uop

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_reuses_cache(self):
    schedule_cache.clear()
    v = Variable('v', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    # first run with v=5
    t1 = (x + Tensor(v.bind(5))).sum()
    self.assertEqual(t1.item(), 60.0)
    cache_size_after_first = len(schedule_cache)

    # second run with v=10 should reuse cache
    t2 = (x + Tensor(v.bind(10))).sum()
    self.assertEqual(t2.item(), 110.0)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_custom_kernel(self):
    for i in range(4):
      a = Tensor.empty(1)
      a = Tensor.custom_kernel(a, fxn=functools.partial(custom_set0_kernel, num=i))[0]
      a.realize()
      self.assertEqual(a.item(), i)

  def test_same_custom_function_reuses_cache(self):
    schedule_cache.clear()
    fxn = functools.partial(custom_set0_kernel, num=10)

    # first run
    a = Tensor.empty(1)
    a = Tensor.custom_kernel(a, fxn=fxn)[0]
    a.realize()
    self.assertEqual(a.item(), 10)
    cache_size_after_first = len(schedule_cache)

    # second run with same function should reuse cache
    b = Tensor.empty(1)
    b = Tensor.custom_kernel(b, fxn=fxn)[0]
    b.realize()
    self.assertEqual(b.item(), 10)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_simple(self):
    a = Tensor.ones(10).contiguous()
    b = Tensor.ones(10).contiguous()
    Tensor.realize(a, b)

    # warm up
    for _ in range(2):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)

    # confirm schedule cache doesn't grow
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)

  @unittest.expectedFailure
  def test_simple_precompile(self):
    @function(precompile=True, precompile_backward=True)
    def f(x:Tensor) -> Tensor:
      out = Tensor.invalids(*x.shape, dtype=x.dtype, device=x.device)
      out = Tensor.custom_kernel(out, x, fxn=functools.partial(custom_set0_kernel, num=10), grad_fxn=custom_set0_backward)[0]
      return out + x

    # warmup
    x = Tensor.ones(1).realize()
    out = f(x)
    out.sum().backward()
    self.assertEqual(out.item(), 11)
    self.assertEqual(x.grad.item(), 1)

    # use the cache next time function is called
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      x = Tensor.ones(1).realize()
      out = f(x)
      out.sum().backward()
      self.assertEqual(out.item(), 11)
      self.assertEqual(x.grad.item(), 1)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)

if __name__ == "__main__":
  unittest.main()
