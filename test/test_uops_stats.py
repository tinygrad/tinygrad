import unittest
from tinygrad import Tensor
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import lower_schedule_item
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.ops import BinaryOps, TernaryOps, flops_mem, UOps, UOp
from tinygrad.dtype import dtypes
from tinygrad.codegen.kernel import Kernel, Opt, OptOps, KernelOptError

# **************** new FlopCounter ****************

def get_stats(x:Tensor):
  si = create_schedule([x.lazydata])[-1]
  ei = lower_schedule_item(si)
  return ei.prg.op_estimate, ei.prg.mem_estimate

class TestMemoryCount(unittest.TestCase):
  def test_add(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    b = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024*3)  # 2 reads + 1 write

  def test_add_const(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+3)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_add_slice(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)[:512]
    _, mem = get_stats(a+3)
    self.assertEqual(mem, 512*1024*2)  # 1 read + 1 write

  def test_expanded(self):
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024*2 + 1024)  # 1 full read + 1 lil read + 1 write

  def test_both_expanded(self):
    # TODO: this probably should be a full write
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024 + 2*1024)  # 2 lil reads + 1 write

  def test_self_add(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+a)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_self_add_transposed(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+a.T)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_self_add_assign(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8).realize()
    _, mem = get_stats(a.assign(a+a))
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

# NOTE: this still isn't testing unroll using the acc
@unittest.skipUnless(getenv("PYTHON"), "only run test on emulated tensor cores")
class TestUOpsStatsMatmulHalf(unittest.TestCase):
  def test_simple_matmul_half(self, N=16):
    GlobalCounters.reset()
    a, b = Tensor.empty(N, N, dtype=dtypes.half), Tensor.empty(N, N, dtype=dtypes.half)
    c = a.matmul(b)
    c.realize()
    expected_ops = N ** 3 * 2
    self.assertEqual(expected_ops, GlobalCounters.global_ops)

  def test_bigger_matmul_half(self): self.test_simple_matmul_half(64)

  def test_batched_matmul_half(self, N=16):
    GlobalCounters.reset()
    a, b = Tensor.empty(4, N, N, dtype=dtypes.half), Tensor.empty(1, N, N, dtype=dtypes.half)
    c = a.matmul(b)
    c.realize()
    expected_ops = 4 * N ** 3 * 2
    self.assertEqual(expected_ops, GlobalCounters.global_ops)

class TestUOpsStats(unittest.TestCase):
  @unittest.skipIf(getenv("PTX"), "wrong in PTX")
  def test_simple_add(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = a+b
    ops, mem = get_stats(c)
    expected_ops = c.numel()
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  @unittest.skipIf(getenv("PTX"), "wrong in PTX")
  def test_simple_add_sq(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = (a+b)*(a+b)
    ops, mem = get_stats(c)
    expected_ops = c.numel()*2
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  def test_simple_matmul(self):
    a = Tensor.empty(1024,1024)
    b = Tensor.empty(1024,1024)
    c = a@b
    ops, mem = get_stats(c)
    expected_ops = c.numel() * 1024 * 2
    required_mem = a.nbytes() + b.nbytes() + c.nbytes()
    assert expected_ops <= ops and ops <= expected_ops * 1.2
    # NOTE: it's hard to assert on the memory here, all depends on caching
    assert required_mem <= mem

  #MULACC should have the same stats as MUL + ADD
  def test_mulacc(self):
    globl = UOp(UOps.DEFINE_GLOBAL, dtypes.int.ptr(), tuple())
    o1 = UOp(UOps.CONST, dtypes.int, tuple(), 1)
    o2 = UOp(UOps.CONST, dtypes.int, tuple(), 2)
    u1 = UOp(UOps.LOAD, dtypes.int, (globl, o1))
    u2 = UOp(UOps.LOAD, dtypes.int, (globl, o2))
    u3 = UOp(UOps.CONST, dtypes.int, tuple(), 3)
    u4 = UOp(UOps.ALU, dtypes.int, (u1,u2), BinaryOps.MUL)
    u5 = UOp(UOps.ALU, dtypes.int, (u4,u3), BinaryOps.ADD)
    uops = linearize_uop(u5.sink())

    globl = UOp(UOps.DEFINE_GLOBAL, dtypes.int.ptr(), tuple())
    o1 = UOp(UOps.CONST, dtypes.int, tuple(), 1)
    o2 = UOp(UOps.CONST, dtypes.int, tuple(), 2)
    u1 = UOp(UOps.LOAD, dtypes.int, (globl, o1))
    u2 = UOp(UOps.LOAD, dtypes.int, (globl, o2))
    u3 = UOp(UOps.CONST, dtypes.int, tuple(), 3)
    u4 = UOp(UOps.ALU, dtypes.int, (u1,u2,u3), TernaryOps.MULACC)
    uops_fma = linearize_uop(u4.sink())

    self.assertEqual(flops_mem(uops), flops_mem(uops_fma))

N = 100
@unittest.skipIf(getenv("PTX"), "wrong in PTX") # maybe?
class TestStatsOptimized(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ast_gemm = (Tensor.empty(N, N) @ Tensor.empty(N, N)).schedule()[-1].ast
    cls.ast_reduce = (Tensor.empty(N*N).sum()).schedule()[-1].ast

  def check_gemm(self, p, extra_flops=0):
    #p.uops.print()
    #print(p.src)
    print(p.name, p.op_estimate, p.mem_estimate, p.lds_estimate)
    self.assertEqual(p.op_estimate, 2*N*N*N + extra_flops)  # N**3 mulaccs
    self.assertEqual(p.mem_estimate, 3*N*N*4) # 3 NxN mats with floats

  def test_gemm(self):
    p = Kernel(self.ast_gemm).to_program()
    self.check_gemm(p)
    self.assertEqual(p.lds_estimate, 2*N*N*N*4 + 4*N*N)

  # this is a good lesson about why UPCASTing is a good idea

  def test_gemm_one_upcasted(self):
    k = Kernel(self.ast_gemm)
    k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
    p = k.to_program()
    self.check_gemm(p)
    self.assertEqual(p.lds_estimate, N*N*N*4 + N*N*N*4//4 + 4*N*N)

  def test_gemm_upcasted(self):
    k = Kernel(self.ast_gemm)
    k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
    k.apply_opt(Opt(OptOps.UPCAST, 1, 4))
    k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
    p = k.to_program()
    self.check_gemm(p)
    self.assertEqual(p.lds_estimate, 2*N*N*N*4//4 + 4*N*N)

  def test_gemm_upcasted_locals(self):
    k = Kernel(self.ast_gemm)
    k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
    k.apply_opt(Opt(OptOps.UPCAST, 1, 4))
    try:
      k.apply_opt(Opt(OptOps.LOCAL, 0, 5))
      k.apply_opt(Opt(OptOps.LOCAL, 1, 5))
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    p = k.to_program()
    self.check_gemm(p)
    self.assertEqual(p.lds_estimate, 2*N*N*N*4//4 + 4*N*N)

  def test_gemm_group(self):
    k = Kernel(self.ast_gemm)
    try:
      k.apply_opt(Opt(OptOps.GROUP, 0, 4))
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    SZ = N*N*4
    p = k.to_program()
    # NOTE: these are sort of wrong. they aren't honoring the IF statement
    self.check_gemm(p, extra_flops=SZ*4)
    self.assertEqual(p.lds_estimate, 2*N*N*N*4 + SZ*4 + (SZ*4 + 4*N*N)*4)

  def test_reduce(self):
    k = Kernel(self.ast_reduce)
    p = k.to_program()
    print(p.name, p.op_estimate, p.mem_estimate, p.lds_estimate)
    self.assertEqual(p.op_estimate, N*N)
    self.assertEqual(p.mem_estimate, N*N*4 + 4)

  def test_reduce_group(self):
    k = Kernel(self.ast_reduce)
    try:
      k.apply_opt(Opt(OptOps.GROUP, 0, 50))
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    p = k.to_program()
    # NOTE: these are wrong, they don't respect the if statement
    print(p.name, p.op_estimate, p.mem_estimate, p.lds_estimate)

if __name__ == '__main__':
  unittest.main(verbosity=2)
