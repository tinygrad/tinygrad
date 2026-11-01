import unittest
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters, Target
from tinygrad.engine.realize import compile_linear, estimate_uop
from tinygrad.codegen import to_program
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import Ops, UOp, AxisType, KernelInfo, AddrSpace
from tinygrad.dtype import dtypes
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.device import Device
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.x86 import X86Renderer
from test.helpers import replace_opts

def flops_mem(uop, ignore_indexing=False):
  est = Estimates.from_uops(uop, ignore_indexing)
  return est.ops, est.lds

# **************** new FlopCounter ****************

def get_stats(x:Tensor):
  est = estimate_uop(compile_linear(x.schedule_linear()).src[-1])
  return est.ops, est.mem

@unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu does extra load/store for packed types")
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

  @unittest.skip("depends on subbuffer working")
  def test_add_slice(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)[:512]
    _, mem = get_stats(a+3)
    self.assertEqual(mem, 512*1024*2)  # 1 read + 1 write

  def test_expanded(self):
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024*2 + 1024)  # 1 full read + 1 lil read + 1 write

  @unittest.skip("no longer supported")
  def test_both_expanded(self):
    # TODO: this probably should be a full write
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    _, mem = get_stats(a+b)
    # rangeify is smart!
    self.assertEqual(mem, 1024 + 2*1024)  # 2 lil reads + 1 lil write

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

  def test_copyout(self):
    a = Tensor.empty(32, dtype=dtypes.uint8, device="NULL").to("NULL:1")
    _, mem = get_stats(a)
    self.assertEqual(mem, 32*1)
    a = Tensor.empty(32, dtype=dtypes.uint32, device="NULL").to("NULL:1")
    _, mem = get_stats(a)
    self.assertEqual(mem, 32*4)

# NOTE: this still isn't testing unroll using the acc
@unittest.skipUnless(Device.DEFAULT == "PYTHON", "only run test on emulated tensor cores")
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

class TestUOpsStatsInlineCall(unittest.TestCase):
  def test_inline_call(self):
    buf = UOp.param(0, dtypes.int32, 4)
    p0 = buf.param_like(0)
    body = p0[0].store(UOp.param(1, dtypes.int32, addrspace=AddrSpace.ALU) + p0[0].load())
    out = body.sink().call(buf, UOp.const(5, dtypes.int32), name="test_call").sink(arg=KernelInfo(optimized=True))
    self.assertEqual(flops_mem(out), (1, 8))

  def test_inline_range_bound(self):
    buf = UOp.param(0, dtypes.int32, 4)
    p0 = buf.param_like(0)
    n = UOp.param(1, dtypes.int32, addrspace=AddrSpace.ALU, vmin_vmax=(0,3))
    i = UOp.range(n + 1, 0, dtype=dtypes.int32)
    body = p0[i].store(i).end(i).sink()
    for value in (0, 3):
      with self.subTest(value=value):
        out = body.call(buf, UOp.const(value, dtypes.int32), name="bound_expr").sink()
        est = Estimates.from_uops(out)
        self.assertEqual((est.ops, est.lds, est.mem), (1, 4*(value+1), 4*(value+1)))

  def test_multiple_inline_call_sites(self):
    buf = UOp.param(0, dtypes.int32, 4)
    p0, p1 = buf.param_like(0), UOp.param(1, dtypes.int32, addrspace=AddrSpace.ALU, vmin_vmax=(0,3))
    val, i = (p0[p1].load() * p1) + 10, UOp.range(4, 0, dtype=dtypes.i32)
    body, buf1 = p0[p1].store(val).sink(), UOp.param(1, dtypes.int32, 4)
    call0 = body.call(buf.after(i), i, name="test_call")
    call1 = body.call(buf1.after(i), i, name="test_call")
    out = UOp.sink(call0, call1).end(i).sink(arg=KernelInfo(optimized=True))
    est = Estimates.from_uops(out, ignore_indexing=True)
    self.assertEqual((est.ops,est.lds,est.mem), (16,64,64))

  def test_sequential_inline_calls(self):
    buf = UOp.param(0, dtypes.int32, 1)
    p0 = buf.param_like(0)
    body = p0[0].store(p0[0].load() + 1).sink()
    first = body.call(buf, name="increment")
    second = body.call(buf.after(first), name="increment")
    est = Estimates.from_uops(second.sink())
    self.assertEqual((est.ops, est.lds, est.mem), (2, 16, 8))

  def test_nested_inline_call(self):
    buf = UOp.param(0, dtypes.int32, 4)
    p0 = buf.param_like(0)
    p1 = UOp.param(1, dtypes.int32, addrspace=AddrSpace.ALU, vmin_vmax=(0,3))
    inner = p0[p1].store(p0[p1].load() * p1 + UOp.const(10, dtypes.int32)).sink()
    outer = inner.call(p0, p1, name="inner_call").sink()
    i = UOp.range(4, 0, dtype=dtypes.i32)
    out = outer.call(buf, i, name="outer_cal").end(i).sink(arg=KernelInfo(optimized=True))
    self.assertEqual(flops_mem(out), (8, 32))

class TestUOpsStats(unittest.TestCase):
  def test_isa_store_estimate(self):
    buf = UOp.param(0, dtypes.int32, 4)
    prg = to_program(buf.index(1).store(5).sink(arg=KernelInfo()), X86Renderer(Target("CPU", arch="x86_64")))
    self.assertEqual(prg.src[0].arg.estimates.mem, 4)
    self.assertEqual(prg.src[0].arg.estimates.lds, 4)

  def test_triangular_loop_estimate(self):
    buf = UOp.param(0, dtypes.int32, 6)
    i = UOp.range(3, 0)
    j = UOp.range(i+1, 1)
    prg = to_program(buf.index(i*(i+1)//2+j).store(j).end(j, i).sink(arg=KernelInfo()), X86Renderer(Target("CPU", arch="x86_64")))
    self.assertEqual(prg.src[0].arg.estimates.lds, 3*3*4)

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

  def test_cat_equal_pieces(self):
    # concatenating equal-size pieces lowers to STACK: pure data movement, no arithmetic
    equal = [Tensor.empty(256, 128) for _ in range(4)]
    self.assertEqual(get_stats(Tensor.cat(*equal, dim=1))[0], 0)
    # a mismatched piece falls back to pad+usum, which sums N zero-padded copies and pays their adds
    unequal = equal[:3] + [Tensor.empty(256, 129)]
    self.assertGreater(get_stats(Tensor.cat(*unequal, dim=1))[0], 0)

  def test_xor_gather(self):
    self.assertEqual(get_stats(Tensor.empty(32)[Tensor.arange(32) ^ 16])[0], 0)

  def test_simple_matmul(self, M=1024, N=1024, K=1024):
    a = Tensor.empty(M,N)
    b = Tensor.empty(N,K)
    c = a@b
    ops, mem = get_stats(c)
    expected_ops = c.numel() * N * 2
    required_mem = a.nbytes() + b.nbytes() + c.nbytes()
    assert expected_ops <= ops and ops <= expected_ops * 1.2
    # NOTE: it's hard to assert on the memory here, all depends on caching
    assert required_mem <= mem

  def test_simple_matmul_8192(self): self.test_simple_matmul(8192, 8192, 8192)

  #MULACC should have the same stats as MUL + ADD
  def test_mulacc(self):
    globl = UOp.param(0, dtypes.int, 3)
    o1 = UOp.const(1, dtypes.int)
    o2 = UOp.const(2, dtypes.int)
    u1 = globl.index(o1)
    u2 = globl.index(o2)
    u3 = UOp.const(3, dtypes.int)
    u4 = UOp(Ops.MUL, src=(u1,u2))
    u5 = UOp(Ops.ADD, src=(u4,u3))

    globl = UOp.param(0, dtypes.int, 3)
    o1 = UOp.const(1, dtypes.int)
    o2 = UOp.const(2, dtypes.int)
    u1 = globl.index(o1)
    u2 = globl.index(o2)
    u3 = UOp.const(3, dtypes.int)
    u4 = UOp(Ops.MULACC, src=(u1,u2,u3))

    self.assertEqual(flops_mem(u5), flops_mem(u4))

N = 64
@unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "wrong in PTX") # maybe?
class TestStatsOptimized(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ast_gemm = (Tensor.empty(N, N) @ Tensor.empty(N, N)).schedule_linear().src[-1].src[0]
    cls.ast_gemm_half = (Tensor.empty(N, N, dtype=dtypes.half) @ Tensor.empty(N, N, dtype=dtypes.half)).schedule_linear().src[-1].src[0]
    cls.ast_reduce = (Tensor.empty(N*N).sum()).schedule_linear().src[-1].src[0]

  def check_gemm(self, p:UOp, extra_flops=0, half=False):
    est = p.src[0].arg.estimates
    print(p.src[0].arg.name, est.ops, est.mem, est.lds)
    self.assertEqual(est.ops, 2*N*N*N + extra_flops)  # N**3 mulaccs
    self.assertEqual(est.mem, 3*N*N*(2 if half else 4)) # 3 NxN mats with floats

  def test_gemm(self):
    p = to_program(replace_opts(self.ast_gemm, []), renderer=Device[Device.DEFAULT].renderer)
    self.check_gemm(p)
    self.assertEqual(p.src[0].arg.estimates.lds, 2*N*N*N*4 + 4*N*N)

  @unittest.skip("fails locally on AMD")
  def test_gemm_tc_unroll_half(self):
    try:
      p = to_program(replace_opts(self.ast_gemm_half, [Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.SPLIT, 4, (2, AxisType.UPCAST))]),
                      renderer=Device[Device.DEFAULT].renderer)
    except KernelOptError:
      raise unittest.SkipTest("no tensor cores")
    print(p.src[2].arg)
    self.check_gemm(p, half=True)

  def test_gemm_tc_unroll(self):
    try:
      p = to_program(replace_opts(self.ast_gemm, [Opt(OptOps.TC, 0, (-1, 0, 1)), Opt(OptOps.SPLIT, 4, (2, AxisType.UPCAST))]),
                      renderer=Device[Device.DEFAULT].renderer)
    except KernelOptError:
      raise unittest.SkipTest("no tensor cores")
    print(p.src[2].arg)
    self.check_gemm(p)

  # this is a good lesson about why UPCASTing is a good idea

  def test_gemm_one_upcasted(self):
    p = to_program(replace_opts(self.ast_gemm, [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))]), renderer=Device[Device.DEFAULT].renderer)
    self.check_gemm(p)
    self.assertEqual(p.src[0].arg.estimates.lds, N*N*N*4 + N*N*N*4//4 + 4*N*N)

  def test_gemm_upcasted(self):
    p = to_program(replace_opts(self.ast_gemm, [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)),
                                                Opt(OptOps.SPLIT, 4, (4, AxisType.UPCAST))]),
                    renderer=Device[Device.DEFAULT].renderer)
    self.check_gemm(p)
    self.assertEqual(p.src[0].arg.estimates.lds, 2*N*N*N*4//4 + 4*N*N)

  def test_gemm_upcasted_locals(self):
    try:
      p = to_program(replace_opts(self.ast_gemm, [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)),
                                                  Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (4, AxisType.LOCAL))]),
                     renderer=Device[Device.DEFAULT].renderer)
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    self.check_gemm(p)
    self.assertEqual(p.src[0].arg.estimates.lds, 2*N*N*N*4//4 + 4*N*N)

  def test_gemm_group(self):
    try:
      p = to_program(replace_opts(self.ast_gemm, [Opt(OptOps.SPLIT, 2, (4, AxisType.LOCAL))]), renderer=Device[Device.DEFAULT].renderer)
    except KernelOptError:
      raise unittest.SkipTest("no locals")
    SZ = N*N*4
    # NOTE: these are sort of wrong. they aren't honoring the IF statement
    self.check_gemm(p, extra_flops=SZ*5)
    self.assertEqual(p.src[0].arg.estimates.lds, 2*N*N*N*4 + SZ*4 + (SZ*4 + 4*N*N)*4)

  def test_reduce(self):
    p = to_program(replace_opts(self.ast_reduce, []), renderer=Device[Device.DEFAULT].renderer)
    est = p.src[0].arg.estimates
    print(p.src[0].arg.name, est.ops, est.mem, est.lds)
    self.assertEqual(est.ops, N*N)
    self.assertEqual(est.mem, N*N*4 + 4)

if __name__ == '__main__':
  unittest.main(verbosity=2)
