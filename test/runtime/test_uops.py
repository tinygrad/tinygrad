from typing import Optional, Any
import unittest, math, itertools
import numpy as np
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.helpers import Context
from tinygrad.dtype import dtypes, DType, AddrSpace, ConstFloat  # noqa: F401
from tinygrad.device import Buffer, Device
from tinygrad.uop.ops import Ops, UOp, KernelInfo, AxisType, ProgramInfo
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime.ops_python import PythonRenderer

def run_uops(uops_list:list[UOp], bufs:list[Buffer]):
  buf_uops = [UOp.from_buffer(b) for b in bufs]
  run_linear(UOp(Ops.LINEAR, src=(UOp.sink(*uops_list, arg=KernelInfo()).call(*buf_uops),)))

def uop(uops:list[UOp], op:Ops, dtype:Optional[DType], src:tuple[UOp, ...], arg:Any=None) -> UOp:
  if op is Ops.CONST: uops.append(UOp.const(arg).cast(dtype))
  elif op is Ops.PARAM: uops.append(UOp.param(arg, dtype, 1))
  else: uops.append(UOp(op, tuple(src), arg))
  return uops[-1]

def _test_single_value(vals, op, dts):
  uops = []
  output_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else dts[-1]
  buf_store = uop(uops, Ops.PARAM, output_dtype, (), 0)
  buf_loads = [uop(uops, Ops.PARAM, dtype, (), i+1) for i,dtype in enumerate(dts)]
  loads = (buf_loads[i].index(uop(uops, Ops.CONST, dtypes.int32, (), 0)) for i, dtype in enumerate(dts))
  alu = uop(uops, op, output_dtype, loads)
  out = uop(uops, Ops.STORE, dtypes.void, (buf_store.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  buf2 = [Buffer(Device.DEFAULT, 1, dtype, initial_value=np.array([a], dtype=_to_np_dtype(dtype)).tobytes()) for a,dtype in zip(vals, dts)]
  run_uops([out], [buf]+buf2)
  return np.frombuffer(buf.as_memoryview(), _to_np_dtype(output_dtype))[0]

def _test_single_value_const(vals, op, dts):
  uops = []
  output_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.CMPNE) else dts[-1]
  buf_store = uop(uops, Ops.PARAM, output_dtype, (), 0)
  loads = (uop(uops, Ops.CONST, dtype, [], a) for a,dtype in zip(vals, dts))
  alu = uop(uops, op, output_dtype, loads)
  out = buf_store[UOp.const(0).cast(dtypes.int32)].store(alu)
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  run_uops([out], [buf])
  return np.frombuffer(buf.as_memoryview(), _to_np_dtype(output_dtype))[0]

def _test_uops_result(output_dtype, uops, res):
  # uops = []
  buf_store = uop(uops, Ops.PARAM, output_dtype, (), 0)
  # res = output_fn(uops)
  out = uop(uops, Ops.STORE, dtypes.void, (buf_store.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), res))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  run_uops([out], [buf])
  return np.frombuffer(buf.as_memoryview(), _to_np_dtype(output_dtype))[0]

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, (CStyleLanguage, PythonRenderer)) and
                     dtypes.uint64 in Device[Device.DEFAULT].renderer.supported_dtypes(), "requires buffer bitcast and 64-bit ints")
class TestBitcastBufferView(unittest.TestCase):
  @Context(SPEC=2)
  def test_load(self):
    val = 0x1122334455667788
    src, out = UOp.param(0, dtypes.uint32, 4), UOp.param(1, dtypes.uint64, 1)
    ibuf = Buffer(Device.DEFAULT, 4, dtypes.uint32, initial_value=np.array([0, 0x55667788, 0x11223344, 0], dtype=np.uint32).tobytes())
    obuf = Buffer(Device.DEFAULT, 1, dtypes.uint64).allocate()
    run_uops([out.index(0).store(src.shrink(((1, 3),)).bitcast(dtypes.uint64).index(0))], [ibuf, obuf])
    self.assertEqual(np.frombuffer(obuf.as_memoryview(), dtype=np.uint64)[0], val)

  @Context(SPEC=2)
  def test_store(self):
    val = 0x1122334455667788
    dst = UOp.param(0, dtypes.uint32, 6)
    buf = Buffer(Device.DEFAULT, 6, dtypes.uint32, initial_value=bytes(24))
    view = dst.shrink(((1, 5),)).bitcast(dtypes.uint64)  # two stores through one view: it must inline, not get a declared vector-pointer
    run_uops([view.index(0).store(val ^ 0xff), view.index(1).store(val)], [buf])
    self.assertEqual(np.frombuffer(buf.as_memoryview(), dtype=np.uint64, count=2, offset=4).tolist(), [val ^ 0xff, val])

  def test_vector_load_store(self):
    for src_dt, dst_dt in [(dtypes.uint8, dtypes.uint32), (dtypes.uint32, dtypes.uint8)]:
      with self.subTest(src=src_dt, dst=dst_dt):
        src, dst = [UOp.param(i, dt, 16 // dt.itemsize) for i, dt in enumerate((src_dt, dst_dt))]
        src, dst = [b.bitcast(dtypes.uint32).index(UOp.stack(*[UOp.const(i) for i in range(4)])) for b in (src, dst)]
        bufs = [Buffer(Device.DEFAULT, 16 // dt.itemsize, dt, initial_value=bytes(range(16)) if i == 0 else bytes(16))
                for i, dt in enumerate((src_dt, dst_dt))]
        run_uops([dst.store(src.load())], bufs)
        self.assertEqual(bytes(bufs[1].as_memoryview()), bytes(range(16)))

class TestArgumentOrder(unittest.TestCase):
  def _test_order(self, scalar_dtype):
    for order in itertools.permutations(range(4)):
      with self.subTest(order=order, dtype=scalar_dtype):
        # Skip an unused buffer in the middle of the call.
        slots = [i if i < 2 else i+1 for i in range(4)]
        params = [UOp.param(slots[order.index(i)], dt, (1,) if i < 2 else (), addrspace=AddrSpace.GLOBAL if i < 2 else AddrSpace.ALU)
                  for i,dt in enumerate((scalar_dtype, dtypes.int, dtypes.int, scalar_dtype))]
        out, inp, scale, bias = params
        sink = out[0].store((inp[0] * scale + bias).cast(scalar_dtype)).sink(arg=KernelInfo())
        obuf = Buffer(Device.DEFAULT, 1, scalar_dtype).allocate()
        ibuf = Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=np.array([5], dtype=np.int32).tobytes())
        unused = UOp.from_buffer(Buffer(Device.DEFAULT, 1, dtypes.int).allocate())
        bias_value = 13 if scalar_dtype == dtypes.int else 2**35+13
        for v in (-7, 11):
          args = [UOp.from_buffer(obuf), UOp.from_buffer(ibuf), UOp.const(v, dtypes.int), UOp.const(bias_value, scalar_dtype)]
          args = [args[i] for i in order]
          args.insert(2, unused)
          run_linear(UOp(Ops.LINEAR, src=(sink.call(*args),)))
          self.assertEqual(obuf.as_memoryview().cast(scalar_dtype.fmt)[0], 5*v+bias_value)
          # Reverse the ABI order as well as the call order.
          prg = to_program(sink, Device[Device.DEFAULT].renderer)
          lin = prg.src[1]
          if not any(u.op is Ops.INS for u in lin.src):
            params = [u for u in lin.src if u.op is Ops.PARAM]
            end = max(i for i,u in enumerate(lin.src) if u.op is Ops.PARAM)+1
            lin = lin.replace(src=tuple(u for u in lin.src[:end] if u.op is not Ops.PARAM)+tuple(reversed(params))+lin.src[end:])
            prg = to_program(prg.replace(src=(prg.src[0], lin)), Device[Device.DEFAULT].renderer)
            obuf.copy_from(Buffer(Device.DEFAULT, 1, scalar_dtype, initial_value=bytes(scalar_dtype.itemsize)))
            run_linear(UOp(Ops.LINEAR, src=(prg.call(*args),)))
            self.assertEqual(obuf.as_memoryview().cast(scalar_dtype.fmt)[0], 5*v+bias_value)

  def test_eight_used_arguments(self):
    for mixed in (False, True):
      with self.subTest(mixed=mixed):
        params = [UOp.param(i, dtypes.int, () if mixed and i%2 else (1,),
                            addrspace=AddrSpace.ALU if mixed and i%2 else AddrSpace.GLOBAL) for i in range(8)]
        values = [p if mixed and i%2 else p[0].load() for i,p in enumerate(params[1:], 1)]
        sink = params[0][0].store(sum((v*i for i,v in enumerate(values, 1)), UOp.const(0, dtypes.int))).sink(arg=KernelInfo())
        buffers = [Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=np.array([i], dtype=np.int32).tobytes())
                   for i in range(8) if not mixed or i%2 == 0]
        buf_iter = iter(buffers)
        args = [UOp.const(i, dtypes.int) if mixed and i%2 else UOp.from_buffer(next(buf_iter)) for i in range(8)]
        prg = to_program(sink, Device[Device.DEFAULT].renderer)
        self.assertEqual(len(prg.to_elf().signature), 8)
        run_linear(UOp(Ops.LINEAR, src=(prg.call(*args),)))
        self.assertEqual(buffers[0].as_memoryview().cast('i')[0], 140)

  @unittest.skipUnless(Device.DEFAULT in ("CPU", "PYTHON") and
                       isinstance(Device[Device.DEFAULT].renderer, (CStyleLanguage, LLVMRenderer, PythonRenderer)), "requires nested calls")
  def test_nested_rendered_order(self):
    renderer = Device[Device.DEFAULT].renderer
    out = UOp.param(0, dtypes.int, (1,))
    a, b = [UOp.param(i, dtypes.int, addrspace=AddrSpace.ALU) for i in (1, 2)]
    inner = to_program(out[0].store(a*10+b).sink(arg=KernelInfo()), renderer).src[1]
    inner = inner.replace(src=tuple(u for u in inner.src if u.op is not Ops.SINK))
    params = [u for u in inner.src if u.op is Ops.PARAM]
    end = max(i for i,u in enumerate(inner.src) if u.op is Ops.PARAM)+1
    inner = inner.replace(src=tuple(u for u in inner.src[:end] if u.op is not Ops.PARAM)+tuple(reversed(params))+inner.src[end:], arg="nested")
    sink = inner.call(out, UOp.const(5, dtypes.int), UOp.const(7, dtypes.int)).sink(arg=KernelInfo())
    obuf = Buffer(Device.DEFAULT, 1, dtypes.int).allocate()
    prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort(enter_calls=False)))),
              arg=ProgramInfo.from_sink(sink, renderer.target))
    run_linear(UOp(Ops.LINEAR, src=(prg.call(UOp.from_buffer(obuf)),)))
    self.assertEqual(obuf.as_memoryview().cast('i')[0], 57)

  @Context(VALIDATE_WITH_CPU=1)
  def test_scheduled_scalar_first(self):
    out = UOp.param(1, dtypes.int, (1,))
    scale = UOp.param(0, dtypes.int, addrspace=AddrSpace.ALU)
    sink = out[0].store(scale*2).sink(arg=KernelInfo())
    for scale_arg,expected in ((UOp.const(-3, dtypes.int), -6), (UOp.variable("scale", -10, 10).bind(4), 8)):
      buf = UOp.from_buffer(Buffer(Device.DEFAULT, 1, dtypes.int).allocate())
      self.assertEqual(Tensor(buf.after(sink.call(scale_arg, buf))).tolist(), [expected])

  def test_buffers_and_args(self): self._test_order(dtypes.int)

  @unittest.skipUnless(dtypes.long in Device[Device.DEFAULT].renderer.supported_dtypes(), "requires 64-bit ints")
  def test_buffers_and_mixed_width_args(self): self._test_order(dtypes.long)

class TestUOps(unittest.TestCase):
  def _equal(self, v1, v2):
    assert isinstance(v2, (float, int, bool))
    if isinstance(v2, float):
      np.testing.assert_allclose(v1, v2, rtol=2e-7)
    else:
      np.testing.assert_equal(v1, v2)

  def _test_uop_fxn(self, op, fxn, dts=(dtypes.float32, )):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        a = dts[0].const(a)
        self._equal(f([a], op, dts), fxn(a))

  def _test_bop_fxn(self, op, fxn, dts=(dtypes.float32, )*2, no_b_zero=False, no_b_neg=False):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        for b in [-3.0, 1.0] + ([] if no_b_zero else [0.0]):
          a = dts[0].const(a)
          b = dts[1].const(abs(b) if no_b_neg else b)
          self._equal(f([a,b], op, dts), fxn(a,b))

  def _test_top_fxn(self, op, fxn, dts=(dtypes.float32, )*3):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0, 1]:
        for b in [-3.0, 3.0]:
          for c in [-4.0, 4.0]:
            a = dts[0].const(a)
            b = dts[1].const(b)
            c = dts[2].const(c)
            self._equal(f([a,b,c], op, dts), fxn(a,b,c))

class TestFloatUOps(TestUOps):
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_exp2(self): self._test_uop_fxn(Ops.EXP2, lambda a: np.exp2(a))
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_log2(self): self._test_uop_fxn(Ops.LOG2, lambda a: math.log2(a) if a > 0 else float('-inf' if a==0 else 'nan'))
  @unittest.skipIf(Device.DEFAULT == "CPU", 'not supported as uop')
  def test_sin(self): self._test_uop_fxn(Ops.SIN, lambda a: math.sin(a))
  def test_recip(self): self._test_uop_fxn(Ops.RECIPROCAL, lambda a: 1/a if a != 0 else float('inf'))
  def test_sqrt(self): self._test_uop_fxn(Ops.SQRT, lambda a: math.sqrt(a) if a >= 0 else float('nan'))

  def test_add(self): self._test_bop_fxn(Ops.ADD, lambda a,b: a+b)
  def test_mul(self): self._test_bop_fxn(Ops.MUL, lambda a,b: a*b)
  def test_max(self): self._test_bop_fxn(Ops.MAX, lambda a,b: max(a,b))
  def test_cmplt(self): self._test_bop_fxn(Ops.CMPLT, lambda a,b: a<b)
  def test_cmpne(self): self._test_bop_fxn(Ops.CMPNE, lambda a,b: a!=b)
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU doesn't support NaN comparison correctly")
  def test_cmpne_nan(self):  # NaN != x for any x (IEEE 754)
    for a, b in [(math.nan, 1.0), (1.0, math.nan), (math.nan, math.nan)]:
      self.assertTrue(_test_single_value(
        [dtypes.float32.const(a), dtypes.float32.const(b)],
        Ops.CMPNE, (dtypes.float32, dtypes.float32)))
  # MOD isn't tested on floats

  def test_where(self):
    self._test_top_fxn(Ops.WHERE, lambda a,b,c: b if a!=0 else c, (dtypes.bool, dtypes.float, dtypes.float))

class TestNonFloatUOps(TestUOps):
  def test_add_int32(self): self._test_bop_fxn(Ops.ADD, lambda a,b: int(a)+int(b), (dtypes.int32, dtypes.int32))
  def test_mul_int32(self): self._test_bop_fxn(Ops.MUL, lambda a,b: int(a)*int(b), (dtypes.int32, dtypes.int32))
  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, CStyleLanguage)), "only ptx and cstyle use bitshifts")
  def test_shr_int32(self): self._test_bop_fxn(Ops.SHR, lambda a,b: int(a)>>int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  @unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, CStyleLanguage)), "only ptx and cstyle use bitshifts")
  def test_shl_int32(self): self._test_bop_fxn(Ops.SHL, lambda a,b: int(a)<<int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  def test_div_int32(self):
    self._test_bop_fxn(Ops.CDIV, lambda a,b: int(a/b), (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_and_int32(self): self._test_bop_fxn(Ops.AND, lambda a,b: int(a)&int(b), (dtypes.int32, dtypes.int32))
  def test_or_int32(self): self._test_bop_fxn(Ops.OR, lambda a,b: int(a)|int(b), (dtypes.int32, dtypes.int32))
  def test_mod_int32(self):
    self._test_bop_fxn(Ops.CMOD,
                       lambda a,b: abs(int(a))%abs(int(b))*(1,-1)[a<0], (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_cmplt_int32(self): self._test_bop_fxn(Ops.CMPLT, lambda a,b: int(a)<int(b), (dtypes.int32, dtypes.int32))
  def test_cmpne_int32(self): self._test_bop_fxn(Ops.CMPNE, lambda a,b: int(a)!=int(b), (dtypes.int32, dtypes.int32))
  def test_where_float16(self):
    self._test_top_fxn(Ops.WHERE, lambda a,b,c: b if a!=0 else c, (dtypes.bool, dtypes.float16, dtypes.float16))

class TestBoolUOps(TestUOps):
  def _test_uop_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        self._equal(f([a], op, (dtypes.bool, )*1), fxn(a))

  def _test_bop_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        for b in [False, True]:
          self._equal(f([a,b], op, (dtypes.bool, )*2), fxn(a,b))

  def _test_top_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        for b in [False, True]:
          for c in [False, True]:
            self._equal(f([a,b,c], op, (dtypes.bool, )*3), fxn(a,b,c))

  def test_add_bool(self): self._test_bop_bool_fxn(Ops.ADD, lambda a,b: a or b)
  def test_mul_bool(self): self._test_bop_bool_fxn(Ops.MUL, lambda a,b: a and b)
  def test_xor_bool(self): self._test_bop_bool_fxn(Ops.XOR, lambda a,b: a != b)
  def test_and_bool(self): self._test_bop_bool_fxn(Ops.AND, lambda a,b: a & b)
  def test_or_bool(self): self._test_bop_bool_fxn(Ops.OR, lambda a,b: a | b)
  def test_cmpne_bool(self): self._test_bop_bool_fxn(Ops.CMPNE, lambda a,b: a != b)
  def test_cmplt_bool(self): self._test_bop_bool_fxn(Ops.CMPLT, lambda a,b: a < b)
  def test_where_bool(self): self._test_top_bool_fxn(Ops.WHERE, lambda a,b,c: b if a else c)

class TestLocalAccess(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  def test_local_basic(self):
    uops = []
    smem = UOp.placeholder((16,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
    uops.append(smem)
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.float32, (), 42.0)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = uop(uops, Ops.LOAD, dtypes.float32, (smem.after(barr).index(uop(uops, Ops.CONST, dtypes.int32, (), 0)),))
    self.assertEqual(_test_uops_result(dtypes.float32, uops, sres), 42)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  @unittest.skip("tinygrad doesn't support this behavior")
  def test_local_indirect(self):
    uops = []
    smem = UOp.placeholder((16,), dtypes.int32, slot=0, addrspace=AddrSpace.LOCAL)
    uops.append(smem)
    st1 = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 1)), uop(uops, Ops.CONST, dtypes.int32, (), 2)))
    st2 = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 2)), uop(uops, Ops.CONST, dtypes.int32, (), 42)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st1,st2))
    ofs = uop(uops, Ops.LOAD, dtypes.int32, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 1)), barr))
    sres = uop(uops, Ops.LOAD, dtypes.int32, (smem.index(ofs),))
    self.assertEqual(_test_uops_result(dtypes.int32, uops, sres), 42)

class TestZeroRange(unittest.TestCase):
  def test_reduce_variable(self):
    for i in range(3,-1,-1):
      v = UOp.variable("i", 0, 5).bind(i)
      out = Tensor.ones(10, dtype=dtypes.int).contiguous().shrink(((0,v),)).sum()
      self.assertEqual(out.item(), i)

class TestUOpPrograms(unittest.TestCase):
  def _run(self, prog:UOp, *tensors:Tensor):
    run_linear(UOp(Ops.LINEAR, src=(prog.call(*[t.realize().uop.buf_uop for t in tensors]),)), update_stats=False)

  def test_simple(self):
    out = Tensor.empty(10,10,dtype=dtypes.int)

    ptr = UOp.placeholder(out.shape, out.dtype, slot=0)
    i, j = UOp.range(10, axis_id=0), UOp.range(10, axis_id=1)
    prog = ptr[i,j].store(42).end(i,j)
    self._run(prog.sink(arg=KernelInfo()), out)

    with Context(DEBUG=0): self.assertTrue((out == 42).all().item())

  def test_matmul(self):
    a = Tensor.randn(10,10)
    b = Tensor.randn(10,10)
    c = Tensor.empty(10,10)
    ref = (a@b)
    with Context(DEBUG=0): Tensor.realize(a, b, c, ref)

    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Shapes: A[M,K], B[K,N], C[M,N]
    M = N = K = 10
    DT = dtypes.float32

    # Placeholders (bind slots explicitly)
    A = UOp.placeholder((M, K), DT, slot=0)
    B = UOp.placeholder((K, N), DT, slot=1)
    C = UOp.placeholder((M, N), DT, slot=2)

    # Axes: i,j are spatial; k is a reduction axis over the shared dim K
    i = UOp.range(M, axis_id=0)                             # rows of A/C
    j = UOp.range(N, axis_id=1)                             # cols of B/C
    k = UOp.range(K, axis_id=2, axis_type=AxisType.LOOP)  # reduction over K

    # Zero-init: write a scalar 0 to each (i,j).
    C = C[i, j].set(0.0)

    # Accumulate: end the store, with C.after(k) enforcing the dependency along the reduction axis
    prog = C[i, j].store(C.after(k)[i, j] + A[i, k] * B[k, j]).end(i, j, k)

    # run program
    self._run(prog.sink(arg=KernelInfo()), a, b, c)

    with Context(DEBUG=0): self.assertLessEqual((c-ref).square().mean().item(), 1e-6)

  def test_matmul_relu(self):
    a, b, c = Tensor.randn(10,10), Tensor.randn(10,10), Tensor.empty(10,10)
    ref = (a@b).relu()
    with Context(DEBUG=0): Tensor.realize(a, b, c, ref)

    A, B, C = a.uop.placeholder_like(0), b.uop.placeholder_like(1), c.uop.placeholder_like(2)
    i, j, k = UOp.range(10, 0), UOp.range(10, 1), UOp.range(10, 2, axis_type=AxisType.LOOP)

    C = C[i, j].set(0.0)
    C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
    prog = C[i, j].store(C[i, j].maximum(0.0)).end(i, j)

    self._run(prog.sink(arg=KernelInfo(opts_to_apply=())), a, b, c)
    with Context(DEBUG=0): self.assertLessEqual((c-ref).square().mean().item(), 1e-6)

if __name__ == '__main__':
  unittest.main(verbosity=2)
