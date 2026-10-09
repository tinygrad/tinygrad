import struct, unittest, weakref
from dataclasses import replace
from tinygrad import Device, Tensor, TinyJit
from tinygrad.codegen import to_program
from tinygrad.device import TinyELF
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import get_call_outs_ins, run_linear
from tinygrad.helpers import Context, cpu_events, ProfilePointEvent
from tinygrad.schedule import resolve_linear_call
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from test.helpers import needs_second_gpu

class TestKernelArgs(unittest.TestCase):
  def test_custom_kernel_jit(self):
    def kernel(n, out, x, bias):
      idx = UOp.range(n, 0, AxisType.GLOBAL if Device.DEFAULT.split(':')[0] in ('AMD', 'NV', 'CUDA', 'METAL', 'NULL') else AxisType.LOOP)
      return out[idx].store(x[idx]*n + bias).end(idx).sink(arg=KernelInfo(name='jit_mixed_args', opts_to_apply=()))
    @TinyJit
    def run(n, x, unrelated):
      out = Tensor.custom_kernel(Tensor(n), Tensor.zeros(4, dtype=dtypes.int32).contiguous(), x,
                                 Tensor(UOp.variable('constant_bias', 0, 10, dtypes.int32).bind(5)), fxn=kernel)[1]
      other = (Tensor([1], dtype=dtypes.int32) + Tensor(unrelated)).contiguous()
      return out, other
    # Warmup, capture, then shrinking and growing replay: both scalar and buffer inputs change.
    for n in (2, 4, 1, 3):
      data = [n, 2*n, 4*n, 8*n]
      out, other = run(UOp.variable('caller_extent', 1, 4, dtypes.int32).bind(n), Tensor(data, dtype=dtypes.int32).realize(),
                       UOp.variable('other', 1, 4, dtypes.int32).bind(5-n))
      self.assertEqual(out.tolist(), [v*n+5 for v in data[:n]] + [0]*(4-n))
      self.assertEqual(other.tolist(), [6-n])

  def test_scalar_launch_extent(self):
    out = Tensor.zeros(4, dtype=dtypes.int32).contiguous().realize().uop
    p, n = UOp.param(1, dtypes.int32, 4), UOp.param(0, dtypes.int32, name='extent', vmin_vmax=(1, 4), addrspace=AddrSpace.ALU)
    idx = UOp.range(n, 0, AxisType.GLOBAL if Device.DEFAULT.split(':')[0] in ('AMD', 'NV', 'CUDA', 'METAL', 'NULL') else AxisType.LOOP)
    prg = to_program(p[idx].store(7).end(idx).sink(arg=KernelInfo(name='scalar_extent', opts_to_apply=())), Device[Device.DEFAULT].renderer)
    run_linear(UOp(Ops.LINEAR, src=(prg.call(UOp.variable('caller_extent', 1, 4, dtypes.int32).bind(3), out),)), wait=True)
    self.assertEqual(out.buffer.as_memoryview().cast('i').tolist(), [7, 7, 7, 0])

  def test_signature_does_not_own_uops(self):
    p = UOp.param(997, dtypes.int32, 997, name='signature_lifetime')
    ref, signature = weakref.ref(p), (p.kernel_param,)
    del p
    self.assertIsNone(ref())
    self.assertEqual(signature[0].arg.slot, 997)

  def test_stack_arguments(self):
    inputs = [Tensor([i], dtype=dtypes.int32).realize().uop for i in range(1, 9)]
    out = UOp.new_buffer(Device.DEFAULT, 1, dtypes.int32)
    scalar = UOp.param(0, dtypes.int32, addrspace=AddrSpace.ALU)
    params = [UOp.param(i, dtypes.int32, 1) for i in range(1, 10)]
    sink = params[-1].index(0).store(sum((i+1)*p.index(0).load() for i,p in enumerate(params[:-1])) + scalar).sink(
      arg=KernelInfo(name='stack_arguments'), tag=1)
    prg = to_program(sink, Device[Device.DEFAULT].renderer)
    self.assertEqual([p.arg.slot for p in prg.to_elf().signature], list(range(10)))
    run_linear(UOp(Ops.LINEAR, src=(prg.call(UOp.variable('factor', 0, 10, dtypes.int32).bind(2), *inputs, out),)), wait=True)
    self.assertEqual(out.buffer.as_memoryview().cast('i').tolist(), [206])

  def test_mixed_arguments(self):
    x = Tensor([1, 2, 3], dtype=dtypes.int32).realize().uop
    unused = Tensor([999], dtype=dtypes.int32).realize().uop
    orders = (('unused', 'a', 'out', 'b', 'x'), ('out', 'unused', 'x', 'a', 'b'), ('b', 'x', 'unused', 'a', 'out'))
    for names, dtype in zip(orders, (dtypes.int16, dtypes.int32, dtypes.int64)):
      # WGSL uniforms only support 32-bit integers; the layouts still exercise the same CALL slots.
      if Device.DEFAULT == 'WEBGPU': dtype = dtypes.int32
      with self.subTest(names=names, dtype=dtype):
        value = -(1<<33)-3 if dtype == dtypes.int64 else -3
        actual = {'unused': unused, 'out': UOp.new_buffer(Device.DEFAULT, 3, dtypes.int32), 'x': x,
                  'a': UOp.variable('caller_a', value, 10, dtype).bind(value), 'b': UOp.const(7, dtypes.int32)}
        params = {name: UOp.param(i, actual[name].dtype, name=name, addrspace=AddrSpace.ALU,
                                  vmin_vmax=(value if name == 'a' else -10, 10)) if name in ('a', 'b') else
                  UOp.param(i, dtypes.int32, 3) for i,name in enumerate(names)}
        idx, factor = UOp.range(3, 0), params['a'] >> 32 if dtype == dtypes.int64 else params['a']
        sink = params['out'][idx].store((params['x'][idx]*factor + params['b']).cast(dtypes.int32)).end(idx).sink(
          arg=KernelInfo(name='mixed_arguments'), tag=1)
        call, start = sink.call(*(actual[name] for name in names)), len(cpu_events)
        with Context(VALIDATE_WITH_CPU=1, PROFILE=1): run_linear(UOp(Ops.LINEAR, src=(call,)), wait=True)
        self.assertEqual(actual['out'].buffer.as_memoryview().cast('i').tolist(), [4, 1, -2])
        prg = to_program(sink, Device[Device.DEFAULT].renderer)
        self.assertEqual([p.arg.slot for p in prg.to_elf().signature], sorted(names.index(k) for k in ('out', 'x', 'a', 'b')))
        self.assertEqual(get_call_outs_ins(call.replace(src=(prg, *call.src[1:])))[0], (names.index('out'),))
        buffers = [actual[k].buffer for k in names if k in ('out', 'x')]
        event = next(e for e in cpu_events[start:] if isinstance(e, ProfilePointEvent) and e.name == 'exec' and e.arg['name'] == 'mixed_arguments')
        self.assertEqual(event.arg['bufs'], [b.trace_num for b in buffers])
        self.assertEqual(event.arg['outputs'], (buffers.index(actual['out'].buffer),))
        self.assertEqual(event.arg['inputs'], (buffers.index(x.buffer),))

  def test_scalar_scopes(self):
    out = UOp.new_buffer(Device.DEFAULT, 1, dtypes.int32)
    p = UOp.param(0, dtypes.int32, 1)
    a, b = UOp.variable('free_a', 0, 20, dtypes.int32), UOp.variable('free_b', 0, 20, dtypes.int32)
    prg = to_program(p.index(0).store(a*10 + b).sink(arg=KernelInfo(name='slotless'), tag=1), Device[Device.DEFAULT].renderer)
    self.assertEqual([v.arg.slot for v in prg.arg.vars], [-1, -1])
    run_linear(UOp(Ops.LINEAR, src=(prg.call(out),)), var_vals={'free_a': 2, 'free_b': 3}, wait=True)
    self.assertEqual(out.buffer.as_memoryview().cast('i').tolist(), [23])
    scalar = UOp.param(1, dtypes.int32, name='callee', vmin_vmax=(0, 10), addrspace=AddrSpace.ALU)
    call = p[0].store(scalar).sink(arg=KernelInfo(name='named_scope')).call(out, UOp.variable('caller', 0, 10, dtypes.int32).bind(3))
    outer = UOp(Ops.LINEAR, src=(call,)).call(out, UOp.variable('other', 0, 10, dtypes.int32).bind(5))
    run_linear(resolve_linear_call(outer), var_vals={'other': 5}, wait=True)
    self.assertEqual(out.buffer.as_memoryview().cast('i').tolist(), [3])

  @needs_second_gpu
  def test_multi_device_scalar_first(self):
    devices = (Device.DEFAULT, f'{Device.DEFAULT}:1')
    x = Tensor([1, 2, 3], dtype=dtypes.int32).shard(devices, axis=None).realize().uop
    out = UOp.new_buffer(devices, 3, dtypes.int32)
    a, p, q = UOp.param(0, dtypes.int32, addrspace=AddrSpace.ALU), UOp.param(1, dtypes.int32, 3), UOp.param(2, dtypes.int32, 3)
    idx, dnum = UOp.range(3, 0), UOp.variable('_device_num', 0, 1, dtypes.int32)
    sink = p.index(idx).store(q.index(idx).load()*a + dnum).end(idx).sink(arg=KernelInfo(name='multi_argument_order'), tag=1)
    run_linear(UOp(Ops.LINEAR, src=(sink.call(UOp.variable('factor', 0, 10, dtypes.int32).bind(3), out, x),)), wait=True)
    for i,buf in enumerate(out.buffer.bufs): self.assertEqual(buf.as_memoryview().cast('i').tolist(), [3+i, 6+i, 9+i])

  def test_pack_repeated_slot(self):
    # IMAGE can have distinct ABI descriptors that share a CALL slot.
    p = UOp.param(2, dtypes.float32, 4).kernel_param
    image = p._replace(arg=replace(p.arg, image=(1, 1)))
    scalar = UOp.param(0, dtypes.int16, addrspace=AddrSpace.ALU).kernel_param
    signature = (scalar, image, p)
    self.assertEqual(TinyELF.pack(signature, (-3, 0x1000, 0x1000)),
                     struct.pack('<h6xQQ', -3, 0x1000, 0x1000))
    self.assertEqual(TinyELF.pack((p, scalar), (0x1000, -3), 12), bytearray(12) + struct.pack('<Qh', 0x1000, -3))
    self.assertEqual(TinyELF.pack((), (), 12), bytearray(12))
    for args in ((-3, 0x1000), (-3, 0x1000, 0x1000, 7)):
      with self.assertRaises(ValueError): TinyELF.pack(signature, args)

if __name__ == '__main__': unittest.main()
