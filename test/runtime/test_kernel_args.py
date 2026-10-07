import itertools, struct, unittest
from dataclasses import replace
from tinygrad import Device, Tensor
from tinygrad.codegen import to_program
from tinygrad.device import TinyELF
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import get_call_outs_ins, run_linear
from tinygrad.helpers import Context, Target
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from test.helpers import needs_second_gpu

class TestKernelArgs(unittest.TestCase):
  def test_arbitrary_order(self):
    x = Tensor([1, 2, 3], dtype=dtypes.int32).realize().uop
    unused = Tensor([999], dtype=dtypes.int32).realize().uop
    for order in itertools.permutations(('out', 'x', 'a', 'b')):
      for scalar_dtype in (dtypes.int16, dtypes.int32, dtypes.int64):
        with self.subTest(order=order, scalar_dtype=scalar_dtype):
          names = ('unused', *order)
          out = UOp.new_buffer(Device.DEFAULT, 3, dtypes.int32)
          actual = {'unused': unused, 'out': out, 'x': x,
                    'a': UOp.variable('caller_a', -10, 10, scalar_dtype).bind(-3),
                    'b': UOp.variable('caller_b', -10, 10, dtypes.int32).bind(7)}
          params = {name: UOp.param(i, actual[name].dtype, addrspace=AddrSpace.ALU, name=name) if name in ('a', 'b') else
                    UOp.param(i, dtypes.int32, 3) for i,name in enumerate(names)}
          idx = UOp.range(3, 0)
          sink = params['out'].index(idx).store((params['x'].index(idx).load()*params['a'] + params['b']).cast(dtypes.int32)).end(idx).sink(
            arg=KernelInfo(name='argument_order'), tag=1)
          call = sink.call(*(actual[name] for name in names))
          run_linear(UOp(Ops.LINEAR, src=(call,)), wait=True)
          self.assertEqual(out.buffer.numpy().tolist(), [4, 1, -2])
          prg = to_program(sink, Device[Device.DEFAULT].renderer)
          sig = prg.to_elf().signature
          self.assertEqual([p.arg.slot for p in sig], sorted(names.index(name) for name in ('out', 'x', 'a', 'b')))
          self.assertEqual(get_call_outs_ins(call.replace(src=(prg, *call.src[1:])))[0], (names.index('out'),))

  def test_free_variables_remain_slotless(self):
    out = UOp.new_buffer(Device.DEFAULT, 1, dtypes.int32)
    p = UOp.param(0, dtypes.int32, 1)
    a, b = UOp.variable('free_a', 0, 20, dtypes.int32), UOp.variable('free_b', 0, 20, dtypes.int32)
    prg = to_program(p.index(0).store(a*10 + b).sink(arg=KernelInfo(name='slotless'), tag=1), Device[Device.DEFAULT].renderer)
    self.assertEqual([v.arg.slot for v in prg.arg.vars], [-1, -1])
    run_linear(UOp(Ops.LINEAR, src=(prg.call(out),)), var_vals={'free_a': 2, 'free_b': 3}, wait=True)
    self.assertEqual(out.buffer.numpy().tolist(), [23])

  def test_bound_scalar_param(self):
    for scalar_slot in (0, 1):
      out = UOp.new_buffer(Device.DEFAULT, 1, dtypes.int32)
      scalar = UOp.param(scalar_slot, dtypes.int32, vmin_vmax=(0, 10), addrspace=AddrSpace.ALU)
      p = UOp.param(1-scalar_slot, dtypes.int32, 1)
      prg = to_program(p.index(0).store(scalar).sink(arg=KernelInfo(name='bounded_param'), tag=1), Device[Device.DEFAULT].renderer)
      args = (UOp.variable('caller', 0, 10, dtypes.int32).bind(4), out)
      run_linear(UOp(Ops.LINEAR, src=(prg.call(*(args if scalar_slot == 0 else args[::-1])),)), wait=True)
      self.assertEqual(out.buffer.numpy().tolist(), [4])

  def test_validate_order(self):
    x = Tensor([1, 2, 3], dtype=dtypes.int32).realize().uop
    out = UOp.new_buffer(Device.DEFAULT, 3, dtypes.int32)
    a, p, q = UOp.param(0, dtypes.int32, addrspace=AddrSpace.ALU), UOp.param(1, dtypes.int32, 3), UOp.param(2, dtypes.int32, 3)
    idx = UOp.range(3, 0)
    sink = p.index(idx).store(q.index(idx).load()*a).end(idx).sink(arg=KernelInfo(name='validate_order'), tag=1)
    with Context(VALIDATE_WITH_CPU=1):
      run_linear(UOp(Ops.LINEAR, src=(sink.call(UOp.variable('factor', 0, 10, dtypes.int32).bind(3), out, x),)), wait=True)
    self.assertEqual(out.buffer.numpy().tolist(), [3, 6, 9])

  @needs_second_gpu
  def test_multi_device_scalar_first(self):
    devices = (Device.DEFAULT, f'{Device.DEFAULT}:1')
    x = Tensor([1, 2, 3], dtype=dtypes.int32).shard(devices, axis=None).realize().uop
    out = UOp.new_buffer(devices, 3, dtypes.int32)
    a, p, q = UOp.param(0, dtypes.int32, addrspace=AddrSpace.ALU), UOp.param(1, dtypes.int32, 3), UOp.param(2, dtypes.int32, 3)
    idx, dnum = UOp.range(3, 0), UOp.variable('_device_num', 0, 1, dtypes.int32)
    sink = p.index(idx).store(q.index(idx).load()*a + dnum).end(idx).sink(arg=KernelInfo(name='multi_argument_order'), tag=1)
    run_linear(UOp(Ops.LINEAR, src=(sink.call(UOp.variable('factor', 0, 10, dtypes.int32).bind(3), out, x),)), wait=True)
    for i,buf in enumerate(out.buffer.bufs): self.assertEqual(buf.numpy().tolist(), [3+i, 6+i, 9+i])

  def test_pack_repeated_slot(self):
    # IMAGE can have distinct ABI descriptors that share a CALL slot.
    p = UOp.param(2, dtypes.float32, 4)
    image = p.replace(arg=replace(p.arg, image=(1, 1)))
    scalar = UOp.param(0, dtypes.int16, addrspace=AddrSpace.ALU)
    obj = TinyELF(b'', 'pack', Target(), (scalar, image, p))
    self.assertEqual(TinyELF.pack(obj.signature, {scalar: -3, image: 0x1000, p: 0x1000})[:24],
                     struct.pack('<h6xQQ', -3, 0x1000, 0x1000))
    self.assertEqual(TinyELF.pack((p, scalar), {scalar: -3, p: 0x1000}, 12), bytearray(12) + struct.pack('<Qh', 0x1000, -3))
    self.assertEqual(TinyELF.pack((), {}, 12), bytearray(12))

if __name__ == '__main__': unittest.main()
