import itertools, pickle, struct, unittest
from tinygrad import Device, UOp, dtypes
from tinygrad.device import Buffer, TinyELF
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.helpers import Context, Target
from tinygrad.uop.ops import KernelInfo, Ops, ParamArg, ProgramInfo

def _read_int(buf:Buffer) -> int: return struct.unpack("i", buf.as_memoryview())[0]

def _run(sink:UOp, bufs:list[Buffer], vals:dict[str, int]):
  run_linear(UOp(Ops.LINEAR, src=(sink.call(*(UOp.from_buffer(x) for x in bufs)),)), var_vals=vals, update_stats=False)

class TestArgOrder(unittest.TestCase):
  def test_two_buffers_two_scalars_all_orders(self):
    roles = ("out", "inp", "a", "b")
    for perm in itertools.permutations(roles):
      with self.subTest(order=perm):
        params = {role:UOp.param(slot, dtypes.int, () if role in ("a", "b") else (1,),
                                 name=role if role in ("a", "b") else None,
                                 addrspace=AddrSpace.ALU if role in ("a", "b") else AddrSpace.GLOBAL)
                  for slot,role in enumerate(perm)}
        sink = params["out"][0].store(params["inp"][0].load() + params["a"]*11 + params["b"]*101).sink(arg=KernelInfo("arg_order"))
        info, buf_roles = ProgramInfo.from_sink(sink), [role for role in perm if role in ("out", "inp")]
        self.assertEqual((info.globals, info.outs, info.ins), ((0, 1), (buf_roles.index("out"),), (buf_roles.index("inp"),)))
        out, inp = Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=bytes(4)), \
                   Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=struct.pack("i", 5))
        _run(sink, [out if role == "out" else inp for role in perm if role in ("out", "inp")], {"a":2, "b":3})
        self.assertEqual(_read_int(out), 330)

  def test_repeated_launch(self):
    scalar_spec = [("a", dtypes.int, 11), ("b", dtypes.int, 101)]
    spec = scalar_spec[:1] + [("out", dtypes.int, 0)] + scalar_spec[1:] + [("inp", dtypes.int, 0)]
    params = {name:UOp.param(slot, dt, (1,) if name in ("out", "inp") else (), name=None if name in ("out", "inp") else name,
                             addrspace=AddrSpace.GLOBAL if name in ("out", "inp") else AddrSpace.ALU)
              for slot,(name,dt,_) in enumerate(spec)}
    val = params["inp"][0].load()
    for name,_,weight in scalar_spec: val += params[name].cast(dtypes.int)*weight
    sink = params["out"][0].store(val).sink(arg=KernelInfo("arg_dtypes"))
    cases = [(5, {name:i+1 for i,(name,_,_) in enumerate(scalar_spec)}),
             (7, {name:len(scalar_spec)-i for i,(name,_,_) in enumerate(scalar_spec)})]
    for inp_val, vals in cases:
      out = Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=bytes(4))
      inp = Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=struct.pack("i", inp_val))
      _run(sink, [out, inp], vals)
      self.assertEqual(_read_int(out), inp_val + sum(vals[name]*weight for name,_,weight in scalar_spec))

  def test_mixed_scalar_widths(self):
    if not all(dt in Device[Device.DEFAULT].renderer.supported_dtypes() for dt in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64)):
      self.skipTest("requires 8/16/32/64-bit integer support")
    spec = (("a", dtypes.int8), ("out", dtypes.int64), ("b", dtypes.int16), ("inp", dtypes.int64),
            ("c", dtypes.int32), ("d", dtypes.int64))
    params = {name:UOp.param(slot, dt, (1,) if name in ("out", "inp") else (), name=None if name in ("out", "inp") else name,
                             addrspace=AddrSpace.GLOBAL if name in ("out", "inp") else AddrSpace.ALU)
              for slot,(name,dt) in enumerate(spec)}
    value = params["inp"][0].load()
    for name in ("a", "b", "c", "d"): value += params[name].cast(dtypes.int64)
    sink = params["out"][0].store(value).sink(arg=KernelInfo("arg_widths"))
    vals = {"a":-3, "b":2000, "c":-100000, "d":2**35}
    out = Buffer(Device.DEFAULT, 1, dtypes.int64, initial_value=bytes(8))
    inp = Buffer(Device.DEFAULT, 1, dtypes.int64, initial_value=struct.pack("q", 17))
    _run(sink, [out, inp], vals)
    self.assertEqual(struct.unpack("q", out.as_memoryview())[0], 17 + sum(vals.values()))

  @unittest.skipUnless(Device.DEFAULT == "CL", "requires OpenCL images")
  def test_image_buffer_interleaving(self):
    a = UOp.param(0, dtypes.int, (), name="a", addrspace=AddrSpace.ALU)
    out = UOp.param(1, dtypes.float, (32,))
    inp = UOp.param(2, dtypes.float, (1,))
    b = UOp.param(3, dtypes.int, (), name="b", addrspace=AddrSpace.ALU)
    image = UOp.param(4, dtypes.float, (32,))
    value = inp[0].load() + a.cast(dtypes.float) + b.cast(dtypes.float)
    sink = UOp.sink(*(out.index(i).store(image.index(i).load() + value) for i in range(4)), arg=KernelInfo("arg_order_image"))
    out_buf = Buffer(Device.DEFAULT, 32, dtypes.float, initial_value=bytes(128))
    inp_buf = Buffer(Device.DEFAULT, 1, dtypes.float, initial_value=struct.pack("f", 4.0))
    image_buf = Buffer(Device.DEFAULT, 32, dtypes.float, initial_value=struct.pack("4f", 1.0, 2.0, 3.0, 4.0) + bytes(112))
    with Context(IMAGE=1): _run(sink, [out_buf, inp_buf, image_buf], {"a":2, "b":3})
    self.assertEqual(struct.unpack("4f", out_buf.as_memoryview()[:16]), (10.0, 11.0, 12.0, 13.0))

  def test_graph(self):
    if Device[Device.DEFAULT].graph is None: self.skipTest("graph support required")
    a, inp, mid = UOp.param(0, dtypes.int, (), name="a", addrspace=AddrSpace.ALU), UOp.param(1, dtypes.int, (1,)), \
                  UOp.param(2, dtypes.int, (1,))
    first = mid[0].store(inp[0].load() + a).sink(arg=KernelInfo("arg_order_graph_first"))
    second_inp, b, out = UOp.param(0, dtypes.int, (1,)), UOp.param(1, dtypes.int, (), name="b", addrspace=AddrSpace.ALU), \
                         UOp.param(2, dtypes.int, (1,))
    second = out[0].store(second_inp[0].load() + b).sink(arg=KernelInfo("arg_order_graph_second"))
    inputs = tuple(UOp.param(i, dtypes.int, (1,), device=Device.DEFAULT) for i in range(3))
    first_call, second_call = first.call(*inputs[:2]), second.call(inputs[1], inputs[2])
    def make_inputs(inp_value:int):
      bufs = (Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=struct.pack("i", inp_value)).ensure_allocated(),
              Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=bytes(4)).ensure_allocated(),
              Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=bytes(4)).ensure_allocated())
      return bufs, tuple(UOp.from_buffer(buf) for buf in bufs)
    _, initial_inputs = make_inputs(5)
    graph = Device[Device.DEFAULT].graph(
      UOp(Ops.CUSTOM_FUNCTION, src=(compile_linear(UOp(Ops.LINEAR, src=(first_call, second_call))),), arg="graph"), initial_inputs)
    for inp_value, vals in ((5, {"a":2, "b":3}), (7, {"a":4, "b":1})):
      bufs, input_uops = make_inputs(inp_value)
      graph(input_uops, vals, wait=True)
      self.assertEqual(_read_int(bufs[2]), inp_value + vals["a"] + vals["b"])

  def test_signature_layout_and_pickle(self):
    signature = ((ParamArg(0, dtypes.int8, name="i8", addrspace=AddrSpace.ALU), ()),
                 (ParamArg(1, dtypes.float), (2, 2, 4)),
                 (ParamArg(2, dtypes.int16, name="i16", addrspace=AddrSpace.ALU), ()),
                 (ParamArg(3, dtypes.int), ()),
                 (ParamArg(4, dtypes.int32, name="i32", addrspace=AddrSpace.ALU), ()),
                 (ParamArg(5, dtypes.int64, name="i64", addrspace=AddrSpace.ALU), ()))
    elf = TinyELF(b"", "layout", Target(), signature)
    self.assertEqual([(off, arg.addrspace, shape, idx) for off,arg,shape,idx in TinyELF.iter_sig(elf.signature)],
                     [(0, AddrSpace.ALU, (), 0), (8, AddrSpace.GLOBAL, (2, 2, 4), 0), (16, AddrSpace.ALU, (), 1),
                      (24, AddrSpace.GLOBAL, (), 1), (32, AddrSpace.ALU, (), 2), (40, AddrSpace.ALU, (), 3)])
    self.assertEqual(pickle.loads(pickle.dumps(elf)), elf)

  def test_manual_program_signature(self):
    out = UOp.param(0, dtypes.int, (1,))
    val = UOp.param(1, dtypes.int, (), name="val", addrspace=AddrSpace.ALU)
    inp = UOp.param(2, dtypes.int, (1,))
    sink = out[0].store(inp[0].load() + val).sink(arg=KernelInfo("manual_signature"))
    prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=b"")), arg=ProgramInfo.from_sink(sink))
    self.assertEqual([(arg.slot, arg.name, arg.addrspace) for arg,_ in prg.to_elf().signature],
                     [(0, None, AddrSpace.GLOBAL), (1, "val", AddrSpace.ALU), (2, None, AddrSpace.GLOBAL)])

if __name__ == "__main__": unittest.main()
