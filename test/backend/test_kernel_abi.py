import array, ctypes, struct, unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad import Context, UOp, dtypes, Tensor, TinyJit
from tinygrad.codegen import to_program
from tinygrad.device import Buffer, Device, TinyELF
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import compile_linear, get_call_arg_uops, get_call_outs_ins, run_linear
from tinygrad.helpers import Target, HCQ2
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.runtime.support.hcq2 import encode_kernargs_clike
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo


@dataclass(frozen=True)
class FakeProgramData:
  kernargs_alloc_size:int


def mixed_signature() -> tuple:
  return ((None, 0, dtypes.float, (1,), AddrSpace.GLOBAL), ("val", 1, dtypes.int, (), AddrSpace.ALU),
          (None, 2, dtypes.float, (1,), AddrSpace.GLOBAL))


def make_mixed_call(order:tuple[str, ...], out_buf:Buffer, inp_buf:Buffer, scalar:int) -> UOp:
  params = {
    name: UOp.param(slot, dtypes.int if name == "val" else dtypes.float, () if name == "val" else 1,
                    vmin_vmax=(0, 100) if name == "val" else None, name="mixed_val" if name == "val" else None,
                    addrspace=AddrSpace.ALU if name == "val" else AddrSpace.GLOBAL)
    for slot,name in enumerate(order)
  }
  sink = params["out"][0].store(params["inp"][0].load() + params["val"].cast(dtypes.float)).sink(arg=KernelInfo("mixed_abi"))
  unused_buf = Buffer(out_buf.device, 1, dtypes.float, preallocate=True)
  call_args = {"out": UOp.from_buffer(out_buf), "inp": UOp.from_buffer(inp_buf), "unused": UOp.from_buffer(unused_buf),
               "val": UOp.variable("mixed_val", 0, 100, dtypes.int).bind(scalar)}
  return sink.call(*(call_args[name] for name in order))


def make_two_scalar_call(order:tuple[str, ...], out_buf:Buffer, inp_buf:Buffer, x:int, y:int) -> UOp:
  params = {
    name: UOp.param(slot, dtypes.int if name in {"x", "y"} else dtypes.float, () if name in {"x", "y"} else 1,
                    vmin_vmax=(0, 100) if name in {"x", "y"} else None, name=name if name in {"x", "y"} else None,
                    addrspace=AddrSpace.ALU if name in {"x", "y"} else AddrSpace.GLOBAL)
    for slot,name in enumerate(order)
  }
  result = params["inp"][0].load() + params["x"].cast(dtypes.float) + params["y"].cast(dtypes.float)
  sink = params["out"][0].store(result).sink(arg=KernelInfo("mixed_abi_two_scalars"))
  call_args = {"out": UOp.from_buffer(out_buf), "inp": UOp.from_buffer(inp_buf),
               "x": UOp.variable("x", 0, 100, dtypes.int).bind(x), "y": UOp.variable("y", 0, 100, dtypes.int).bind(y)}
  return sink.call(*(call_args[name] for name in order))


def cpu_buffer(values:list[float]) -> Buffer:
  buf = Buffer("CPU", len(values), dtypes.float, preallocate=True)
  buf.as_memoryview(force_zero_copy=True).cast("f")[:] = array.array("f", values)
  return buf


def device_buffer(device:str, values:list[float]) -> Buffer:
  buf = Buffer(device, len(values), dtypes.float, preallocate=True)
  src = Buffer("PYTHON", len(values), dtypes.float, opaque=memoryview(bytearray(array.array("f", values).tobytes())))
  buf.copy_from(src)
  return buf


def run_mixed_cpu(order:tuple[str, ...], scalar:int=3, inp:float=5, hcq2:int=0, debug:int=0) -> float:
  out_buf, inp_buf = cpu_buffer([0]), cpu_buffer([inp])
  call = make_mixed_call(order, out_buf, inp_buf, scalar)
  with Context(HCQ2=hcq2, DEBUG=debug):
    run_linear(UOp(Ops.LINEAR, src=(call,)), var_vals={"mixed_val": scalar}, update_stats=debug == 0)
  return float(out_buf.numpy()[0])


def run_mixed_device(device:str, order:tuple[str, ...], scalar:int=3, inp:float=5) -> float:
  out_buf, inp_buf = device_buffer(device, [0]), device_buffer(device, [inp])
  call = make_mixed_call(order, out_buf, inp_buf, scalar)
  run_linear(UOp(Ops.LINEAR, src=(call,)), var_vals={"mixed_val": scalar}, update_stats=False)
  return float(out_buf.numpy()[0])


def run_two_scalar_device(device:str, order:tuple[str, ...], x:int=3, y:int=4, inp:float=5) -> float:
  out_buf, inp_buf = device_buffer(device, [0]), device_buffer(device, [inp])
  call = make_two_scalar_call(order, out_buf, inp_buf, x, y)
  run_linear(UOp(Ops.LINEAR, src=(call,)), var_vals={"x": x, "y": y}, update_stats=False)
  return float(out_buf.numpy()[0])


class TestKernelABIMetadata(unittest.TestCase):
  def test_signature_preserves_interleaved_slots(self):
    out = UOp.param(0, dtypes.float, 1)
    val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, 100), name="val", addrspace=AddrSpace.ALU)
    inp = UOp.param(2, dtypes.float, 1)
    sink = out[0].store(inp[0].load() + val.cast(dtypes.float)).sink(arg=KernelInfo("mixed_args"))

    elf = to_program(sink, PythonRenderer(Target("PYTHON"))).to_elf()
    self.assertEqual([slot for _,slot,_,_,_ in elf.signature], [0, 1, 2])
    self.assertEqual([space for *_,space in elf.signature], [AddrSpace.GLOBAL, AddrSpace.ALU, AddrSpace.GLOBAL])
    self.assertEqual([offset for offset,_,_ in TinyELF.iter_sig(elf.signature)], [0, 8, 16])

  def test_signature_preserves_buffers_first(self):
    out = UOp.param(0, dtypes.float, 1)
    inp = UOp.param(1, dtypes.float, 1)
    val = UOp.param(2, dtypes.int, (), vmin_vmax=(0, 100), name="val", addrspace=AddrSpace.ALU)
    sink = out[0].store(inp[0].load() + val.cast(dtypes.float)).sink(arg=KernelInfo("buffers_first"))
    elf = to_program(sink, PythonRenderer(Target("PYTHON"))).to_elf()
    self.assertEqual([slot for _,slot,_,_,_ in elf.signature], [0, 1, 2])

  def test_dependency_indices_use_compact_buffer_positions(self):
    out_buf, inp_buf = cpu_buffer([0]), cpu_buffer([5])
    call = make_mixed_call(("out", "unused", "val", "inp"), out_buf, inp_buf, 3)
    program = to_program(call.src[0], PythonRenderer(Target("PYTHON")))
    compiled_call = program.call(*call.src[1:])
    self.assertEqual(get_call_arg_uops(compiled_call), (compiled_call.src[1], compiled_call.src[2], compiled_call.src[4]))
    self.assertEqual(get_call_outs_ins(compiled_call), ((0,), (2,)))

  def test_signature_maps_split_runtime_arguments_in_slot_order(self):
    self.assertEqual([arg for _,arg in TinyELF.iter_args(mixed_signature(), ("out", "inp"), (7,))], ["out", 7, "inp"])
    scalar_first = (("val", 0, dtypes.int, (), AddrSpace.ALU), (None, 1, dtypes.float, (1,), AddrSpace.GLOBAL))
    self.assertEqual([arg for _,arg in TinyELF.iter_args(scalar_first, ("out",), (9,))], [9, "out"])

  def test_signature_rejects_extra_or_missing_runtime_arguments(self):
    with self.assertRaises(AssertionError): list(TinyELF.iter_args(mixed_signature(), ("out",), (7,)))
    with self.assertRaises(AssertionError): list(TinyELF.iter_args(mixed_signature(), ("out", "inp", "extra"), (7,)))


class TestCPUKernelABI(unittest.TestCase):
  def test_timing_scratch_buffers_preserve_unused_slots(self):
    from tinygrad.codegen.opt.postrange import args_from_ast
    from tinygrad.codegen.opt.search import _time_program
    out, inp = cpu_buffer([0]), cpu_buffer([5])
    call = make_mixed_call(("val", "unused", "out", "inp"), out, inp, 3)
    program = to_program(call.src[0], Device["CPU"].renderer)
    scratch, values = args_from_ast(program.src[0], "CPU")
    scratch[3].ensure_allocated().as_memoryview(force_zero_copy=True).cast('f')[:] = array.array('f', [5])
    with Context(HCQ2=0):
      self.assertEqual(len(_time_program(program, values, scratch, cnt=1, allow_test_size=False)), 1)
    self.assertEqual(scratch[2].numpy().tolist(), [5 + values['mixed_val']])

  def test_direct_execution_argument_orders(self):
    for order in (("out", "val", "inp"), ("val", "out", "inp"), ("out", "inp", "val")):
      with self.subTest(order=order): self.assertEqual(run_mixed_cpu(order), 8)

  def test_hcq2_execution_argument_orders(self):
    for order in (("out", "val", "inp"), ("val", "out", "inp"), ("out", "inp", "val")):
      with self.subTest(order=order): self.assertEqual(run_mixed_cpu(order, hcq2=1), 8)

  def test_unused_slot_does_not_shift_arguments(self):
    for hcq2 in (0, 1):
      with self.subTest(hcq2=hcq2): self.assertEqual(run_mixed_cpu(("out", "unused", "val", "inp"), hcq2=hcq2), 8)

  def test_repeated_execution_updates_buffers_and_scalar(self):
    self.assertEqual(run_mixed_cpu(("out", "val", "inp"), scalar=3, inp=5), 8)
    self.assertEqual(run_mixed_cpu(("out", "val", "inp"), scalar=9, inp=4), 13)

  def test_scalar_first_debug_stats_path(self):
    self.assertEqual(run_mixed_cpu(("val", "out", "inp"), debug=2), 8)

  def test_cpu_validation_preserves_mixed_and_unused_slots(self):
    with Context(VALIDATE_WITH_CPU=1):
      for order in (("out", "val", "inp"), ("val", "unused", "out", "inp")):
        with self.subTest(order=order): self.assertEqual(run_mixed_cpu(order), 8)

  def test_core_id_uses_compact_signature_position(self):
    for hcq2 in (0, 1):
      with self.subTest(hcq2=hcq2):
        out = UOp.param(0, dtypes.float, 4)
        core = UOp.param(2, dtypes.int, (), vmin_vmax=(0, 3), name="core_id", addrspace=AddrSpace.ALU)
        sink = out[core].store(core.cast(dtypes.float)).sink(arg=KernelInfo("core_slot"))
        out_buf, unused_buf = cpu_buffer([0, 0, 0, 0]), cpu_buffer([0])
        args = (UOp.from_buffer(out_buf), UOp.from_buffer(unused_buf), UOp.variable("core_id", 0, 3, dtypes.int).bind(0))
        with Context(HCQ2=hcq2): run_linear(UOp(Ops.LINEAR, src=(sink.call(*args),)), update_stats=False)
        self.assertEqual(out_buf.numpy().tolist(), [0, 1, 2, 3])

  def test_native_runtime_packs_interleaved_arguments(self):
    program = object.__new__(CPUProgram)
    program.lvp, program.runtimevars = False, {}
    program.signature = ((None, 0, dtypes.float, (1,), AddrSpace.GLOBAL), ("val", 1, dtypes.int, (), AddrSpace.ALU),
                         (None, 2, dtypes.float, (1,), AddrSpace.GLOBAL))
    captured = []
    program.fxn = lambda *args: captured.append(tuple(x.value for x in args))
    program(SimpleNamespace(va_addr=0x1111), SimpleNamespace(va_addr=0x2222), vals=(7,))
    self.assertEqual(captured, [(0x1111, 7, 0x2222)])

  def test_lvp_runtime_packs_interleaved_payload(self):
    program = object.__new__(CPUProgram)
    program.lvp = True
    program.signature = ((None, 0, dtypes.float, (1,), AddrSpace.GLOBAL), ("val", 1, dtypes.int, (), AddrSpace.ALU),
                         (None, 2, dtypes.float, (1,), AddrSpace.GLOBAL))
    captured:dict[str, bytes|int] = {}

    def capture(header_addr:int):
      lo, hi, dwords = struct.unpack("<3I", ctypes.string_at(header_addr, 12))
      captured["dwords"] = dwords
      captured["payload"] = ctypes.string_at(lo | hi << 32, dwords * 4)

    program.fxn = capture
    program(SimpleNamespace(va_addr=0x1111), SimpleNamespace(va_addr=0x2222), vals=(7,))
    payload = captured["payload"]
    assert isinstance(payload, bytes)
    self.assertEqual(captured["dwords"], 6)
    self.assertEqual(struct.unpack_from("<Q", payload, 0)[0], 0x1111)
    self.assertEqual(struct.unpack_from("<i", payload, 8)[0], 7)
    self.assertEqual(payload[12:16], bytes(4))
    self.assertEqual(struct.unpack_from("<Q", payload, 16)[0], 0x2222)


class TestHCQ2KernelABI(unittest.TestCase):
  def test_clike_kernargs_preserve_slots_and_alignment(self):
    out = UOp.placeholder((1,), dtypes.float, 100, device="CPU")
    inp = UOp.placeholder((1,), dtypes.float, 200, device="CPU")
    val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, 100), name="val", addrspace=AddrSpace.ALU)
    info = ProgramInfo(vars=(val,), globals=(0, 2), target=Target("CPU"))
    program = UOp(Ops.PROGRAM, arg=(FakeProgramData(24), info))
    call = program.call(out, UOp.variable("val", 0, 100, dtypes.int).bind(7), inp)
    captured = []
    with patch("tinygrad.runtime.support.hcq2.make_patches", side_effect=lambda _, patches: captured.extend(patches) or ()):
      encode_kernargs_clike(call, program, "CPU")
    self.assertEqual([offset for offset,_ in captured], [0, 8, 16])
    self.assertEqual([value.op for _,value in captured], [Ops.GETADDR, Ops.PARAM, Ops.GETADDR])
    self.assertIs(captured[1][1], val)


class TestHIPKernelABI(unittest.TestCase):
  def test_runtime_struct_preserves_interleaved_layout(self):
    from tinygrad.runtime.ops_hip import encode_args
    c_args, _ = encode_args((0x1111, 0x2222), (7,), mixed_signature())
    self.assertEqual(ctypes.sizeof(c_args), 24)
    self.assertEqual([(name, offset) for name,_,offset in c_args._real_fields_], [("f0", 0), ("v0", 8), ("f1", 16)])
    self.assertEqual((c_args.f0, c_args.v0, c_args.f1), (0x1111, 7, 0x2222))

  def test_runtime_struct_handles_multiple_interleaved_scalars(self):
    from tinygrad.runtime.ops_hip import encode_args
    signature = (("x", 0, dtypes.int, (), AddrSpace.ALU), (None, 1, dtypes.float, (1,), AddrSpace.GLOBAL),
                 ("y", 2, dtypes.int, (), AddrSpace.ALU), (None, 3, dtypes.float, (1,), AddrSpace.GLOBAL))
    c_args, _ = encode_args((0x1111, 0x2222), (3, 4), signature)
    self.assertEqual(ctypes.sizeof(c_args), 32)
    self.assertEqual([(name, offset) for name,_,offset in c_args._real_fields_], [("v0", 0), ("f0", 8), ("v1", 16), ("f1", 24)])
    self.assertEqual((c_args.v0, c_args.f0, c_args.v1, c_args.f1), (3, 0x1111, 4, 0x2222))

  def test_cached_runtime_struct_updates_buffers_and_scalar(self):
    from tinygrad.runtime import ops_hip
    program = object.__new__(ops_hip.HIPProgram)
    program.dev, program.prg, program.signature = SimpleNamespace(device_id=0), object(), mixed_signature()
    with patch.object(ops_hip, "check"), patch.object(ops_hip.hip, "hipSetDevice", return_value=0), \
         patch.object(ops_hip.hip, "hipModuleLaunchKernel", return_value=0):
      program(0x1111, 0x2222, vals=(7,))
      program(0x3333, 0x4444, vals=(9,))
    self.assertEqual((program.c_args.f0, program.c_args.v0, program.c_args.f1), (0x3333, 9, 0x4444))


class TestCUDAKernelABI(unittest.TestCase):
  def test_runtime_struct_preserves_interleaved_layout(self):
    from tinygrad.runtime.ops_cuda import encode_args
    c_args, _ = encode_args((0x1111, 0x2222), (7,), mixed_signature())
    self.assertEqual(ctypes.sizeof(c_args), 24)
    self.assertEqual([(name, offset) for name,_,offset in c_args._real_fields_], [("f0", 0), ("v0", 8), ("f1", 16)])
    self.assertEqual((c_args.f0, c_args.v0, c_args.f1), (0x1111, 7, 0x2222))


@unittest.skipUnless(Device.DEFAULT in {"CUDA", "HIP", "CL", "METAL", "WEBGPU", "AMD", "NV", "QCOM", "DSP"}, "mixed runtime backend required")
class TestRuntimeKernelABI(unittest.TestCase):
  def test_direct_execution_argument_orders(self):
    for order in (("out", "val", "inp"), ("val", "out", "inp"), ("out", "inp", "val")):
      with self.subTest(device=Device.DEFAULT, order=order): self.assertEqual(run_mixed_device(Device.DEFAULT, order), 8)

  def test_repeated_execution_updates_buffers_and_scalar(self):
    self.assertEqual(run_mixed_device(Device.DEFAULT, ("out", "val", "inp"), scalar=3, inp=5), 8)
    self.assertEqual(run_mixed_device(Device.DEFAULT, ("out", "val", "inp"), scalar=9, inp=4), 13)

  def test_unused_slot_does_not_shift_runtime_bindings(self):
    self.assertEqual(run_mixed_device(Device.DEFAULT, ("out", "unused", "val", "inp")), 8)

  def test_scalar_buffer_scalar_buffer_order(self):
    self.assertEqual(run_two_scalar_device(Device.DEFAULT, ("x", "out", "y", "inp")), 12)


class TestMixedABIReplay(unittest.TestCase):
  def test_jit_replays_changed_input_buffers_and_scalars(self):
    out = UOp.param(0, dtypes.float, 1)
    val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, 100), name="mixed_val", addrspace=AddrSpace.ALU)
    inp = UOp.param(2, dtypes.float, 1)
    sink = out[0].store(inp[0].load() + val.cast(dtypes.float)).sink(arg=KernelInfo("mixed_jit"))

    @TinyJit
    def run(inp_tensor, scalar):
      output = Tensor.empty_like(inp_tensor)
      return Tensor(output.uop.after(sink.call(output.uop, scalar, inp_tensor.uop))).realize()

    for value, scalar in ((5, 3), (4, 9), (8, 2), (10, 7), (1, 6)):
      actual = run(Tensor([value], dtype=dtypes.float).realize(), UOp.variable("mixed_val", 0, 100).bind(scalar))
      self.assertEqual(actual.item(), value + scalar)

  @unittest.skipUnless(Device.DEFAULT in {"METAL", "AMD", "NV", "QCOM"}, "graph backend required")
  def test_graph_replays_scalar_first_unused_slots_and_input_buffers(self):
    if HCQ2 and Device.DEFAULT == "AMD": self.skipTest("legacy HCQ graph needs an AMD device initialized with HCQ2=0")
    from tinygrad.runtime.graph.hcq import HCQGraph
    graph_type = HCQGraph
    if Device.DEFAULT == "METAL":
      from tinygrad.runtime.graph.metal import MetalGraph
      graph_type = MetalGraph
    for order in (("val", "out", "inp"), ("val", "unused", "out", "inp"), ("out", "val", "inp")):
      with self.subTest(order=order), Context(HCQ2=0):
        output, first = device_buffer(Device.DEFAULT, [0]), device_buffer(Device.DEFAULT, [5])
        call = make_mixed_call(order, output, first, 3)
        linear = compile_linear(UOp(Ops.LINEAR, src=(call,)))
        linear = linear.substitute({UOp.from_buffer(first): UOp.param(0, dtypes.float, 1, Device.DEFAULT)}, walk=True)
        graph = graph_type(UOp.custom_function("graph", linear), input_uops=(UOp.from_buffer(first),))
        for value, scalar in ((5, 3), (4, 9), (8, 2)):
          new_input = device_buffer(Device.DEFAULT, [value])
          graph((UOp.from_buffer(new_input),), {"mixed_val": scalar}, wait=True)
          self.assertEqual(float(output.numpy()[0]), value + scalar)


@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required")
class TestMetalGraphKernelABI(unittest.TestCase):
  def test_graph_execution_and_scalar_replay_preserve_interleaved_order(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    out_buf, inp_buf = device_buffer("METAL", [0]), device_buffer("METAL", [5])
    call = make_mixed_call(("out", "val", "inp"), out_buf, inp_buf, 3)
    linear = compile_linear(UOp(Ops.LINEAR, src=(call,)))
    graph = MetalGraph(UOp.custom_function("graph", linear))
    graph((), {"mixed_val": 3}, wait=True)
    self.assertEqual(float(out_buf.numpy()[0]), 8)
    graph((), {"mixed_val": 9}, wait=True)
    self.assertEqual(float(out_buf.numpy()[0]), 14)


if __name__ == "__main__":
  unittest.main()
