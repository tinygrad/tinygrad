import unittest
from tinygrad import Device, Tensor
from tinygrad.engine.jit import TinyJit, GraphRunner
from tinygrad.engine.realize import CompiledRunner
from tinygrad.engine.schedule import linear_to_schedule
from tinygrad.engine.memory import memory_plan_rewrite
from tinygrad.uop.ops import UOp, Ops, buffers
from tinygrad.runtime.graph.hcq import HCQGraph
from tinygrad.runtime.support.hcq import HCQCompiled
from tinygrad.runtime.support.usb import USBMMIOInterface
from test.mockgpu.usb import MockUSB

@unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "HCQ device required to run")
class TestHCQUnit(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT == "CPU", "requires non-CPU HCQ device")
  def test_supports_exec_item(self):
    d0, cpu_dev = Device[Device.DEFAULT], Device["CPU"]

    @TinyJit
    def f(inp, inp_cpu):
      return (inp + 1.0).contiguous().realize(), (inp_cpu + 1.0).contiguous().realize()
    inp, inp_cpu = Tensor.randn(10, 10, device=Device.DEFAULT).realize(), Tensor.randn(10, 10, device="CPU").realize()
    for _ in range(5): f(inp, inp_cpu)

    # find GPU and CPU CALL UOps from the jit_cache graph's sub_linear
    gpu_call, cpu_call, gpu_devs = None, None, []
    for ji in f.captured.jit_cache:
      if hasattr(ji.prg, 'jit_cache'):
        for inner in ji.prg.jit_cache:
          if isinstance(inner.prg, CompiledRunner):
            # reconstruct a CALL-like UOp from the ExecItem for supports_exec_item
            buf_uops = [UOp.new_buffer(b.device, b.size, b.dtype) for b in inner.bufs if b is not None]
            call = inner.ast.call(*buf_uops)
            if inner.prg.dev._is_cpu(): cpu_call = call
            else:
              gpu_call = call
              if inner.prg.dev not in gpu_devs: gpu_devs.append(inner.prg.dev)
      elif isinstance(ji.prg, CompiledRunner):
        buf_uops = [UOp.new_buffer(ji.prg.device, b.size, b.dtype) for b in ji.bufs if b is not None]
        call = ji.ast.call(*buf_uops)
        if ji.prg.dev._is_cpu(): cpu_call = call
        else:
          gpu_call = call
          if ji.prg.dev not in gpu_devs: gpu_devs.append(ji.prg.dev)
    assert gpu_call is not None and cpu_call is not None and len(gpu_devs) > 0

    # local MMIO: GPU works alone and with CPU in batch (cpu_support=True)
    assert HCQGraph.supports_exec_item(gpu_devs, gpu_call) is True
    assert HCQGraph.supports_exec_item(gpu_devs, cpu_call) is True
    assert HCQGraph.supports_exec_item(gpu_devs + [cpu_dev], gpu_call) is True

    # USB MMIO: GPU-only still works, but CPU batching must be rejected (cpu_support=False)
    orig_view = d0.timeline_signal.base_buf.view
    try:
      d0.timeline_signal.base_buf.view = USBMMIOInterface(MockUSB(bytearray(256)), 0, 16, fmt='B')
      assert HCQGraph.supports_exec_item(gpu_devs, gpu_call) is True
      assert HCQGraph.supports_exec_item(gpu_devs, cpu_call) is False
      assert HCQGraph.supports_exec_item(gpu_devs + [cpu_dev], gpu_call) is False
    finally:
      d0.timeline_signal.base_buf.view = orig_view

if __name__ == "__main__":
  unittest.main()
