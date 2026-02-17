import unittest
from tinygrad import Device, Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.realize import CompiledRunner
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

    gpu_ei, cpu_ei, gpu_devs = None, None, []
    for ji in f.captured.jit_cache:
      if isinstance(ji.prg, CompiledRunner):
        if ji.prg.dev._is_cpu(): cpu_ei = ji
        else:
          gpu_ei = ji
          if ji.prg.dev not in gpu_devs: gpu_devs.append(ji.prg.dev)
    assert gpu_ei is not None and cpu_ei is not None and len(gpu_devs) > 0

    # local MMIO: GPU works alone and with CPU in batch (cpu_support=True)
    assert HCQGraph.supports_exec_item(gpu_devs, gpu_ei) is True
    assert HCQGraph.supports_exec_item(gpu_devs, cpu_ei) is True
    assert HCQGraph.supports_exec_item(gpu_devs + [cpu_dev], gpu_ei) is True

    # USB MMIO: GPU-only still works, but CPU batching must be rejected (cpu_support=False)
    orig_view = d0.timeline_signal.base_buf.view
    try:
      d0.timeline_signal.base_buf.view = USBMMIOInterface(MockUSB(bytearray(256)), 0, 16, fmt='B')
      assert HCQGraph.supports_exec_item(gpu_devs, gpu_ei) is True
      assert HCQGraph.supports_exec_item(gpu_devs, cpu_ei) is False
      assert HCQGraph.supports_exec_item(gpu_devs + [cpu_dev], gpu_ei) is False
    finally:
      d0.timeline_signal.base_buf.view = orig_view

if __name__ == "__main__":
  unittest.main()
