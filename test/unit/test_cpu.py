import unittest, io, os, subprocess, sys
from contextlib import redirect_stdout
from tinygrad import Tensor, Device, UOp
from tinygrad.helpers import Target
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.codegen import to_program
from tinygrad.runtime.ops_cpu import RING_SLOTS
from tinygrad.uop.ops import AxisType, KernelInfo

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestCPU(unittest.TestCase):
  def test_parallel_workers_exit_cleanly(self):
    env = os.environ.copy()
    env.update(DEV="CPU", CPU_PARALLEL_UOPS="1")
    proc = subprocess.run([sys.executable, "-c",
      "from tinygrad import Tensor; assert Tensor.arange(32).sum().item() == 496"], env=env, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, proc.stderr)

  def test_32_buffer_kernel(self):
    def add_inputs(out:UOp, *inputs:UOp) -> UOp:
      return out[0].store(sum((x[0] for x in inputs), start=UOp.const(out.dtype, 0))).sink(
        arg=KernelInfo(name="add_31_inputs", opts_to_apply=()))
    inputs = [Tensor([i], device="CPU").realize() for i in range(31)]
    out = Tensor.custom_kernel(Tensor.empty(1, device="CPU"), *inputs, fxn=add_inputs)[0]
    self.assertEqual(out.item(), sum(range(31)))

  def test_command_ring_backpressure(self):
    dev, count = Device["CPU"], RING_SLOTS + 257
    signal, queue = dev.new_signal(value=0), dev.hw_compute_queue_t()
    for value in range(1, count + 1): queue.signal(signal, value)
    queue.submit(dev)
    signal.wait(count, timeout=10000)
    self.assertEqual(signal.value, count)

  def test_parallel_launch(self):
    def fill(out:UOp) -> UOp:
      idx = UOp.range(67, 0, AxisType.GLOBAL)
      return out[idx].store(idx).end(idx).sink(arg=KernelInfo(name="parallel_launch", optimize=False, parallel=True))
    probe = Tensor.custom_kernel(Tensor.empty(67, device="CPU"), fxn=fill)[0]
    self.assertTrue(to_program(probe.schedule_linear().src[-1].src[0], Device["CPU"].renderer).arg.parallel)
    out = Tensor.custom_kernel(Tensor.empty(67, device="CPU"), fxn=fill)[0]
    self.assertEqual(out.tolist(), list(range(67)))

  def test_arch_feats(self):
    ast = (Tensor.empty(16) + Tensor.empty(16)).schedule_linear().src[-1].src[0]
    for ren in Device[Device.DEFAULT].renderers:
      for arch, expect_vmov in [("x86_64,x86-64,avx", True), ("x86_64,x86-64,-avx", False)]:
        with self.subTest(arch=arch):
          if ren is X86Renderer: continue # X86 requires avx support
          if ren is LVPRenderer: continue # LVP does not play nice with cross compilation
          r = ren(Target(device="CPU", arch=arch))
          p = to_program(ast, r)
          lib = r.compiler.compile(p.src[2].arg)
          out = io.StringIO()
          with redirect_stdout(out): r.compiler.disassemble(lib)
          self.assertEqual("vmov" in out.getvalue(), expect_vmov, out.getvalue())

if __name__ == '__main__':
  unittest.main()
