import functools, queue
from tinygrad.helpers import capstone_flatdump
from tinygrad.renderer.isa import X86Renderer
from tinygrad.runtime.support.hcq import HCQCompiled
from tinygrad.runtime.ops_cpu import CPUWorker, CPUAllocator, CPUProgram, Compiler, CPUSignal, CPUComputeQueue
from tinygrad.muop import MUOp

class X86Compiler(Compiler):
  def __init__(self): super().__init__(None)
  def compile(self, src:list[MUOp]): return MUOp.assemble(src)
  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

class X86Device(HCQCompiled):
  def __init__(self, device:str):
    self.tasks:queue.Queue = queue.Queue()
    CPUWorker(self, self.tasks, thread_id=0).start()
    super().__init__(device, CPUAllocator(self), X86Renderer(), X86Compiler(), functools.partial(CPUProgram, self), CPUSignal, CPUComputeQueue)
