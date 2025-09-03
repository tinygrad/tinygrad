# import ctypes, platform, functools, queue
# from tinygrad.device import Compiler
# from tinygrad.runtime.support.hcq import HCQCompiled, HCQSignal
# from tinygrad.runtime.ops_cpu import CPUAllocator, CPUProgram, CPUComputeQueue, CPUWorker
# from tinygrad.helpers import OSX, getenv, capstone_flatdump, DEBUG
# from tinygrad.runtime.support.elf import jit_loader

# class LLVMDevice(HCQCompiled):
#   def __init__(self, device:str=""):
#     self.tasks:queue.Queue = queue.Queue()
#     CPUWorker(self).start()
#     super().__init__(device, CPUAllocator(self), LLVMRenderer(), HostLLVMCompiler(), functools.partial(CPUProgram, self), HCQSignal, CPUComputeQueue)
