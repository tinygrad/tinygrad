from __future__ import annotations
import ctypes, functools
from typing import Tuple
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import DEBUG, cpu_time_execution, cpu_objdump, getenv
from tinygrad.renderer.llvmir import LLVMRenderer
import llvmlite.binding as llvm

class LLVMCompiler(Compiler):
  def __init__(self, device:LLVMDevice, opt:bool=False):
    self.device = device
    self.optimizer: llvm.passmanagers.ModulePassManager = llvm.create_module_pass_manager()
    self.device.target_machine.add_analysis_passes(self.optimizer)
    if opt:
      with llvm.create_pass_manager_builder() as builder:
        builder.opt_level = 3; builder.size_level = 0; builder.loop_vectorize = True; builder.slp_vectorize = True  # noqa: E702
        builder.populate(self.optimizer)
    super().__init__("compile_llvm_opt" if opt else "compile_llvm")
  def compile(self, src:str) -> bytes:
    mod = llvm.parse_assembly(src)
    mod.verify()
    self.optimizer.run(mod)
    if DEBUG >= 5: print(self.device.target_machine.emit_assembly(mod))
    return self.device.target_machine.emit_object(mod)

class LLVMProgram:
  def __init__(self, device:LLVMDevice, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    device.engine.add_object_file(llvm.object_file.ObjectFileRef.from_data(lib))
    self.fxn = device.engine.get_function_address(name)
    assert self.fxn != 0, "LLVM failed to get function address"

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    if not hasattr(self, 'cfunc'):
      self.cfunc = ctypes.CFUNCTYPE(ctypes.c_int, *([ctypes.c_void_p]*len(bufs)), *([ctypes.c_int32]*len(vals)))(self.fxn)
    return cpu_time_execution(lambda: self.cfunc(*bufs, *vals), enable=wait)

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    # this opt actually can change things. ex: opt=3 means no FMA, opt=2 means FMA
    self.target_machine: llvm.targets.TargetMachine = llvm.Target.from_triple(llvm.get_process_triple()).create_target_machine(opt=2)
    backing_mod = llvm.parse_assembly(str())
    backing_mod.triple = llvm.get_process_triple()
    self.engine: llvm.executionengine.ExecutionEngine = llvm.create_mcjit_compiler(backing_mod, self.target_machine)
    super().__init__(device, MallocAllocator, LLVMRenderer(), LLVMCompiler(self, getenv("LLVMOPT")), functools.partial(LLVMProgram, self))
