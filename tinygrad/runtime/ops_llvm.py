from __future__ import annotations
import ctypes, functools
from typing import Tuple
from tinygrad.device import Compiled, MallocAllocator
from tinygrad.helpers import DEBUG, cpu_time_execution
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.llvmir import uops_to_llvm_ir
import llvmlite.binding as llvm

def compile_llvm(device, prg) -> bytes:
  mod = llvm.parse_assembly(prg)
  mod.verify()
  device.optimizer.run(mod)
  if DEBUG >= 5: print(device.target_machine.emit_assembly(mod))
  return device.target_machine.emit_object(mod)

class LLVMProgram:
  def __init__(self, device:LLVMDevice, name:str, lib:bytes):
    self.name, self.lib = name, lib
    device.engine.add_object_file(llvm.object_file.ObjectFileRef.from_data(lib))
    self.fxn = device.engine.get_function_address(name)

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    self.cfunc = ctypes.CFUNCTYPE(ctypes.c_int, *([ctypes.c_void_p]*len(bufs)), *([ctypes.c_int32]*len(vals)))(self.fxn)
    return cpu_time_execution(lambda: self.cfunc(*bufs, *vals), enable=wait)

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    self.optimizer: llvm.passmanagers.ModulePassManager = llvm.create_module_pass_manager()
    # this opt actually can change things. ex: opt=3 means no FMA, opt=2 means FMA
    self.target_machine: llvm.targets.TargetMachine = llvm.Target.from_triple(llvm.get_process_triple()).create_target_machine(opt=2)
    self.target_machine.add_analysis_passes(self.optimizer)
    self.target_machine.set_asm_verbosity(True)
    backing_mod = llvm.parse_assembly(str())
    backing_mod.triple = llvm.get_process_triple()
    self.engine: llvm.executionengine.ExecutionEngine = llvm.create_mcjit_compiler(backing_mod, self.target_machine)
    super().__init__(MallocAllocator, LinearizerOptions("LLVM", supports_float4=False, has_local=False, has_shared=False),
                     uops_to_llvm_ir, functools.partial(compile_llvm, self), "compile_llvm", functools.partial(LLVMProgram, self))
