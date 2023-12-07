import ctypes
from typing import ClassVar, Tuple
from tinygrad.device import Compiled, MallocAllocator
from tinygrad.helpers import getenv, DEBUG, diskcache, cpu_time_execution
from ctypes import CFUNCTYPE
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.llvmir import uops_to_llvm_ir

import llvmlite.binding as llvm

LLVMOPT = bool(getenv("LLVMOPT"))

class LLVM:
  target_machine: ClassVar[llvm.targets.TargetMachine] = None
  engine: ClassVar[llvm.executionengine.ExecutionEngine] = None
  optimizer: ClassVar[llvm.passmanagers.ModulePassManager] = None

  def __init__(self):
    if LLVM.engine is not None: return
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    target = llvm.Target.from_triple(llvm.get_process_triple())
    LLVM.optimizer = llvm.create_module_pass_manager()
    LLVM.target_machine = target.create_target_machine(opt=2)  # this opt actually can change things. ex: opt=3 means no FMA, opt=2 means FMA
    LLVM.target_machine.add_analysis_passes(LLVM.optimizer)

    # TODO: this makes compile times so much faster
    if LLVMOPT:
      llvm.set_option(str(), '-force-vector-interleave=4')  # this makes sum the same speed as torch, it also doubles the (slow) conv speed
      if DEBUG >= 4: llvm.set_option(str(), '--debug-only=loop-vectorize')
      #llvm.set_option(str(), '--debug')

      # does this do anything?
      builder = llvm.create_pass_manager_builder()
      builder.opt_level = 3
      builder.size_level = 0
      builder.loop_vectorize = True
      builder.slp_vectorize = True
      builder.populate(LLVM.optimizer)

    LLVM.target_machine.set_asm_verbosity(True)
    backing_mod = llvm.parse_assembly(str())
    backing_mod.triple = llvm.get_process_triple()
    LLVM.engine = llvm.create_mcjit_compiler(backing_mod, LLVM.target_machine)

@diskcache
def compile_llvm(prg, llvmopt=LLVMOPT) -> bytes:
  mod = llvm.parse_assembly(prg)
  mod.verify()
  LLVM().optimizer.run(mod)
  if DEBUG >= 5: print(LLVM.target_machine.emit_assembly(mod))
  return LLVM.target_machine.emit_object(mod)

class LLVMProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    LLVM().engine.add_object_file(llvm.object_file.ObjectFileRef.from_data(lib))
    self.fxn = LLVM.engine.get_function_address(name)

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    self.cfunc = CFUNCTYPE(ctypes.c_int, *([ctypes.c_void_p]*len(bufs)), *([ctypes.c_int32]*len(vals)))(self.fxn)
    return cpu_time_execution(lambda: self.cfunc(*bufs, *vals), enable=wait)

LLVMDevice = Compiled(MallocAllocator, LinearizerOptions(supports_float4=False, has_local=False, has_shared=False), uops_to_llvm_ir, compile_llvm, LLVMProgram)
