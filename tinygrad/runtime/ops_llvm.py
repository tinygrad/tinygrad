import platform, llvmlite.binding as llvm
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.helpers import getenv, capstone_flatdump
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.runtime.support.elf import jit_loader

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    triple = llvm.Target.from_triple(f'{platform.machine()}-none-unknown-elf')
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    features = '+reserve-x18' if platform.machine() == 'arm64' else ''
    # this opt actually can change things. ex: opt=3 means no FMA, opt=2 means FMA
    self.target_machine: llvm.targets.TargetMachine = triple.create_target_machine(opt=2, reloc='pic', codemodel='default', features=features)
    super().__init__(device, MallocAllocator, LLVMRenderer(), LLVMJITCompiler(self, getenv("LLVMOPT")), CPUProgram)

class LLVMJITCompiler(Compiler):
  def __init__(self, dev:LLVMDevice, opt:bool=False):
    self.dev = dev
    self.optimizer: llvm.passmanagers.ModulePassManager = llvm.create_module_pass_manager()
    self.dev.target_machine.add_analysis_passes(self.optimizer)
    if opt:
      with llvm.create_pass_manager_builder() as builder:
        builder.opt_level = 3; builder.size_level = 0; builder.loop_vectorize = True; builder.slp_vectorize = True  # noqa: E702
        builder.populate(self.optimizer)
    super().__init__("compile_llvm_jit_opt" if opt else "compile_llvm_jit")

  def compile(self, src:str) -> bytes:
    mod = llvm.parse_assembly(src)
    mod.verify()
    self.optimizer.run(mod)
    obj = self.dev.target_machine.emit_object(mod)
    return jit_loader(obj)

  def disassemble(self, lib:bytes): return capstone_flatdump(lib)
