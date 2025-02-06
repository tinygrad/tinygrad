import ctypes, platform, sys
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.helpers import OSX, getenv, capstone_flatdump
from tinygrad.renderer.llvmir import LLVMRenderer
import tinygrad.runtime.autogen.llvm as llvm
from tinygrad.runtime.support.elf import jit_loader

def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))

def expect(x, err, ret=None):
  if x: raise RuntimeError(llvm.string_cast(err.contents) if not isinstance(err, str) else err)
  return ret

class LLVMCompiler(Compiler):
  def __init__(self, host_arch:str, opt:bool):
    for component in ['Target', 'TargetInfo', 'TargetMC', 'AsmPrinter']: getattr(llvm, f'LLVMInitialize{host_arch}{component}')()

    triple = {'AArch64': b'aarch64', 'X86': b'x86_64'}[host_arch] + b'-none-unknown-elf'
    target = expect(llvm.LLVMGetTargetFromTriple(triple, ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=cerr()), err, tgt)
    cpu, feats = ctypes.string_at(llvm.LLVMGetHostCPUName()), ctypes.string_at(llvm.LLVMGetHostCPUFeatures())
    # +reserve-x18 here does the same thing as -ffixed-x18 in ops_clang.py, see comments there for why it's needed on arm osx
    self.target_machine = llvm.LLVMCreateTargetMachine(target, triple, cpu, b'+reserve-x18,' + feats if OSX and host_arch == 'AArch64' else feats,
                                                       llvm.LLVMCodeGenLevelDefault, llvm.LLVMRelocPIC, llvm.LLVMCodeModelDefault)

    self.pbo = llvm.LLVMCreatePassBuilderOptions()
    if opt:
      self.passes = b'default<O2>'
      llvm.LLVMPassBuilderOptionsSetLoopUnrolling(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetLoopVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetSLPVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetVerifyEach(self.pbo, True)
    else:
      self.passes = b'default<O0>'

    super().__init__(f"compile_llvm_jit{'_opt' if opt else ''}")

  def __del__(self):
    llvm.LLVMDisposePassBuilderOptions(self.pbo)

  def compile(self, src:str) -> bytes:
    src_buf = llvm.LLVMCreateMemoryBufferWithMemoryRangeCopy(ctypes.create_string_buffer(src_bytes:=src.encode()), len(src_bytes), b'src')
    mod = expect(llvm.LLVMParseIRInContext(llvm.LLVMGetGlobalContext(), src_buf, ctypes.pointer(m:=llvm.LLVMModuleRef()), err:=cerr()), err, m)
    expect(llvm.LLVMVerifyModule(mod, llvm.LLVMReturnStatusAction, err:=cerr()), err)
    expect(llvm.LLVMRunPasses(mod, self.passes, self.target_machine, self.pbo), 'failed to run passes')
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(self.target_machine, mod, llvm.LLVMObjectFile, err:=cerr(),
                                                              ctypes.pointer(buf:=llvm.LLVMMemoryBufferRef())), err, buf)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))
    llvm.LLVMDisposeModule(mod)
    llvm.LLVMDisposeMemoryBuffer(obj_buf)
    return jit_loader(obj)

  def disassemble(self, lib:bytes): capstone_flatdump(lib)

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    compiler = LLVMCompiler({'arm64': 'AArch64', 'aarch64': 'AArch64', 'x86_64': 'X86', 'AMD64': 'X86'}[platform.machine()], bool(getenv("LLVMOPT")))
    super().__init__(device, MallocAllocator, LLVMRenderer('win64cc' if sys.platform == 'win32' else None), compiler, CPUProgram)
